"""
Router pipeline handlers — deterministic stages for the router pipeline.

Handles: extract_context (stage 1), select_context (stage 2),
         validate_response (stage 5).

Stages 3-4 (classify_action, synthesize_response) are LLM-driven and
handled by the pipeline compiler's standard Classify/Synthesize execution.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline_compiler import PipelineCompiler

logger = logging.getLogger(__name__)

HISTORY_DIR = Path(os.environ.get("MESH_HISTORY_DIR", str(Path.home() / ".mesh" / "history")))

# ── Stage 1: Extract Context ────────────────────────────────────────────

_MENTION_RE = re.compile(r"@(\w[\w-]*)")
_URGENCY_WORDS = {
    "urgent", "asap", "immediately", "critical", "broken", "down",
    "outage", "emergency", "now", "crash", "crashed", "fix",
}
_QUESTION_PATTERNS = [
    re.compile(r"\?\s*$", re.MULTILINE),
    re.compile(r"^(what|who|where|when|why|how|can you|could you|is there|are there|do we|did we|have we|should we)", re.IGNORECASE | re.MULTILINE),
]
_COMMAND_PATTERNS = [
    re.compile(r"^(run|start|stop|restart|kill|check|deploy|build|commit|push|pull|delete|remove|create|add|update|fix|rerun|re-run|launch|disable|enable)", re.IGNORECASE | re.MULTILINE),
]


def _extract_topic_keywords(text: str) -> list[str]:
    """Extract likely topic keywords from message text."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "don", "now", "and",
        "but", "or", "if", "that", "this", "these", "those", "it", "its",
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
        "them", "his", "her", "what", "which", "who", "whom", "up", "about",
        "also", "please", "thanks", "thank", "okay", "ok", "yeah", "yes",
        "hi", "hello", "hey", "let", "us", "get", "got", "go", "going",
        "want", "need", "think", "know", "see", "look", "make", "take",
        "come", "give", "tell", "say", "said", "try", "use", "right",
        "still", "well", "back", "like", "actually", "really", "though",
    }
    words = re.findall(r"[a-zA-Z_][\w.-]*", text.lower())
    keywords = []
    seen = set()
    for w in words:
        if w not in stop_words and len(w) > 2 and w not in seen:
            seen.add(w)
            keywords.append(w)
    return keywords[:30]


def _classify_intent(text: str) -> str:
    """Heuristic intent classification: question / command / status / greeting / statement."""
    stripped = text.strip()
    if not stripped:
        return "empty"
    lower = stripped.lower()

    if re.match(r"^(hi|hey|hello|morning|afternoon|evening|sup)\b", lower):
        if len(stripped) < 40:
            return "greeting"

    for pat in _QUESTION_PATTERNS:
        if pat.search(stripped):
            return "question"

    for pat in _COMMAND_PATTERNS:
        if pat.search(stripped):
            return "command"

    if any(w in lower for w in ("status", "update", "report", "check on", "how's", "how is")):
        return "status"

    return "statement"


def _coerce_task_payload(task: object) -> object:
    """Return a structured task payload when the pipeline input is JSON."""
    if isinstance(task, str):
        stripped = task.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return task
    return task


def _message_and_context(task: object) -> tuple[dict, dict]:
    payload = _coerce_task_payload(task)
    if isinstance(payload, dict):
        msg = payload.get("message", payload)
        context = payload.get("context", {})
        if isinstance(msg, dict):
            return msg, context if isinstance(context, dict) else {}
    return {}, {}


def handler_extract_context(inputs: dict, compiler: "PipelineCompiler") -> dict:
    """Stage 1: Parse raw message into structured context signal.

    Input bindings:
      task: raw message text (string or dict with message fields)
    """
    task = _coerce_task_payload(inputs.get("task", ""))

    if isinstance(task, dict):
        msg = task.get("message", task)
        content = msg.get("content", "")
        sender = msg.get("from_node", "unknown")
        to_node = msg.get("to_node", "")
        timestamp = msg.get("timestamp", "")
    elif isinstance(task, str):
        content = task
        sender = "unknown"
        to_node = ""
        timestamp = ""
    else:
        content = str(task)
        sender = "unknown"
        to_node = ""
        timestamp = ""

    sender_type = "user" if sender.startswith("user:") else (
        "agent" if sender.startswith("agent:") else "system"
    )
    is_channel = to_node.startswith("channel:") if to_node else False
    channel = to_node.split(":", 1)[1] if is_channel else ""

    mentions = _MENTION_RE.findall(content)
    keywords = _extract_topic_keywords(content)

    words_lower = set(content.lower().split())
    urgency = bool(words_lower & _URGENCY_WORDS)
    intent_type = _classify_intent(content)

    preview = content[:500] if len(content) > 500 else content

    return {
        "sender": sender,
        "sender_type": sender_type,
        "to_node": to_node,
        "channel": channel,
        "is_channel": is_channel,
        "mentions": mentions,
        "topic_keywords": keywords,
        "intent_type": intent_type,
        "urgency": urgency,
        "timestamp": timestamp,
        "content_preview": preview,
    }


# ── Stage 2: Select Context ────────────────────────────────────────────

def _search_memory(keywords: list[str], max_results: int = 10) -> list[dict]:
    """Call mesh-tool memory_search with topic keywords."""
    if not keywords:
        return []
    query = " ".join(keywords[:8])
    try:
        result = subprocess.run(
            ["mesh-tool", "memory_search", "--query", query, "--k", str(max_results)],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if isinstance(data, list):
                return [
                    {
                        "id": m.get("id", ""),
                        "summary": m.get("summary", ""),
                        "tags": m.get("tags", []),
                        "date": m.get("date", ""),
                        "score": m.get("score", 0),
                    }
                    for m in data[:max_results]
                ]
    except Exception as e:
        logger.warning(f"memory_search failed: {e}")
    return []


def _history_file_candidates(agent_name: str) -> list[Path]:
    """Return candidate legacy agent-history files for a router identity.

    AgentNode persists legacy history by nickname (`agent-{nickname}.json`),
    while router contexts often carry richer identities such as
    `sysadmin:bob`, `test:pipeline-router`, or `agent:test:pipeline-router`.
    Prefer exact matches, then the nickname suffix, then the filesystem-safe
    node-id form used by the base Node fallback.
    """
    raw = str(agent_name or "").strip()
    keys: list[str] = []

    def add(key: str) -> None:
        key = key.strip()
        if key and key not in keys:
            keys.append(key)

    add(raw)
    if raw.startswith("agent:"):
        add(raw.removeprefix("agent:"))
    if ":" in raw:
        add(raw.split(":")[-1])
        add(raw.replace(":", "-"))

    if not keys:
        keys.append("agent")
    return [HISTORY_DIR / f"agent-{key}.json" for key in keys]


def _history_file_for_agent(agent_name: str) -> Path:
    candidates = _history_file_candidates(agent_name)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _search_history_file(
    agent_name: str,
    keywords: list[str],
    max_snippets: int = 15,
    context_lines: int = 5,
) -> list[dict]:
    """Regex search over the agent's persisted conversation history file.

    Reads the JSONL history file line by line, searching message content
    for keyword matches. Returns snippets with surrounding context.
    """
    history_file = _history_file_for_agent(agent_name)
    if not history_file.exists():
        tried = ", ".join(str(p) for p in _history_file_candidates(agent_name))
        logger.info(f"History file not found for {agent_name!r}; tried: {tried}")
        return []

    if not keywords:
        return []

    pattern = re.compile(
        "|".join(re.escape(kw) for kw in keywords[:10]),
        re.IGNORECASE,
    )

    snippets = []
    try:
        with open(history_file) as f:
            for line_num, line in enumerate(f):
                if len(snippets) >= max_snippets:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg = entry.get("message", {})
                content = msg.get("content", "")
                if not content or len(content) < 10:
                    continue

                matches = pattern.findall(content)
                if not matches:
                    continue

                preview_start = max(0, content.lower().find(matches[0].lower()) - 200)
                preview_end = min(len(content), preview_start + 600)
                snippet_text = content[preview_start:preview_end]
                if preview_start > 0:
                    snippet_text = "..." + snippet_text
                if preview_end < len(content):
                    snippet_text = snippet_text + "..."

                snippets.append({
                    "line": line_num,
                    "from_node": msg.get("from_node", ""),
                    "to_node": msg.get("to_node", ""),
                    "timestamp": msg.get("timestamp", ""),
                    "direction": entry.get("direction", ""),
                    "matched_keywords": list(set(m.lower() for m in matches)),
                    "snippet": snippet_text,
                })
    except Exception as e:
        logger.warning(f"History search failed for {history_file}: {e}")

    return snippets


def _get_rolling_window(
    agent_name: str,
    window_size: int = 10,
    before_index: int | None = None,
) -> list[dict]:
    """Get the last N messages from the agent's history as rolling window.

    If before_index is provided, get N messages before that index.
    Otherwise, get the last N messages from the file.
    """
    history_file = _history_file_for_agent(agent_name)
    if not history_file.exists():
        return []

    try:
        entries = []
        with open(history_file) as f:
            if before_index is not None:
                for i, line in enumerate(f):
                    if i >= before_index:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                entries = entries[-window_size:]
            else:
                all_lines = []
                for line in f:
                    all_lines.append(line.strip())
                for line in all_lines[-window_size:]:
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        window = []
        for entry in entries:
            msg = entry.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, dict):
                content = str(content)
            if not content:
                continue
            ts = msg.get("timestamp", "")[:19]
            from_node = msg.get("from_node", "?")
            truncated = content[:200].replace("\n", " ")
            if len(content) > 200:
                truncated += "..."
            window.append(
                f'<message from="{from_node}" timestamp="{ts}">'
                f"{truncated}</message>"
            )
        return window
    except Exception as e:
        logger.warning(f"Rolling window failed for {history_file}: {e}")
        return []


# ── Tool Execution (for iterative classify loop) ─────────────────────


def _xml_escape_attr(text: str) -> str:
    """Escape text for use in XML attributes."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def execute_tool_calls(tool_calls: list[dict], timeout: int = 15) -> str:
    """Execute tool calls and return XML-formatted results with timestamp.

    Each tool call dict:
      - tool: "bash" | "mesh_status" | "agent_status" | "file_read"
      - command: (bash) shell command to run
      - target: (agent_status) agent node ID
      - path: (file_read) absolute file path

    Returns XML like:
      <tool_results timestamp="2026-06-18T01:30:00">
        <result tool="bash" command="pgrep -fa sobek" exit_code="0">
          output here
        </result>
      </tool_results>
    """
    now = datetime.datetime.now().isoformat()[:19]
    parts = []

    for tc in tool_calls[:5]:
        tool = tc.get("tool", "")

        if tool == "bash":
            command = tc.get("command", "")
            if not command:
                continue
            try:
                proc = subprocess.run(
                    ["bash", "-c", command],
                    capture_output=True, text=True, timeout=timeout,
                )
                output = (proc.stdout + proc.stderr).strip()
                if len(output) > 2000:
                    output = output[:2000] + "\n... (truncated)"
                parts.append(
                    f'  <result tool="bash" command="{_xml_escape_attr(command)}" '
                    f'exit_code="{proc.returncode}">\n{output}\n  </result>'
                )
            except subprocess.TimeoutExpired:
                parts.append(
                    f'  <result tool="bash" command="{_xml_escape_attr(command)}" '
                    f'error="timeout after {timeout}s" />'
                )
            except Exception as e:
                parts.append(
                    f'  <result tool="bash" command="{_xml_escape_attr(command)}" '
                    f'error="{_xml_escape_attr(str(e))}" />'
                )

        elif tool == "mesh_status":
            try:
                proc = subprocess.run(
                    ["mesh-tool", "mesh_status"],
                    capture_output=True, text=True, timeout=timeout,
                )
                output = proc.stdout.strip()
                if len(output) > 3000:
                    output = output[:3000] + "\n... (truncated)"
                parts.append(f'  <result tool="mesh_status">\n{output}\n  </result>')
            except Exception as e:
                parts.append(f'  <result tool="mesh_status" error="{_xml_escape_attr(str(e))}" />')

        elif tool == "agent_status":
            target = tc.get("target", "self")
            try:
                proc = subprocess.run(
                    ["mesh-tool", "agent_status", "--target", target],
                    capture_output=True, text=True, timeout=timeout,
                )
                output = proc.stdout.strip()
                if len(output) > 3000:
                    output = output[:3000] + "\n... (truncated)"
                parts.append(
                    f'  <result tool="agent_status" target="{_xml_escape_attr(target)}">'
                    f'\n{output}\n  </result>'
                )
            except Exception as e:
                parts.append(
                    f'  <result tool="agent_status" target="{_xml_escape_attr(target)}" '
                    f'error="{_xml_escape_attr(str(e))}" />'
                )

        elif tool == "file_read":
            fpath = tc.get("path", "")
            if not fpath:
                continue
            try:
                content = Path(fpath).read_text()
                if len(content) > 3000:
                    content = content[:3000] + "\n... (truncated)"
                parts.append(
                    f'  <result tool="file_read" path="{_xml_escape_attr(fpath)}">'
                    f'\n{content}\n  </result>'
                )
            except Exception as e:
                parts.append(
                    f'  <result tool="file_read" path="{_xml_escape_attr(fpath)}" '
                    f'error="{_xml_escape_attr(str(e))}" />'
                )

    if not parts:
        return f'<tool_results timestamp="{now}">\n  <result error="no valid tool calls" />\n</tool_results>'

    return f'<tool_results timestamp="{now}">\n' + "\n".join(parts) + "\n</tool_results>"


def handler_select_context(inputs: dict, compiler: "PipelineCompiler") -> dict:
    """Stage 2: Retrieve scoped memories, rolling window, and search history.

    Input bindings:
      extract_context: structured context from stage 1
      task: raw message (for agent name extraction)
      rolling_window: (optional) pre-extracted preceding messages
    """
    ctx = inputs.get("extract_context", {})
    task = inputs.get("task", "")
    msg, task_context = _message_and_context(task)
    keywords = ctx.get("topic_keywords", [])
    defaults = compiler.plan.config.get("defaults", {})
    max_memory = defaults.get("max_memory_results", 10)
    max_snippets = defaults.get("max_history_snippets", 15)
    snippet_lines = defaults.get("history_snippet_lines", 5)
    window_size = defaults.get("rolling_window_size", 25)

    nickname = str(task_context.get("nickname") or task_context.get("agent_nickname") or "")
    agent_name = str(task_context.get("agent_name") or "")
    node_id = str(task_context.get("node_id") or task_context.get("agent_node_id") or "")

    if not nickname or not agent_name:
        to_node = msg.get("to_node", "") if msg else ""
        if to_node.startswith("agent:"):
            parts = to_node.split(":")
            if not nickname and parts:
                nickname = parts[-1]
            if not agent_name and len(parts) >= 3:
                agent_name = ":".join(parts[1:])
    nickname = nickname or "bob"
    agent_name = agent_name or nickname

    rolling_window = inputs.get("rolling_window")
    if rolling_window is None:
        trigger_index = inputs.get("trigger_index")
        rolling_window = _get_rolling_window(nickname, window_size, trigger_index)

    memories = _search_memory(keywords, max_memory)
    history_snippets = _search_history_file(
        agent_name, keywords, max_snippets, snippet_lines
    )

    return {
        "memories": memories,
        "history_snippets": history_snippets,
        "rolling_window": rolling_window,
        "current_time": task_context.get("current_time") or datetime.datetime.now().isoformat()[:19],
        "personality": task_context.get("personality", ""),
        "agent_tools": task_context.get("agent_tools", ""),
        "project_context": task_context.get("project_context", ""),
        "todo_context": task_context.get("todo_context", ""),
        "project_maps": task_context.get("project_maps", []),
        "relevant_memories": task_context.get("relevant_memories", ""),
        "system_prompt": task_context.get("system_prompt", ""),
        "memory_toc": task_context.get("memory_toc", ""),
        "conversation_summary": task_context.get("conversation_summary", ""),
        "worker_status": task_context.get("worker_status", ""),
        "agent_name": agent_name,
        "nickname": nickname,
        "node_id": node_id,
    }


def AUTO_PROMPT_VARS(step_name: str, inputs: dict) -> dict[str, object]:
    select_context = inputs.get("select_context", {})
    if not isinstance(select_context, dict):
        select_context = {}
    return {
        "agent_name": select_context.get("agent_name", "agent"),
        "nickname": select_context.get("nickname", "agent"),
        "classify_history": inputs.get("classify_history", "(no prior passes)"),
    }


# ── Stage 5: Validate Response ─────────────────────────────────────────

_XML_LEAK_PATTERNS = [
    re.compile(r"<tool_call\b", re.IGNORECASE),
    re.compile(r"<thinking\b", re.IGNORECASE),
    re.compile(r"<anthr?opic", re.IGNORECASE),
    re.compile(r"<function_calls?\b", re.IGNORECASE),
    re.compile(r"<mesh_call\b", re.IGNORECASE),
    re.compile(r"<system-reminder\b", re.IGNORECASE),
    re.compile(r"</?(invoke|result|parameter)\b", re.IGNORECASE),
]

_VALID_ACTIONS = {"respond", "dispatch", "sleep", "tool_call"}


def handler_validate_response(inputs: dict, compiler: "PipelineCompiler") -> dict:
    """Stage 5: Validate synthesized output for common failure modes.

    Input bindings:
      synthesize_response: response text from stage 4
      classify_action: action decision from stage 3
    """
    synth = inputs.get("synthesize_response", {})
    action_data = inputs.get("classify_action", {})

    if isinstance(synth, dict):
        response_text = synth.get("response_text", "")
    elif isinstance(synth, str):
        response_text = synth
    else:
        response_text = str(synth)

    if isinstance(action_data, dict):
        action = str(action_data.get("action") or "respond").strip().lower()
        task_spec = str(action_data.get("task_spec") or "")
        tool_plan = str(action_data.get("tool_plan") or "")
    else:
        action = "respond"
        task_spec = ""
        tool_plan = ""

    tools_executed = []
    if isinstance(synth, dict):
        tools_executed = synth.get("tools_executed", [])

    issues = []

    if action not in _VALID_ACTIONS:
        issues.append(f"invalid_action: '{action}' not in {_VALID_ACTIONS}")
        action = "respond"

    if not response_text.strip():
        if action == "dispatch" and task_spec.strip():
            response_text = "On it. I am dispatching this as a worker task."
        elif action == "tool_call" and tool_plan.strip():
            response_text = "I need to check current state before answering."
        elif action == "respond":
            response_text = (
                "I could not synthesize a full response from the router pipeline "
                "output. Please retry or ask me to check directly."
            )

    for pat in _XML_LEAK_PATTERNS:
        if pat.search(response_text):
            issues.append(f"xml_leak: {pat.pattern}")

    if action == "respond" and not response_text.strip():
        issues.append("empty_response: action is 'respond' but response text is empty")

    if action == "dispatch" and not task_spec.strip():
        issues.append("empty_task_spec: action is 'dispatch' but no task spec provided")

    if action == "tool_call" and not tool_plan.strip():
        issues.append("empty_tool_plan: action is 'tool_call' but no tool_plan provided")

    if len(response_text) > 20000:
        issues.append(f"response_too_long: {len(response_text)} chars")

    return {
        "valid": len(issues) == 0,
        "response_text": response_text,
        "action": action,
        "task_spec": task_spec,
        "tool_plan": tool_plan,
        "tools_executed": tools_executed,
        "issues": issues,
    }


# ── Handler Registration ────────────────────────────────────────────────

HANDLERS = {
    "extract_context": handler_extract_context,
    "select_context": handler_select_context,
    "validate_response": handler_validate_response,
}
