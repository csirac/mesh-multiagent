"""
Core auto-tool loop — extracted from agent_node.py.

This is a vendor-neutral TAOR (Think-Act-Observe-Repeat) loop that:
1. Sends history + system prompt to an LLM via mesh.llm.LLMClient
2. Parses tool calls from the response
3. Executes tool calls via mesh.tools.ToolRegistry
4. Appends results to history
5. Repeats until the LLM produces a final text response or hits max iterations

All output is emitted as JSONL events on stdout (see protocol.py).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

import aiohttp

from ..llm import LLMClient, LLMConfig, HistoryMessage, estimate_history_tokens, estimate_tokens
from ..tools import ToolRegistry, ToolCall, parse_tool_calls, has_tool_call
from . import protocol
from .tools import PLANNER_TOOLS, MESH_READ_ONLY_TOOLS
from .tools.phase_complete import PhaseCompleteSignal

AGENT_LOCAL_TOOLS = {
    "send_message", "channel_list", "channel_members",
    "schedule_wake", "schedule_list", "schedule_cancel",
    "agent_shutdown", "mesh_status", "agent_status",
}

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 100
DEFAULT_SOFT_LIMIT = 500_000  # tokens
KEEP_RECENT_RESULTS = 4
FORCED_SYNTHESIS_FRACTION = 0.97
LLM_RETRIES = 4
LLM_RETRY_BACKOFF = (2.0, 4.0, 8.0, 16.0)
TRANSIENT_HTTP_CODES = {400, 429, 500, 502, 503, 504}

WRITE_TOOLS = {"apply_patch", "file_edit", "file_create", "file_write"}
SHELL_WRITE_PATTERNS = re.compile(
    r"""\b(sed\s+-i|tee\s|patch\s|mv\s|rm\s|cp\s|chmod\s|chown\s|mkdir\s|touch\s)"""
    r"""|[^|]>(?!&)"""
    r"""|>>"""
)


def _is_write_call(tc: ToolCall) -> bool:
    if tc.name in WRITE_TOOLS:
        return True
    if tc.name == "shell":
        cmd = tc.arguments.get("command", "")
        return bool(SHELL_WRITE_PATTERNS.search(cmd))
    return False


_HALLUCINATED_MESH_CALL_RE = re.compile(r'<mesh_call\s+name="([^"]+)"')
_HALLUCINATED_BARE_TAG_RE = re.compile(
    r'<(apply_patch|file_read|file_edit|shell|list_dir|file_create|file_write)\b'
)
_TOOL_RESULT_RE = re.compile(r'<tool_result name="([^"]+)">\n?(.*?)\n?</tool_result>', re.DOTALL)
_FILE_PATH_RE = re.compile(r'(/[^\s<>"\']+\.\w+)')


def _detect_hallucinated_tools(
    response_text: str, available_tools: list[str] | None
) -> list[str]:
    """Return sorted, deduplicated list of tool names found in *response_text*
    that are NOT in *available_tools*.

    If *available_tools* is ``None`` (meaning all tools are available),
    nothing can be hallucinated — return ``[]``.
    """
    if available_tools is None:
        return []

    found: set[str] = set()
    for m in _HALLUCINATED_MESH_CALL_RE.finditer(response_text):
        found.add(m.group(1))
    for m in _HALLUCINATED_BARE_TAG_RE.finditer(response_text):
        found.add(m.group(1))

    allowed = set(available_tools)
    hallucinated = sorted(found - allowed)
    return hallucinated


READ_ONLY_TOOLS = {"file_read", "list_dir", "grep", "find_files"}

DEFAULT_CHECKPOINT_PROMPT = (
    "CHECKPOINT: Answer both questions.\n"
    "Line 1: Have you completed the deliverables for this phase? YES or NO\n"
    "Line 2: Are you stuck in a degenerate loop (reading the same files repeatedly "
    "without making progress)? YES or NO\n"
    "Line 3+: Brief rationale."
)

ASSESSMENT_MAX_RETRIES = 2
PLAN_MAX_RETRIES = 2
HALLUCINATION_MAX_RETRIES = 2

_ASSESSMENT_RETRY_PREFIX = (
    "**YOUR PREVIOUS RESPONSE WAS MALFORMED.** The harness could not parse it. "
    "Do NOT write <mesh_call> or any tool-call XML — you have no tools. "
    "Output ONLY the <assessment> block with <decision>, <reasoning>, "
    "and either <next_phase_prompt> (continue) or <summary> (done). "
    "Nothing else.\n\n"
)

_PLAN_RETRY_PREFIX = (
    "**YOUR PREVIOUS RESPONSE DID NOT CONTAIN THE REQUIRED TAGS.** "
    "You MUST include <phase_1_prompt>...</phase_1_prompt> tags in your output. "
    "Do NOT write tool-call XML — you have no tools. "
    "Write your plan in plain text, then wrap the Phase 1 executor prompt "
    "in <phase_1_prompt> tags at the end.\n\n"
)


def _is_malformed_assessment(decision: AssessmentDecision) -> bool:
    """Check if an assessment was malformed (fell through to default done)."""
    return decision.reasoning.startswith("[Malformed assessment")


def _build_evaluator_context(
    task_prompt: str,
    plan: str,
    history: list[HistoryMessage],
    phase_start_idx: int,
    phase_number: int,
    phase_prompt: str = "",
    prior_phase_summaries: list[dict[str, str]] | None = None,
) -> str:
    """Build structured XML context for checkpoint and assessment evaluators.

    Produces a single XML document with task, plan, prior phase summaries,
    current phase objective, tool history, and files modified so evaluators
    audit actual evidence, not raw history.  Tool results are copied
    faithfully — truncation is the executor's responsibility (via
    max_result_chars).
    """

    parts = ["<evaluator_context>"]
    parts.append(f"<task>\n{task_prompt}\n</task>")
    if plan:
        parts.append(f"<plan>\n{plan}\n</plan>")

    # Prior phase summaries — prefer explicit list, fall back to history scan
    if prior_phase_summaries:
        parts.append("<prior_phases>")
        for ps in prior_phase_summaries:
            p_num = ps.get("phase", "?")
            p_summary = ps.get("summary", "")
            parts.append(f'<phase_summary phase="{p_num}">\n{p_summary}\n</phase_summary>')
        parts.append("</prior_phases>")
    else:
        scanned = []
        for msg in history[:phase_start_idx]:
            if "<phase_summary" in msg.content:
                scanned.append(msg.content)
        if scanned:
            parts.append("<prior_phases>")
            parts.extend(scanned)
            parts.append("</prior_phases>")

    # Current phase with objective, tool history, and files modified
    current_msgs = history[phase_start_idx:]
    if current_msgs or phase_prompt:
        parts.append(f'<current_phase phase="{phase_number}">')
        if phase_prompt:
            parts.append(f"<objective>\n{phase_prompt}\n</objective>")

        # Extract structured tool history (faithful copy of executor results)
        files_modified: set[str] = set()
        tool_entries: list[str] = []
        for msg in current_msgs:
            # Parse tool results from message content
            for m in _TOOL_RESULT_RE.finditer(msg.content):
                tool_name = m.group(1)
                result_text = m.group(2)
                tool_entries.append(f'<tool_call name="{tool_name}">\n{result_text}\n</tool_call>')

                # Track files modified by write tools
                if tool_name in WRITE_TOOLS:
                    for fp in _FILE_PATH_RE.findall(m.group(2)):
                        files_modified.add(fp)

            # Also include agent reasoning turns (non-tool content)
            if msg.from_node not in ("system", "tool") and not _TOOL_RESULT_RE.search(msg.content):
                tool_entries.append(f'<turn from="{msg.from_node}">\n{msg.content}\n</turn>')

        if tool_entries:
            parts.append("<tool_history>")
            parts.extend(tool_entries)
            parts.append("</tool_history>")

        if files_modified:
            parts.append("<files_modified>")
            for fp in sorted(files_modified):
                parts.append(f"  {fp}")
            parts.append("</files_modified>")

        parts.append("</current_phase>")

    parts.append("</evaluator_context>")
    return "\n".join(parts)


@dataclass
class CheckpointConfig:
    """Configuration for structured checkpoint evaluators."""
    task_prompt: str
    plan: str
    phase_number: int
    phase_start_idx: int
    phase_prompt: str = ""


_WATCHDOG_PROMPT = """\
{evaluator_context}

WATCHDOG: You are evaluating an autonomous code executor for progress.
The full tool history for this phase is shown above. Recent activity
(since turn {recent_since}) is demarcated with <recent_activity>.
Based on the FULL history and especially the recent activity, is the
executor making meaningful progress toward the phase objective?

Consider:
- Is it exploring new ground or repeating the same actions?
- Is there a clear trajectory toward completing the objective?
- Are tool calls productive (edits, tests, targeted reads) vs. circular?
- Has the executor modified any files, or only been reading?

Line 1: YES (making progress) or NO (stuck/degenerate)
Line 2+: Brief rationale (1-2 sentences)."""

WATCHDOG_MAX_CONTEXT_TOKENS = 100_000


class CCWatchdog:
    """LLM-based progress watchdog for CC executor sessions.

    Called periodically (every 8 CC turns) with new turn batches.
    Accumulates all turns across calls and builds full phase history
    (same XML format as _build_evaluator_context) with a
    <recent_activity> demarcation for the latest window.
    Uses an assessor LLM to evaluate whether the executor is making
    progress.  Returns True (kill) after 2 consecutive NO evaluations.
    """

    RECENT_WINDOW = 16

    def __init__(
        self,
        assessor: LLMClient,
        task_prompt: str,
        plan: str,
        phase_prompt: str,
    ):
        self.assessor = assessor
        self.task_prompt = task_prompt
        self.plan = plan
        self.phase_prompt = phase_prompt
        self._consecutive_no = 0
        self._all_turns: list[list[dict]] = []
        self._prev_turn_count = 0

    async def __call__(self, turns: list[list[dict]]) -> bool:
        """Evaluate executor progress. Returns True to kill the CC process."""
        self._all_turns.extend(turns)
        prompt = self._build_prompt()
        try:
            response = await self.assessor.complete(prompt)
            _wdog_usage = getattr(self.assessor, "_last_usage", None)
            if isinstance(_wdog_usage, dict):
                protocol.emit_usage(_wdog_usage)
        except Exception as e:
            logger.warning("CC watchdog assessor call failed: %s — defaulting to continue", e)
            return False

        first_line = (response or "").strip().split("\n")[0].strip().upper()
        making_progress = first_line.startswith("YES")

        if making_progress:
            logger.info("CC watchdog: YES (progress) — resetting counter")
            self._consecutive_no = 0
            self._prev_turn_count = len(self._all_turns)
            return False

        self._consecutive_no += 1
        logger.info(
            "CC watchdog: NO (%d/2 consecutive) — %s",
            self._consecutive_no,
            (response or "").strip()[:200],
        )
        self._prev_turn_count = len(self._all_turns)
        return self._consecutive_no >= 2

    def _build_prompt(self) -> str:
        recent_since = max(1, len(self._all_turns) - self.RECENT_WINDOW + 1)
        ctx = self._build_evaluator_context(recent_since)

        tok_estimate = estimate_tokens(ctx)
        if tok_estimate > WATCHDOG_MAX_CONTEXT_TOKENS:
            ctx = self._truncate_oldest(ctx, tok_estimate)

        return _WATCHDOG_PROMPT.format(
            evaluator_context=ctx,
            recent_since=recent_since,
        )

    def _build_evaluator_context(self, recent_since: int) -> str:
        parts = ["<evaluator_context>"]
        parts.append(f"<task>\n{self.task_prompt}\n</task>")
        if self.plan:
            parts.append(f"<plan>\n{self.plan}\n</plan>")
        parts.append(f'<current_phase phase="executor">')
        parts.append(f"<objective>\n{self.phase_prompt}\n</objective>")

        files_modified: set[str] = set()
        parts.append(f"<tool_history total_turns=\"{len(self._all_turns)}\">")

        for i, turn in enumerate(self._all_turns, 1):
            if i == recent_since:
                parts.append(f'<recent_activity since_turn="{recent_since}">')

            for evt in turn:
                if evt["type"] == "call":
                    parts.append(
                        f'<tool_call name="{evt["name"]}" turn="{i}">\n'
                        f'<input>{evt["input"]}</input>'
                    )
                elif evt["type"] == "result":
                    parts.append(
                        f'<result>{evt["output"]}</result>\n'
                        f'</tool_call>'
                    )
                    if evt["name"] in WRITE_TOOLS:
                        for fp in _FILE_PATH_RE.findall(str(evt["output"])):
                            files_modified.add(fp)

        if recent_since <= len(self._all_turns):
            parts.append("</recent_activity>")

        parts.append("</tool_history>")

        if files_modified:
            parts.append("<files_modified>")
            for fp in sorted(files_modified):
                parts.append(f"  {fp}")
            parts.append("</files_modified>")

        parts.append("</current_phase>")
        parts.append("</evaluator_context>")
        return "\n".join(parts)

    def _truncate_oldest(self, ctx: str, tok_estimate: int) -> str:
        """Re-build context with oldest turn results truncated to fit budget."""
        excess = tok_estimate - WATCHDOG_MAX_CONTEXT_TOKENS
        recent_since = max(1, len(self._all_turns) - self.RECENT_WINDOW + 1)
        chars_to_trim = excess * 4

        trimmed = 0
        for turn in self._all_turns[: recent_since - 1]:
            if trimmed >= chars_to_trim:
                break
            for evt in turn:
                if evt["type"] == "result" and len(str(evt["output"])) > 200:
                    original = str(evt["output"])
                    evt["output"] = original[:100] + f"\n[TRUNCATED: {len(original)} chars]\n" + original[-100:]
                    trimmed += len(original) - 200
                    if trimmed >= chars_to_trim:
                        break

        return self._build_evaluator_context(recent_since)


def _is_transient(exc: Exception) -> bool:
    """Check if an exception is a transient HTTP error worth retrying."""
    # Malformed provider responses (missing choices, etc.) — emit by mesh.llm
    # for OpenRouter load-balanced backends that occasionally return empty bodies.
    if type(exc).__name__ == "MalformedResponseError":
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in TRANSIENT_HTTP_CODES
    if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout,
                        httpx.PoolTimeout, ConnectionError, TimeoutError)):
        return True
    # KeyError on top-level keys ('choices', 'message', etc.) — defensive: even
    # if upstream code paths regress and stop wrapping these, retry rather than crash.
    if isinstance(exc, (KeyError, IndexError)):
        return True
    msg = str(exc).lower()
    for code in TRANSIENT_HTTP_CODES:
        if f"http {code}" in msg or f"{code}" in msg and ("error" in msg or "status" in msg):
            return True
    return False


def _backoff_for_attempt(attempt: int, exc: Exception | None = None) -> float:
    """Pick the backoff for the given attempt.

    On HTTP 429 with a retry-after header, honor the header (capped at 60s).
    Otherwise use the configured exponential backoff schedule.
    """
    if exc is not None and isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        header = exc.response.headers.get("retry-after")
        if header:
            try:
                return min(60.0, max(0.0, float(header.strip())))
            except (ValueError, TypeError):
                pass
    return LLM_RETRY_BACKOFF[min(attempt, len(LLM_RETRY_BACKOFF) - 1)]


def _manage_in_flight_context(
    history: list[HistoryMessage],
    soft_limit: int = DEFAULT_SOFT_LIMIT,
    keep_recent: int = KEEP_RECENT_RESULTS,
    threshold_fraction: float = 0.8,
) -> tuple[list[HistoryMessage], int]:
    """Prune old in-flight tool results when context grows too large.

    Returns (possibly-pruned history, number of entries pruned).
    """
    threshold = int(soft_limit * threshold_fraction)
    estimated = estimate_history_tokens(history)

    if estimated <= threshold:
        return history, 0

    in_flight_indices = [
        i for i, msg in enumerate(history)
        if getattr(msg, "source", "persisted") == "in_flight"
    ]

    if len(in_flight_indices) <= keep_recent:
        return history, 0

    indices_to_prune = set(in_flight_indices[:-keep_recent])

    pruned: list[HistoryMessage] = []
    pruned_count = 0
    for i, msg in enumerate(history):
        if i in indices_to_prune:
            pruned_count += 1
        else:
            pruned.append(msg)

    if pruned_count > 0:
        insert_idx = None
        for i, msg in enumerate(pruned):
            if getattr(msg, "source", "persisted") == "in_flight":
                insert_idx = i
                break
        if insert_idx is not None:
            marker = HistoryMessage(
                from_node="system",
                content=f"[{pruned_count} previous tool result(s) omitted for context management]",
                timestamp=history[0].timestamp if history else "",
                source="in_flight",
            )
            pruned.insert(insert_idx, marker)

        before = estimated
        after = estimate_history_tokens(pruned)
        protocol.emit_context_pruned(pruned_count, before, after)
        logger.info(f"Pruned {pruned_count} in-flight entries: {before} -> {after} tokens")

    return pruned, pruned_count


def _truncate_extreme_result(result: str, soft_limit: int = DEFAULT_SOFT_LIMIT) -> str:
    """Cap a single tool result at soft_limit * 3 chars."""
    max_chars = soft_limit * 3
    if len(result) <= max_chars:
        return result
    original_size = len(result)
    truncated = result[:max_chars]
    marker = f"\n\n[TRUNCATED: Original size {original_size:,} chars, kept first {max_chars:,} chars]"
    logger.warning(f"Extreme result truncation: {original_size:,} -> {max_chars:,} chars")
    return truncated + marker


async def _call_agent_socket(socket_path: str, name: str, arguments: dict) -> str:
    """Route a tool call to the parent agent_node via Unix domain socket."""
    connector = aiohttp.UnixConnector(path=socket_path)
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                "http://localhost/tool",
                json={"name": name, "arguments": arguments},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                return data.get("result", "No result")
    finally:
        await connector.close()


async def _execute_tool_calls(
    tool_calls: list[ToolCall],
    registry: ToolRegistry,
    agent_socket_path: str | None = None,
    max_result_chars: int = 0,
    effective_tool_names: list[str] | None = None,
    force_synthesis: bool = False,
    read_tools_stripped: bool = False,
) -> list[tuple[str, str, bool]]:
    """Execute tool calls and return list of (name, result_str, success)."""
    _allowed = set(effective_tool_names) if effective_tool_names is not None else None
    results: list[tuple[str, str, bool]] = []
    for call in tool_calls:
        call_id = getattr(call, "call_id", None) or uuid.uuid4().hex[:8]
        protocol.emit_tool_call(call.name, call.arguments, call_id)

        if _allowed is not None and call.name not in _allowed:
            if force_synthesis:
                err = (
                    f"DISALLOWED: '{call.name}' is NOT available. "
                    f"Your context budget is exhausted and all tools are disabled. "
                    f"Produce your final response now using what you have gathered."
                )
            elif read_tools_stripped and call.name in READ_ONLY_TOOLS:
                err = (
                    f"DISALLOWED: '{call.name}' is NOT available this turn. "
                    f"You appear to be stuck in a degenerate read-only loop, "
                    f"so read tools are disabled until you make a write. "
                    f"Use file_edit, apply_patch, or shell to make progress."
                )
            else:
                err = (
                    f"DISALLOWED: '{call.name}' is not in your available toolset. "
                    f"Available tools: {sorted(_allowed)}."
                )
            logger.warning(
                "Blocked disallowed tool '%s' (allowed_count=%d, has_socket=%s)",
                call.name, len(_allowed), bool(agent_socket_path),
            )
            protocol.emit_tool_result(call.name, err, call_id, success=False)
            results.append((call.name, err, False))
            continue

        # Try local harness tool first, then delegate to agent socket
        tool_def = registry.get(call.name)

        if tool_def is not None and tool_def.handler is not None:
            pass  # fall through to local execution below
        elif agent_socket_path:
            # Delegate to parent agent_node via socket (mesh tools, agent-local tools)
            try:
                result_str = await _call_agent_socket(
                    agent_socket_path, call.name, call.arguments,
                )
                if max_result_chars and len(result_str) > max_result_chars:
                    err = (
                        f"Error: tool output too large ({len(result_str):,} chars, "
                        f"limit is {max_result_chars:,} chars). "
                        f"Use more specific arguments to reduce output size — "
                        f"e.g., read a specific line range, filter results, or query a subset of the data."
                    )
                    logger.warning(f"Size gate on '{call.name}': {len(result_str):,} > {max_result_chars:,}")
                    protocol.emit_tool_result(call.name, err, call_id, success=False)
                    results.append((call.name, err, False))
                else:
                    protocol.emit_tool_result(call.name, result_str[:2000], call_id, success=True)
                    results.append((call.name, result_str, True))
            except Exception as e:
                err = f"Error calling agent socket for '{call.name}': {e}"
                logger.error(err)
                protocol.emit_tool_result(call.name, err, call_id, success=False)
                results.append((call.name, err, False))
            continue
        else:
            err = f"Error: Unknown tool '{call.name}'"
            if tool_def is not None:
                err = f"Error: Tool '{call.name}' has no handler and no agent socket available"
            results.append((call.name, err, False))
            continue

        try:
            if asyncio.iscoroutinefunction(tool_def.handler):
                result = await tool_def.handler(**call.arguments)
            else:
                result = await asyncio.to_thread(tool_def.handler, **call.arguments)
            result_str = str(result)
            if max_result_chars and len(result_str) > max_result_chars:
                err = (
                    f"Error: tool output too large ({len(result_str):,} chars, "
                    f"limit is {max_result_chars:,} chars). "
                    f"Use more specific arguments to reduce output size — "
                    f"e.g., read a specific line range, filter results, or query a subset of the data."
                )
                logger.warning(f"Size gate on '{call.name}': {len(result_str):,} > {max_result_chars:,}")
                protocol.emit_tool_result(call.name, err, call_id, success=False)
                results.append((call.name, err, False))
            else:
                protocol.emit_tool_result(call.name, result_str[:2000], call_id, success=True)
                results.append((call.name, result_str, True))
        except PhaseCompleteSignal:
            raise
        except TypeError as e:
            err = f"Error: Invalid arguments for '{call.name}': {e}"
            protocol.emit_tool_result(call.name, err, call_id, success=False)
            results.append((call.name, err, False))
        except Exception as e:
            err = f"Error executing '{call.name}': {e}"
            logger.exception(err)
            protocol.emit_tool_result(call.name, err, call_id, success=False)
            results.append((call.name, err, False))

    return results


async def run_loop(
    llm_client: LLMClient,
    history: list[HistoryMessage],
    system_prompt: str,
    tool_registry: ToolRegistry,
    tool_names: list[str] | None = None,
    node_id: str = "harness",
    max_iterations: int = MAX_ITERATIONS,
    soft_limit: int = DEFAULT_SOFT_LIMIT,
    mcp_config: str | None = None,
    instructions: str = "",
    agent_socket_path: str | None = None,
    controller_mode: str = "standard",
    checkpoint_enabled: bool = False,
    checkpoint_prompt: str = DEFAULT_CHECKPOINT_PROMPT,
    checkpoint_config: CheckpointConfig | None = None,
    checkpoint_llm_client: LLMClient | None = None,
    checkpoint_periodic_interval: int = 8,
    checkpoint_periodic_start: int = 8,
    assessor_llm_client: LLMClient | None = None,
    cc_watchdog: "Any | None" = None,
    codex_assessor_config: dict[str, str] | None = None,
) -> str:
    """Run the auto-tool loop until completion.

    Args:
        llm_client: Configured LLMClient instance.
        history: Initial conversation history (mutated in place).
        system_prompt: System prompt for the LLM.
        tool_registry: Registry of available tools.
        tool_names: Subset of tool names to enable (None = all).
        node_id: Identity string passed to format_history_xml.
        max_iterations: Safety cap on loop iterations.
        soft_limit: Token threshold for context pruning.
        mcp_config: Optional MCP JSON config (CC/zai backends).
        instructions: Task-specific instructions appended to prompt.
        agent_socket_path: Unix socket for routing agent-local tools to parent.
        controller_mode: "standard" (TAOR), "plan_and_execute", or "decompose".
        checkpoint_enabled: If True, fire toolless "are you done?" checkpoints
            when degeneration signals are detected (post-write stall or
            duplicate read). Breaks the loop on YES.
        checkpoint_prompt: The question asked during checkpoints. Callers
            can customize this per-phase (e.g. observe vs execute).
        checkpoint_llm_client: Optional separate LLMClient for checkpoint
            calls. Falls back to llm_client when None.
        assessor_llm_client: Optional separate LLMClient for assessor/judge
            calls. Threaded through to _run_plan_and_execute_loop.
        cc_watchdog: Optional CCWatchdog instance for CC executor sessions.
            Monitors streaming tool activity and kills on degenerate loops.

    Returns:
        The final assistant text response.
    """
    if controller_mode == "plan_and_execute":
        return await _run_plan_and_execute_loop(
            llm_client=llm_client,
            history=history,
            system_prompt=system_prompt,
            tool_registry=tool_registry,
            tool_names=tool_names,
            node_id=node_id,
            soft_limit=soft_limit,
            mcp_config=mcp_config,
            instructions=instructions,
            agent_socket_path=agent_socket_path,
            assessor_llm_client=assessor_llm_client,
            codex_assessor_config=codex_assessor_config,
        )

    if controller_mode == "decompose":
        return await _run_decompose_loop(
            llm_client=llm_client,
            history=history,
            system_prompt=system_prompt,
            tool_registry=tool_registry,
            tool_names=tool_names,
            node_id=node_id,
            soft_limit=soft_limit,
            mcp_config=mcp_config,
            instructions=instructions,
            agent_socket_path=agent_socket_path,
            assessor_llm_client=assessor_llm_client,
            codex_assessor_config=codex_assessor_config,
        )

    # Standard mode with native multi-turn for OpenAI backends
    if llm_client.config.backend == "openai":
        native_messages: list[dict] = []
        tool_prompt = tool_registry.format_tools_prompt(tool_names, backend="openai")
        sys_content = system_prompt
        if tool_prompt:
            sys_content += "\n\n" + tool_prompt
        native_messages.append({"role": "system", "content": sys_content})
        task_content = "\n\n".join(msg.content for msg in history)
        if instructions:
            task_content += "\n\n" + instructions
        native_messages.append({"role": "user", "content": task_content})
        return await run_native_loop(
            llm_client=llm_client,
            messages=native_messages,
            tool_registry=tool_registry,
            tool_names=tool_names,
            node_id=node_id,
            max_iterations=max_iterations,
            soft_limit=soft_limit,
            instructions=instructions,
            agent_socket_path=agent_socket_path,
            checkpoint_enabled=checkpoint_enabled,
            checkpoint_prompt=checkpoint_prompt,
            checkpoint_config=checkpoint_config,
            checkpoint_llm_client=checkpoint_llm_client,
            checkpoint_periodic_interval=checkpoint_periodic_interval,
            checkpoint_periodic_start=checkpoint_periodic_start,
        )

    thread_id = uuid.uuid4().hex[:12]
    backend = llm_client.config.backend
    model = llm_client.config.model
    protocol.emit_thread_started(thread_id, backend, model)

    cumulative_usage: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "llm_calls": 0,
        "backend": backend,
        "model": model,
    }

    is_cc = backend in ("claude-code", "zai")
    final_text = ""
    _loop_start = time.monotonic()
    _last_heartbeat = _loop_start
    synthesis_threshold = int(soft_limit * FORCED_SYNTHESIS_FRACTION)

    # Checkpoint degeneration trackers
    CHECKPOINT_STALL_TURNS = 3
    CHECKPOINT_PERIODIC_INTERVAL = checkpoint_periodic_interval
    CHECKPOINT_PERIODIC_START = checkpoint_periodic_start
    DEGENERATE_ESCALATION_LIMIT = 3
    _ckpt_has_written = False
    _ckpt_read_only_streak = 0
    _ckpt_last_tool_sig: tuple[str, ...] | None = None
    _read_tools_stripped = False
    _consecutive_degenerate = 0

    for iteration in range(1, max_iterations + 1):
        _now = time.monotonic()
        if iteration % 5 == 0 or (_now - _last_heartbeat) >= 60:
            logger.info(
                "Harness loop: iteration %d/%d, elapsed=%.0fs, llm_calls=%d, cumulative_in=%d tokens",
                iteration, max_iterations, _now - _loop_start,
                cumulative_usage["llm_calls"], cumulative_usage["input_tokens"],
            )
            _last_heartbeat = _now
        protocol.emit_turn_started(iteration)

        # --- Budget awareness + forced synthesis ---
        estimated_tokens = estimate_history_tokens(history)
        force_synthesis = estimated_tokens >= synthesis_threshold

        budget_line = (
            f"[Context: ~{estimated_tokens // 1000}K / {soft_limit // 1000}K tokens. "
            f"At ~{synthesis_threshold // 1000}K your tools will be removed and you must deliver your final answer.]"
        )

        if force_synthesis:
            forced_msg = (
                "**YOUR CONTEXT BUDGET IS EXHAUSTED.** Produce your FINAL response now "
                "using what you have gathered. No further tool calls are available — "
                "write your complete answer."
            )
            effective_instructions = f"{forced_msg}\n\n{budget_line}"
            if instructions:
                effective_instructions += f"\n\n{instructions}"
            effective_tool_names: list[str] | None = []
            logger.info(
                "Forced synthesis triggered at iteration %d: estimated=%d tokens, threshold=%d",
                iteration, estimated_tokens, synthesis_threshold,
            )
            protocol.emit_error(
                f"Context budget exhausted ({estimated_tokens} >= {synthesis_threshold} tokens). "
                f"Forcing final synthesis at iteration {iteration}.",
                iteration, fatal=False,
            )
        else:
            effective_instructions = f"{budget_line}\n\n{instructions}" if instructions else budget_line
            effective_tool_names = tool_names

        if _read_tools_stripped and effective_tool_names:
            effective_tool_names = [
                t for t in effective_tool_names if t not in READ_ONLY_TOOLS
            ]
            logger.info("Iteration %d: read tools stripped (degeneration detected at prior checkpoint)", iteration)

        response = None
        tool_calls = []
        llm_ok = False
        for attempt in range(1 + LLM_RETRIES):
            try:
                response, tool_calls = await llm_client.complete_with_tools(
                    history=history,
                    node_id=node_id,
                    system_prompt=system_prompt,
                    tool_registry=tool_registry,
                    tool_names=effective_tool_names,
                    instructions=effective_instructions,
                    mcp_config=mcp_config,
                    cc_watchdog=cc_watchdog,
                )
                llm_ok = True
                break
            except Exception as e:
                if attempt < LLM_RETRIES and _is_transient(e):
                    delay = _backoff_for_attempt(attempt, e)
                    protocol.emit_error(
                        f"Transient LLM error (attempt {attempt + 1}/{1 + LLM_RETRIES}), "
                        f"retrying in {delay:.0f}s: {e}",
                        iteration, fatal=False,
                    )
                    logger.warning(f"Transient LLM error at iteration {iteration}, "
                                   f"attempt {attempt + 1}: {e}")
                    await asyncio.sleep(delay)
                else:
                    protocol.emit_error(f"LLM call failed: {e}", iteration, fatal=True)
                    logger.exception(f"LLM call failed at iteration {iteration}")
                    break
        if not llm_ok:
            break

        # --- Hallucinated tool-call retry ---
        if (
            response
            and not tool_calls
            and effective_tool_names is not None
        ):
            bad_tools = _detect_hallucinated_tools(response, effective_tool_names)
            if bad_tools:
                for h_attempt in range(HALLUCINATION_MAX_RETRIES):
                    names_str = ", ".join(f"`{t}`" for t in bad_tools)
                    if effective_tool_names:
                        avail_str = ", ".join(effective_tool_names)
                        nudge = (
                            f"You attempted to call {names_str} which is not available. "
                            f"Your available tools are: {avail_str}. "
                            f"Do NOT emit XML for unavailable tools."
                        )
                    else:
                        nudge = (
                            f"You attempted to call {names_str} but you have NO tools available. "
                            f"Do NOT emit tool-call XML. Respond with plain text only."
                        )
                    logger.info(
                        "Hallucinated tool retry %d/%d at iteration %d: %s",
                        h_attempt + 1, HALLUCINATION_MAX_RETRIES, iteration, bad_tools,
                    )
                    now_ts = history[-1].timestamp if history else ""
                    history.append(HistoryMessage(
                        from_node=node_id,
                        content=response,
                        timestamp=now_ts,
                        source="in_flight",
                    ))
                    history.append(HistoryMessage(
                        from_node="system",
                        content=nudge,
                        timestamp=now_ts,
                        source="in_flight",
                    ))
                    try:
                        response, tool_calls = await llm_client.complete_with_tools(
                            history=history,
                            node_id=node_id,
                            system_prompt=system_prompt,
                            tool_registry=tool_registry,
                            tool_names=effective_tool_names,
                            instructions=effective_instructions,
                            mcp_config=mcp_config,
                            cc_watchdog=cc_watchdog,
                        )
                    except Exception:
                        logger.warning("Hallucinated tool retry LLM call failed, using last response")
                        break
                    if not response or tool_calls:
                        break
                    bad_tools = _detect_hallucinated_tools(response, effective_tool_names)
                    if not bad_tools:
                        break

        # Accumulate usage
        if llm_client._last_usage:
            u = llm_client._last_usage
            for key in ("input_tokens", "output_tokens", "cache_creation_tokens",
                        "cache_read_tokens", "reasoning_tokens", "total_tokens"):
                cumulative_usage[key] += u.get(key, 0)
            cumulative_usage["llm_calls"] += 1
            cumulative_usage["backend"] = u.get("backend", backend)
            cumulative_usage["model"] = u.get("model", model)

        protocol.emit_assistant_message(response[:500] if response else "", iteration)

        # Forced synthesis: tools were stripped, accept whatever the model produced
        if force_synthesis:
            final_text = (response or "").strip()
            if not final_text:
                final_text = "[Worker produced no output during forced synthesis]"
            logger.info("Forced synthesis complete at iteration %d: %d chars", iteration, len(final_text))
            break

        if not tool_calls:
            final_text = response.strip()
            break

        # CC/zai with MCP: tools handled internally, no XML tool calls
        if is_cc and mcp_config and not tool_calls:
            final_text = response.strip()
            break

        # Execute tool calls
        max_result_chars = int(soft_limit * 0.4)  # 10% of context window (~4 chars/token)
        try:
            results = await _execute_tool_calls(
                tool_calls, tool_registry, agent_socket_path, max_result_chars,
                effective_tool_names, force_synthesis=force_synthesis,
                read_tools_stripped=_read_tools_stripped,
            )
        except PhaseCompleteSignal as sig:
            logger.info("Phase complete signal received: %s", sig.summary[:200])
            final_text = sig.summary
            break

        # Format results for history
        result_parts: list[str] = []
        for name, result_str, _ok in results:
            result_parts.append(f'<tool_result name="{name}">\n{result_str}\n</tool_result>')
        tool_results_str = "\n\n".join(result_parts)
        tool_results_str = _truncate_extreme_result(tool_results_str, soft_limit)

        # Proactive budget guard: truncate tool results that would blow context
        response_est = len(response or "") // 4
        results_est = len(tool_results_str) // 4
        headroom = synthesis_threshold - estimated_tokens - response_est
        if headroom > 0 and results_est > headroom:
            max_chars = max(headroom * 4, 200)
            tool_results_str = tool_results_str[:max_chars] + (
                f"\n\n[Tool results truncated to fit context budget: "
                f"~{results_est // 1000}K tokens of output, "
                f"~{headroom // 1000}K budget remaining. "
                f"Use start_line/num_lines to read specific file sections.]"
            )

        # Build response for history (include reasoning if available)
        response_for_history = response
        if not response and tool_calls:
            response_for_history = "\n".join(
                getattr(tc, "raw_xml", f"<tool_call>{tc.name}</tool_call>")
                for tc in tool_calls
            )
        reasoning = getattr(llm_client, "_last_reasoning_content", None)
        if reasoning:
            response_for_history = f"<reasoning>\n{reasoning}\n</reasoning>\n{response_for_history}"

        # Append assistant turn + tool results
        now_ts = history[-1].timestamp if history else ""
        history.append(HistoryMessage(
            from_node=node_id,
            content=response_for_history,
            timestamp=now_ts,
            source="in_flight",
        ))
        history.append(HistoryMessage(
            from_node="system",
            content=f"Tool execution results:\n{tool_results_str}",
            timestamp=now_ts,
            source="in_flight",
        ))

        # --- Update checkpoint degeneration trackers ---
        if checkpoint_enabled and tool_calls:
            if any(_is_write_call(tc) for tc in tool_calls):
                _ckpt_has_written = True
                _ckpt_read_only_streak = 0
                _consecutive_degenerate = 0
                if _read_tools_stripped:
                    logger.info("Write detected — restoring read tools")
                    _read_tools_stripped = False
            else:
                _ckpt_read_only_streak += 1

            cur_sig = tuple(
                sorted(
                    (tc.name, json.dumps(tc.arguments, sort_keys=True))
                    for tc in tool_calls
                )
            )
            trigger_reason = ""
            if _ckpt_has_written and _ckpt_read_only_streak >= CHECKPOINT_STALL_TURNS:
                trigger_reason = "post_write_stall"
            elif _ckpt_last_tool_sig is not None and cur_sig == _ckpt_last_tool_sig:
                trigger_reason = "duplicate_read"
            elif iteration >= CHECKPOINT_PERIODIC_START and (iteration - CHECKPOINT_PERIODIC_START) % CHECKPOINT_PERIODIC_INTERVAL == 0:
                trigger_reason = "periodic"
            _ckpt_last_tool_sig = cur_sig

            if trigger_reason:
                logger.info("Checkpoint at iteration %d (%s): asking model if phase is complete",
                            iteration, trigger_reason)
                protocol.emit_checkpoint_query(iteration, trigger_reason)

                ckpt_ts = history[-1].timestamp if history else ""

                # Build side-channel checkpoint history (NOT appended to main history)
                if checkpoint_config:
                    eval_ctx = _build_evaluator_context(
                        task_prompt=checkpoint_config.task_prompt,
                        plan=checkpoint_config.plan,
                        history=history,
                        phase_start_idx=checkpoint_config.phase_start_idx,
                        phase_number=checkpoint_config.phase_number,
                        phase_prompt=checkpoint_config.phase_prompt,
                    )
                    ckpt_history: list[HistoryMessage] = [
                        HistoryMessage(
                            from_node="user",
                            content=f"{eval_ctx}\n\n{checkpoint_prompt}",
                            timestamp=ckpt_ts,
                            source="in_flight",
                        ),
                    ]
                else:
                    ckpt_history = list(history)
                    ckpt_history.append(HistoryMessage(
                        from_node="user",
                        content=checkpoint_prompt,
                        timestamp=ckpt_ts,
                        source="in_flight",
                    ))

                _ckpt_client = checkpoint_llm_client or llm_client
                ckpt_response = None
                ckpt_ok = False
                for ckpt_attempt in range(1 + LLM_RETRIES):
                    try:
                        ckpt_response, _ = await _ckpt_client.complete_with_tools(
                            history=ckpt_history,
                            node_id=node_id,
                            system_prompt=system_prompt,
                            tool_registry=tool_registry,
                            tool_names=[],
                            instructions="",
                            mcp_config=mcp_config,
                        )
                        _ckpt_usage = getattr(_ckpt_client, "_last_usage", None)
                        if isinstance(_ckpt_usage, dict):
                            protocol.emit_usage(_ckpt_usage)
                        ckpt_ok = True
                        break
                    except Exception as e:
                        if ckpt_attempt < LLM_RETRIES and _is_transient(e):
                            delay = _backoff_for_attempt(ckpt_attempt, e)
                            logger.warning("Checkpoint LLM error (attempt %d): %s, retrying in %.0fs",
                                           ckpt_attempt + 1, e, delay)
                            await asyncio.sleep(delay)
                        else:
                            logger.warning("Checkpoint LLM call failed: %s — treating as NO", e)
                            break
                # --- Hallucinated tool-call retry for checkpoint ---
                if ckpt_ok and ckpt_response:
                    ckpt_bad = _detect_hallucinated_tools(ckpt_response, [])
                    if ckpt_bad:
                        for ch_attempt in range(HALLUCINATION_MAX_RETRIES):
                            ch_names = ", ".join(f"`{t}`" for t in ckpt_bad)
                            ch_nudge = (
                                f"You attempted to call {ch_names} but you have NO tools available. "
                                f"Do NOT emit tool-call XML. Respond with plain text only."
                            )
                            logger.info(
                                "Checkpoint hallucinated tool retry %d/%d at iteration %d: %s",
                                ch_attempt + 1, HALLUCINATION_MAX_RETRIES, iteration, ckpt_bad,
                            )
                            ckpt_history.append(HistoryMessage(
                                from_node=node_id,
                                content=ckpt_response,
                                timestamp=ckpt_ts,
                                source="in_flight",
                            ))
                            ckpt_history.append(HistoryMessage(
                                from_node="system",
                                content=ch_nudge,
                                timestamp=ckpt_ts,
                                source="in_flight",
                            ))
                            try:
                                ckpt_response, _ = await _ckpt_client.complete_with_tools(
                                    history=ckpt_history,
                                    node_id=node_id,
                                    system_prompt=system_prompt,
                                    tool_registry=tool_registry,
                                    tool_names=[],
                                    instructions="",
                                    mcp_config=mcp_config,
                                )
                                _ckpt_usage = getattr(_ckpt_client, "_last_usage", None)
                                if isinstance(_ckpt_usage, dict):
                                    protocol.emit_usage(_ckpt_usage)
                            except Exception:
                                logger.warning("Checkpoint hallucination retry LLM call failed, using last response")
                                break
                            if not ckpt_response:
                                break
                            ckpt_bad = _detect_hallucinated_tools(ckpt_response, [])
                            if not ckpt_bad:
                                break

                ckpt_text = (ckpt_response or "").strip() if ckpt_ok else ""
                # Checkpoint Q&A is NOT appended to main history (side-channel)
                ckpt_lines = ckpt_text.split("\n") if ckpt_text else []
                first_line = ckpt_lines[0].strip().upper() if ckpt_lines else ""
                second_line = ckpt_lines[1].strip().upper() if len(ckpt_lines) > 1 else ""
                completed = first_line.startswith("YES")
                degenerate = second_line.startswith("YES")
                decision = "yes" if completed else "no"
                protocol.emit_checkpoint_response(iteration, ckpt_text, decision)
                if completed:
                    logger.info("Checkpoint at iteration %d: phase complete — %s", iteration, ckpt_text[:200])
                    _consecutive_degenerate = 0
                    final_text = ckpt_text
                    break
                elif degenerate:
                    _consecutive_degenerate += 1
                    logger.info(
                        "Checkpoint at iteration %d: degenerate loop detected (%d/%d) — stripping read tools until write",
                        iteration, _consecutive_degenerate, DEGENERATE_ESCALATION_LIMIT,
                    )
                    _read_tools_stripped = True
                    _ckpt_read_only_streak = 0
                    if _consecutive_degenerate == 2:
                        _reasoning_re = re.compile(r"<reasoning>.*?</reasoning>\s*", re.DOTALL)
                        _stripped_count = 0
                        for _msg in history:
                            if _msg.content and "<reasoning>" in _msg.content:
                                _new = _reasoning_re.sub("", _msg.content)
                                if _new != _msg.content:
                                    _msg.content = _new
                                    _stripped_count += 1
                        logger.info(
                            "Degenerate escalation (%d/%d): stripped <reasoning> tokens from %d history messages",
                            _consecutive_degenerate, DEGENERATE_ESCALATION_LIMIT, _stripped_count,
                        )
                    if _consecutive_degenerate >= DEGENERATE_ESCALATION_LIMIT:
                        logger.warning(
                            "Plan-and-execute: phase force-terminated after %d consecutive degenerate checkpoints",
                            _consecutive_degenerate,
                        )
                        final_text = (
                            "[Phase terminated: %d consecutive degenerate loop detections "
                            "without forward progress. The executor was unable to complete "
                            "this phase's objectives. Assess whether to retry with a "
                            "different approach or rescope the remaining work.]"
                            % _consecutive_degenerate
                        )
                        break
                else:
                    logger.info("Checkpoint at iteration %d: continuing — %s", iteration, first_line[:100])
                    _consecutive_degenerate = 0
                    _ckpt_read_only_streak = 0

    else:
        # Hit max iterations
        protocol.emit_error(
            f"Hit max iterations ({max_iterations}) without completing",
            max_iterations,
        )
        final_text = response.strip() if response else ""

    protocol.emit_usage(cumulative_usage)
    protocol.emit_thread_finished(thread_id, iteration, final_text, cumulative_usage)
    return final_text


def _truncate_at_word(text: str, max_chars: int) -> str:
    """Truncate text at most max_chars, breaking on word boundary if possible."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Find last whitespace; fall back to hard cut if no space in last 40 chars
    space_idx = cut.rfind(" ")
    if space_idx >= max_chars - 40:
        cut = cut[:space_idx]
    return cut + "…"


def _iso_now() -> str:
    """Return current UTC ISO8601 timestamp string."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Native Multi-Turn Executor Loop
# ---------------------------------------------------------------------------


def estimate_native_tokens(messages: list[dict]) -> int:
    """Estimate token count for native OpenAI messages."""
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        total += len(content) // 4
        for tc in msg.get("tool_calls", []):
            args = tc.get("function", {}).get("arguments", "")
            total += len(args) // 4
    return total


def _tool_name_for_call_id(messages: list[dict], call_id: str) -> str:
    """Find the tool name that matches a tool_call_id in the message history."""
    for msg in messages:
        for tc in msg.get("tool_calls", []):
            if tc.get("id") == call_id:
                return tc.get("function", {}).get("name", "unknown")
    return "unknown"


def _build_evaluator_context_from_native(
    task_prompt: str,
    plan: str,
    messages: list[dict],
    phase_start_idx: int,
    phase_number: int,
    phase_prompt: str = "",
    prior_phase_summaries: list[dict[str, str]] | None = None,
) -> str:
    """Build structured XML context for assessor from native OpenAI messages."""
    parts = ["<evaluator_context>"]
    parts.append(f"<task>\n{task_prompt}\n</task>")
    if plan:
        parts.append(f"<plan>\n{plan}\n</plan>")

    if prior_phase_summaries:
        parts.append("<prior_phases>")
        for ps in prior_phase_summaries:
            p_num = ps.get("phase", "?")
            p_summary = ps.get("summary", "")
            parts.append(f'<phase_summary phase="{p_num}">\n{p_summary}\n</phase_summary>')
        parts.append("</prior_phases>")
    else:
        scanned = []
        for msg in messages[:phase_start_idx]:
            c = msg.get("content") or ""
            if msg.get("role") == "user" and "<phase_summary" in c:
                scanned.append(c)
        if scanned:
            parts.append("<prior_phases>")
            parts.extend(scanned)
            parts.append("</prior_phases>")

    current_msgs = messages[phase_start_idx:]
    if current_msgs or phase_prompt:
        parts.append(f'<current_phase phase="{phase_number}">')
        if phase_prompt:
            parts.append(f"<objective>\n{phase_prompt}\n</objective>")

        tool_entries: list[str] = []
        files_modified: set[str] = set()
        for msg in current_msgs:
            if msg.get("role") == "tool":
                tool_name = _tool_name_for_call_id(messages, msg.get("tool_call_id", ""))
                result_text = msg.get("content", "")
                tool_entries.append(f'<tool_call name="{tool_name}">\n{result_text}\n</tool_call>')
                if tool_name in WRITE_TOOLS:
                    for fp in _FILE_PATH_RE.findall(result_text):
                        files_modified.add(fp)
            elif msg.get("role") == "assistant" and not msg.get("tool_calls"):
                content = msg.get("content") or ""
                if content.strip():
                    tool_entries.append(f'<turn from="executor">\n{content}\n</turn>')

        if tool_entries:
            parts.append("<tool_history>")
            parts.extend(tool_entries)
            parts.append("</tool_history>")

        if files_modified:
            parts.append("<files_modified>")
            for fp in sorted(files_modified):
                parts.append(f"  {fp}")
            parts.append("</files_modified>")

        parts.append("</current_phase>")

    parts.append("</evaluator_context>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Codex Assessor — subprocess-based controller using codex exec
# ---------------------------------------------------------------------------

CODEX_ASSESSOR_TIMEOUT = 300  # 5 minutes max per assessor call
CODEX_DEFAULT_BINARY = shutil.which("codex") or "codex"


async def _run_codex_assessor(
    eval_ctx: str,
    controller_instructions: str,
    *,
    codex_binary: str = "",
    codex_model: str = "o3",
    codex_effort: str = "high",
    cwd: str = "",
    timeout: int = CODEX_ASSESSOR_TIMEOUT,
) -> str:
    """Run codex exec as the assessor controller.

    Passes the evaluator context + controller instructions as the prompt,
    reads the structured output (intent XML) from the response.
    Returns the raw text output from codex.
    """
    if not codex_binary:
        codex_binary = shutil.which("codex") or CODEX_DEFAULT_BINARY

    prompt = f"{eval_ctx}\n\n---\n\n{controller_instructions}"

    cmd = [
        codex_binary, "exec",
        "--model", codex_model,
        "-s", "read-only",
        "--ephemeral",
        "--json",
    ]
    if codex_effort:
        cmd.extend(["--config", f"reasoning_effort={codex_effort}"])
    if cwd:
        cmd.extend(["-C", cwd])
    cmd.append("-")  # read prompt from stdin

    logger.info(
        "Codex assessor: model=%s effort=%s binary=%s",
        codex_model, codex_effort, codex_binary,
    )

    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                ),
            ),
            timeout=timeout + 10,
        )
    except (asyncio.TimeoutError, subprocess.TimeoutExpired):
        logger.error("Codex assessor timed out after %ds", timeout)
        return "<intent>terminate</intent>\n<summary>Codex assessor timed out.</summary>"

    if result.returncode != 0:
        logger.error(
            "Codex assessor failed (exit %d): %s",
            result.returncode, result.stderr[:500],
        )
        return "<intent>terminate</intent>\n<summary>Codex assessor failed.</summary>"

    # Parse JSONL output — extract the last assistant message content
    output_text = ""
    for line in result.stdout.strip().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        # codex exec --json emits events; the final message is in a "message" event
        if event.get("type") == "message" and event.get("role") == "assistant":
            parts = event.get("content", [])
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    output_text = part.get("text", "")
                elif isinstance(part, str):
                    output_text = part
        # Also check for "response.completed" or similar patterns
        elif event.get("type") == "response.completed":
            resp = event.get("response", {})
            for item in resp.get("output", []):
                if item.get("type") == "message" and item.get("role") == "assistant":
                    for c in item.get("content", []):
                        if isinstance(c, dict) and c.get("type") == "output_text":
                            output_text = c.get("text", "")

    # Fallback: if JSONL parsing didn't find structured content, use raw stdout
    if not output_text:
        # Try to find intent tags directly in the raw output
        raw = result.stdout
        if "<intent>" in raw:
            output_text = raw
        else:
            logger.warning("Codex assessor: no structured output found in JSONL")
            output_text = raw

    logger.info("Codex assessor output: %s", output_text[:300])
    return output_text


async def _compact_native_history(
    messages: list[dict],
    compact_after: int,
    phase_id: int,
    phase_prompt: str,
    executor_output: str,
    llm_client: LLMClient,
    soft_limit: int,
) -> str:
    """Compact native messages accumulated since the protected zone."""
    # Guard: don't split in the middle of an assistant→tool_result group.
    # If we'd cut right after an assistant with tool_calls (leaving orphaned
    # tool_calls without their results), or inside a sequence of tool results,
    # walk backward to include the full group in the compactable region.
    while compact_after > 0:
        msg_at = messages[compact_after - 1] if compact_after <= len(messages) else None
        if msg_at is None:
            break
        if msg_at.get("role") == "assistant" and msg_at.get("tool_calls"):
            # Would leave an assistant with orphaned tool_calls. Include it
            # in the compactable region by moving compact_after backward.
            compact_after -= 1
        elif msg_at.get("role") == "tool":
            # Inside a tool result sequence — keep walking back to include
            # the assistant that issued these calls.
            compact_after -= 1
        else:
            break

    compactable = messages[compact_after:]
    if not compactable:
        return ""

    parts = []
    for msg in compactable:
        role = msg.get("role", "?")
        if role == "tool":
            tool_name = _tool_name_for_call_id(messages, msg.get("tool_call_id", ""))
            parts.append(f"[tool:{tool_name}]: {msg.get('content', '')}")
        elif role == "assistant":
            content = msg.get("content") or ""
            tc_list = msg.get("tool_calls", [])
            if tc_list:
                tc_names = ", ".join(tc.get("function", {}).get("name", "?") for tc in tc_list)
                parts.append(f"[assistant → {tc_names}]: {content}")
            elif content.strip():
                parts.append(f"[assistant]: {content}")
        elif role == "user":
            parts.append(f"[user]: {msg.get('content', '')}")
    compactable_text = "\n\n".join(parts)

    prompt = _COMPACTION_TEMPLATE.format(
        phase_id=phase_id,
        phase_prompt=phase_prompt,
    ) + "\n\n---\n\n" + compactable_text

    summary = ""
    try:
        response, _ = await llm_client.complete_with_tools(
            history=[],
            node_id="compaction",
            system_prompt="You are a context compaction agent. Produce only the requested summary.",
            tool_registry=ToolRegistry(),
            tool_names=[],
            instructions=prompt,
        )
        summary = response.strip() if response else ""
    except Exception as e:
        logger.error(f"Native compaction LLM call failed: {e}")
        summary = f"[Compaction failed: {e}. Raw executor output preserved.]\n{executor_output[:2000]}"

    tag_match = re.search(r"<phase_summary[^>]*>(.*?)</phase_summary>", summary, re.DOTALL)
    summary_content = tag_match.group(1).strip() if tag_match else summary

    compact_msg = (
        f'<phase_summary phase="{phase_id}">\n'
        f"<prompt>{phase_prompt[:200]}</prompt>\n"
        f"{summary_content}\n"
        f"</phase_summary>"
    )

    before_tokens = estimate_native_tokens(messages[:compact_after]) + len(compactable_text) // 4
    del messages[compact_after:]
    messages.append({"role": "user", "content": compact_msg})
    after_tokens = estimate_native_tokens(messages)
    logger.info(
        "Native phase %d compaction: ~%dK → ~%dK tokens",
        phase_id, before_tokens // 1000, after_tokens // 1000,
    )
    return compact_msg


async def run_native_loop(
    llm_client: LLMClient,
    messages: list[dict],
    tool_registry: ToolRegistry,
    tool_names: list[str] | None = None,
    node_id: str = "harness",
    max_iterations: int = MAX_ITERATIONS,
    soft_limit: int = DEFAULT_SOFT_LIMIT,
    instructions: str = "",
    agent_socket_path: str | None = None,
    checkpoint_enabled: bool = False,
    checkpoint_prompt: str = DEFAULT_CHECKPOINT_PROMPT,
    checkpoint_config: CheckpointConfig | None = None,
    checkpoint_llm_client: LLMClient | None = None,
    checkpoint_periodic_interval: int = 8,
    checkpoint_periodic_start: int = 8,
) -> str:
    """Native multi-turn executor loop using OpenAI Chat Completions format.

    Unlike run_loop, this maintains a native messages list (mutated in place)
    instead of XML-serialized HistoryMessage objects. Tool results appear as
    native {"role": "tool"} messages so the model treats them as its own work.
    """
    thread_id = uuid.uuid4().hex[:12]
    backend = llm_client.config.backend
    model = llm_client.config.model
    protocol.emit_thread_started(thread_id, backend, model)

    cumulative_usage: dict[str, Any] = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_creation_tokens": 0, "cache_read_tokens": 0,
        "reasoning_tokens": 0, "total_tokens": 0,
        "llm_calls": 0, "backend": backend, "model": model,
    }

    openai_tools = tool_registry.get_openai_tools(tool_names) if tool_names else None
    synthesis_threshold = int(soft_limit * FORCED_SYNTHESIS_FRACTION)
    final_text = ""
    _loop_start = time.monotonic()
    _last_heartbeat = _loop_start

    # Checkpoint degeneration trackers
    CHECKPOINT_STALL_TURNS = 3
    CHECKPOINT_PERIODIC_INTERVAL = checkpoint_periodic_interval
    CHECKPOINT_PERIODIC_START = checkpoint_periodic_start
    DEGENERATE_ESCALATION_LIMIT = 3
    _ckpt_has_written = False
    _ckpt_read_only_streak = 0
    _ckpt_last_tool_sig: tuple[str, ...] | None = None
    _read_tools_stripped = False
    _consecutive_degenerate = 0

    for iteration in range(1, max_iterations + 1):
        _now = time.monotonic()
        if iteration % 5 == 0 or (_now - _last_heartbeat) >= 60:
            logger.info(
                "Native loop: iteration %d/%d, elapsed=%.0fs, llm_calls=%d",
                iteration, max_iterations, _now - _loop_start,
                cumulative_usage["llm_calls"],
            )
            _last_heartbeat = _now
        protocol.emit_turn_started(iteration)

        estimated_tokens = estimate_native_tokens(messages)
        force_synthesis = estimated_tokens >= synthesis_threshold

        budget_line = (
            f"[Context: ~{estimated_tokens // 1000}K / {soft_limit // 1000}K tokens. "
            f"At ~{synthesis_threshold // 1000}K your tools will be removed and you must deliver your final answer.]"
        )

        effective_tool_names = tool_names
        effective_tools = openai_tools
        if force_synthesis:
            effective_tool_names = []
            effective_tools = None
            forced_msg = (
                "**YOUR CONTEXT BUDGET IS EXHAUSTED.** Produce your FINAL response now "
                "using what you have gathered. No further tool calls are available — "
                "write your complete answer."
            )
            messages.append({"role": "user", "content": f"{forced_msg}\n\n{budget_line}"})
            logger.info(
                "Forced synthesis triggered at iteration %d: estimated=%d tokens, threshold=%d",
                iteration, estimated_tokens, synthesis_threshold,
            )
        else:
            # Inject budget line (and instructions if provided) as a transient
            # user message. Removed after the LLM call so it doesn't accumulate.
            per_iter_parts = [budget_line]
            if instructions:
                per_iter_parts.append(instructions)
            messages.append({"role": "user", "content": "\n\n".join(per_iter_parts)})

        if _read_tools_stripped and effective_tool_names:
            effective_tool_names = [t for t in effective_tool_names if t not in READ_ONLY_TOOLS]
            effective_tools = tool_registry.get_openai_tools(effective_tool_names)

        # LLM call with retry
        response_content = ""
        tool_calls: list[ToolCall] = []
        llm_ok = False
        for attempt in range(1 + LLM_RETRIES):
            try:
                content, tc_list, usage = await llm_client.complete_multi_turn(
                    messages=messages,
                    tools=effective_tools,
                )
                response_content = content
                tool_calls = tc_list
                llm_ok = True
                break
            except Exception as e:
                if attempt < LLM_RETRIES and _is_transient(e):
                    delay = _backoff_for_attempt(attempt, e)
                    protocol.emit_error(
                        f"Transient LLM error (attempt {attempt + 1}/{1 + LLM_RETRIES}), "
                        f"retrying in {delay:.0f}s: {e}",
                        iteration, fatal=False,
                    )
                    await asyncio.sleep(delay)
                else:
                    protocol.emit_error(f"LLM call failed: {e}", iteration, fatal=True)
                    logger.exception(f"Native loop LLM call failed at iteration {iteration}")
                    break
        if not llm_ok:
            # Remove transient budget message on failure too
            if not force_synthesis and messages and messages[-1].get("role") == "user":
                last_content = messages[-1].get("content", "")
                if last_content.startswith("[Context:") or last_content.startswith(budget_line[:20]):
                    messages.pop()
            break

        # Remove the transient per-iteration budget/instructions message.
        # Forced synthesis messages stay (they're part of the conversation).
        if not force_synthesis and messages:
            last_msg = messages[-1]
            if last_msg.get("role") == "user" and budget_line in (last_msg.get("content") or ""):
                messages.pop()

        # Accumulate usage
        if llm_client._last_usage:
            u = llm_client._last_usage
            for key in ("input_tokens", "output_tokens", "cache_creation_tokens",
                        "cache_read_tokens", "reasoning_tokens", "total_tokens"):
                cumulative_usage[key] += u.get(key, 0)
            cumulative_usage["llm_calls"] += 1

        protocol.emit_assistant_message(response_content[:500] if response_content else "", iteration)

        if force_synthesis:
            final_text = (response_content or "").strip()
            if not final_text:
                final_text = "[Worker produced no output during forced synthesis]"
            break

        if not tool_calls:
            final_text = response_content.strip()
            break

        # Build the assistant message with tool_calls for the messages list
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if response_content:
            assistant_msg["content"] = response_content
        assistant_msg["tool_calls"] = [
            {
                "id": tc.call_id or uuid.uuid4().hex[:12],
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in tool_calls
        ]
        messages.append(assistant_msg)

        # Emit protocol events for tool calls (dashboard visibility)
        for tc in tool_calls:
            tc_id = tc.call_id or ""
            protocol.emit_tool_call(tc.name, tc.arguments, tc_id)

        # Execute tool calls
        max_result_chars = int(soft_limit * 0.4)
        try:
            results = await _execute_tool_calls(
                tool_calls, tool_registry, agent_socket_path, max_result_chars,
                effective_tool_names, force_synthesis=force_synthesis,
                read_tools_stripped=_read_tools_stripped,
            )
        except PhaseCompleteSignal as sig:
            logger.info("Phase complete signal received: %s", sig.summary[:200])
            final_text = sig.summary
            break

        # Append native tool result messages (_execute_tool_calls already emits protocol events)
        for i, (name, result_str, _ok) in enumerate(results):
            tc = tool_calls[i] if i < len(tool_calls) else None
            call_id = (tc.call_id if tc else None) or assistant_msg["tool_calls"][i]["id"]
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result_str,
            })

        # Checkpoint degeneration tracking
        if checkpoint_enabled and tool_calls:
            if any(_is_write_call(tc) for tc in tool_calls):
                _ckpt_has_written = True
                _ckpt_read_only_streak = 0
                _consecutive_degenerate = 0
                if _read_tools_stripped:
                    logger.info("Write detected — restoring read tools")
                    _read_tools_stripped = False
                    effective_tools = openai_tools
            else:
                _ckpt_read_only_streak += 1

            cur_sig = tuple(
                sorted(
                    (tc.name, json.dumps(tc.arguments, sort_keys=True))
                    for tc in tool_calls
                )
            )
            trigger_reason = ""
            if _ckpt_has_written and _ckpt_read_only_streak >= CHECKPOINT_STALL_TURNS:
                trigger_reason = "post_write_stall"
            elif _ckpt_last_tool_sig is not None and cur_sig == _ckpt_last_tool_sig:
                trigger_reason = "duplicate_read"
            elif iteration >= CHECKPOINT_PERIODIC_START and (iteration - CHECKPOINT_PERIODIC_START) % CHECKPOINT_PERIODIC_INTERVAL == 0:
                trigger_reason = "periodic"
            _ckpt_last_tool_sig = cur_sig

            if trigger_reason:
                logger.info("Native checkpoint at iteration %d (%s)", iteration, trigger_reason)
                protocol.emit_checkpoint_query(iteration, trigger_reason)

                if checkpoint_config:
                    eval_ctx = _build_evaluator_context_from_native(
                        task_prompt=checkpoint_config.task_prompt,
                        plan=checkpoint_config.plan,
                        messages=messages,
                        phase_start_idx=checkpoint_config.phase_start_idx,
                        phase_number=checkpoint_config.phase_number,
                        phase_prompt=checkpoint_config.phase_prompt,
                    )
                    ckpt_history: list[HistoryMessage] = [
                        HistoryMessage(
                            from_node="user",
                            content=f"{eval_ctx}\n\n{checkpoint_prompt}",
                            timestamp=_iso_now(),
                            source="in_flight",
                        ),
                    ]
                else:
                    ckpt_history = [
                        HistoryMessage(
                            from_node="user",
                            content=checkpoint_prompt,
                            timestamp=_iso_now(),
                            source="in_flight",
                        ),
                    ]

                _ckpt_client = checkpoint_llm_client or llm_client
                ckpt_response = None
                ckpt_ok = False
                for ckpt_attempt in range(1 + LLM_RETRIES):
                    try:
                        ckpt_response, _ = await _ckpt_client.complete_with_tools(
                            history=ckpt_history,
                            node_id=node_id,
                            system_prompt="You are a checkpoint evaluator.",
                            tool_registry=tool_registry,
                            tool_names=[],
                            instructions="",
                        )
                        ckpt_ok = True
                        break
                    except Exception as e:
                        if ckpt_attempt < LLM_RETRIES and _is_transient(e):
                            delay = _backoff_for_attempt(ckpt_attempt, e)
                            await asyncio.sleep(delay)
                        else:
                            logger.warning("Checkpoint LLM failed: %s — treating as NO", e)
                            break

                ckpt_text = (ckpt_response or "").strip() if ckpt_ok else ""
                ckpt_lines = ckpt_text.split("\n") if ckpt_text else []
                first_line = ckpt_lines[0].strip().upper() if ckpt_lines else ""
                second_line = ckpt_lines[1].strip().upper() if len(ckpt_lines) > 1 else ""
                completed = first_line.startswith("YES")
                degenerate = second_line.startswith("YES")
                decision = "yes" if completed else "no"
                protocol.emit_checkpoint_response(iteration, ckpt_text, decision)
                if completed:
                    logger.info("Native checkpoint: phase complete — %s", ckpt_text[:200])
                    _consecutive_degenerate = 0
                    final_text = ckpt_text
                    break
                elif degenerate:
                    _consecutive_degenerate += 1
                    logger.info(
                        "Native checkpoint: degenerate (%d/%d) — stripping read tools",
                        _consecutive_degenerate, DEGENERATE_ESCALATION_LIMIT,
                    )
                    _read_tools_stripped = True
                    _ckpt_read_only_streak = 0
                    if _consecutive_degenerate >= DEGENERATE_ESCALATION_LIMIT:
                        logger.warning("Native loop: force-terminated after %d degenerate checkpoints", _consecutive_degenerate)
                        final_text = (
                            "[Phase terminated: %d consecutive degenerate loop detections "
                            "without forward progress.]" % _consecutive_degenerate
                        )
                        break
                else:
                    _consecutive_degenerate = 0
                    _ckpt_read_only_streak = 0

    else:
        protocol.emit_error(f"Hit max iterations ({max_iterations})", max_iterations)
        final_text = response_content.strip() if response_content else ""

    protocol.emit_usage(cumulative_usage)
    protocol.emit_thread_finished(thread_id, iteration, final_text, cumulative_usage)
    return final_text


# ---------------------------------------------------------------------------
# Plan-and-Execute Controller
# ---------------------------------------------------------------------------

DEFAULT_MAX_PHASES = 15
DEFAULT_COMPACTION_THRESHOLD_FRACTION = 0.40

OBSERVE_INSTRUCTIONS = """\
## Observe the Codebase

You are the observer in a plan-and-execute controller. Your job is to \
explore the codebase and gather enough context to write a comprehensive \
implementation plan.

### Your tools

You have read-only tools: `file_read`, `list_dir`, `grep`, and `find_files`.

- **`grep`** — search file contents by regex pattern. Use this to locate \
classes, functions, imports, and string literals across the codebase. \
Much faster than paginating through files with `file_read`.
- **`find_files`** — find files by glob pattern (e.g., `*.py`, `test_*.py`). \
Use this to discover project structure instead of recursive `list_dir` calls.
- **`file_read`** — read specific line ranges of a file. Use after `grep` \
or `find_files` to inspect the relevant sections.
- **`list_dir`** — list directory contents. Good for initial orientation.

Start with `grep` or `find_files` to locate relevant code, then use \
`file_read` to inspect specific sections. Avoid reading entire files \
line-by-line — search first, read targeted ranges second.

### Your goal

Understand the codebase well enough to specify concrete implementation \
phases. Focus on:
- The files and classes that need to be modified
- The API surfaces and interfaces the implementation must conform to
- Any test files that define expected behavior

Do NOT produce a plan yet — just gather information. You will be asked \
to write the plan in a separate step after you have finished exploring.\
"""

PLAN_INSTRUCTIONS = """\
## Write the Implementation Plan

You have just finished exploring the codebase. Now synthesize your \
findings into a structured plan for an executor to follow.

### Your output

You MUST produce TWO things:

1. **The plan.** Natural prose describing each phase the executor \
should complete, in order. One paragraph per phase — describe the \
goal and the approach. The plan can have 1–15 phases depending on \
task complexity. A simple task gets a one-phase plan.

2. **The Phase 1 executor prompt.** Wrapped in `<phase_1_prompt>` tags. \
This is the executor's first instruction — a `user` message telling \
it exactly what to do for Phase 1. Be specific: name files, line \
ranges, and expected outcomes.

Example output:

Phase 1 — Fix the serialization bug in models.py
Read the `serialize()` method and fix the off-by-one error on the \
boundary check. Add a test case.

Phase 2 — Update the API handler
Update `handlers/api.py` to use the fixed serializer. Verify the \
response shape matches the OpenAPI spec.

<phase_1_prompt>
Read `models.py` lines 45-80 to find the `serialize()` method. The \
boundary check on line 62 has an off-by-one error (uses `<` instead of \
`<=`). Fix it, then add a test in `tests/test_models.py` that covers \
the boundary value. Run the test to confirm it passes.
</phase_1_prompt>

### Important

- You MUST end with `<phase_1_prompt>...</phase_1_prompt>` tags. \
Without them, the harness cannot proceed.
- The plan does not need to be perfectly detailed — you will reassess \
after every phase and can revise the plan as you learn more.
- The `<phase_1_prompt>` is NOT a copy of the plan text. It is a \
separate, actionable instruction tailored for the executor. Write it \
as a clear task description the executor can follow without reading \
the plan.
- You have no tools available. Produce the plan from your observations.\
"""

_ASSESSMENT_TEMPLATE = """\
## Assess Phase {phase_id}

You are the planner in a plan-and-execute controller. The executor \
just completed Phase {phase_id}. Your job is to evaluate the result, \
decide what to do next, and — if continuing — write the prompt for \
the next executor phase.

### Your current plan
{current_plan}

(This is the plan you wrote during Phase 0, or your most recent \
revision if you revised it in a prior assessment. It is preserved \
through compaction and always available to you.)

### The prompt you wrote for this phase
{last_executor_prompt}

(This is the exact `<next_phase_prompt>` you wrote in your last \
assessment — or the initial phase prompt from the plan if this is \
Phase 1. It is preserved through compaction so you can evaluate \
whether the executor followed your instructions.)

### Phase evidence
{phase_result}

### Your responsibilities

1. **Evaluate the phase result.** Did the executor achieve the goal? \
What key information was produced? What failed?

2. **Decide your strategy internally.** You have full autonomy over \
plan management:
   - The phase succeeded and the plan is on track → continue to the \
next phase.
   - The phase revealed something the plan didn't account for → \
revise your remaining phases, include `<revised_plan>` with the \
updated plan, and continue with the new direction.
   - The phase failed but might succeed with a different approach → \
continue with a retry prompt (different angle, different files, \
narrower scope).
   - The phase failed and the approach is fundamentally wrong → \
replan from scratch, include `<revised_plan>`, and continue.
   - The task is complete (all phases done, or the answer was found \
early) → terminate and write a summary for the user.
   - The task is beyond capability (repeated failures, missing \
information, impossible constraint) → terminate and explain why.

3. **Recognize when you're stuck.** If you've retried the same phase \
more than twice without progress, or if you keep replanning without \
the executor producing useful output, terminate. A clear explanation \
of what went wrong is more valuable than another failed attempt.

4. **Output your decision.** You MUST output exactly one of:

   **To continue:**
   ```xml
   <assessment>
   <decision>continue</decision>
   <reasoning>Your evaluation and strategy reasoning.</reasoning>
   <next_phase_prompt>The executor's instructions for the next phase. \
Write this fresh — incorporate what was just learned. Include the \
specific goal, which files or resources to examine, and what the \
deliverable should be. This is injected as a `user` message to \
the executor, so write it as a clear task description.</next_phase_prompt>
   <revised_plan>Optional — include ONLY if you are changing the plan. \
Write the full updated plan (remaining phases, revised goals). \
The harness replaces the current plan with this text. Omit this \
element entirely if the plan is unchanged.</revised_plan>
   </assessment>
   ```

   **To terminate (task complete, impossible, or stuck):**
   ```xml
   <assessment>
   <decision>done</decision>
   <reasoning>Your evaluation and why you're terminating.</reasoning>
   <summary>A user-facing summary of all work completed: what was \
accomplished, what was discovered, key decisions made. If \
terminating early, explain what remains unfinished and why. \
This is the ONLY text the user will see — make it complete \
and useful.</summary>
   </assessment>
   ```

### Important

- The `<next_phase_prompt>` is your sole steering mechanism. The \
executor does exactly what you write there — no more, no less. \
Be specific: name files, line ranges, and expected outcomes.
- The `<summary>` on `done` is the user's final deliverable. It \
should stand alone — the user hasn't seen any of the intermediate \
phase results.
- Your current plan and last executor prompt are always available \
to you — they are preserved through compaction and injected into \
every assessment prompt. Use them to track whether execution is \
following your intended direction.
- When you revise the plan (change direction, drop phases, add \
phases), include `<revised_plan>` so the harness can track your \
current strategy. The harness replaces `current_plan` with your \
revision. If the plan hasn't changed, omit `<revised_plan>` \
entirely — don't repeat the existing plan.\
"""

_COMPACTION_TEMPLATE = """\
You are a context compaction agent. The executor just completed \
Phase {phase_id} of a multi-phase plan.

Phase prompt (what the planner asked the executor to do):
{phase_prompt}

Below is the raw executor trace for this phase (tool calls, results, \
and reasoning). Compress it into a concise summary that preserves \
the information the planner needs to assess progress and the executor \
needs to continue in subsequent phases.

PRESERVE (do not omit or paraphrase):
1. What was attempted — the specific actions taken and tools called
2. What succeeded — key facts, values, file contents, and conclusions \
the executor reached
3. What failed and why — error messages VERBATIM, root causes identified
4. Artifacts produced — file paths created/modified, decisions made, \
code written (include key snippets, not full files)
5. Open sub-questions — anything unresolved that affects later phases

OUTPUT FORMAT: Write a single block wrapped in <phase_summary> tags. \
Be concise but complete. Err on the side of preserving detail over \
brevity — the planner uses this to decide what to do next.\
"""


_CONTROLLER_TEMPLATE = """\
You are the controller in a plan-and-execute system. Examine the \
evaluator context above — it contains the task, any existing plan, \
your action history, and evidence from the most recent action.

Choose exactly ONE intent for your next action:

1. **observe** — Gather information by reading files, searching code, \
or querying APIs. Use when you need more context before planning or \
executing.
   Output:
   <intent>observe</intent>
   <prompt>Describe specifically what to explore and why.</prompt>

2. **plan** — Write or revise the implementation plan. Use after enough \
observation to know what needs to be done. The plan should list concrete \
numbered phases.
   Output:
   <intent>plan</intent>
   <plan>Your complete plan with numbered phases.</plan>

3. **execute** — Run the next step of the plan. Provide a specific prompt \
for the executor describing exactly what to do: name files, line ranges, \
and expected outcomes.
   Output:
   <intent>execute</intent>
   <prompt>Specific instructions for the executor.</prompt>

4. **terminate** — Terminate when the objective described in the task \
is satisfied by the current workspace state. Your plan is a means to \
the objective, not the objective itself. \
Produce a detailed user-facing summary of all work accomplished.
   Output:
   <intent>terminate</intent>
   <summary>Complete summary of results.</summary>

## Guidelines

- If the task is a simple factual question you can answer now, \
terminate immediately with the answer.
- For information retrieval (email lookup, status check, code review, \
literature search): observe to gather findings, then terminate.
- For implementation tasks, the typical flow is: observe → plan → \
execute (repeated) → terminate. Skip observe if the task is clear.
- Observe uses read-only tools. Execute uses full tools (edits, shell, etc).
- Plan is toolless — you write the plan inline in your output.
- You may revise the plan by choosing plan again at any point.
- If you have been executing without progress, replan or terminate \
with an explanation.

Emit your intent now.\
"""

_CONTROLLER_TEMPLATE_NO_OBSERVE = """\
You are the controller in a plan-and-execute system. Examine the \
evaluator context above — it contains the task, any existing plan, \
your action history, and evidence from the most recent action.

**observe is temporarily disabled** — you have already used your \
observation budget. Choose plan, execute, or terminate.

Choose exactly ONE intent for your next action:

1. **plan** — Write or revise the implementation plan. Use after enough \
observation to know what needs to be done. The plan should list concrete \
numbered phases.
   Output:
   <intent>plan</intent>
   <plan>Your complete plan with numbered phases.</plan>

2. **execute** — Run the next step of the plan. Provide a specific prompt \
for the executor describing exactly what to do: name files, line ranges, \
and expected outcomes.
   Output:
   <intent>execute</intent>
   <prompt>Specific instructions for the executor.</prompt>

3. **terminate** — Terminate when the objective described in the task \
is satisfied by the current workspace state. Your plan is a means to \
the objective, not the objective itself. \
Produce a detailed user-facing summary of all work accomplished.
   Output:
   <intent>terminate</intent>
   <summary>Complete summary of results.</summary>

## Guidelines

- If the task is a simple factual question you can answer now, \
terminate immediately with the answer.
- Observe uses read-only tools. Execute uses full tools (edits, shell, etc).
- Plan is toolless — you write the plan inline in your output.
- You may revise the plan by choosing plan again at any point.
- If you have been executing without progress, replan or terminate \
with an explanation.

Emit your intent now.\
"""


def _strip_observe_from_instructions(instructions: str) -> str:
    """Replace the controller template with the no-observe variant.

    Raises RuntimeError if the controller template isn't present in the
    instructions — that means _CONTROLLER_TEMPLATE drifted without a matching
    update here, and the cap would silently stop working.
    """
    result = instructions.replace(
        _CONTROLLER_TEMPLATE, _CONTROLLER_TEMPLATE_NO_OBSERVE
    )
    if result == instructions:
        raise RuntimeError(
            "_strip_observe_from_instructions: _CONTROLLER_TEMPLATE not "
            "found in instructions. The template likely drifted — keep "
            "this function and the templates in sync."
        )
    return result


_EXECUTOR_FRAMING = """\
## Execute Phase {phase_id} of the Plan

You are executing phase {phase_id} of a multi-phase plan.

### Current plan
{current_plan}

### Your goal for this phase
{phase_prompt}

Focus only on this phase's goal. The harness will check in periodically \
to verify your progress — when it determines the phase is complete, it \
will advance to the next step. Do not start the next phase.\
"""


_TRIAGE_TEMPLATE = """\
<evaluator_context>
<task>
{task}
</task>
</evaluator_context>

You are a task assessor. No work has been done yet on the above task. \
Based on the task description and any context provided, decide how to proceed.

There are three possible classifications:

1. **Done** — You can answer the task directly from your knowledge or the \
provided context. No tools needed.
   Output: <decision>done</decision><answer>your answer</answer>

2. **Search-only** — The task is PURELY informational: answering a question, \
looking something up, reading/reviewing code, searching emails, fetching \
references, surveying literature, or summarizing existing content. The \
findings ARE the deliverable — no code changes, no file edits, no new \
artifacts need to be produced.
   Output: <decision>continue</decision><needs_plan>false</needs_plan>\
<next_phase_prompt>describe what to search for</next_phase_prompt>

3. **Needs plan** — The task requires producing something: code changes, bug \
fixes, feature implementations, new files, documents, presentations, reports, \
or any other artifact. Even if the change seems simple, it needs a plan.
   Output: <decision>continue</decision><needs_plan>true</needs_plan>\
<next_phase_prompt>describe what to explore and plan</next_phase_prompt>

## Classification examples

- "Pull my latest email from Google" → continue, needs_plan=false \
(information retrieval — findings are the deliverable)
- "What's the capital of France?" → done (you know this already)
- "What's the mesh agent status?" → continue, needs_plan=false \
(status check — read-only lookup)
- "Review router_v2.py for concurrency bugs" → continue, needs_plan=false \
(code review — read-only analysis, findings are the deliverable)
- "Check mesh/llm.py for error handling gaps" → continue, needs_plan=false \
(code audit — read and report)
- "Survey recent papers on submodular optimization" → continue, needs_plan=false \
(literature search — findings are the deliverable)
- "Fix the bug where synthesis drops messages" → continue, needs_plan=true \
(bug fix — requires code changes)
- "Add heartbeat status to the /status command" → continue, needs_plan=true \
(feature — requires implementation)
- "Implement a CLI interval merge tool from scratch" → continue, needs_plan=true \
(greenfield implementation)
- "Write a 10-slide beamer deck on TSP" → continue, needs_plan=true \
(document production — artifact must be created)
- "Produce a technical report comparing FLMI vs greedy" → continue, needs_plan=true \
(writing task — artifact must be created)

Emit your classification now.\
"""


@dataclass
class TriageDecision:
    type: str  # "done" or "continue"
    answer: str = ""
    next_phase_prompt: str = ""
    needs_plan: bool = False


def parse_triage(text: str) -> TriageDecision:
    """Parse triage assessor output into a structured decision.

    Trusts the LLM's <needs_plan> tag. Defaults to needs_plan=True when
    the tag is missing (safer — implementation tasks are the common case).
    """
    decision_match = re.search(r"<decision>\s*(done|continue)\s*</decision>", text)
    if not decision_match:
        return TriageDecision(type="continue", next_phase_prompt=text.strip(), needs_plan=True)

    dtype = decision_match.group(1)

    if dtype == "done":
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else text.strip()
        return TriageDecision(type="done", answer=answer)

    prompt_match = re.search(r"<next_phase_prompt>(.*?)</next_phase_prompt>", text, re.DOTALL)
    if not prompt_match:
        return TriageDecision(type="continue", next_phase_prompt=text.strip(), needs_plan=True)

    phase_prompt = prompt_match.group(1).strip()

    needs_plan_match = re.search(r"<needs_plan>\s*(true|false)\s*</needs_plan>", text, re.IGNORECASE)
    needs_plan = needs_plan_match.group(1).lower() == "true" if needs_plan_match else True

    return TriageDecision(type="continue", next_phase_prompt=phase_prompt, needs_plan=needs_plan)


@dataclass
class AssessmentDecision:
    type: str  # "continue" or "done"
    next_phase_prompt: str = ""
    revised_plan: str = ""
    summary: str = ""
    reasoning: str = ""


def extract_plan_and_phase_prompt(plan_output: str) -> tuple[str, str]:
    """Split Phase 0 output into plan text and Phase 1 executor prompt."""
    match = re.search(r"<phase_1_prompt>(.*?)</phase_1_prompt>", plan_output, re.DOTALL)
    if not match:
        raise ValueError("Phase 0 output missing <phase_1_prompt> tags")
    phase_prompt = match.group(1).strip()
    plan_text = plan_output[:match.start()].strip()
    return plan_text, phase_prompt


def build_assessment_prompt(
    phase_id: int,
    current_plan: str,
    last_executor_prompt: str,
    phase_result: str,
) -> str:
    """Build the planner assessment instructions for post-phase evaluation."""
    return _ASSESSMENT_TEMPLATE.format(
        phase_id=phase_id,
        current_plan=current_plan,
        last_executor_prompt=last_executor_prompt,
        phase_result=phase_result,
    )


def parse_assessment(assessment_text: str) -> AssessmentDecision:
    """Parse the planner's assessment output into a structured decision.

    Falls back to 'done' with raw text as summary if parsing fails.
    """
    decision_match = re.search(r"<decision>\s*(continue|done)\s*</decision>", assessment_text)
    if not decision_match:
        return AssessmentDecision(
            type="done",
            summary=assessment_text.strip(),
            reasoning="[Malformed assessment — could not parse decision tag]",
        )

    decision_type = decision_match.group(1)
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", assessment_text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    if decision_type == "done":
        summary_match = re.search(r"<summary>(.*?)</summary>", assessment_text, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else assessment_text.strip()
        return AssessmentDecision(type="done", summary=summary, reasoning=reasoning)

    # decision_type == "continue"
    prompt_match = re.search(r"<next_phase_prompt>(.*?)</next_phase_prompt>", assessment_text, re.DOTALL)
    if not prompt_match:
        return AssessmentDecision(
            type="done",
            summary=assessment_text.strip(),
            reasoning="[Malformed assessment — decision=continue but no <next_phase_prompt>]",
        )
    next_prompt = prompt_match.group(1).strip()

    revised_match = re.search(r"<revised_plan>(.*?)</revised_plan>", assessment_text, re.DOTALL)
    revised_plan = revised_match.group(1).strip() if revised_match else ""

    return AssessmentDecision(
        type="continue",
        next_phase_prompt=next_prompt,
        revised_plan=revised_plan,
        reasoning=reasoning,
    )


@dataclass
class ControllerIntent:
    intent: str  # "observe", "plan", "execute", "terminate"
    prompt: str = ""
    plan: str = ""
    summary: str = ""


def parse_controller_intent(text: str) -> ControllerIntent:
    """Parse controller output into a structured intent.

    Defaults to observe on parse failure (safe — read-only).
    """
    intent_match = re.search(
        r"<intent>\s*(observe|plan|execute|terminate)\s*</intent>",
        text, re.IGNORECASE,
    )
    if not intent_match:
        return ControllerIntent(intent="observe", prompt=text.strip())

    intent = intent_match.group(1).lower()

    if intent == "terminate":
        summary_match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else text.strip()
        return ControllerIntent(intent="terminate", summary=summary)

    if intent == "plan":
        plan_match = re.search(r"<plan>(.*?)</plan>", text, re.DOTALL)
        plan = plan_match.group(1).strip() if plan_match else text.strip()
        return ControllerIntent(intent="plan", plan=plan)

    # observe or execute
    prompt_match = re.search(r"<prompt>(.*?)</prompt>", text, re.DOTALL)
    prompt = prompt_match.group(1).strip() if prompt_match else text.strip()
    return ControllerIntent(intent=intent, prompt=prompt)


async def _compact_history(
    history: list[HistoryMessage],
    compact_after: int,
    phase_id: int,
    phase_prompt: str,
    executor_output: str,
    llm_client: LLMClient,
    soft_limit: int,
) -> str:
    """Compact history turns that accumulated since the protected zone.

    Everything before compact_after is NEVER touched.  Everything after
    is replaced with a single summary message.

    Returns the compaction summary text.
    """
    compactable = history[compact_after:]
    if not compactable:
        return ""

    compactable_text = "\n\n".join(
        f"[{msg.from_node}]: {msg.content}" for msg in compactable
    )

    prompt = _COMPACTION_TEMPLATE.format(
        phase_id=phase_id,
        phase_prompt=phase_prompt,
    ) + "\n\n---\n\n" + compactable_text

    summary = ""
    try:
        response, _ = await llm_client.complete_with_tools(
            history=[],
            node_id="compaction",
            system_prompt="You are a context compaction agent. Produce only the requested summary.",
            tool_registry=ToolRegistry(),
            tool_names=[],
            instructions=prompt,
        )
        summary = response.strip() if response else ""
    except Exception as e:
        logger.error(f"Compaction LLM call failed: {e}")
        summary = f"[Compaction failed: {e}. Raw executor output preserved.]\n{executor_output[:2000]}"

    # Extract just the summary content if wrapped in tags
    tag_match = re.search(r"<phase_summary[^>]*>(.*?)</phase_summary>", summary, re.DOTALL)
    summary_content = tag_match.group(1).strip() if tag_match else summary

    compact_msg = (
        f'<phase_summary phase="{phase_id}">\n'
        f"<prompt>{phase_prompt[:200]}</prompt>\n"
        f"{summary_content}\n"
        f"</phase_summary>"
    )

    # Replace compactable region with single summary message
    del history[compact_after:]
    history.append(HistoryMessage(
        from_node="system",
        content=compact_msg,
        timestamp=history[-1].timestamp if history else "",
        source="in_flight",
    ))

    before_tokens = estimate_history_tokens(history[:compact_after]) + len(compactable_text) // 4
    after_tokens = estimate_history_tokens(history)
    logger.info(
        "Phase %d compaction: ~%dK → ~%dK tokens",
        phase_id, before_tokens // 1000, after_tokens // 1000,
    )

    return compact_msg


async def _run_plan_and_execute_loop(
    llm_client: LLMClient,
    history: list[HistoryMessage],
    system_prompt: str,
    tool_registry: ToolRegistry,
    *,
    tool_names: list[str] | None = None,
    node_id: str = "harness",
    soft_limit: int = DEFAULT_SOFT_LIMIT,
    max_phases: int = DEFAULT_MAX_PHASES,
    compaction_threshold_fraction: float = DEFAULT_COMPACTION_THRESHOLD_FRACTION,
    agent_socket_path: str | None = None,
    instructions: str = "",
    mcp_config: str | None = None,
    assessor_llm_client: LLMClient | None = None,
    codex_assessor_config: dict[str, str] | None = None,
) -> str:
    """Assessor-controller loop.

    The assessor acts as a flexible controller, choosing one of four
    intents each turn: observe, plan, execute, or terminate.

    - observe: read-only exploration (search tools, checkpoint-terminated)
    - plan: write or revise the implementation plan (toolless, inline)
    - execute: run next step with full tools (checkpoint-terminated)
    - terminate: produce final summary and return
    """
    _assessor = assessor_llm_client or llm_client

    thread_id = uuid.uuid4().hex[:12]
    backend = str(llm_client.config.backend)
    model = str(llm_client.config.model)
    protocol.emit_thread_started(thread_id, backend, model)

    use_native = backend == "openai"

    compact_after = len(history)
    task_prompt = "\n\n".join(msg.content for msg in history[:compact_after])

    initial_tokens = estimate_history_tokens(history[:compact_after])
    workspace = max(soft_limit - initial_tokens, 50_000)
    compaction_threshold = int(workspace * compaction_threshold_fraction)
    observe_limit = min(int(workspace * 0.50), 250_000)
    logger.info(
        "Plan-and-execute: soft_limit=%dK, initial_prompt=%dK, workspace=%dK, "
        "compaction_threshold=%dK, observe_limit=%dK, native=%s",
        soft_limit // 1000, initial_tokens // 1000, workspace // 1000,
        compaction_threshold // 1000, observe_limit // 1000, use_native,
    )

    # Initialize native messages list for OpenAI backends
    native_messages: list[dict] | None = None
    native_compact_after: int = 0
    if use_native:
        native_messages = []
        tool_prompt = tool_registry.format_tools_prompt(tool_names, backend="openai")
        sys_content = system_prompt
        if tool_prompt:
            sys_content += "\n\n" + tool_prompt
        native_messages.append({"role": "system", "content": sys_content})
        task_content = task_prompt
        if instructions:
            task_content += "\n\n" + instructions
        native_messages.append({"role": "user", "content": task_content})
        native_compact_after = len(native_messages)

    # Build search/observe tool list
    search_tools = list(PLANNER_TOOLS)
    if tool_names:
        for tn in tool_names:
            if tn in MESH_READ_ONLY_TOOLS and tn not in search_tools:
                search_tools.append(tn)
    if len(search_tools) > len(PLANNER_TOOLS):
        logger.info(
            "Plan-and-execute: search tools expanded with %d mesh tools: %s",
            len(search_tools) - len(PLANNER_TOOLS),
            sorted(t for t in search_tools if t not in PLANNER_TOOLS),
        )
    else:
        logger.info(
            "Plan-and-execute: no mesh tools in search path (tool_names=%d, matches=%d)",
            len(tool_names) if tool_names else 0, 0,
        )

    controller_instructions = _CONTROLLER_TEMPLATE
    if instructions:
        controller_instructions = f"{instructions}\n\n{controller_instructions}"

    current_plan: str | None = None
    action_history: list[str] = []
    last_phase_start_idx: int | None = None
    last_native_phase_start_idx: int | None = None
    last_phase_prompt: str = ""
    observe_count: int = 0

    for phase_id in range(1, max_phases + 1):
        # Build controller context using evaluator context + action history
        if use_native and native_messages is not None:
            eval_ctx = _build_evaluator_context_from_native(
                task_prompt=task_prompt,
                plan=current_plan or "",
                messages=native_messages,
                phase_start_idx=(
                    last_native_phase_start_idx if last_native_phase_start_idx is not None
                    else len(native_messages)
                ),
                phase_number=phase_id - 1,
                phase_prompt=last_phase_prompt,
            )
        else:
            eval_ctx = _build_evaluator_context(
                task_prompt=task_prompt,
                plan=current_plan or "",
                history=history,
                phase_start_idx=(
                    last_phase_start_idx if last_phase_start_idx is not None
                    else len(history)
                ),
                phase_number=phase_id - 1,
                phase_prompt=last_phase_prompt,
            )
        if action_history:
            ah_entries = "\n".join(f"- {e}" for e in action_history)
            ah_section = f"<action_history>\n{ah_entries}\n</action_history>"
            eval_ctx = eval_ctx.replace(
                "</evaluator_context>",
                f"{ah_section}\n</evaluator_context>",
            )

        # Observe cap: after 2 observe phases, disable observe until execute
        phase_instructions = controller_instructions
        if observe_count >= 2:
            phase_instructions = _strip_observe_from_instructions(
                controller_instructions
            )

        controller_ts = history[-1].timestamp if history else _iso_now()
        logger.info("Plan-and-execute: controller call for phase %d", phase_id)

        if codex_assessor_config:
            # Codex exec subprocess as assessor
            controller_output = await _run_codex_assessor(
                eval_ctx=eval_ctx,
                controller_instructions=phase_instructions,
                codex_binary=codex_assessor_config.get("binary", CODEX_DEFAULT_BINARY),
                codex_model=codex_assessor_config.get("model", "o3"),
                codex_effort=codex_assessor_config.get("effort", "high"),
                cwd=codex_assessor_config.get("cwd", ""),
                timeout=int(codex_assessor_config.get("timeout", str(CODEX_ASSESSOR_TIMEOUT))),
            )
        else:
            # Standard LLM-based assessor
            controller_output = await run_loop(
                llm_client=_assessor,
                history=[HistoryMessage(
                    from_node="user",
                    content=eval_ctx,
                    timestamp=controller_ts,
                    source="in_flight",
                )],
                system_prompt=system_prompt,
                tool_registry=tool_registry,
                tool_names=[],
                node_id=node_id,
                max_iterations=1,
                soft_limit=soft_limit,
                mcp_config=mcp_config,
                instructions=phase_instructions,
                agent_socket_path=agent_socket_path,
            )

        intent = parse_controller_intent(controller_output or "")

        # Enforce observe cap: if observe is disabled but the LLM ignored the
        # no-observe template and returned observe anyway, skip this phase
        # entirely. Plan/execute/terminate are first-class intents the
        # controller can choose on the next iteration.
        if observe_count >= 2 and intent.intent == "observe":
            logger.warning(
                "Plan-and-execute: observe cap exceeded at phase %d "
                "(observe_count=%d), skipping phase. Controller ignored "
                "no-observe template.",
                phase_id, observe_count,
            )
            action_history.append(
                f"Phase {phase_id} (skipped): observe cap exceeded"
            )
            continue

        logger.info(
            "Plan-and-execute: phase %d intent=%s",
            phase_id, intent.intent,
        )

        if phase_id == 1:
            protocol.emit_assessor_triage(
                decision="done" if intent.intent == "terminate" else "continue",
                needs_plan=intent.intent not in ("terminate", "observe"),
                summary=intent.summary or intent.prompt or intent.plan[:300] or "",
            )

        protocol.emit_assessor_phase_start(
            phase_id, intent.intent,
            intent.prompt or intent.summary or intent.plan[:300] or "",
        )

        if intent.intent == "terminate":
            protocol.emit_assessor_phase_complete(phase_id, "terminate")
            protocol.emit_thread_finished(thread_id, phase_id, intent.summary, {})
            return intent.summary

        elif intent.intent == "observe":
            observe_count += 1
            observe_prompt = intent.prompt + "\n\n" + OBSERVE_INSTRUCTIONS
            _observe_ckpt_prompt = (
                "CHECKPOINT: Answer both questions.\n"
                "Line 1: Have you gathered enough information to proceed? "
                "YES or NO\n"
                "Line 2: Are you stuck in a degenerate loop (reading the "
                "same files repeatedly without making progress)? YES or NO\n"
                "Line 3+: Brief rationale."
            )

            if use_native and native_messages is not None:
                native_phase_start = len(native_messages)
                _roles = [m.get("role") for m in native_messages]
                _prior_tool = sum(1 for r in _roles[:native_phase_start] if r == "tool")
                logger.info(
                    "phase_boundary: phase=%d type=observe msgs=%d "
                    "user=%d assistant=%d tool=%d prior_phase_tool=%d",
                    phase_id, len(native_messages),
                    _roles.count("user"), _roles.count("assistant"),
                    _roles.count("tool"), _prior_tool,
                )
                native_messages.append({"role": "user", "content": observe_prompt})
                await run_native_loop(
                    llm_client=llm_client,
                    messages=native_messages,
                    tool_registry=tool_registry,
                    tool_names=list(search_tools),
                    node_id=node_id,
                    max_iterations=MAX_ITERATIONS,
                    soft_limit=observe_limit,
                    agent_socket_path=agent_socket_path,
                    checkpoint_enabled=True,
                    checkpoint_prompt=_observe_ckpt_prompt,
                    checkpoint_config=CheckpointConfig(
                        task_prompt=task_prompt,
                        plan=current_plan or "",
                        phase_number=phase_id,
                        phase_start_idx=native_phase_start,
                    ),
                    checkpoint_llm_client=_assessor if assessor_llm_client else None,
                    checkpoint_periodic_interval=5,
                    checkpoint_periodic_start=8,
                )
                _new = len(native_messages) - native_phase_start
                _new_tool = sum(
                    1 for m in native_messages[native_phase_start:]
                    if m.get("role") == "tool"
                )
                _new_asst = sum(
                    1 for m in native_messages[native_phase_start:]
                    if m.get("role") == "assistant"
                )
                logger.info(
                    "phase_complete: phase=%d type=observe new_msgs=%d "
                    "new_tool=%d new_assistant=%d total_msgs=%d",
                    phase_id, _new, _new_tool, _new_asst, len(native_messages),
                )
                last_native_phase_start_idx = native_phase_start
            else:
                phase_start_idx = len(history)
                await run_loop(
                    llm_client=llm_client,
                    history=history,
                    system_prompt=system_prompt,
                    tool_registry=tool_registry,
                    tool_names=list(search_tools),
                    node_id=node_id,
                    max_iterations=MAX_ITERATIONS,
                    soft_limit=observe_limit,
                    mcp_config=mcp_config,
                    instructions=observe_prompt,
                    agent_socket_path=agent_socket_path,
                    checkpoint_enabled=True,
                    checkpoint_prompt=_observe_ckpt_prompt,
                    checkpoint_config=CheckpointConfig(
                        task_prompt=task_prompt,
                        plan=current_plan or "",
                        phase_number=phase_id,
                        phase_start_idx=phase_start_idx,
                    ),
                    checkpoint_llm_client=_assessor if assessor_llm_client else None,
                    checkpoint_periodic_interval=5,
                    checkpoint_periodic_start=8,
                )
                last_phase_start_idx = phase_start_idx

            protocol.emit_assessor_phase_complete(phase_id, "observe")

            action_history.append(
                f"Phase {phase_id} (observe): "
                f"{_truncate_at_word(intent.prompt, 200)}"
            )
            last_phase_prompt = intent.prompt

        elif intent.intent == "plan":
            current_plan = intent.plan
            protocol.emit_assessor_phase_complete(phase_id, "plan")
            action_history.append(
                f"Phase {phase_id} (plan): Plan written ({len(current_plan)} chars)"
            )
            last_phase_start_idx = None
            last_phase_prompt = ""

        elif intent.intent == "execute":
            executor_instructions = _EXECUTOR_FRAMING.format(
                phase_id=phase_id,
                current_plan=current_plan or "(no plan)",
                phase_prompt=intent.prompt,
            )

            if use_native and native_messages is not None:
                native_phase_start = len(native_messages)
                _roles = [m.get("role") for m in native_messages]
                _prior_tool = sum(1 for r in _roles[:native_phase_start] if r == "tool")
                logger.info(
                    "phase_boundary: phase=%d type=execute msgs=%d "
                    "user=%d assistant=%d tool=%d prior_phase_tool=%d",
                    phase_id, len(native_messages),
                    _roles.count("user"), _roles.count("assistant"),
                    _roles.count("tool"), _prior_tool,
                )
                native_messages.append({"role": "user", "content": executor_instructions})
                executor_result = await run_native_loop(
                    llm_client=llm_client,
                    messages=native_messages,
                    tool_registry=tool_registry,
                    tool_names=tool_names,
                    node_id=node_id,
                    soft_limit=soft_limit,
                    agent_socket_path=agent_socket_path,
                    checkpoint_enabled=True,
                    checkpoint_config=CheckpointConfig(
                        task_prompt=task_prompt,
                        plan=current_plan or "",
                        phase_number=phase_id,
                        phase_start_idx=native_phase_start,
                        phase_prompt=intent.prompt,
                    ),
                    checkpoint_llm_client=_assessor if assessor_llm_client else None,
                    checkpoint_periodic_interval=5,
                    checkpoint_periodic_start=8,
                )
                protocol.emit_assessor_phase_complete(phase_id, "execute")

                _new = len(native_messages) - native_phase_start
                _new_tool = sum(
                    1 for m in native_messages[native_phase_start:]
                    if m.get("role") == "tool"
                )
                _new_asst = sum(
                    1 for m in native_messages[native_phase_start:]
                    if m.get("role") == "assistant"
                )
                logger.info(
                    "phase_complete: phase=%d type=execute new_msgs=%d "
                    "new_tool=%d new_assistant=%d total_msgs=%d",
                    phase_id, _new, _new_tool, _new_asst, len(native_messages),
                )

                estimated_tokens = estimate_native_tokens(native_messages)
                scratchpad_tokens = estimated_tokens - initial_tokens
                if scratchpad_tokens > compaction_threshold:
                    _pre_compact_roles = [m.get("role") for m in native_messages]
                    logger.info(
                        "compaction_boundary: phase=%d before_msgs=%d "
                        "compact_after=%d user=%d assistant=%d tool=%d "
                        "scratchpad=%dK threshold=%dK",
                        phase_id, len(native_messages), native_compact_after,
                        _pre_compact_roles.count("user"),
                        _pre_compact_roles.count("assistant"),
                        _pre_compact_roles.count("tool"),
                        scratchpad_tokens // 1000, compaction_threshold // 1000,
                    )
                    await _compact_native_history(
                        native_messages, native_compact_after, phase_id,
                        phase_prompt=intent.prompt,
                        executor_output=executor_result,
                        llm_client=llm_client,
                        soft_limit=soft_limit,
                    )
                    _post_roles = [m.get("role") for m in native_messages]
                    logger.info(
                        "compaction_boundary: phase=%d after_msgs=%d "
                        "user=%d assistant=%d tool=%d",
                        phase_id, len(native_messages),
                        _post_roles.count("user"),
                        _post_roles.count("assistant"),
                        _post_roles.count("tool"),
                    )
                    # After compaction, messages list is truncated. Point to end
                    # so next assessor sees the compacted summary as "last phase."
                    last_native_phase_start_idx = len(native_messages)
                else:
                    last_native_phase_start_idx = native_phase_start
            else:
                phase_start_idx = len(history)

                _cc_watchdog = None
                if llm_client.config.backend in ("claude-code", "zai") and assessor_llm_client:
                    _cc_watchdog = CCWatchdog(
                        assessor=_assessor,
                        task_prompt=task_prompt,
                        plan=current_plan or "",
                        phase_prompt=intent.prompt,
                    )
                    logger.info(
                        "Plan-and-execute: CC watchdog enabled for Phase %d",
                        phase_id,
                    )

                executor_result = await run_loop(
                    llm_client=llm_client,
                    history=history,
                    system_prompt=system_prompt,
                    tool_registry=tool_registry,
                    tool_names=tool_names,
                    node_id=node_id,
                    soft_limit=soft_limit,
                    mcp_config=mcp_config,
                    instructions=executor_instructions,
                    agent_socket_path=agent_socket_path,
                    checkpoint_enabled=True,
                    checkpoint_config=CheckpointConfig(
                        task_prompt=task_prompt,
                        plan=current_plan or "",
                        phase_number=phase_id,
                        phase_start_idx=phase_start_idx,
                        phase_prompt=intent.prompt,
                    ),
                    checkpoint_llm_client=_assessor if assessor_llm_client else None,
                    checkpoint_periodic_interval=5,
                    checkpoint_periodic_start=8,
                    cc_watchdog=_cc_watchdog,
                )
                protocol.emit_assessor_phase_complete(phase_id, "execute")

                estimated_tokens = estimate_history_tokens(history)
                scratchpad_tokens = estimated_tokens - initial_tokens
                if scratchpad_tokens > compaction_threshold:
                    logger.info(
                        "Plan-and-execute: compacting after Phase %d "
                        "(scratchpad ~%dK > %dK threshold, total ~%dK)",
                        phase_id, scratchpad_tokens // 1000,
                        compaction_threshold // 1000, estimated_tokens // 1000,
                    )
                    await _compact_history(
                        history, compact_after, phase_id,
                        phase_prompt=intent.prompt,
                        executor_output=executor_result,
                        llm_client=llm_client,
                        soft_limit=soft_limit,
                    )
                last_phase_start_idx = phase_start_idx

            action_history.append(
                f"Phase {phase_id} (execute): "
                f"{_truncate_at_word(intent.prompt, 200)}"
            )
            last_phase_prompt = intent.prompt

    last_action = action_history[-1] if action_history else "none"
    final_text = (
        f"[Task exceeded {max_phases} phases. "
        f"Last action: {last_action}]"
    )
    protocol.emit_error(f"Phase cap of {max_phases} exceeded", max_phases)
    protocol.emit_thread_finished(thread_id, max_phases, final_text, {})
    return final_text


# ---------------------------------------------------------------------------
# Decompose Controller
# ---------------------------------------------------------------------------

DECOMPOSE_MAX_STEPS = 8
DECOMPOSE_MAX_RETRIES = 2
DECOMPOSE_PLAN_MAX_ATTEMPTS = 3

DECOMPOSE_STEP_TYPES = frozenset({"search", "extract", "classify", "synthesize", "translate"})
DECOMPOSE_READONLY_TYPES = frozenset({"search", "extract", "classify"})

_DECOMPOSE_CLASSIFY_TEMPLATE = """\
You are a task complexity classifier. Given the following task, classify \
it as SIMPLE or COMPLEX.

SIMPLE: Can be completed in a single pass with no intermediate planning. \
Examples: write a function, fix a known bug, answer a factual question, \
look up a single piece of information.

COMPLEX: Requires multiple distinct steps — discovering information, \
extracting structured data, evaluating or classifying findings, \
synthesizing understanding, or transforming one representation into \
another. Examples: investigate a bug across files, research and write \
a report, multi-file refactor, review a proposal against criteria, \
survey literature and summarize findings.

Output exactly one word: SIMPLE or COMPLEX\
"""

_DECOMPOSE_PLAN_TEMPLATE = """\
You are a task decomposition agent. Break the following task into a \
sequence of typed steps. Each step must have one of these types:

- **search**: Discover, locate, and gather information from available \
sources. Find relevant artifacts, query for facts, scan for patterns.
- **extract**: Pull specific structured information from unstructured \
sources — fields from documents, patterns from logs, data points from \
prose, entities from text.
- **classify**: Categorize, label, evaluate, or sort items against \
defined criteria. Triage by severity, score against a rubric, assign \
to buckets.
- **synthesize**: Analyze gathered information and produce structured \
understanding — a plan, specification, evaluation, summary, decision \
framework, or report.
- **translate**: Transform one representation into another. A spec into \
an implementation, an outline into prose, raw data into a formatted \
report, a description into code.

Rules:
- Use as few steps as needed. {max_steps} is the absolute maximum, not a \
target. Most tasks need 3-5 steps.
- A single search step can have multiple objectives — do not create \
separate search steps for different files or directories.
- Steps execute in order. No branching or loops.
- The plan is immutable once produced — it will not be modified during \
execution.
- For each step, write a clear executor_prompt — the exact instructions \
the executor will receive. Be specific: name files, locations, criteria, \
or formats as applicable.

Output your plan in this XML format:

<plan>
<step index="1" type="search">
<description>Brief description of what this step accomplishes.</description>
<executor_prompt>Detailed instructions for the executor. Be specific about \
what to search for, where to look, and what to report back.</executor_prompt>
</step>
<step index="2" type="synthesize">
<description>Brief description.</description>
<executor_prompt>Detailed instructions for the executor.</executor_prompt>
</step>
</plan>

Task:
{task}\
"""

_DECOMPOSE_STEP_ASSESS_TEMPLATE = """\
You are a step assessment agent. The executor just completed step \
{step_index} of {total_steps} in a multi-step task.

### Overall objective
{task}

### Step {step_index} ({step_type}): {step_description}

### Executor prompt given
{executor_prompt}

### Executor output
{executor_output}

### Your responsibilities

1. Evaluate whether the step was completed successfully. Did the executor \
accomplish what was asked? Is the output sufficient for subsequent steps?

2. Output exactly one verdict:

**PASS** — The step was completed successfully.
```xml
<assessment>
<verdict>PASS</verdict>
</assessment>
```

**RETRY** — The step was not completed successfully but may succeed with \
a different approach. Provide a hint to guide the retry.
```xml
<assessment>
<verdict>RETRY</verdict>
<hint>Specific guidance on what to do differently.</hint>
</assessment>
```

**FAIL** — The step cannot be completed. The task should terminate early.
```xml
<assessment>
<verdict>FAIL</verdict>
<reason>Why this step cannot be completed.</reason>
</assessment>
```

Emit your assessment now.\
"""

_DECOMPOSE_SUMMARY_TEMPLATE = """\
You are a task summary agent. A multi-step task has just {completion_status}. \
Produce a structured report summarizing what was done.

### Original task
{task}

### Plan ({total_steps} steps)
{plan_summary}

### Step outcomes
{step_outcomes}

### Step results
{step_results}

### Files modified
{files_modified}

### Your output

Write a clear, structured summary covering:
1. What was accomplished (or attempted)
2. Whether the task succeeded or failed, and why
3. Key findings or artifacts produced
4. The current state of the environment (files changed, tests run, etc.)

This summary is the ONLY text the user will see. Make it complete and \
useful. Include specific data, numbers, and findings from the step results.\
"""


@dataclass
class DecomposeStep:
    """A single step in a decompose plan."""
    index: int
    type: str
    description: str
    executor_prompt: str


@dataclass
class DecomposeAssessment:
    """Assessment of a completed step."""
    verdict: str  # "pass", "retry", "fail"
    hint: str = ""
    reason: str = ""


def parse_decompose_plan(text: str) -> list[DecomposeStep]:
    """Parse the assessor's decomposition output into a list of steps.

    Raises ValueError if no valid steps are found or step count exceeds
    DECOMPOSE_MAX_STEPS.
    """
    steps: list[DecomposeStep] = []

    step_pattern = re.compile(
        r'<step\s+index="(\d+)"\s+type="(\w+)">\s*'
        r"<description>(.*?)</description>\s*"
        r"<executor_prompt>(.*?)</executor_prompt>\s*"
        r"</step>",
        re.DOTALL,
    )

    for m in step_pattern.finditer(text):
        idx = int(m.group(1))
        stype = m.group(2).lower().strip()
        desc = m.group(3).strip()
        prompt = m.group(4).strip()

        if stype not in DECOMPOSE_STEP_TYPES:
            logger.warning("Decompose: unknown step type %r, defaulting to 'search'", stype)
            stype = "search"

        steps.append(DecomposeStep(index=idx, type=stype, description=desc, executor_prompt=prompt))

    if not steps:
        raise ValueError("No valid <step> elements found in decomposition output")

    if len(steps) > DECOMPOSE_MAX_STEPS:
        logger.warning(
            "Decompose: plan has %d steps, truncating to %d",
            len(steps), DECOMPOSE_MAX_STEPS,
        )
        steps = steps[:DECOMPOSE_MAX_STEPS]

    return steps


def parse_decompose_assessment(text: str) -> DecomposeAssessment:
    """Parse the assessor's step assessment into a verdict."""
    verdict_match = re.search(
        r"<verdict>\s*(PASS|RETRY|FAIL)\s*</verdict>",
        text, re.IGNORECASE,
    )
    if not verdict_match:
        return DecomposeAssessment(verdict="pass")

    verdict = verdict_match.group(1).lower()

    if verdict == "retry":
        hint_match = re.search(r"<hint>(.*?)</hint>", text, re.DOTALL)
        hint = hint_match.group(1).strip() if hint_match else ""
        return DecomposeAssessment(verdict="retry", hint=hint)

    if verdict == "fail":
        reason_match = re.search(r"<reason>(.*?)</reason>", text, re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else ""
        return DecomposeAssessment(verdict="fail", reason=reason)

    return DecomposeAssessment(verdict="pass")


def _build_decompose_step_prompt(
    step: DecomposeStep,
    total_steps: int,
    task: str,
    carry_forward: str,
    retry_hint: str = "",
) -> str:
    """Build the full prompt injected as executor instructions for a step."""
    parts = [
        f"## Step {step.index} of {total_steps}: {step.description}",
    ]
    if step.type not in DECOMPOSE_READONLY_TYPES:
        parts.append(f"\n### Overall objective\n{task}")
    if carry_forward:
        parts.append(f"\n### Context from prior steps\n{carry_forward}")
    if retry_hint:
        parts.append(
            f"\n### Retry guidance\n"
            f"A previous attempt at this step did not succeed. "
            f"Guidance: {retry_hint}"
        )
    parts.append(f"\n### Instructions\n{step.executor_prompt}")
    remaining = total_steps - step.index
    if remaining > 0:
        parts.append(
            f"\n{remaining} step{'s' if remaining != 1 else ''} remaining after this one."
        )
    else:
        parts.append("\nThis is the final step.")
    return "\n".join(parts)


async def _call_assessor(
    assessor: LLMClient,
    prompt: str,
    system_prompt: str,
    tool_registry: ToolRegistry,
    node_id: str,
    soft_limit: int,
    mcp_config: str | None,
    agent_socket_path: str | None,
) -> str:
    """Make a single-turn, toolless assessor call and return the response."""
    ts = _iso_now()
    return await run_loop(
        llm_client=assessor,
        history=[HistoryMessage(
            from_node="user",
            content=prompt,
            timestamp=ts,
            source="in_flight",
        )],
        system_prompt=system_prompt,
        tool_registry=tool_registry,
        tool_names=[],
        node_id=node_id,
        max_iterations=1,
        soft_limit=soft_limit,
        mcp_config=mcp_config,
        instructions="",
        agent_socket_path=agent_socket_path,
    )


async def _run_decompose_loop(
    llm_client: LLMClient,
    history: list[HistoryMessage],
    system_prompt: str,
    tool_registry: ToolRegistry,
    *,
    tool_names: list[str] | None = None,
    node_id: str = "harness",
    soft_limit: int = DEFAULT_SOFT_LIMIT,
    compaction_threshold_fraction: float = DEFAULT_COMPACTION_THRESHOLD_FRACTION,
    agent_socket_path: str | None = None,
    instructions: str = "",
    mcp_config: str | None = None,
    assessor_llm_client: LLMClient | None = None,
    codex_assessor_config: dict[str, str] | None = None,
) -> str:
    """Decompose controller loop.

    Phase 0: Classify task complexity (SIMPLE → standard mode).
    Phase 1: Decompose task into typed steps.
    Phase 2: Execute each step sequentially with per-step assessment.
    Phase 3: Produce a summary report.
    """
    _assessor = assessor_llm_client or llm_client

    thread_id = uuid.uuid4().hex[:12]
    backend = str(llm_client.config.backend)
    model = str(llm_client.config.model)
    protocol.emit_thread_started(thread_id, backend, model)

    use_native = backend == "openai"

    compact_after = len(history)
    task_prompt = "\n\n".join(msg.content for msg in history[:compact_after])
    if instructions:
        task_prompt += "\n\n" + instructions

    initial_tokens = estimate_history_tokens(history[:compact_after])
    workspace = max(soft_limit - initial_tokens, 50_000)
    compaction_threshold = int(workspace * compaction_threshold_fraction)

    logger.info(
        "Decompose: soft_limit=%dK, initial_prompt=%dK, workspace=%dK, "
        "compaction_threshold=%dK, native=%s",
        soft_limit // 1000, initial_tokens // 1000, workspace // 1000,
        compaction_threshold // 1000, use_native,
    )

    # Build read-only and full tool lists
    search_tools = list(PLANNER_TOOLS)
    if tool_names:
        for tn in tool_names:
            if tn in MESH_READ_ONLY_TOOLS and tn not in search_tools:
                search_tools.append(tn)

    # ---- Phase 0: Complexity classification ----
    logger.info("Decompose: Phase 0 — classifying task complexity")
    protocol.emit_assessor_phase_start(0, "classify", "Classifying task complexity")

    classify_prompt = _DECOMPOSE_CLASSIFY_TEMPLATE + f"\n\nTask:\n{task_prompt}"
    classify_output = await _call_assessor(
        _assessor, classify_prompt, system_prompt, tool_registry,
        node_id, soft_limit, mcp_config, agent_socket_path,
    )

    is_simple = "SIMPLE" in classify_output.upper().split()
    protocol.emit_assessor_phase_complete(0, "classify")
    protocol.emit_assessor_triage(
        decision="simple" if is_simple else "complex",
        needs_plan=not is_simple,
        summary=f"Classification: {'SIMPLE' if is_simple else 'COMPLEX'}",
    )

    if is_simple:
        logger.info("Decompose: task classified as SIMPLE — falling through to standard mode")
        protocol.emit_thread_finished(thread_id, 0, "[routed to standard]", {})
        # Standard mode — delegate to the existing standard executor
        if use_native:
            native_messages: list[dict] = []
            tool_prompt = tool_registry.format_tools_prompt(tool_names, backend="openai")
            sys_content = system_prompt
            if tool_prompt:
                sys_content += "\n\n" + tool_prompt
            native_messages.append({"role": "system", "content": sys_content})
            task_content = "\n\n".join(msg.content for msg in history)
            if instructions:
                task_content += "\n\n" + instructions
            native_messages.append({"role": "user", "content": task_content})
            return await run_native_loop(
                llm_client=llm_client,
                messages=native_messages,
                tool_registry=tool_registry,
                tool_names=tool_names,
                node_id=node_id,
                soft_limit=soft_limit,
                instructions=instructions,
                agent_socket_path=agent_socket_path,
            )
        else:
            return await run_loop(
                llm_client=llm_client,
                history=history,
                system_prompt=system_prompt,
                tool_registry=tool_registry,
                tool_names=tool_names,
                node_id=node_id,
                soft_limit=soft_limit,
                mcp_config=mcp_config,
                instructions=instructions,
                agent_socket_path=agent_socket_path,
                controller_mode="standard",
            )

    # ---- Phase 1: Task decomposition (with retry) ----
    logger.info("Decompose: Phase 1 — decomposing task into steps")
    protocol.emit_assessor_phase_start(1, "decompose", "Decomposing task into typed steps")

    decompose_prompt = _DECOMPOSE_PLAN_TEMPLATE.format(
        max_steps=DECOMPOSE_MAX_STEPS,
        task=task_prompt,
    )

    steps: list[DecomposeStep] | None = None
    for plan_attempt in range(1, DECOMPOSE_PLAN_MAX_ATTEMPTS + 1):
        if plan_attempt > 1:
            decompose_prompt += (
                "\n\nIMPORTANT: Your previous response could not be parsed. "
                "You MUST respond with a <plan> containing <step> elements "
                "in this exact XML format:\n"
                "<plan>\n"
                '<step index="1" type="search">\n'
                "<description>Brief description.</description>\n"
                "<executor_prompt>Detailed instructions.</executor_prompt>\n"
                "</step>\n"
                "</plan>"
            )

        decompose_output = await _call_assessor(
            _assessor, decompose_prompt, system_prompt, tool_registry,
            node_id, soft_limit, mcp_config, agent_socket_path,
        )

        try:
            steps = parse_decompose_plan(decompose_output)
            break
        except ValueError as e:
            logger.warning(
                "Decompose: Phase 1 parse failed (attempt %d/%d): %s",
                plan_attempt, DECOMPOSE_PLAN_MAX_ATTEMPTS, e,
            )
            logger.debug(
                "Decompose: raw decomposition output (attempt %d): %.2000s",
                plan_attempt, decompose_output,
            )

    if steps is None:
        logger.error(
            "Decompose: plan parsing failed after %d attempts — "
            "falling through to standard", DECOMPOSE_PLAN_MAX_ATTEMPTS,
        )
        protocol.emit_assessor_phase_complete(1, "decompose")
        protocol.emit_thread_finished(
            thread_id, 1,
            f"[decomposition failed after {DECOMPOSE_PLAN_MAX_ATTEMPTS} attempts]",
            {},
        )
        return await run_loop(
            llm_client=llm_client,
            history=history,
            system_prompt=system_prompt,
            tool_registry=tool_registry,
            tool_names=tool_names,
            node_id=node_id,
            soft_limit=soft_limit,
            mcp_config=mcp_config,
            instructions=instructions,
            agent_socket_path=agent_socket_path,
            controller_mode="standard",
        )

    protocol.emit_assessor_phase_complete(1, "decompose")
    logger.info(
        "Decompose: plan has %d steps: %s",
        len(steps), ", ".join(f"{s.index}:{s.type}" for s in steps),
    )

    # Build a plan summary string for logging and the summary phase
    plan_summary = "\n".join(
        f"  Step {s.index} ({s.type}): {s.description}" for s in steps
    )

    # ---- Phase 2: Sequential step execution ----
    carry_forward_parts: list[str] = []
    step_outcomes: list[dict] = []
    failed = False
    failure_reason = ""
    last_completed_step = 0

    # Initialize native messages if needed
    native_msgs: list[dict] | None = None
    native_compact_after: int = 0
    if use_native:
        native_msgs = []
        tool_prompt = tool_registry.format_tools_prompt(tool_names, backend="openai")
        sys_content = system_prompt
        if tool_prompt:
            sys_content += "\n\n" + tool_prompt
        native_msgs.append({"role": "system", "content": sys_content})
        task_content = "\n\n".join(msg.content for msg in history)
        if instructions:
            task_content += "\n\n" + instructions
        native_msgs.append({"role": "user", "content": task_content})
        native_compact_after = len(native_msgs)

    for step in steps:
        step_phase_id = step.index + 1  # phase 0=classify, 1=decompose, 2+=steps
        logger.info(
            "Decompose: executing step %d/%d (%s): %s",
            step.index, len(steps), step.type,
            _truncate_at_word(step.description, 100),
        )
        protocol.emit_assessor_phase_start(
            step_phase_id, f"step:{step.type}",
            _truncate_at_word(step.description, 200),
        )

        # Determine tool access based on step type
        if step.type in DECOMPOSE_READONLY_TYPES:
            step_tools = list(search_tools)
        else:
            step_tools = list(tool_names) if tool_names else None

        # Build carry-forward context
        carry_forward = "\n\n---\n\n".join(carry_forward_parts) if carry_forward_parts else ""

        # Retry loop for this step
        attempt = 0
        retry_hint = ""
        step_passed = False

        while attempt <= DECOMPOSE_MAX_RETRIES:
            attempt += 1
            logger.info(
                "Decompose: step %d attempt %d/%d",
                step.index, attempt, DECOMPOSE_MAX_RETRIES + 1,
            )

            step_prompt = _build_decompose_step_prompt(
                step=step,
                total_steps=len(steps),
                task=task_prompt,
                carry_forward=carry_forward,
                retry_hint=retry_hint if attempt > 1 else "",
            )

            # Execute the step — each step gets the full soft_limit.
            # Compaction between steps (below) handles freeing space
            # when carry-forward context grows too large.
            if use_native and native_msgs is not None:
                native_phase_start = len(native_msgs)
                native_msgs.append({"role": "user", "content": step_prompt})
                executor_result = await run_native_loop(
                    llm_client=llm_client,
                    messages=native_msgs,
                    tool_registry=tool_registry,
                    tool_names=step_tools,
                    node_id=node_id,
                    soft_limit=soft_limit,
                    instructions="",
                    agent_socket_path=agent_socket_path,
                )
            else:
                # Clean slate per attempt: start from the protected history zone
                step_history = list(history[:compact_after])
                step_history.append(HistoryMessage(
                    from_node="user",
                    content=step_prompt,
                    timestamp=_iso_now(),
                    source="in_flight",
                ))
                executor_result = await run_loop(
                    llm_client=llm_client,
                    history=step_history,
                    system_prompt=system_prompt,
                    tool_registry=tool_registry,
                    tool_names=step_tools,
                    node_id=node_id,
                    soft_limit=soft_limit,
                    mcp_config=mcp_config,
                    instructions="",
                    agent_socket_path=agent_socket_path,
                    controller_mode="standard",
                )

            # Assess the step
            assess_prompt = _DECOMPOSE_STEP_ASSESS_TEMPLATE.format(
                step_index=step.index,
                total_steps=len(steps),
                task=task_prompt,
                step_type=step.type,
                step_description=step.description,
                executor_prompt=step.executor_prompt,
                executor_output=executor_result[:8000],
            )
            assess_output = await _call_assessor(
                _assessor, assess_prompt, system_prompt, tool_registry,
                node_id, soft_limit, mcp_config, agent_socket_path,
            )
            assessment = parse_decompose_assessment(assess_output)

            logger.info(
                "Decompose: step %d attempt %d verdict=%s",
                step.index, attempt, assessment.verdict,
            )

            if assessment.verdict == "pass":
                step_passed = True
                carry_forward_parts.append(
                    f"Step {step.index} ({step.type}) — {step.description}:\n"
                    f"{executor_result}"
                )
                step_outcomes.append({
                    "step": step.index,
                    "type": step.type,
                    "description": step.description,
                    "verdict": "PASS",
                    "attempts": attempt,
                })
                last_completed_step = step.index

                # Compaction check (non-native only — native accumulates in native_msgs)
                if not use_native:
                    total_carry = sum(len(c) // 4 for c in carry_forward_parts)
                    if total_carry > compaction_threshold:
                        logger.info(
                            "Decompose: compacting carry-forward after step %d "
                            "(%dK > %dK threshold)",
                            step.index, total_carry // 1000,
                            compaction_threshold // 1000,
                        )
                        compact_text = "\n\n---\n\n".join(carry_forward_parts)
                        compact_prompt = (
                            f"Summarize the following outputs from steps 1-{step.index} "
                            f"of a multi-step task. Preserve all key findings, file paths, "
                            f"data points, and decisions.\n\n{compact_text}"
                        )
                        compact_result = await _call_assessor(
                            _assessor, compact_prompt, system_prompt,
                            tool_registry, node_id, soft_limit, mcp_config,
                            agent_socket_path,
                        )
                        carry_forward_parts = [compact_result]

                if use_native and native_msgs is not None:
                    est = estimate_native_tokens(native_msgs)
                    scratchpad = est - initial_tokens
                    if scratchpad > compaction_threshold:
                        logger.info(
                            "Decompose: compacting native history after step %d "
                            "(%dK > %dK threshold)",
                            step.index, scratchpad // 1000,
                            compaction_threshold // 1000,
                        )
                        await _compact_native_history(
                            native_msgs, native_compact_after, step.index,
                            phase_prompt=step.executor_prompt,
                            executor_output=executor_result,
                            llm_client=llm_client,
                            soft_limit=soft_limit,
                        )

                break

            elif assessment.verdict == "retry":
                retry_hint = assessment.hint
                if use_native and native_msgs is not None:
                    # Discard the failed attempt from native messages
                    del native_msgs[native_phase_start:]
                logger.info(
                    "Decompose: step %d retry (hint: %s)",
                    step.index, _truncate_at_word(retry_hint, 100),
                )

            elif assessment.verdict == "fail":
                failed = True
                failure_reason = assessment.reason
                step_outcomes.append({
                    "step": step.index,
                    "type": step.type,
                    "description": step.description,
                    "verdict": "FAIL",
                    "reason": failure_reason,
                    "attempts": attempt,
                })
                logger.warning(
                    "Decompose: step %d FAILED: %s",
                    step.index, failure_reason,
                )
                break

        if not step_passed and not failed:
            # Exhausted retries without passing
            failed = True
            failure_reason = f"Step {step.index} failed after {DECOMPOSE_MAX_RETRIES + 1} attempts"
            step_outcomes.append({
                "step": step.index,
                "type": step.type,
                "description": step.description,
                "verdict": "FAIL",
                "reason": failure_reason,
                "attempts": attempt,
            })
            logger.warning("Decompose: step %d exhausted retries", step.index)

        protocol.emit_assessor_phase_complete(step_phase_id, f"step:{step.type}")

        if failed:
            # Add skipped steps to outcomes
            for remaining_step in steps[step.index:]:
                if remaining_step.index > step.index:
                    step_outcomes.append({
                        "step": remaining_step.index,
                        "type": remaining_step.type,
                        "description": remaining_step.description,
                        "verdict": "SKIPPED",
                    })
            break

    # ---- Phase 3: Summary synthesis ----
    logger.info("Decompose: Phase 3 — synthesizing summary")
    summary_phase_id = len(steps) + 2
    protocol.emit_assessor_phase_start(summary_phase_id, "summarize", "Producing task summary")

    completion_status = "completed successfully" if not failed else "been terminated early due to a failure"

    outcomes_text = "\n".join(
        f"  Step {o['step']} ({o['type']}): {o['description']} — "
        f"{o['verdict']}"
        + (f" (attempts: {o.get('attempts', '?')})" if o.get('attempts') else "")
        + (f"\n    Reason: {o.get('reason', '')}" if o.get('reason') else "")
        for o in step_outcomes
    )

    # Collect modified files from step history
    files_mod: set[str] = set()
    for cf in carry_forward_parts:
        for fp in _FILE_PATH_RE.findall(cf):
            if any(fp.endswith(ext) for ext in (".py", ".js", ".ts", ".c", ".h", ".yaml", ".json", ".md", ".txt", ".tex", ".html", ".css", ".sh")):
                files_mod.add(fp)

    step_results_text = "\n\n---\n\n".join(carry_forward_parts) if carry_forward_parts else "(no results captured)"
    step_results_text = step_results_text[:12000]

    summary_prompt = _DECOMPOSE_SUMMARY_TEMPLATE.format(
        completion_status=completion_status,
        task=task_prompt[:4000],
        total_steps=len(steps),
        plan_summary=plan_summary,
        step_outcomes=outcomes_text,
        step_results=step_results_text,
        files_modified="\n".join(f"  {f}" for f in sorted(files_mod)) if files_mod else "  (none detected)",
    )

    summary_output = await _call_assessor(
        _assessor, summary_prompt, system_prompt, tool_registry,
        node_id, soft_limit, mcp_config, agent_socket_path,
    )

    protocol.emit_assessor_phase_complete(summary_phase_id, "summarize")
    protocol.emit_thread_finished(thread_id, summary_phase_id, summary_output, {})

    logger.info(
        "Decompose: finished — %d/%d steps passed, failed=%s",
        last_completed_step, len(steps), failed,
    )
    return summary_output
