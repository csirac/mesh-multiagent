# SPDX-License-Identifier: Apache-2.0
"""
Agent node - LLM-driven autonomous participant.

Processes incoming messages using an LLM, handles tool calls internally,
and routes responses to the appropriate destination.

Supports tool confirmation for sensitive operations: when a tool has
`requires_confirmation=True`, the agent sends a CONFIRM_REQUEST to the
original user and waits for their CONFIRM_RESPONSE before executing.

Node ID format: agent:{type}:{nickname}
  - type: The agent type (e.g., "coder", "researcher")
  - nickname: A unique, human-friendly name for addressing
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import logging
import os
import re
import socket
import uuid
from pathlib import Path
from typing import Any, Callable, Awaitable

from .node import Node, SummaryState
from .protocol import (
    Attachment, Message, MessageType, ControlAction, make_confirm_request,
    make_message, build_agent_node_id, make_status_response,
    make_status_request, make_tool_activity, make_todo_get, make_todo_mutate,
)
from .config import NodeConfig, ControllerConfig, ControllerConfigV02, RelevanceRouterConfig
from .controller import get_controller, get_controller_v02, BaseController, ControllerDecision, ControllerContext, StreamingObserver, PhaseFlowController
from .llm import (
    LLMClient, LLMConfig, HistoryMessage, ImageAttachment, CCToolEvent, LLMStreamCallback,
    estimate_tokens, estimate_history_tokens, SUMMARIZATION_PROMPT,
)
from .tools import (
    ToolRegistry, ToolCall, get_registry,
)
from .memory import MemorySystem, MemorySystemV2
from .preferences import PreferenceExtractor
from .storage import MessageStore
from .relevance_router import RelevanceRouter, RelevanceResult
from .router_v2 import RouterV2, RouterV2Config, WorkerResult
from .conversation_history import ConversationHistory, Turn

logger = logging.getLogger(__name__)


# =============================================================================
# Status Diagnostic Formatters
# =============================================================================


def _format_uptime(seconds: float) -> str:
    """Format uptime seconds into human-readable string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes:02d}m"


def _format_status_report(sections: dict, node_id: str) -> str:
    """Format a full diagnostic report into human-readable text."""
    lines = [f"=== Agent Status: {node_id} ===", ""]

    # Identity
    if "identity" in sections:
        i = sections["identity"]
        lines.append("IDENTITY")
        lines.append(f"  Node:      {i.get('node_id', '?')}")
        lines.append(f"  Host:      {i.get('hostname', '?')} (PID {i.get('pid', '?')})")
        lines.append(f"  Uptime:    {_format_uptime(i.get('uptime_seconds', 0))}")
        lines.append(f"  Directory: {i.get('working_directory', '?')}")
        lines.append("")

    # LLM
    if "llm" in sections:
        ll = sections["llm"]
        lines.append("LLM")
        lines.append(f"  Worker:  {ll.get('backend', '?')} / {ll.get('model', '?')}")
        lines.append(f"  Router:  {ll.get('router_llm_backend', '?')} / {ll.get('router_llm_model', '?')}")
        lines.append("")

    # Router
    if "router" in sections:
        r = sections["router"]
        lines.append("ROUTER")
        state = r.get("state", "?").upper()
        lines.append(f"  State:   {state}")
        if r.get("worker_active"):
            elapsed = r.get("worker_elapsed_seconds")
            wid = r.get("worker_id", "?")
            lines.append(f"  Worker:  active ({wid}, {elapsed:.0f}s)" if elapsed else f"  Worker:  active ({wid})")
            snap = r.get("worker_snapshot_turns")
            if snap is not None:
                lines.append(f"  Snapshot: {snap} turns")
        else:
            lines.append("  Worker:  inactive")
        if r.get("session_stats"):
            ss = r["session_stats"]
            lines.append(f"  Session: {ss.get('user_turns', 0)} user turns, {ss.get('tool_calls', 0)} tool calls, {ss.get('total_chars', 0)} chars")
        lines.append("")

    # History
    if "history" in sections:
        h = sections["history"]
        if h.get("detail"):
            lines.append("HISTORY")
            lines.append(f"  {h['detail']}")
        else:
            turns = h.get("window_turns", 0)
            tokens = h.get("estimated_tokens", 0)
            soft = h.get("soft_limit_tokens", 0)
            hard = h.get("hard_limit_tokens", 0)
            pct = h.get("utilization_pct", 0)
            lines.append("HISTORY")
            lines.append(f"  Window:  {turns} turns (~{tokens:,} tokens)")
            lines.append(f"  Limits:  {soft:,} soft / {hard:,} hard ({pct:.0f}% utilized)")
            summ = "none (rolling window mode)" if not h.get("summarization_enabled") else "active"
            if h.get("summary_present"):
                summ = "present"
            lines.append(f"  Summary: {summ}")
            oldest = h.get("oldest_turn_timestamp", "?")
            newest = h.get("newest_turn_timestamp", "?")
            lines.append(f"  Range:   {oldest} -> {newest}")
        lines.append("")

    # Memory
    if "memory" in sections:
        m = sections["memory"]
        lines.append("MEMORY")
        if not m.get("enabled"):
            lines.append(f"  {m.get('detail', 'disabled')}")
        else:
            version = m.get("version", 1)
            lines.append(f"  Version: v{version}")
            lines.append(f"  Pool:    {m.get('pool_size', 0)} entries (max {m.get('pool_max_entries', '?')})")
            lines.append(f"  Active:  {m.get('active_set_size', 0)} / {m.get('active_set_target', '?')} target")
            # Active map (v2 only)
            active_proj = m.get("active_project")
            if active_proj:
                map_chars = m.get("active_map_chars", 0)
                map_words = map_chars // 5 if map_chars else 0
                lines.append(f"  Map:     {active_proj} ({map_chars:,} chars, ~{map_words:,} words)")
                map_count = m.get("map_count", 0)
                if map_count > 1:
                    lines.append(f"  Maps:    {map_count} total")
            elif version == 2:
                lines.append("  Map:     none")
            ago = m.get("last_reflection_ago_seconds")
            if ago is not None:
                lines.append(f"  Last reflection: {_format_uptime(ago)} ago")
            else:
                lines.append("  Last reflection: none")
        lines.append("")

    # Context Health
    if "context_health" in sections:
        ch = sections["context_health"]
        checks = ch.get("checks", [])
        if checks:
            lines.append("HEALTH CHECKS")
            for check in checks:
                icon = "+" if check.get("ok") else "!"
                lines.append(f"  {icon} {check.get('name', '?')} ({check.get('detail', '')})")
            lines.append("")

    return "\n".join(lines)


# =============================================================================
# v0.2 Controller Response Cleanup
# =============================================================================


def strip_controller_xml(response: str) -> str:
    """
    Strip v0.2 controller XML blocks from LLM response before sending to user.

    Removes <assessment>, <validation>, <plan>, and similar internal XML
    that the controller uses for flow decisions but shouldn't be shown to users.
    """
    # XML tags to strip (controller-internal blocks)
    xml_patterns = [
        r'<assessment>.*?</assessment>',
        r'<validation>.*?</validation>',
        r'<plan>.*?</plan>',
        r'<info_result>.*?</info_result>',
        r'<reasoning>.*?</reasoning>',
    ]

    cleaned = response
    for pattern in xml_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Clean up extra whitespace left behind
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()


# =============================================================================
# Worker Instructions (Phase 2: Prompt Unification)
# =============================================================================

WORKER_INSTRUCTIONS = """\
You are executing a task dispatched by the routing layer.
{routing_context}
Your conversation history contains the full prior conversation including
the request that triggered this work. Process the request using your
available tools. When finished, send your response to the requester
using send_message.

IMPORTANT: Call send_message exactly ONCE — only when you have your
final, complete answer. Do NOT send intermediate progress updates or
partial results. Gather all information first, then send one message.

NOTE: Tilde expansion (~/) is unreliable in this environment. Always use
absolute paths (e.g., /home/youruser/...) when referencing files.

TOOL USAGE — use the `mesh-tool` CLI for mesh tools:
The `mesh-tool` command is on your PATH. Use it to access mesh services
(email, calendar, notes, web search, memory, messaging, etc.).

Usage:
  mesh-tool                              # list all available tools
  mesh-tool <name>                       # show usage for a specific tool
  mesh-tool <name> --arg1 val1 --arg2 val2   # call a tool (returns JSON)

Examples:
  mesh-tool send_message --to "user:yourname" --content "Done — here are the results."
  mesh-tool gmail_search_emails --query "from:user subject:deploy" --limit 5
  mesh-tool notes_search --query "mesh architecture" --db personal
  mesh-tool exa_search --query "submodular optimization survey" --num_results 3
  mesh-tool memory_search --query "router restart incident"
  mesh-tool memory_get --id m_xxxx
  mesh-tool current_time

Return codes: exit 0 + JSON on success, exit 1 + error on failure.

IMPORTANT: Use `mesh-tool` for all mesh services — do NOT try to replicate
them with Bash/curl/Python scripts or any other workaround.

- For email: `mesh-tool gmail_search_emails`, `mesh-tool gmail_list_from_date`, etc.
- For calendar: `mesh-tool calendar_list_on_date`, etc.
- For scheduling/reminders: use mesh `schedule_wake` (with optional `recurrence`
  parameter for recurring timers) — NOT CC-native CronCreate. schedule_wake
  persists in SQLite across restarts; CronCreate is session-scoped and expires.
- For project map updates: use `map_review` — NOT manual map_get + map_edit.
- For web search/fetch: prefer `mesh-tool exa_search`, `mesh-tool extract_url`\
"""

WORKER_INSTRUCTIONS_MCP = """\
You are executing a task dispatched by the routing layer.
{routing_context}
Your conversation history contains the full prior conversation including
the request that triggered this work. Process the request using your
available tools. When finished, send your response to the requester
using send_message.

IMPORTANT: Call send_message exactly ONCE — only when you have your
final, complete answer. Do NOT send intermediate progress updates or
partial results. Gather all information first, then send one message.

NOTE: Tilde expansion (~/) is unreliable in this environment. Always use
absolute paths (e.g., /home/youruser/...) when referencing files.

TOOL USAGE — mesh tools are available as native MCP tools:
Your MCP tools include mesh-specific tools (gmail, calendar, send_message,
schedule_wake, etc.). Call them directly — they work like any other tool.
- For scheduling/reminders: use mesh `schedule_wake` (with optional `recurrence`
  parameter for recurring timers) — NOT CC-native CronCreate. schedule_wake
  persists in SQLite across restarts; CronCreate is session-scoped and expires.
- For project map updates: use `map_review` — NOT manual map_get + map_edit.\
"""

WORKER_BRIEFING_INSTRUCTIONS = """\
You are executing a task dispatched by the routing layer.
{routing_context}
Your system prompt contains:
- A **project map** with the project's architecture, key decisions, and current state.
- A **briefing** with condensed context from the conversation that led to this task.

Use these as your strategic anchor. If your work would contradict a decision in the
project map or briefing, stop and flag the conflict to the user.

IMPORTANT: Call send_message exactly ONCE — only when you have your
final, complete answer. Do NOT send intermediate progress updates or
partial results. Gather all information first, then send one message.

NOTE: Tilde expansion (~/) is unreliable in this environment. Always use
absolute paths (e.g., /home/youruser/...) when referencing files.

TOOL USAGE — use the `mesh-tool` CLI for mesh tools:
The `mesh-tool` command is on your PATH. Use it to access mesh services
(email, calendar, notes, web search, memory, messaging, etc.).

Usage:
  mesh-tool                              # list all available tools
  mesh-tool <name>                       # show usage for a specific tool
  mesh-tool <name> --arg1 val1 --arg2 val2   # call a tool (returns JSON)

Examples:
  mesh-tool send_message --to "user:yourname" --content "Done — here are the results."
  mesh-tool gmail_search_emails --query "from:user subject:deploy" --limit 5
  mesh-tool notes_search --query "mesh architecture" --db personal
  mesh-tool exa_search --query "submodular optimization survey" --num_results 3
  mesh-tool memory_search --query "router restart incident"
  mesh-tool memory_get --id m_xxxx
  mesh-tool current_time

Return codes: exit 0 + JSON on success, exit 1 + error on failure.

IMPORTANT: Use `mesh-tool` for all mesh services — do NOT try to replicate
them with Bash/curl/Python scripts or any other workaround.

- For email: `mesh-tool gmail_search_emails`, `mesh-tool gmail_list_from_date`, etc.
- For calendar: `mesh-tool calendar_list_on_date`, etc.
- For scheduling/reminders: use mesh `schedule_wake` (with optional `recurrence`
  parameter for recurring timers) — NOT CC-native CronCreate. schedule_wake
  persists in SQLite across restarts; CronCreate is session-scoped and expires.
- For project map updates: use `map_review` — NOT manual map_get + map_edit.
- For web search/fetch: prefer `mesh-tool exa_search`, `mesh-tool extract_url`\
"""

WORKER_BRIEFING_INSTRUCTIONS_MCP = """\
You are executing a task dispatched by the routing layer.
{routing_context}
Your system prompt contains:
- A **project map** with the project's architecture, key decisions, and current state.
- A **briefing** with condensed context from the conversation that led to this task.

Use these as your strategic anchor. If your work would contradict a decision in the
project map or briefing, stop and flag the conflict to the user.

IMPORTANT: Call send_message exactly ONCE — only when you have your
final, complete answer. Do NOT send intermediate progress updates or
partial results. Gather all information first, then send one message.

NOTE: Tilde expansion (~/) is unreliable in this environment. Always use
absolute paths (e.g., /home/youruser/...) when referencing files.

TOOL USAGE — mesh tools are available as native MCP tools:
Your MCP tools include mesh-specific tools (gmail, calendar, send_message,
schedule_wake, etc.). Call them directly — they work like any other tool.
- For scheduling/reminders: use mesh `schedule_wake` (with optional `recurrence`
  parameter for recurring timers) — NOT CC-native CronCreate. schedule_wake
  persists in SQLite across restarts; CronCreate is session-scoped and expires.
- For project map updates: use `map_review` — NOT manual map_get + map_edit.\
"""

BRIEFING_GENERATION_PROMPT = """\
You are preparing a briefing for a worker agent that will execute a specific task.
The worker will NOT see the full conversation history — only this briefing, the
project map, and the task description.

Your briefing must capture everything the worker needs to avoid contradicting
prior decisions or losing strategic context during extended execution.

Include:
1. **Project state**: What is the current state of the project? What has been accomplished?
2. **Key decisions**: What decisions have been made? What constraints apply?
3. **Recent context**: What was the user working on in the last few exchanges?
4. **Open questions**: What is unresolved or being explored?
5. **File/artifact references**: What specific files, paths, or artifacts are relevant?

Do NOT include:
- Routine greetings or acknowledgments
- Tool call details or error messages
- Mesh infrastructure details (routing, channels, agent management)

Target length: 1000-2000 words. Be specific and concrete — the worker needs
actionable context, not a vague summary.

<project_map>
{map_summary}
</project_map>

<conversation_history>
{history}
</conversation_history>

<upcoming_task>
{task_description}
</upcoming_task>

Write the briefing now. Start directly with the content.\
"""

BRIEFING_UPDATE_PROMPT = """\
You are updating a worker briefing with new conversation context.
The existing briefing was accurate when written. Revise it to incorporate
the new turns below. Preserve all existing decisions and context that
remain valid. Remove anything contradicted by the new turns.

<existing_briefing>
{existing_briefing}
</existing_briefing>

<new_conversation_turns>
{new_turns}
</new_conversation_turns>

<upcoming_task>
{task_description}
</upcoming_task>

Write the updated briefing. Same format and length constraints as the original.
Start directly with the content.\
"""

BRIEFING_STALE_THRESHOLD = 5
BRIEFING_REGEN_THRESHOLD = 20


# Trace-as-history emulation-risk framing (docs/plans/trace-as-history-2026-04-27.md §2.6.3).
# Appended to worker system prompts only when config.trace_as_history_enabled is True.
TRACE_HISTORY_FRAMING = """\
Past <tool_call> and <tool_result> elements appearing in the conversation
history are records of prior tool invocations, not templates for new ones.
To call a tool, use your registered toolset directly — the system will
invoke it natively. Do not emit <tool_call> XML in your final response
text; that XML is for history rendering only.
"""



# =============================================================================
# Scheduled Wake Data Model
# =============================================================================


@dataclass
class ScheduledWake:
    """A scheduled wake-up for the agent."""
    id: str
    wake_time: datetime  # UTC
    prompt: str
    requested_by: str = ""  # node ID of user who triggered the schedule
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    recurrence: str | None = None  # e.g. "daily", "weekly", "weekdays", "every 2 hours"


def parse_wake_time(time_str: str, local_tz: timezone | None = None) -> datetime:
    """
    Parse a wake time string into a UTC datetime.

    Supports:
    - ISO 8601: "2026-01-26T17:00:00-06:00"
    - Relative: "in 30 minutes", "in 2 hours", "in 1 day"
    - Natural time: "5pm", "17:00", "5:30pm" (uses local_tz, defaults to system)

    Returns datetime in UTC.
    """
    time_str = time_str.strip()

    # Try ISO 8601 first
    try:
        dt = datetime.fromisoformat(time_str)
        if dt.tzinfo is None:
            # Assume local timezone if not specified
            if local_tz:
                dt = dt.replace(tzinfo=local_tz)
            else:
                # Use system local timezone
                import time
                local_offset = timedelta(seconds=-time.timezone if time.daylight == 0 else -time.altzone)
                dt = dt.replace(tzinfo=timezone(local_offset))
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    # Try relative formats: "in 30 minutes", "in 2 hours", "in 1 day"
    rel_match = re.match(r'in\s+(\d+)\s+(minute|hour|day)s?', time_str.lower())
    if rel_match:
        amount = int(rel_match.group(1))
        unit = rel_match.group(2)
        if unit == "minute":
            delta = timedelta(minutes=amount)
        elif unit == "hour":
            delta = timedelta(hours=amount)
        elif unit == "day":
            delta = timedelta(days=amount)
        else:
            raise ValueError(f"Unknown time unit: {unit}")
        return datetime.now(timezone.utc) + delta

    # Try natural time: "5pm", "17:00", "5:30pm"
    # First, get the local timezone
    if local_tz is None:
        import time as time_module
        local_offset = timedelta(seconds=-time_module.timezone if time_module.daylight == 0 else -time_module.altzone)
        local_tz = timezone(local_offset)

    now_local = datetime.now(local_tz)

    # Try "5pm", "5:30pm", "5:30 pm"
    time_match = re.match(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', time_str.lower())
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2) or 0)
        ampm = time_match.group(3)

        if ampm:
            if ampm == "pm" and hour != 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0

        # Build datetime for today in local timezone
        target = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)

        # If the time has already passed today, schedule for tomorrow
        if target <= now_local:
            target += timedelta(days=1)

        return target.astimezone(timezone.utc)

    raise ValueError(f"Cannot parse time: {time_str}")


def compute_next_recurrence(current_wake: datetime, recurrence: str) -> datetime | None:
    """
    Compute the next wake time for a recurring schedule.

    Advances from current_wake (not from now) to prevent drift.

    Supported rules:
    - "daily" — same time every day
    - "weekly" — same time every 7 days
    - "weekdays" — same time Mon-Fri, skips weekends
    - "hourly" — every hour
    - "every N minutes" / "every N hours" / "every N days" — fixed interval

    Returns UTC datetime, or None if recurrence is invalid.
    """
    rule = recurrence.strip().lower()

    if rule == "daily":
        return current_wake + timedelta(days=1)
    elif rule == "weekly":
        return current_wake + timedelta(weeks=1)
    elif rule == "hourly":
        return current_wake + timedelta(hours=1)
    elif rule == "weekdays":
        next_time = current_wake + timedelta(days=1)
        # Skip Saturday (5) and Sunday (6)
        while next_time.weekday() >= 5:
            next_time += timedelta(days=1)
        return next_time

    # "every N minutes/hours/days"
    match = re.match(r"every\s+(\d+)\s+(minute|hour|day)s?", rule)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        if unit == "minute":
            return current_wake + timedelta(minutes=amount)
        elif unit == "hour":
            return current_wake + timedelta(hours=amount)
        elif unit == "day":
            return current_wake + timedelta(days=amount)

    return None


def is_nicknamed_mention(content: str, nicknames: list[str]) -> bool:
    """
    Check if content mentions any of the given nicknames (fuzzy matching).

    Fuzzy matching means:
    - Case-insensitive
    - Matches whole word or subword (e.g., "claude" matches "claude-sobek")
    - No @ symbol required

    Examples:
        is_nicknamed_mention("hey claude can you help", ["claude", "sobek"]) -> True
        is_nicknamed_mention("what do you think", ["claude", "alice"]) -> False
    """
    if not content or not nicknames:
        return False

    content_lower = content.lower()

    for nickname in nicknames:
        if not nickname:
            continue
        nickname_lower = nickname.lower()

        # Check if the nickname appears as a whole word or subword
        # Using word boundaries for whole-word match, but also allowing subword matches
        # e.g., "claude" matches "claude-sobek" but also "hey claude"
        pattern = r'\b' + re.escape(nickname_lower) + r'\b'
        if re.search(pattern, content_lower):
            return True

        # Also check for subword matches (e.g., "claude" in "claude-sobek")
        if nickname_lower in content_lower:
            return True

    return False


def is_at_mentioned(content: str, nicknames: list[str]) -> bool:
    """
    Check if content contains an @mention for any of the given nicknames.

    Requires the @ prefix, case-insensitive. This is stricter than
    is_nicknamed_mention() — prevents triggering on casual name references
    like "look at Bob's work" (only triggers on "@bob").

    Examples:
        is_at_mentioned("@bob check this", ["bob"]) -> True
        is_at_mentioned("@Bob check this", ["bob"]) -> True
        is_at_mentioned("look at bob's work", ["bob"]) -> False
        is_at_mentioned("hey everyone", ["bob"]) -> False
    """
    if not content or not nicknames:
        return False

    content_lower = content.lower()

    for nickname in nicknames:
        if not nickname:
            continue
        # Match @nickname with word boundary after (or end of string)
        pattern = r'@' + re.escape(nickname.lower()) + r'\b'
        if re.search(pattern, content_lower):
            return True

    return False


def format_cc_tool_call(tool_name: str, args: dict, max_width: int = 200) -> str:
    """Format a CC tool call like: ● cc:Read(file_path: "/path/to/file.py", limit: 100)"""
    if not args:
        return f"● {tool_name}()"

    param_parts = []
    for key, value in args.items():
        if value is None or value == "":
            continue
        if isinstance(value, str):
            # Truncate long strings
            if len(value) > 80:
                value = value[:77] + "..."
            param_parts.append(f'{key}: "{value}"')
        elif isinstance(value, bool):
            param_parts.append(f'{key}: {str(value).lower()}')
        elif isinstance(value, (int, float)):
            param_parts.append(f'{key}: {value}')
        else:
            # Complex types - compact JSON
            import json
            try:
                val_str = json.dumps(value)
                if len(val_str) > 60:
                    val_str = val_str[:57] + "..."
                param_parts.append(f'{key}: {val_str}')
            except (TypeError, ValueError):
                param_parts.append(f'{key}: ...')

    params_str = ", ".join(param_parts)
    full_str = f"{tool_name}({params_str})"

    if len(full_str) > max_width:
        full_str = full_str[:max_width - 3] + "..."

    return f"● {full_str}"


def format_cc_tool_result(tool_name: str, content: str | list, max_lines: int = 20) -> str:
    """Format a CC tool result with ⎿ prefix.

    For cc:Edit, shows diff preview. For cc:Read/cc:Bash, shows first lines of output.
    """
    import json

    # Extract text from content if it's a list of content blocks
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        content = "\n".join(parts)

    if not content:
        return "  ⎿  (no output)"

    # Try to parse as JSON for special handling (e.g., Edit tool with diff)
    try:
        data = json.loads(content) if isinstance(content, str) else content
        if isinstance(data, dict):
            # Handle file_edit style response with diff preview
            if data.get("ok") and "preview" in data:
                diff_lines = data["preview"].splitlines()[:max_lines]
                if diff_lines:
                    result = ["  ⎿  " + diff_lines[0]]
                    for line in diff_lines[1:]:
                        result.append("     " + line)
                    if len(data["preview"].splitlines()) > max_lines:
                        result.append(f"     ... +{len(data['preview'].splitlines()) - max_lines} lines")
                    return "\n".join(result)
    except (json.JSONDecodeError, TypeError):
        pass

    # Default: show first few lines
    lines = content.strip().splitlines()[:max_lines]
    if not lines:
        return "  ⎿  (no output)"

    result = ["  ⎿  " + lines[0][:200]]
    for line in lines[1:]:
        result.append("     " + line[:200])

    total_lines = len(content.strip().splitlines())
    if total_lines > max_lines:
        result.append(f"     ... +{total_lines - max_lines} lines")

    return "\n".join(result)


class CCToolCollector:
    """Collects CC tool events during LLM processing for later storage."""

    def __init__(
        self,
        realtime_list: list[CCToolEvent] | None = None,
        activity_callback: Callable[[CCToolEvent], Awaitable[None]] | None = None,
    ):
        self.events: list[CCToolEvent] = []
        self.todos: list[dict] = []
        # Optional real-time list for status queries to access during processing
        self._realtime_list = realtime_list
        # Optional async callback to push tool activity to the trigger sender
        self._activity_callback = activity_callback

    def on_cc_tool_event(self, event: CCToolEvent) -> None:
        """Called when a CC tool call or result is observed."""
        self.events.append(event)
        # Also append to real-time list if provided (for status queries)
        if self._realtime_list is not None:
            self._realtime_list.append(event)

        # Print to stdout for real-time visibility using formatted output
        if event.event_type == "tool_call":
            args = event.data if isinstance(event.data, dict) else {}
            print(f"  {format_cc_tool_call(event.tool_name, args)}")
        elif event.event_type == "tool_result":
            content = event.data if isinstance(event.data, str) else str(event.data)
            # For stdout, use fewer lines (3) to keep it concise
            formatted = format_cc_tool_result(event.tool_name, content, max_lines=3)
            print(f"  {formatted}")

        logger.debug(f"CC tool event: {event.event_type} {event.tool_name} call_id={event.call_id}")

        # Push to activity callback if set (for TOOL_ACTIVITY streaming)
        if self._activity_callback is not None:
            # Schedule the callback on the event loop (we're in a sync context)
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._activity_callback(event))
            except RuntimeError:
                pass  # No running loop, skip callback

    def on_todos(self, todos: list[dict]) -> None:
        """Called when TodoWrite updates are observed."""
        self.todos = todos
        logger.debug(f"CC TodoWrite: {len(todos)} todos")

    def clear(self) -> None:
        """Clear collected events."""
        self.events = []
        self.todos = []


class AgentNode(Node):
    """
    LLM-driven agent node.

    When a message arrives:
    1. Add to history
    2. Format history as XML and send to LLM
    3. If LLM requests tool call: execute internally, feed result back to LLM
       - If tool requires confirmation, request from user first
       - send_message tool calls are handled specially to route messages
    4. When LLM produces final response with no tool calls:
       - If there's plain text content, reject it and ask the agent to use send_message
       - Empty responses after send_message are allowed
    """

    # Default timeout for confirmation requests (seconds)
    CONFIRM_TIMEOUT = 60.0

    # Max iterations to prevent runaway loops
    MAX_ITERATIONS = 50

    # Max times to reject plain text without send_message before bailing
    MAX_PLAIN_TEXT_REJECTIONS = 3

    # Default summarization thresholds
    DEFAULT_SOFT_LIMIT = 50_000  # Trigger summarization when context exceeds this
    DEFAULT_TARGET_RATIO = 0.25  # Target = soft_limit * ratio after summarization

    def __init__(
        self,
        config: NodeConfig,
        llm_config: LLMConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        nickname: str | None = None,
        agent_type: str | None = None,
        description: str | None = None,
        history_file: Path | str | None = None,
        persist: bool = False,
        soft_limit: int | None = None,
        target_ratio: float | None = None,
        # Preference extraction settings
        pref_message_threshold: int | None = None,
        pref_context_limit: int | None = None,
        pref_stale_hours: int | None = None,
        pref_extraction_model: str | None = None,
        pref_extraction_backend: str | None = None,
        # In-flight context management
        keep_recent_results: int | None = None,
        # Optional message store for SQLite archiving
        message_store: MessageStore | None = None,
        # Sandbox settings
        sandboxed: bool = False,
        allowed_dirs: list[str] | None = None,
        allow_network: bool = True,
        # Relevance router for channel message filtering
        relevance_router_config: RelevanceRouterConfig | None = None,
    ):
        # Build the node ID from type and nickname
        # Priority: explicit params > config fields > parse from config.id
        self._agent_type = agent_type or config.agent_type
        self._nickname = nickname or config.nickname

        # If we have type and nickname, build the full node ID
        if self._agent_type and self._nickname:
            config.id = build_agent_node_id(self._agent_type, self._nickname)
        elif self._agent_type and not self._nickname:
            # Type but no nickname - auto-generate
            import secrets
            self._nickname = secrets.token_hex(2)
            config.id = build_agent_node_id(self._agent_type, self._nickname)

        # Set MESH_NODE_ID in process env for tool subprocess inheritance
        # (bash_exec, etc.). LLM subprocesses use LLMConfig.node_id explicitly.
        os.environ["MESH_NODE_ID"] = config.id

        # Use nickname for history file path (not full node_id)
        # This ensures the same nickname always uses the same history,
        # regardless of agent type changes
        if not history_file and persist and self._nickname:
            from .node import DEFAULT_HISTORY_DIR
            history_file = DEFAULT_HISTORY_DIR / f"agent-{self._nickname}.json"

        super().__init__(config, history_file=history_file, persist=persist)

        # Store type, nickname, and description for later access
        self.agent_type = self._agent_type
        self.nickname = self._nickname
        self.description = description

        self.llm_config = llm_config
        self.llm_client: LLMClient | None = None
        self.tool_registry = tool_registry or get_registry()
        self.enabled_tools: list[str] = config.tools  # List of tool names from config
        self.system_prompt = config.system_prompt

        # Build tool prompt once at init (only for enabled tools)
        self._tool_prompt = self.tool_registry.format_tools_prompt(
            self.enabled_tools if self.enabled_tools else None
        ) if self.enabled_tools else ""

        # Polling interval for checking inbox (when idle)
        self.poll_interval = 1.0

        # Pending confirmations: msg_id -> asyncio.Event
        self._pending_confirms: dict[str, asyncio.Event] = {}
        # Confirmation results: msg_id -> bool
        self._confirm_results: dict[str, bool] = {}

        # Summarization settings — prefer config, then CLI arg, then default
        config_soft = getattr(config, 'history_soft_limit_tokens', None)
        self._soft_limit = config_soft or soft_limit or self.DEFAULT_SOFT_LIMIT
        self._target_ratio = target_ratio or self.DEFAULT_TARGET_RATIO
        self._target = int(self._soft_limit * self._target_ratio)

        # Summary state
        self._summary: SummaryState | None = None
        self._summarizing = False  # Lock to prevent concurrent summarization
        self._summarization_task: asyncio.Task | None = None

        # Worker briefing state (for cc_worker_briefing feature)
        self._worker_briefing: str | None = None
        self._briefing_history_len: int = 0

        # Shared ConversationHistory for LLM context building + summarization
        # _history remains the canonical append-only store; _conv_history provides
        # summary+window context building and delegates summarization to its logic.
        summary_path = None
        if self._history_file:
            summary_path = self._history_file.with_suffix(".summary.json")
        config_hard = getattr(config, 'history_hard_limit_tokens', 40_000)
        config_window = getattr(config, 'history_window_tokens', None)
        summarization_enabled = getattr(config, 'history_summarization_enabled', False)
        self._conv_history = ConversationHistory(
            soft_token_limit=self._soft_limit,
            hard_token_limit=config_hard,
            target_ratio=self._target_ratio,
            window_budget=config_window,
            summary_persist_path=summary_path,
            summarization_enabled=summarization_enabled,
        )

        # Tracks how far we've synced _history → _conv_history.
        # Incremental sync avoids rebuilding the entire window each call,
        # which was causing hard-limit pruning and summarization trims to
        # be undone on the next _sync_conv_history() invocation.
        self._history_sync_idx: int = 0

        # Preference extraction (defaults to gemini from preferences.py)
        from .preferences import DEFAULT_EXTRACTION_BACKEND, DEFAULT_EXTRACTION_MODEL
        self._preference_extractor = PreferenceExtractor(
            history_file=self._history_file,
            message_threshold=pref_message_threshold or 50,
            context_limit=pref_context_limit or 100_000,
            stale_hours=pref_stale_hours or 24,
            extraction_model=pref_extraction_model or DEFAULT_EXTRACTION_MODEL,
            extraction_backend=pref_extraction_backend or DEFAULT_EXTRACTION_BACKEND,
        )

        # Message store for SQLite archiving (optional)
        self._message_store = message_store

        # In-flight context management: how many recent tool results to always keep
        self._keep_recent_results = keep_recent_results if keep_recent_results is not None else 3

        # Sandbox settings
        self.sandboxed = sandboxed
        self.allowed_dirs = allowed_dirs or []
        self.allow_network = allow_network

        # Configure sandbox for tool implementations
        if self.sandboxed:
            from .tool_implementations import configure_sandbox
            configure_sandbox(
                sandboxed=self.sandboxed,
                allowed_dirs=self.allowed_dirs,
                allow_network=self.allow_network,
            )

        # Real-time CC tool activity tracking (for status queries during LLM processing)
        # _current_cc_events: populated by the WORKER path (_process_with_llm).
        # _router_cc_events: populated by the ROUTER path (_router_process_with_llm).
        # Split prevents the router's own tool calls from leaking into worker
        # activity monitoring — the watchdog reads _current_cc_events only, so
        # router-originated CC events no longer masquerade as worker progress.
        self._current_cc_events: list[CCToolEvent] = []
        self._router_cc_events: list[CCToolEvent] = []
        self._current_cc_events_lock = asyncio.Lock()

        # Worker snapshot: set by _router_v2_worker() to the mutable list[Turn]
        # the router uses for live progress visibility. None when not in worker mode.
        self._worker_snapshot: list | None = None

        # Message queue for serialized processing: messages that arrive during
        # an active LLM loop are queued and incorporated into the current context
        # rather than spawning parallel processing tasks
        self._processing = False
        self._message_queue: list[Message] = []
        self._processing_lock = asyncio.Lock()

        # Abort flag: set by reset_context to interrupt in-flight LLM processing
        self._abort_processing = False

        # Scheduled wakes (agent-local timer management)
        self._scheduled_wakes: dict[str, ScheduledWake] = {}
        self._scheduler_task: asyncio.Task | None = None
        self._scheduler_check_interval = 10.0  # Check every 10 seconds

        # Initialize controller based on config
        # Controller manages message routing and task tracking
        controller_config = config.controller
        if controller_config is None:
            # Default to passthrough mode (preserves existing behavior)
            controller_mode = "passthrough"
        else:
            controller_mode = controller_config.mode

        # v0.2 controller uses get_controller_v02() for streaming support
        # The streaming callback will be set up lazily when we have a connection
        if isinstance(controller_config, ControllerConfigV02) and controller_mode == "phase-flow-v02":
            # Initialize v0.2 controller with logging observer for now
            # Streaming will be set up in _handle_incoming_message when we know the recipient
            self.controller: BaseController = get_controller_v02(config=controller_config)
            self._is_v02_controller = True
            logger.info(f"Controller initialized for {config.id}: mode={controller_mode} (v0.2 phase-flow)")
        else:
            self.controller: BaseController = get_controller(controller_mode, controller_config)
            self._is_v02_controller = False
            logger.info(f"Controller initialized for {config.id}: mode={controller_mode}")

        # Initialize relevance router for channel message filtering
        # This uses LLM scoring to decide if channel messages are relevant
        self._relevance_router: RelevanceRouter | None = None
        self._relevance_router_config = relevance_router_config
        if relevance_router_config is not None:
            self._relevance_router = RelevanceRouter(
                config=relevance_router_config,
                agent_nickname=self._nickname or "",
                agent_description=description or config.system_prompt[:200] if config.system_prompt else "",
                nicknames=self._get_nicknames_for_mention_check() if self._nickname else [],
            )
            logger.info(f"Relevance router enabled for {config.id}: threshold={relevance_router_config.threshold}")

        # Memory system — initialized in connect() if memory_enabled
        self._memory_system: MemorySystem | None = None

        # Memory Formation v3 (rev 6) — task handles + token-pressure counter.
        self._startup_formation_task: asyncio.Task | None = None
        self._formation_timer_task: asyncio.Task | None = None
        self._token_pressure_task: asyncio.Task | None = None
        self._uncommitted_token_count: int = 0

        # Router V2 - mediating router between I/O and LLM processing
        # Initialized lazily in connect() after LLM client is ready
        self._router_v2: RouterV2 | None = None
        # Use unified history fields (70K/90K defaults match mesh.yaml)
        _r_soft = getattr(config, 'history_soft_limit_tokens', 70_000)
        _r_hard = getattr(config, 'history_hard_limit_tokens', 105_000)
        _r_window = getattr(config, 'history_window_tokens', None)
        self._router_v2_config = RouterV2Config(
            llm_enabled=getattr(config, 'router_v2_llm_enabled', True),
            synthesize_enabled=getattr(config, 'synthesize_enabled', True),
            deliver_buffered_verbatim=getattr(config, 'deliver_buffered_verbatim', False),
            worker_digest_max_tokens=getattr(config, 'worker_digest_max_tokens', 15_000),
            synthesis_max_tokens=getattr(config, 'synthesis_max_tokens', 150_000),
            history_window_tokens=_r_window,
            history_soft_limit_tokens=_r_soft,
            history_hard_limit_tokens=_r_hard,
            history_target_ratio=getattr(config, 'router_history_target_ratio', 0.25),
            history_persist=getattr(config, 'router_history_persist', True),
            history_summarization_enabled=getattr(config, 'history_summarization_enabled', False),
            worker_context_window_tokens=getattr(config, 'worker_context_window_tokens', 80_000),
            router_mode=getattr(config, 'router_mode', 'classifier'),
            router_max_iters=getattr(config, 'router_max_iters', 10),
            pipeline_backend=getattr(config, 'pipeline_backend', 'deepseek'),
            pipeline_plan_path=getattr(config, 'pipeline_plan_path', ''),
            trace_as_history_enabled=getattr(config, 'trace_as_history_enabled', False),
            tool_result_max_lines=getattr(config, 'tool_result_max_lines', 80),
            tool_result_max_chars=getattr(config, 'tool_result_max_chars', 6400),
            memory_retrieval_redesign_enabled=getattr(config, 'memory_retrieval_redesign_enabled', False),
            memory_toc_size=getattr(config, 'memory_toc_size', 30),
            standing_digest_enabled=getattr(config, 'standing_digest_enabled', False),
            standing_digest_path=getattr(config, 'standing_digest_path', ''),
        ) if getattr(config, 'use_router_v2', True) else None
        # Separate LLM config for router (if configured, avoids sharing LLM with worker)
        self._router_v2_llm_config: LLMConfig | None = None
        # Separate LLM config for memory operations (formation, etc.)
        self._memory_llm_config: LLMConfig | None = None
        # Resolved LLM config for the native harness session backend (set by
        # run_agent.py from NodeConfig.harness_session_backend). Read by the
        # RouterV2 → HarnessSessionManager to build the session subprocess.
        self._harness_session_llm_config: LLMConfig | None = None
        # Store the original send method so the router can always use it,
        # even when the worker temporarily monkey-patches self.send.
        self._original_send = self.send
        # F5: Track last known user node for fallback routing
        self._last_user_node: str | None = None

    def _get_registration_content(self) -> dict:
        """Add description and backend info to registration message."""
        import socket
        content = super()._get_registration_content()
        if self.description:
            content["description"] = self.description
        # Include LLM backend info for roster display
        if self.llm_config:
            content["llm_backend"] = self.llm_config.backend or "unknown"
            content["llm_model"] = self.llm_config.model or ""
        if self.config.router_v2_llm_backend:
            content["router_v2_llm_backend"] = self.config.router_v2_llm_backend
            content["router_v2_llm_model"] = self.config.router_v2_llm_model or ""
        if self.config.harness_session_backend:
            content["harness_session_backend"] = self.config.harness_session_backend
        if self.config.cc_interactive_tools:
            content["cc_interactive_tools"] = True
            content["cc_interactive_model"] = self.config.cc_interactive_model or ""
            content["cc_interactive_binary"] = self.config.cc_interactive_binary or ""
            content["cc_interactive_effort"] = self.config.cc_interactive_effort or ""
        # Include hostname
        try:
            content["hostname"] = socket.gethostname()
        except Exception:
            pass
        return content

    def load_preferences_from_disk(self) -> bool:
        """Load saved preferences from disk if available."""
        return self._preference_extractor.load_preferences()

    async def connect(self) -> None:
        """Connect to router and initialize LLM client."""
        import time as _time
        self._start_time = _time.monotonic()
        await super().connect()

        # Load preferences and check for staleness
        self._preference_extractor.load_preferences()
        if self.llm_config and self._persist:
            await self._preference_extractor.maybe_extract_on_startup(
                self._history, self.llm_config
            )

        # Load controller state (tasks, etc.)
        await self.controller.load_state()
        logger.info(f"Controller state loaded for {self.node_id}")

        # Initialize LLM client if config provided
        if self.llm_config:
            backend = self.llm_config.backend
            # Check if we have necessary credentials for the backend
            can_init = False
            if backend in ("openai", "openai-reasoning"):
                # Allow if we have api_key OR base_url (local endpoints don't need api_key)
                if self.llm_config.api_key or self.llm_config.base_url:
                    can_init = True
            elif backend == "anthropic" and self.llm_config.api_key:
                can_init = True
            elif backend in ("claude-code", "claude-interactive", "zai", "codex", "mesh-harness"):
                # These use subprocess, don't require api_key in config
                can_init = True

            if can_init:
                self.llm_client = LLMClient(self.llm_config)
                self.llm_client.config.agent_label = self.nickname
                self.llm_client.config.node_id = self.node_id
                logger.info(f"LLM client initialized for {self.node_id} (backend={backend})")
            else:
                logger.warning(f"LLM backend {backend} not configured for {self.node_id}, will echo messages")
        else:
            logger.warning(f"No LLM config for {self.node_id}, will echo messages")

        # Summarization (rolling-window context compression) uses the main LLM client.
        if getattr(self.config, 'history_summarization_enabled', False):
            logger.info("Summarization enabled (uses main LLM client)")
        else:
            logger.info("Worker summarization disabled (rolling window mode)")

        # Log tool configuration
        if self.enabled_tools:
            logger.info(f"Tools enabled for {self.node_id}: {self.enabled_tools}")
        else:
            logger.info(f"No tools configured for {self.node_id}")

        # Start the scheduler loop for scheduled wakes
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info(f"Scheduler loop started for {self.node_id}")

        # Start tool socket for any backend that spawns subprocesses needing
        # agent-local tool access (MCP sidecar, mesh-harness, or mesh-tool CLI).
        _needs_tool_socket = (
            getattr(self.llm_config, 'cc_use_mcp', False)
            or self.llm_config.backend in ("mesh-harness", "claude-code", "zai", "codex", "claude-interactive")
        ) if self.llm_config else False
        # Native harness sessions spawn a subprocess that reaches mesh/agent-local
        # tools over this socket, regardless of the worker backend.
        if getattr(self.config, 'harness_session_tools', False):
            _needs_tool_socket = True
        if _needs_tool_socket:
            self._tool_socket_path = await self._start_tool_socket()
            os.environ["MESH_SOCKET_PATH"] = self._tool_socket_path
            self._current_trigger_msg = None  # initialized; set by worker loop before LLM call
        else:
            self._tool_socket_path = None

        # Check if context is bloated and run summarization at startup
        await self._maybe_summarize_on_startup()

        # Initialize memory system if enabled
        if self.config.memory_enabled and self.llm_client:
            # Build a dedicated LLM client for memory formation if configured
            memory_llm_client = None
            if self._memory_llm_config:
                memory_llm_client = LLMClient(self._memory_llm_config)
                memory_llm_client.config.agent_label = f"{self.nickname}-memory"
                memory_llm_client.config.node_id = self.node_id
                logger.info(
                    "Memory LLM client: backend=%s model=%s",
                    self._memory_llm_config.backend,
                    self._memory_llm_config.model,
                )

            if self.config.memory_version >= 2:
                self._memory_system = MemorySystemV2(
                    nickname=self._nickname or self.node_id,
                    llm_client=self.llm_client,
                    active_size=self.config.memory_active_size,
                    pool_max_entries=self.config.memory_pool_max_entries,
                    embedding_backend=self.config.memory_embedding_backend,
                    embedding_model=self.config.memory_embedding_model,
                    reflection_min_tools=self.config.memory_reflection_min_tools,
                    retrieval_k=self.config.memory_retrieval_k,
                    trace_max_tokens=self.config.memory_trace_max_tokens,
                    reflection_min_discussion_turns=self.config.memory_reflection_min_discussion_turns,
                    reflection_min_discussion_chars=self.config.memory_reflection_min_discussion_chars,
                    reflection_min_brainstorm_response_chars=self.config.memory_reflection_min_brainstorm_response_chars,
                    reflection_max_brainstorm_tools=self.config.memory_reflection_max_brainstorm_tools,
                    reflection_cooldown_secs=self.config.memory_reflection_cooldown_secs,
                    recent_log_count=self.config.memory_recent_log_count,
                    retrieve_budget_tokens=self.config.memory_retrieve_budget_tokens,
                    retrieve_max_rounds=self.config.memory_retrieve_max_rounds,
                    curation_audit_max_tool_calls=self.config.memory_curation_audit_max_tool_calls,
                    review_max_tool_calls=self.config.memory_review_max_tool_calls,
                    formation_v3_enabled=getattr(self.config, "memory_formation_v3_enabled", False),
                    formation_v3_window_size=getattr(self.config, "memory_v3_window_size", 60),
                    formation_v3_overlap=getattr(self.config, "memory_v3_overlap", 20),
                    formation_v3_defer_tail=getattr(self.config, "memory_v3_defer_tail", 10),
                    formation_v3_model=getattr(self.config, "memory_v3_model", None) or None,
                    formation_v3_parse_failure_fallback_threshold=getattr(
                        self.config, "memory_v3_parse_failure_fallback_threshold", 3,
                    ),
                    payload_max_chars=getattr(
                        self.config, "memory_get_payload_max_chars", 6000,
                    ),
                    formation_llm_client=memory_llm_client,
                )
                logger.info("Using MemorySystemV2 (memory_version=2)")
            else:
                self._memory_system = MemorySystem(
                    nickname=self._nickname or self.node_id,
                    llm_client=self.llm_client,
                    active_size=self.config.memory_active_size,
                    pool_max_entries=self.config.memory_pool_max_entries,
                    embedding_backend=self.config.memory_embedding_backend,
                    embedding_model=self.config.memory_embedding_model,
                    reflection_min_tools=self.config.memory_reflection_min_tools,
                    retrieval_k=self.config.memory_retrieval_k,
                    worker_full_reflections=self.config.memory_worker_full_reflections,
                    router_full_reflections=self.config.memory_router_full_reflections,
                    router_recent_reflections=self.config.memory_router_recent_reflections,
                    worker_recent_reflections=self.config.memory_worker_recent_reflections,
                    trace_max_tokens=self.config.memory_trace_max_tokens,
                    reflection_max_tokens=self.config.memory_reflection_max_tokens,
                    reflection_min_discussion_turns=self.config.memory_reflection_min_discussion_turns,
                    reflection_min_discussion_chars=self.config.memory_reflection_min_discussion_chars,
                    reflection_min_brainstorm_response_chars=self.config.memory_reflection_min_brainstorm_response_chars,
                    reflection_max_brainstorm_tools=self.config.memory_reflection_max_brainstorm_tools,
                    reflection_cooldown_secs=self.config.memory_reflection_cooldown_secs,
                    light_profile_config=self.config.memory_profile_light,
                    deep_profile_config=self.config.memory_profile_deep,
                    router_profile_config=self.config.memory_router_profile,
                    worker_profile_config=self.config.memory_worker_profile,
                )
            await self._memory_system.initialize()

            # ── Memory Formation v3: wire triggers ────────────────────
            # The cursor-advance callback resets the agent-node token counter
            # (§2.7.9). Set after initialize() since the lock is created there.
            if getattr(self._memory_system, "_formation_v3_enabled", False):
                def _reset_token_counter():
                    self._uncommitted_token_count = 0
                self._memory_system._on_cursor_advance = _reset_token_counter

                # Kick off the startup formation chain (migration + form_un_formed)
                # as a background task so connect() returns promptly. Cancelled
                # in disconnect() if the agent is shut down before it finishes.
                self._startup_formation_task = asyncio.create_task(
                    self._v3_startup_formation_chain(),
                    name=f"{self.node_id}-v3-startup-formation",
                )

                # Time-based timer task (default 1800s).
                self._formation_timer_task = asyncio.create_task(
                    self._v3_formation_timer_loop(),
                    name=f"{self.node_id}-v3-formation-timer",
                )

            # Seed personality from config (only if DB has none yet)
            if self.config.personality:
                self._memory_system.seed_personality(self.config.personality)
            # Set module-level singleton for tool access
            import mesh.tool_implementations as _ti
            _ti._memory_system = self._memory_system
            _ti._memory_search_mode = self.config.memory_search_mode
            logger.info(f"Memory system enabled for {self.node_id}")

            # Load persisted scheduled wakes from SQLite
            try:
                persisted_wakes = self._memory_system._store.load_wakes()
                now = datetime.now(timezone.utc)
                loaded = 0
                expired = 0
                for w in persisted_wakes:
                    wake_time = datetime.fromisoformat(w["wake_time"])
                    created_at = datetime.fromisoformat(w["created_at"])
                    recurrence = w.get("recurrence")
                    if wake_time <= now:
                        if recurrence:
                            # Recurring wake expired — advance to next future occurrence
                            next_time = wake_time
                            while next_time <= now:
                                computed = compute_next_recurrence(next_time, recurrence)
                                if computed is None:
                                    break
                                next_time = computed
                            if next_time > now:
                                self._scheduled_wakes[w["id"]] = ScheduledWake(
                                    id=w["id"],
                                    wake_time=next_time,
                                    prompt=w["prompt"],
                                    requested_by=w.get("requested_by", ""),
                                    created_at=created_at,
                                    recurrence=recurrence,
                                )
                                self._memory_system._store.save_wake(
                                    wake_id=w["id"],
                                    wake_time=next_time.isoformat(),
                                    prompt=w["prompt"],
                                    requested_by=w.get("requested_by", ""),
                                    created_at=created_at.isoformat(),
                                    recurrence=recurrence,
                                )
                                loaded += 1
                                logger.info(
                                    f"Advanced recurring wake {w['id']} to {next_time.isoformat()} "
                                    f"(rule={recurrence})"
                                )
                                continue
                        # One-shot or invalid recurrence — purge
                        self._memory_system._store.delete_wake(w["id"])
                        expired += 1
                        logger.warning(
                            f"Expired scheduled wake {w['id']} "
                            f"(was due {wake_time.isoformat()}, requested_by={w['requested_by']})"
                        )
                    else:
                        self._scheduled_wakes[w["id"]] = ScheduledWake(
                            id=w["id"],
                            wake_time=wake_time,
                            prompt=w["prompt"],
                            requested_by=w.get("requested_by", ""),
                            created_at=created_at,
                            recurrence=recurrence,
                        )
                        loaded += 1
                if loaded or expired:
                    logger.info(
                        f"Loaded {loaded} persisted wakes, {expired} expired "
                        f"(purged) for {self.node_id}"
                    )
            except Exception as e:
                logger.warning(f"Failed to load persisted wakes: {e}")

        # Initialize Router V2 if enabled (needs LLM client to be ready)
        if self._router_v2_config is not None and self.llm_client:
            self._init_router_v2()
            logger.info(
                f"Router V2 enabled for {self.node_id}: "
                f"llm_enabled={self._router_v2_config.llm_enabled}"
            )

        # Auto-join configured channels
        if self.config.channels:
            for channel_name in self.config.channels:
                await self._join_channel(channel_name)

    async def _join_channel(self, channel_name: str) -> None:
        """Join a channel during startup.

        Note: This is fire-and-forget since the main message loop handles ACKs.
        The channel must already exist (only users can create channels).
        """
        from .protocol import ControlAction

        join_msg = Message(
            from_node=self.node_id,
            to_node="router",
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_JOIN.value,
                "channel_name": channel_name,
            },
        )
        await self.send_message(join_msg)
        logger.info(f"Requested channel join: {channel_name}")

    # ── Memory Formation v3 helpers ──────────────────────────────────

    async def _v3_startup_formation_chain(self) -> None:
        """Run embedding migration then startup formation as a background task."""
        try:
            if not self._memory_system:
                return
            try:
                await self._memory_system._maybe_run_v3_embedding_migration()
            except Exception as e:
                logger.warning("v3 embedding migration failed: %s", e)
            try:
                turns = list(self._conv_history.window) if self._conv_history else []
                if turns:
                    await self._memory_system.form_un_formed(turns, "startup")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("v3 startup formation failed: %s", e)
        except asyncio.CancelledError:
            logger.info("v3 startup formation task cancelled")
            raise

    async def _v3_formation_timer_loop(self) -> None:
        """Background task: fire `form_un_formed` every interval_seconds.

        Trims un-formed turns to those older than `defer_tail_seconds` so the
        live conversation isn't stolen mid-flight (§2.7.4).
        """
        interval = max(
            10,
            int(getattr(self.config, "memory_formation_interval_seconds", 1800)),
        )
        defer_tail_secs = max(
            0,
            int(getattr(self.config, "memory_formation_defer_tail_seconds", 300)),
        )
        try:
            while True:
                await asyncio.sleep(interval)
                if not self._memory_system or not getattr(
                    self._memory_system, "_formation_v3_enabled", False,
                ):
                    continue
                # Skip if a formation is already running.
                lock = getattr(self._memory_system, "_formation_lock", None)
                if lock and lock.locked():
                    continue
                try:
                    history = list(self._conv_history.window) if self._conv_history else []
                    if not history:
                        continue
                    if defer_tail_secs > 0:
                        cutoff = datetime.now(timezone.utc).timestamp() - defer_tail_secs
                        trimmed = []
                        for t in history:
                            ts = getattr(t, "timestamp", None)
                            ts_dt = ts if isinstance(ts, datetime) else None
                            if isinstance(ts, str) and ts:
                                try:
                                    ts_dt = datetime.fromisoformat(ts)
                                except ValueError:
                                    ts_dt = None
                            if ts_dt is None or ts_dt.timestamp() <= cutoff:
                                trimmed.append(t)
                            else:
                                break  # turns are ordered chronologically
                        history = trimmed
                    if not history:
                        continue
                    await self._memory_system.form_un_formed(history, "time-based")
                except Exception as e:
                    logger.warning("v3 time-based formation tick failed: %s", e)
        except asyncio.CancelledError:
            logger.debug("v3 formation timer cancelled")
            raise

    def _v3_on_turn_appended(self, turn) -> None:
        """Hook called after a Turn is appended to history.

        Hot path — must be O(1). Increments the un-committed token counter
        and fires `form_un_formed("token-pressure")` when it crosses
        `memory_formation_token_threshold` (default 30000). Disabled by
        a threshold of 0.
        """
        if not (self._memory_system and getattr(
            self._memory_system, "_formation_v3_enabled", False,
        )):
            return
        threshold = int(getattr(
            self.config, "memory_formation_token_threshold", 30000,
        ))
        if threshold <= 0:
            return
        try:
            self._uncommitted_token_count += int(getattr(turn, "token_estimate", 0) or 0)
        except Exception:
            return
        if self._uncommitted_token_count < threshold:
            return
        lock = getattr(self._memory_system, "_formation_lock", None)
        if lock and lock.locked():
            return  # in-flight formation will pick up these turns

        async def _run_token_pressure():
            try:
                history = list(self._conv_history.window) if self._conv_history else []
                if not history:
                    return
                await self._memory_system.form_un_formed(history, "token-pressure")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("v3 token-pressure formation failed: %s", e)

        self._token_pressure_task = asyncio.create_task(
            _run_token_pressure(),
            name=f"{self.node_id}-v3-token-pressure",
        )

    async def disconnect(self) -> None:
        """Disconnect and cleanup LLM clients."""
        # Cancel scheduler task
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        # ── Memory Formation v3: cancel background tasks before shutdown ──
        # Cancel the startup formation task (runs regardless of v3 flag).
        if self._startup_formation_task and not self._startup_formation_task.done():
            self._startup_formation_task.cancel()
            try:
                await self._startup_formation_task
            except (asyncio.CancelledError, Exception):
                pass

        # Cancel the time-based timer task.
        if self._formation_timer_task and not self._formation_timer_task.done():
            self._formation_timer_task.cancel()
            try:
                await self._formation_timer_task
            except (asyncio.CancelledError, Exception):
                pass

        # Cancel any in-flight token-pressure task.
        if self._token_pressure_task and not self._token_pressure_task.done():
            self._token_pressure_task.cancel()
            try:
                await self._token_pressure_task
            except (asyncio.CancelledError, Exception):
                pass

        # Run shutdown formation: BLOCKING, capped by config (default 30s).
        if (
            self._memory_system
            and getattr(self._memory_system, "_formation_v3_enabled", False)
        ):
            shutdown_timeout = float(getattr(
                self.config, "memory_formation_shutdown_timeout", 30.0,
            ))
            try:
                turns = list(self._conv_history.window) if self._conv_history else []
                if turns:
                    n = await asyncio.wait_for(
                        self._memory_system.form_un_formed(turns, "shutdown"),
                        timeout=shutdown_timeout,
                    )
                    logger.info("v3 shutdown formation: %d entries created", n)
            except asyncio.TimeoutError:
                logger.warning(
                    "v3 shutdown formation: timed out at %.1fs, proceeding",
                    shutdown_timeout,
                )
            except Exception as e:
                logger.warning("v3 shutdown formation failed: %s", e)

        if self.llm_client and self.llm_client._client:
            await self.llm_client._client.aclose()

        # Flush any pending session reflection before closing memory
        if self._router_v2:
            self._router_v2._flush_session_reflection()

        # Close memory system
        if self._memory_system:
            await self._memory_system.close()
            self._memory_system = None

        # Save router history before disconnecting
        if self._router_v2:
            try:
                self._router_v2.save_history()
                logger.info(f"RouterV2 history saved for {self.node_id}")
            except Exception as e:
                logger.error(f"Failed to save RouterV2 history: {e}")

        # Save controller state before disconnecting
        await self.controller.save_state()
        logger.info(f"Controller state saved for {self.node_id}")

        # Stop MCP tool socket if running
        await self._stop_tool_socket()

        await super().disconnect()

    # =========================================================================
    # MCP Tool Socket (Phase 2: Unix domain socket for MCP sidecar → agent)
    # =========================================================================

    async def _start_tool_socket(self) -> str:
        """Start Unix domain socket HTTP server for MCP sidecar tool calls.

        Returns the socket path.
        """
        import os
        from aiohttp import web

        # Real home from /etc/passwd — $HOME may be a synthetic CC acct home
        # when the agent was launched from a CC session (see mesh/paths.py).
        from .paths import real_home
        socket_dir = real_home() / ".mesh" / "sockets"
        socket_dir.mkdir(parents=True, exist_ok=True)
        socket_dir.chmod(0o700)
        socket_path = str(socket_dir / f"{self.node_id.replace(':', '_')}.sock")

        # Clean up stale socket from prior crash
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        async def handle_tool_call(request: web.Request) -> web.Response:
            try:
                data = await request.json()
                name = data.get("name", "")
                arguments = data.get("arguments", {})
                req_account = data.get("account")
                trigger_msg = self._current_trigger_msg

                # Switch Gmail account if requested by the caller
                prev_account = None
                if req_account and name.startswith("gmail_"):
                    try:
                        from .tool_implementations import _get_tool_host
                        host = _get_tool_host()
                        if host:
                            prev_account = host.get_current_account()
                            host.set_current_account(req_account)
                    except Exception:
                        pass

                try:
                    result = await self._execute_special_tool(name, arguments, trigger_msg)
                    if result.startswith("Unknown special tool:"):
                        tool_call = ToolCall(name=name, arguments=arguments, raw_xml="")
                        # Socket calls originate from worker subprocesses —
                        # the user already authorized the work by dispatching
                        # the worker, so skip the confirmation gate.
                        result = await self._execute_single_tool_with_confirmation(
                            tool_call,
                            original_sender=trigger_msg.from_node if trigger_msg else self.node_id,
                            skip_confirmation=True,
                        )
                    return web.json_response({"result": result})
                finally:
                    if prev_account is not None:
                        try:
                            from .tool_implementations import _get_tool_host
                            host = _get_tool_host()
                            if host:
                                host.set_current_account(prev_account)
                        except Exception:
                            pass
            except Exception as e:
                logger.error("Socket tool call failed for '%s': %s", name, e)
                return web.json_response(
                    {"error": str(e)}, status=500
                )

        async def handle_list_tools(request: web.Request) -> web.Response:
            tools = []
            seen: set[str] = set()
            for name in (self.enabled_tools or list(self.tool_registry._tools.keys())):
                if name in seen:
                    continue
                seen.add(name)
                tool_def = self.tool_registry.get(name)
                if tool_def:
                    tools.append({
                        "name": tool_def.name,
                        "description": tool_def.description,
                        "parameters": [
                            {
                                "name": p.name,
                                "type": p.type,
                                "description": p.description,
                                "required": p.required,
                                "default": p.default,
                            }
                            for p in tool_def.parameters
                        ],
                    })
            return web.json_response({"tools": tools})

        app = web.Application()
        app.router.add_post("/tool", handle_tool_call)
        app.router.add_get("/tools", handle_list_tools)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.UnixSite(runner, socket_path)
        await site.start()

        self._tool_socket_runner = runner
        self._tool_socket_path = socket_path
        logger.info(f"Tool socket started: {socket_path}")
        return socket_path

    async def _stop_tool_socket(self) -> None:
        """Shut down the tool socket server."""
        import os
        if hasattr(self, '_tool_socket_runner'):
            await self._tool_socket_runner.cleanup()
            del self._tool_socket_runner
        if hasattr(self, '_tool_socket_path') and self._tool_socket_path and os.path.exists(self._tool_socket_path):
            os.unlink(self._tool_socket_path)

    _TODO_TOOL_NAMES = {
        "todo_list", "todo_add", "todo_update",
        "todo_toggle", "todo_remove", "todo_reorder",
        "todo_set_section_order",
    }

    def _todo_conversation_id(self, arguments: dict, trigger_msg) -> str:
        """Resolve the conversation id for a todo tool call."""
        conversation_id = arguments.get("conversation_id") if isinstance(arguments, dict) else None
        if conversation_id:
            return str(conversation_id)
        if not trigger_msg:
            raise ValueError("conversation_id is required when no triggering message is available")
        if not trigger_msg.from_node or not trigger_msg.to_node:
            raise ValueError("triggering message does not identify a conversation")
        if trigger_msg.to_node and trigger_msg.to_node.startswith("channel:"):
            return trigger_msg.to_node
        return MessageStore.compute_conversation_id(trigger_msg.from_node, trigger_msg.to_node)

    async def _execute_todo_tool(self, name: str, arguments: dict, trigger_msg) -> str:
        """Execute a per-conversation todo tool through the router broker."""
        import json

        arguments = arguments or {}
        conversation_id = self._todo_conversation_id(arguments, trigger_msg)

        if name == "todo_list":
            response = await self.send_control_and_wait(
                make_todo_get(
                    self.node_id,
                    [conversation_id],
                    include_done=bool(arguments.get("include_done", True)),
                ).content,
                timeout=15.0,
            )
            content = response.content if isinstance(response.content, dict) else {}
            todos = content.get("todos", {}).get(conversation_id, [])
            section_order = content.get("section_order", {}).get(conversation_id, [])
            limit = arguments.get("limit")
            if limit is not None:
                try:
                    todos = todos[: max(1, int(limit))]
                except (TypeError, ValueError):
                    pass
            return json.dumps(
                {
                    "conversation_id": conversation_id,
                    "todos": todos,
                    "section_order": section_order,
                    "count": len(todos),
                },
                ensure_ascii=False,
                indent=2,
            )

        payload: dict[str, Any]
        expected_version = arguments.get("expected_version")
        if name == "todo_add":
            payload = {
                "text": arguments.get("text", ""),
                "priority": arguments.get("priority", 0),
            }
            if "position" in arguments:
                payload["position"] = arguments.get("position")
            if "section" in arguments:
                payload["section"] = arguments.get("section")
            op = "add"
        elif name == "todo_update":
            todo_id = arguments.get("todo_id") or arguments.get("id")
            payload = {"todo_id": todo_id}
            for key in ("text", "status", "priority", "position", "section"):
                if key in arguments:
                    payload[key] = arguments.get(key)
            op = "update"
        elif name == "todo_toggle":
            todo_id = arguments.get("todo_id") or arguments.get("id")
            payload = {
                "todo_id": todo_id,
                "done": arguments.get("done", True),
            }
            op = "toggle"
        elif name == "todo_remove":
            todo_id = arguments.get("todo_id") or arguments.get("id")
            payload = {"todo_id": todo_id}
            op = "remove"
        elif name == "todo_reorder":
            payload = {"ordered_ids": arguments.get("ordered_ids", [])}
            op = "reorder"
        elif name == "todo_set_section_order":
            payload = {
                "section_order": arguments.get(
                    "section_order",
                    arguments.get("sections", []),
                )
            }
            op = "set_section_order"
        else:
            return f"Unknown todo tool: {name}"

        response = await self.send_control_and_wait(
            make_todo_mutate(
                self.node_id,
                conversation_id,
                op,
                payload=payload,
                expected_version=expected_version,
            ).content,
            timeout=15.0,
        )
        content = response.content if isinstance(response.content, dict) else {}
        return json.dumps(content, ensure_ascii=False, indent=2)

    async def _execute_todo_tool_safe(self, name: str, arguments: dict, trigger_msg) -> str:
        """Execute a todo tool, returning tool-result text instead of aborting the turn."""
        try:
            return await self._execute_todo_tool(name, arguments, trigger_msg)
        except (asyncio.TimeoutError, ConnectionError, ValueError) as e:
            logger.warning("Todo tool %s failed: %s", name, e)
            return f"Error: {name} failed: {e}"

    async def _execute_special_tool(self, name: str, arguments: dict, trigger_msg) -> str:
        """Execute a special (agent-local) tool and return the result string.

        Used by both the XML dispatch loop and the MCP socket handler.
        """
        if name == "send_message":
            return await self._execute_send_message(arguments, trigger_msg)
        elif name == "attach_file":
            return await self._execute_attach_file(arguments)
        elif name == "channel_list":
            return await self._execute_channel_list()
        elif name == "channel_members":
            return await self._execute_channel_members(arguments)
        elif name == "schedule_wake":
            return self._execute_schedule_wake(arguments, requested_by=trigger_msg.from_node)
        elif name == "schedule_list":
            return self._execute_schedule_list()
        elif name == "schedule_cancel":
            return self._execute_schedule_cancel(arguments)
        elif name == "agent_shutdown":
            return await self._execute_agent_shutdown(arguments)
        elif name == "mesh_status":
            return await self._execute_mesh_status()
        elif name == "agent_status":
            return await self._execute_agent_status(arguments)
        elif name in self._TODO_TOOL_NAMES:
            return await self._execute_todo_tool_safe(name, arguments, trigger_msg)
        elif name == "worker_stop":
            reason = arguments.get("reason", "Worker self-stop")
            self._abort_processing = True
            logger.info(f"Worker self-stop initiated: {reason}")
            return f"Worker self-stop initiated: {reason}"
        else:
            return f"Unknown special tool: {name}"

    def _build_mcp_config(self, socket_path: str) -> str:
        """Build MCP config JSON for --mcp-config CLI arg."""
        import json
        import os
        import sys

        config = {
            "mcpServers": {
                "mesh": {
                    "command": sys.executable,
                    "args": [
                        "-m", "mesh.mcp_server",
                        "--router", f"ws://{self.config.router_host}:{self.config.router_ws_port}/ws",
                        "--token", self.config.auth_token,
                        "--node-id", f"{self.node_id}:mcp",
                        "--agent-socket", socket_path,
                    ],
                    "env": {"PYTHONPATH": os.getcwd()},
                },
            },
        }
        # Propagate tool whitelist
        if self.enabled_tools:
            config["mcpServers"]["mesh"]["args"].extend(
                ["--tools"] + list(self.enabled_tools)
            )
        return json.dumps(config)

    def _get_nicknames_for_mention_check(self) -> list[str]:
        """
        Get list of nicknames to check for mentions in channel messages.

        Includes the agent's nickname, type, and common variations.
        This allows fuzzy matching without requiring the @ symbol.
        """
        nicknames = []

        if self.nickname:
            nicknames.append(self.nickname)
            # Add nickname without common suffixes (e.g., "claude" from "claude-sobek")
            base = self.nickname.split("-")[0]
            if base and base != self.nickname:
                nicknames.append(base)

        if self.agent_type:
            nicknames.append(self.agent_type)

        # Also include the full node ID parts for robustness
        if self.node_id:
            # Extract agent:type:nickname -> [type, nickname]
            parts = self.node_id.split(":")
            if len(parts) >= 3:
                nicknames.append(parts[1])  # type
                nicknames.append(parts[2])  # nickname

        return list(set(n for n in nicknames if n))  # Dedupe and filter empty

    async def _should_process_channel_message(self, msg: Message) -> bool:
        """
        Determine if a channel message should be processed by this agent.

        Uses the relevance router (LLM-based scoring) if enabled, otherwise
        falls back to simple nickname matching.

        Returns:
            True if the message should be processed, False to add to context only.
        """
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        # If relevance router is enabled, use LLM-based scoring
        if self._relevance_router is not None:
            # Get current controller state for context
            controller_state = None
            if hasattr(self.controller, 'get_state'):
                controller_state = self.controller.get_state()

            # Get recent messages for context
            recent_messages = None
            if self._history:
                recent_messages = [
                    {"from": entry.message.from_node, "content": entry.message.content}
                    for entry in self._history[-5:]
                ]

            # Classify the message
            result = await self._relevance_router.classify(
                msg,
                controller_state=controller_state,
                recent_messages=recent_messages,
            )

            should_process = self._relevance_router.should_process(result)

            if should_process:
                logger.info(
                    f"Channel message from {msg.from_node} to {msg.to_node}: "
                    f"RELEVANT (score={result.score:.2f}, bypassed={result.bypassed}, reason={result.reason})"
                )
            else:
                logger.debug(
                    f"Channel message from {msg.from_node} to {msg.to_node}: "
                    f"IGNORED (score={result.score:.2f}, reason={result.reason})"
                )

            return should_process

        # Fall back to simple nickname matching (legacy behavior)
        nicknames = self._get_nicknames_for_mention_check()

        if is_nicknamed_mention(content, nicknames):
            return True

        logger.debug(
            f"Channel message from {msg.from_node} to {msg.to_node} "
            f"does not mention our nicknames {nicknames}, adding to context only"
        )
        return False

    _recent_msg_ids: set = set()  # dedup guard for multi-connection delivery
    _recent_msg_ids_order: list = []  # FIFO for eviction

    async def on_message(self, msg: Message) -> None:
        """Handle an incoming message by processing it through the LLM."""
        # Dedup: skip if we already processed this message ID (multi-connection delivery)
        if msg.id in self._recent_msg_ids:
            logger.debug(f"Dedup: skipping already-processed message {msg.id[:12]}")
            return
        self._recent_msg_ids.add(msg.id)
        self._recent_msg_ids_order.append(msg.id)
        # Evict old entries to prevent unbounded growth
        while len(self._recent_msg_ids_order) > 200:
            old_id = self._recent_msg_ids_order.pop(0)
            self._recent_msg_ids.discard(old_id)

        # Skip processing our own messages (echoed back from channels)
        if msg.from_node == self.node_id:
            logger.debug(f"Ignoring own message (echo from channel): {msg.id[:8]}...")
            return

        if msg.type == MessageType.CONTROL:
            # Control messages handled separately
            return

        if msg.type == MessageType.CONFIRM_RESPONSE:
            # Handle confirmation response from user
            logger.info(f"Received CONFIRM_RESPONSE from {msg.from_node}, in_reply_to={msg.in_reply_to}")
            await self._handle_confirm_response(msg)
            return

        if msg.type == MessageType.STATUS_REQUEST:
            # Handle status request (return recent context)
            await self._handle_status_request(msg)
            return

        if msg.type == MessageType.MESSAGE:
            # Router V2: delegate all message handling to RouterV2 if enabled
            # RouterV2 handles classification, relevance filtering, acks, and worker dispatch
            if self._router_v2 is not None:
                # Channel messages: require @mention to trigger LLM classification.
                # Non-mentioned channel messages are added to history for passive
                # awareness but don't cost an LLM call.
                to_node = msg.to_node or ""
                if to_node.startswith("channel:"):
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    nicknames = self._get_nicknames_for_mention_check()
                    if not is_at_mentioned(content, nicknames):
                        logger.debug(
                            f"Channel message from {msg.from_node} has no @mention "
                            f"for {nicknames}, adding to context only"
                        )
                        await self._router_v2.add_to_history_only(msg)
                        return

                logger.debug(f"Delegating message to RouterV2: from={msg.from_node}")
                await self._router_v2.on_message(msg)
                return

            # Channel message filtering:
            # Use relevance router if enabled, otherwise fall back to nickname-based check
            if msg.to_node and msg.to_node.startswith("channel:"):
                should_process = await self._should_process_channel_message(msg)
                if not should_process:
                    # Add to context but don't actively respond
                    await self._add_to_history(msg, "incoming")
                    return

            # Serialize LLM processing: if already processing, queue the message
            # to be incorporated into the current context rather than spawning
            # a parallel processing task that would fork the conversation
            async with self._processing_lock:
                if self._processing:
                    # Don't queue if abort flag is set - this message is stale
                    # (reset_context was called, so current processing will be aborted)
                    if self._abort_processing:
                        logger.info(
                            f"Dropping message from {msg.from_node} (abort flag set): "
                            f"{str(msg.content)[:50]!r}..."
                        )
                        return
                    logger.info(
                        f"Already processing, queuing message from {msg.from_node}: "
                        f"{str(msg.content)[:50]!r}..."
                    )
                    self._message_queue.append(msg)
                    return
                self._processing = True

            try:
                # Apply wall-clock timeout if configured
                timeout = self.config.max_processing_time if self.config else None
                await self._process_with_timeout(msg, timeout)

                # After processing, check if any messages arrived after the last
                # LLM iteration but before we finished. If so, process the first
                # one to continue the conversation (others will be queued and
                # incorporated or processed subsequently)
                while True:
                    async with self._processing_lock:
                        if self._message_queue:
                            next_msg = self._message_queue.pop(0)
                        else:
                            break
                    logger.info(
                        f"Processing queued message from {next_msg.from_node} "
                        f"(post-completion)"
                    )
                    logger.debug(f"📨 Processing queued message from {next_msg.from_node}")
                    await self._process_with_timeout(next_msg, timeout)
            finally:
                async with self._processing_lock:
                    self._processing = False
                    # Clear abort flag now that we're done - ready for next problem
                    self._abort_processing = False

    async def _handle_confirm_response(self, msg: Message) -> None:
        """Process a user's response to a confirmation request.

        When multiple clients are registered under the same user ID (e.g., TUI and Android),
        all receive the CONFIRM_REQUEST and may respond. We accept the first TRUE response,
        or let the timeout handle the case where all respond FALSE.
        """
        reply_to = msg.in_reply_to
        logger.info(f"_handle_confirm_response: reply_to={reply_to}, pending_confirms={list(self._pending_confirms.keys())}")
        if not reply_to or reply_to not in self._pending_confirms:
            logger.warning(f"Received CONFIRM_RESPONSE for unknown request: {reply_to}")
            return

        content = msg.content if isinstance(msg.content, dict) else {}
        confirmed = content.get("confirmed", False)

        logger.info(f"Received confirmation response: {reply_to} -> {confirmed}")

        # Only unblock on TRUE. This way if multiple clients respond, we wait for
        # any TRUE or timeout. A FALSE response is logged but doesn't complete the wait.
        if confirmed:
            self._confirm_results[reply_to] = True
            self._pending_confirms[reply_to].set()

    async def _handle_status_request(self, msg: Message) -> None:
        """Handle a status request by returning recent context and optional diagnostics."""
        req_content = msg.content if isinstance(msg.content, dict) else {}
        num_messages = req_content.get("num_messages", 5)
        want_diagnostics = req_content.get("diagnostics", False)

        # Build context from recent history
        context = []
        entries = self._history[-num_messages:] if num_messages > 0 else []

        for entry in entries:
            m = entry.message
            msg_content = m.content if isinstance(m.content, str) else str(m.content)
            from_node = self.node_id if entry.direction == "outgoing" else m.from_node

            # Include entry type for display formatting
            entry_type = "message"
            if m.metadata.get("tool_calls"):
                entry_type = "tool_call"
            elif m.metadata.get("tool_results"):
                entry_type = "tool_result"

            context.append({
                "from": from_node,
                "content": msg_content,
                "timestamp": m.timestamp,
                "type": entry_type,
            })

        # Include summary if we have one
        summary_text = None
        if self._summary:
            summary_text = self._summary.summary_text

        # Include real-time CC tool activity (if any in progress).
        # Combines worker (_current_cc_events) and router (_router_cc_events)
        # activity so status reflects whoever is currently making tool calls.
        current_activity = None
        combined_events = list(self._current_cc_events) + list(self._router_cc_events)
        if combined_events:
            activity_lines = []
            for evt in combined_events:
                if evt.event_type == "tool_call":
                    args = evt.data if isinstance(evt.data, dict) else {}
                    activity_lines.append(format_cc_tool_call(evt.tool_name, args))
                elif evt.event_type == "tool_result":
                    content_str = evt.data if isinstance(evt.data, str) else str(evt.data)
                    activity_lines.append(format_cc_tool_result(evt.tool_name, content_str))
            current_activity = "\n".join(activity_lines)
        elif hasattr(self._router_v2, 'get_current_activity'):
            # CC-session mode: pull activity from RouterCC's live stream
            cc_events = self._router_v2.get_current_activity()
            if cc_events:
                activity_lines = []
                for evt in cc_events:
                    if evt["event_type"] == "tool_call":
                        activity_lines.append(f"● {evt['tool_name']}: {evt['data'][:120]}")
                    elif evt["event_type"] == "tool_result":
                        activity_lines.append(f"  ⎿ {evt['tool_name']}: {evt['data'][:200]}")
                current_activity = "\n".join(activity_lines)

        # Get system info for status
        try:
            hostname = socket.gethostname()
        except Exception:
            hostname = None

        # Get working directory from tool_implementations
        try:
            from .tool_implementations import _bash_working_directory
            import os
            working_directory = _bash_working_directory or os.getcwd()
        except Exception:
            working_directory = None

        # Build diagnostics if requested
        diagnostics_data = None
        if want_diagnostics:
            try:
                diagnostics_data = self._build_diagnostic_report()
            except Exception as e:
                logger.error(f"Failed to build diagnostic report: {e}")
                diagnostics_data = {"error": str(e)}

        # Include heartbeat-lite status summary (state, tokens, memory, uptime)
        status_summary = self._get_status_summary()

        # Send response
        response = make_status_response(
            from_node=self.node_id,
            to_node=msg.from_node,
            in_reply_to=msg.id,
            context=context,
            summary=summary_text,
            current_activity=current_activity,
            hostname=hostname,
            model=self.llm_config.model if self.llm_config else None,
            backend=self.llm_config.backend if self.llm_config else None,
            working_directory=working_directory,
            diagnostics=diagnostics_data,
            status_summary=status_summary,
        )
        await self._conn.send(response)
        logger.info(f"Sent status response to {msg.from_node}: {len(context)} messages" +
                    (", with diagnostics" if want_diagnostics else ""))

    def _get_status_summary(self) -> dict:
        """Build lightweight status summary for heartbeat pings.

        Overrides Node base to include router state, history stats,
        total context token estimate, and memory metrics.
        """
        import time as _time
        summary: dict = {}

        # Router state
        if self._router_v2:
            summary["state"] = self._router_v2.state.value
            if self._router_v2._worker_start_time and self._router_v2._worker_task and not self._router_v2._worker_task.done():
                summary["worker_elapsed_s"] = round(_time.monotonic() - self._router_v2._worker_start_time, 1)
            else:
                summary["worker_elapsed_s"] = None
        else:
            summary["state"] = "idle"
            summary["worker_elapsed_s"] = None

        # History stats + context token estimate
        if self._router_v2 and self._router_v2.history:
            h = self._router_v2.history
            est_tokens = h.estimate_tokens()
            hard = h._hard_limit
            summary["history_turns"] = len(h)
            summary["history_pct"] = round(est_tokens / hard * 100, 1) if hard else 0
            # Full prompt tokens: use cached value from last _build_router_prompt(),
            # or fall back to static components + history estimate
            if self._router_v2._last_prompt_tokens > 0:
                summary["context_tokens"] = self._router_v2._last_prompt_tokens
            else:
                summary["context_tokens"] = self._router_v2._static_prompt_tokens + est_tokens
        else:
            summary["history_turns"] = 0
            summary["history_pct"] = 0
            summary["context_tokens"] = 0

        # Memory metrics
        if self._memory_system:
            summary["memory_pool"] = len(self._memory_system._pool)
            summary["memory_active"] = len(self._memory_system._active_ids)
            # Active map for v2
            if isinstance(self._memory_system, MemorySystemV2):
                proj = self._memory_system._active_project
                summary["active_map"] = proj if proj else None
            else:
                summary["active_map"] = None
        else:
            summary["memory_pool"] = 0
            summary["memory_active"] = 0
            summary["active_map"] = None

        # Uptime
        if hasattr(self, '_start_time'):
            summary["uptime_s"] = round(_time.monotonic() - self._start_time, 1)
        else:
            summary["uptime_s"] = 0

        return summary

    async def _handle_reset_context(self, msg: Message, content: dict) -> None:
        """
        Override to also abort in-flight processing, clear message queue, and clean workdir.

        When the test harness sends reset_context, we need to:
        1. Set abort flag to interrupt current LLM processing loop
        2. Clear the message queue (pending problems are now stale)
        3. Clean up workdir if configured (remove stale files from previous problem)
        4. Let the base class clear history and send ACK
        """
        logger.info(f"AgentNode received reset_context, setting abort flag")

        # Set abort flag - checked in LLM loop
        self._abort_processing = True

        # Clear any queued messages (they're from old problems)
        async with self._processing_lock:
            queue_size = len(self._message_queue)
            if queue_size > 0:
                logger.info(f"Clearing {queue_size} queued messages due to reset_context")
                self._message_queue.clear()

        # Clean up workdir if configured
        if self.config.workdir:
            import shutil
            from pathlib import Path
            from .paths import resolve_path
            workdir = Path(resolve_path(self.config.workdir))
            if workdir.exists():
                try:
                    # Remove all files in workdir but keep the directory
                    for item in workdir.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    logger.info(f"Cleaned workdir: {workdir}")
                except Exception as e:
                    logger.warning(f"Failed to clean workdir {workdir}: {e}")

        # Reset Router V2 if enabled (clear its context and state)
        if self._router_v2 is not None:
            await self._router_v2.reset()
            logger.info("Reset RouterV2 context and state")

        # Reset ConversationHistory (clear window and summary)
        self._conv_history._window.clear()
        self._conv_history._next_seq_id = 1
        self._conv_history.summary = None

        # Call parent implementation (clears history, sends ACK)
        await super()._handle_reset_context(msg, content)

        # Only clear abort flag if we're NOT currently processing.
        # If processing is active, the finally block at line 876 clears it
        # after the LLM loop actually stops. Clearing here while processing
        # races with the in-flight loop that hasn't seen the abort yet.
        if not self._processing:
            self._abort_processing = False

    def _history_entries_to_messages(
        self,
        entries: list,
        start_idx: int = 0,
    ) -> list[HistoryMessage]:
        """Convert history entries to HistoryMessage format."""
        messages = []
        for entry in entries[start_idx:]:
            msg = entry.message
            if msg.type == MessageType.MESSAGE:
                # Check if content is structured (dict) - might be an image message
                images = None
                if isinstance(msg.content, dict):
                    content_type = msg.content.get("type")
                    if content_type == "image":
                        # Extract image data
                        images = [ImageAttachment(
                            data=msg.content.get("data", ""),
                            mime_type=msg.content.get("mime_type", "image/jpeg"),
                            width=msg.content.get("width"),
                            height=msg.content.get("height"),
                        )]
                        # Use caption as content, or placeholder
                        content = msg.content.get("caption") or "[Image]"
                    else:
                        # Other structured content - stringify
                        content = str(msg.content)
                else:
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)

                from_node = self.node_id if entry.direction == "outgoing" else msg.from_node
                # Include to_node so the LLM knows where messages were sent
                # (important for channel messages so LLM knows to reply to the channel)
                to_node = msg.to_node
                messages.append(HistoryMessage(
                    from_node=from_node,
                    content=content,
                    timestamp=msg.timestamp,
                    to_node=to_node,
                    images=images,
                ))
        return messages

    def _sync_conv_history(self) -> None:
        """Incrementally sync new _history entries into _conv_history.

        _history is the canonical append-only list.  _conv_history provides
        summary + window context building.  This method appends only entries
        added since the last sync, so that summarization trims and hard-limit
        pruning performed by ConversationHistory are preserved across calls.

        The previous implementation rebuilt the entire window on every call,
        which undid hard-limit drops and summarization trims because _history
        was never pruned.
        """
        # On first call after startup, skip entries covered by the summary
        if self._history_sync_idx == 0 and not self._conv_history._window:
            summary = self._conv_history.summary or self._summary
            if summary and summary.messages_summarized > 0:
                self._history_sync_idx = summary.messages_summarized
            # Sync summary state on first call
            if self._summary and not self._conv_history.summary:
                self._conv_history.summary = self._summary

        # Append only new entries since last sync.
        for entry in self._history[self._history_sync_idx:]:
            msg = entry.message
            if msg.type != MessageType.MESSAGE:
                continue
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            from_node = self.node_id if entry.direction == "outgoing" else msg.from_node
            role = "assistant" if entry.direction == "outgoing" else "user"
            meta = {}
            self._conv_history.append(Turn(
                role=role,
                content=content,
                timestamp=msg.timestamp,
                from_node=from_node,
                to_node=msg.to_node,
                meta=meta,
            ))

        self._history_sync_idx = len(self._history)

    def _build_history_for_llm(self) -> list[HistoryMessage]:
        """
        Build history in HistoryMessage format for the LLM.

        Delegates to ConversationHistory for summary + window context building.
        Falls back to _history_entries_to_messages() for image support
        (ConversationHistory Turn doesn't carry image attachments).
        """
        # Check if any history entries have image content — if so, fall back
        # to the legacy path which handles ImageAttachment properly
        has_images = False
        start_idx = self._summary.messages_summarized if self._summary and self._summary.messages_summarized > 0 else 0
        for entry in self._history[start_idx:]:
            if isinstance(getattr(entry.message, 'content', None), dict):
                content_type = entry.message.content.get("type") if isinstance(entry.message.content, dict) else None
                if content_type == "image":
                    has_images = True
                    break

        if has_images:
            # Legacy path: handles ImageAttachment directly
            return self._build_history_for_llm_legacy()

        # Standard path: delegate to ConversationHistory
        self._sync_conv_history()
        messages = self._conv_history.build_context_for_llm()

        # Truncate individual messages if needed to stay within target
        return self._truncate_messages_if_needed(messages)

    def _build_history_for_llm_legacy(self) -> list[HistoryMessage]:
        """Legacy path for building LLM history when images are present."""
        messages = []

        if self._summary and self._summary.messages_summarized > 0:
            summary_msg = HistoryMessage(
                from_node="system",
                content=f"[Earlier summary]\n{self._summary.summary_text}",
                timestamp=self._summary.created_at,
            )
            messages.append(summary_msg)
            recent_msgs = self._history_entries_to_messages(
                self._history,
                start_idx=self._summary.messages_summarized,
            )
            messages.extend(recent_msgs)
        else:
            messages = self._history_entries_to_messages(self._history)

        MAX_CONTEXT_TOKENS = 150_000
        total_tokens = estimate_history_tokens(messages)

        if total_tokens > MAX_CONTEXT_TOKENS:
            logger.warning(
                f"Context too large ({total_tokens} tokens > {MAX_CONTEXT_TOKENS}), "
                f"truncating to most recent messages"
            )
            truncated = []
            running_tokens = 0
            for msg in reversed(messages):
                msg_tokens = estimate_tokens(msg.content)
                if running_tokens + msg_tokens > MAX_CONTEXT_TOKENS:
                    break
                truncated.insert(0, msg)
                running_tokens += msg_tokens
            messages = truncated

        return self._truncate_messages_if_needed(messages)

    def _truncate_messages_if_needed(
        self,
        messages: list[HistoryMessage],
    ) -> list[HistoryMessage]:
        """
        Truncate individual messages that exceed the target token limit.

        This handles the edge case where a single message is longer than
        the entire target context window.
        """
        # Reserve some tokens for system prompt, tools, etc.
        max_per_message = self._target - 2000

        if max_per_message <= 0:
            return messages

        result = []
        for msg in messages:
            tokens = estimate_tokens(msg.content)
            if tokens > max_per_message:
                # Truncate this message
                # Rough estimate: 4 chars per token
                max_chars = max_per_message * 4
                truncated = msg.content[:max_chars] + "\n\n[... content truncated ...]"
                result.append(HistoryMessage(
                    from_node=msg.from_node,
                    content=truncated,
                    timestamp=msg.timestamp,
                ))
                logger.warning(
                    f"Truncated message from {msg.from_node}: "
                    f"{tokens} tokens -> ~{max_per_message} tokens"
                )
            else:
                result.append(msg)

        return result

    async def _process_with_timeout(
        self, msg: Message, timeout: float | None
    ) -> None:
        """Wrap _process_with_llm with optional wall-clock timeout.

        If timeout is set, wraps the entire controller cycle in asyncio.wait_for().
        On timeout, sends an error message to the sender and returns.
        """
        if timeout is None:
            await self._process_with_llm(msg)
            return

        try:
            await asyncio.wait_for(self._process_with_llm(msg), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(
                f"Processing timed out after {timeout}s for message from {msg.from_node}"
            )
            error_msg = (
                f"[{self.node_id}] Error: Processing exceeded {timeout}s wall-clock limit. "
                f"Request aborted."
            )
            await self.send(msg.from_node, error_msg, in_reply_to=msg.id)

    # =========================================================================
    # Worker Briefing (cc_worker_briefing feature)
    # =========================================================================

    def _is_briefing_stale(self) -> bool:
        """Check if the worker briefing needs regeneration."""
        if self._worker_briefing is None:
            return True
        if not hasattr(self, '_router_v2') or not self._router_v2:
            return True
        current_len = len(self._router_v2._history.window)
        delta = current_len - self._briefing_history_len
        return delta >= BRIEFING_STALE_THRESHOLD

    def _format_history_for_briefing(self, window: list) -> str:
        """Format conversation history turns into readable text for briefing generation."""
        lines = []
        for turn in window:
            role = getattr(turn, 'role', 'unknown')
            content = getattr(turn, 'content', '')
            if isinstance(content, str) and content.strip():
                if len(content) > 500:
                    content = content[:500] + "... [truncated]"
                lines.append(f"[{role}]: {content}")
        return "\n\n".join(lines)

    def _get_worker_prompt_logger(self) -> logging.Logger:
        """Get or create a dedicated logger for worker prompt capture."""
        name = f"mesh.worker_prompts.{self._nickname or 'unknown'}"
        prompt_logger = logging.getLogger(name)
        if not prompt_logger.handlers:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            handler = logging.FileHandler(
                log_dir / f"agent-{self._nickname or 'unknown'}-worker-prompts.log"
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            prompt_logger.addHandler(handler)
            prompt_logger.setLevel(logging.DEBUG)
            prompt_logger.propagate = False
        return prompt_logger

    def _log_worker_dispatch(
        self,
        trigger_msg,
        task_desc: str,
        briefing: str | None,
        cc_system_prompt: str,
        slim_prompt: str,
        mcp_config: dict | None,
        history_len: int,
        cc_use_mcp: bool,
    ) -> None:
        """Write a detailed prompt capture entry for a briefing-mode worker dispatch."""
        import json
        from datetime import datetime

        pl = self._get_worker_prompt_logger()
        worker_id = ""
        if hasattr(self, '_router_v2') and self._router_v2:
            worker_id = getattr(self._router_v2, '_current_worker_id', '') or ''

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trigger_id = getattr(trigger_msg, 'id', 'unknown')
        trigger_from = getattr(trigger_msg, 'from_node', 'unknown')

        # Determine briefing provenance
        briefing_status = "none"
        if briefing:
            if self._briefing_history_len == 0:
                briefing_status = "generated_fresh"
            elif hasattr(self, '_briefing_was_updated') and self._briefing_was_updated:
                briefing_status = "updated"
            else:
                briefing_status = "reused"

        sep = "=" * 80
        lines = [
            "",
            sep,
            f"WORKER DISPATCH {worker_id} @ {ts}",
            sep,
            f"Trigger: msg_id={trigger_id} from={trigger_from}",
            f"History: {history_len} turns",
            f"MCP: {'enabled' if cc_use_mcp else 'disabled'}",
            f"Briefing: {briefing_status} ({len(briefing) if briefing else 0} chars)",
            "",
            f"--- TASK DESCRIPTION ---",
            task_desc or "(empty)",
            "",
            f"--- BRIEFING ({briefing_status}) ---",
            briefing or "(no briefing)",
            "",
            f"--- SYSTEM PROMPT (--system-prompt, {len(cc_system_prompt)} chars) ---",
            cc_system_prompt,
            "",
            f"--- USER PROMPT (-p, {len(slim_prompt)} chars) ---",
            slim_prompt,
            "",
        ]

        if mcp_config:
            try:
                mcp_json = json.dumps(mcp_config, indent=2)
            except (TypeError, ValueError):
                mcp_json = str(mcp_config)
            lines.extend([
                f"--- MCP CONFIG ---",
                mcp_json,
                "",
            ])

        lines.append(sep)
        pl.info("\n".join(lines))
        logger.info(
            f"Worker prompt capture logged to logs/agent-{self._nickname}-worker-prompts.log "
            f"(worker={worker_id}, briefing={briefing_status})"
        )

    def _log_worker_dispatch_fallback(self, trigger_msg) -> None:
        """Log when briefing mode fell back to legacy due to an error."""
        pl = self._get_worker_prompt_logger()
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        worker_id = ""
        if hasattr(self, '_router_v2') and self._router_v2:
            worker_id = getattr(self._router_v2, '_current_worker_id', '') or ''
        trigger_id = getattr(trigger_msg, 'id', 'unknown')

        sep = "=" * 80
        lines = [
            "",
            sep,
            f"WORKER DISPATCH {worker_id} @ {ts} — FALLBACK TO LEGACY",
            sep,
            f"Trigger: msg_id={trigger_id}",
            "Briefing generation failed. Worker dispatched in legacy mode (full XML history).",
            sep,
        ]
        pl.info("\n".join(lines))
        logger.warning(f"Worker {worker_id} fell back to legacy mode — see worker-prompts log")

    async def _generate_briefing(self, trigger_msg) -> str:
        """Generate a fresh worker briefing from the conversation history."""
        if not self._router_v2:
            return ""

        history_window = list(self._router_v2._history.window)
        history_text = self._format_history_for_briefing(history_window)

        map_summary = ""
        if self._memory_system and isinstance(self._memory_system, MemorySystemV2):
            map_summary = await self._memory_system.render_maps_block()

        prompt = BRIEFING_GENERATION_PROMPT.format(
            history=history_text,
            map_summary=map_summary or "(no project map)",
            task_description=self._router_v2._current_task_description or "(not yet determined)",
        )

        result = await self.llm_client.complete(prompt, max_tokens=4000)

        self._worker_briefing = result.strip()
        self._briefing_history_len = len(history_window)

        logger.info(
            f"Generated worker briefing: {len(self._worker_briefing)} chars "
            f"from {len(history_window)} history turns"
        )
        return self._worker_briefing

    async def _update_briefing(self, trigger_msg) -> str:
        """Incrementally update an existing briefing with new conversation turns."""
        if not self._router_v2 or not self._worker_briefing:
            return await self._generate_briefing(trigger_msg)

        history_window = list(self._router_v2._history.window)

        delta = len(history_window) - self._briefing_history_len
        if delta >= BRIEFING_REGEN_THRESHOLD:
            return await self._generate_briefing(trigger_msg)

        new_turns = history_window[self._briefing_history_len:]
        new_text = self._format_history_for_briefing(new_turns)

        prompt = BRIEFING_UPDATE_PROMPT.format(
            existing_briefing=self._worker_briefing,
            new_turns=new_text,
            task_description=self._router_v2._current_task_description or "(not yet determined)",
        )

        result = await self.llm_client.complete(prompt, max_tokens=4000)

        self._worker_briefing = result.strip()
        self._briefing_history_len = len(history_window)

        logger.info(
            f"Updated worker briefing: {len(self._worker_briefing)} chars "
            f"(+{delta} new turns)"
        )
        return self._worker_briefing

    async def _ensure_briefing(self, trigger_msg) -> str:
        """Ensure the worker briefing is fresh. Generate, update, or reuse as appropriate."""
        self._briefing_was_updated = False
        if self._worker_briefing is None:
            result = await self._generate_briefing(trigger_msg)
            return result

        if not self._is_briefing_stale():
            logger.debug("Worker briefing is fresh, reusing")
            return self._worker_briefing

        self._briefing_was_updated = True
        return await self._update_briefing(trigger_msg)

    async def _process_with_llm(self, trigger_msg: Message) -> None:
        """
        Process a message through the LLM.

        Tool calls stay internal until a final response is ready.
        """
        import sys
        # Removed stdout prints and flush - causes BrokenPipeError in background processes
        logger.debug(
            f"Processing message from {trigger_msg.from_node}: "
            f"{str(trigger_msg.content)[:200]!r}"
        )

        # F5: Track last known user node for fallback routing
        if trigger_msg.from_node and trigger_msg.from_node.startswith("user:"):
            self._last_user_node = trigger_msg.from_node

        # Check for controller commands (e.g., /tasks, /task)
        content = trigger_msg.content if isinstance(trigger_msg.content, str) else ""
        if content.startswith("/"):
            parts = content.split()
            command = parts[0][1:]  # Remove leading /
            args = parts[1:] if len(parts) > 1 else []

            # Try controller command handling first
            response = await self.controller.handle_command(command, args)
            if response is not None:
                logger.info(f"Controller handled command /{command}: {response[:100]}...")
                await self.send(trigger_msg.from_node, response, in_reply_to=trigger_msg.id)
                await self._add_to_history(trigger_msg, "incoming")
                return

        if self.llm_client is None:
            # No LLM - fall back to echo behavior
            logger.debug("No LLM client configured, using echo mode")
            content = trigger_msg.content if isinstance(trigger_msg.content, str) else str(trigger_msg.content)
            response = f"[{self.node_id}] (no LLM) Received: {content}"
            await self.send(trigger_msg.from_node, response, in_reply_to=trigger_msg.id)
            return

        # Controller pre-processing
        controller_addendum, handled = await self._setup_controller_for_message(trigger_msg)
        if handled:
            return

        # Build history for LLM
        history = self._build_history_for_llm()
        logger.debug(f"Built history with {len(history)} messages")

        # Debug: Log summary presence and message breakdown
        summary_count = sum(1 for msg in history if msg.from_node == "system" and "[Earlier summary]" in msg.content)
        user_msgs = sum(1 for msg in history if "user:" in msg.from_node)
        agent_msgs = sum(1 for msg in history if "agent:" in msg.from_node)
        logger.info(
            f"History breakdown: {len(history)} total "
            f"(summary={summary_count}, user={user_msgs}, agent={agent_msgs})"
        )
        if summary_count > 0:
            summary_msg = next(msg for msg in history if msg.from_node == "system" and "[Earlier summary]" in msg.content)
            logger.info(f"Summary content preview (first 200 chars): {summary_msg.content[:200]}")
        logger.info(f"Last 100 messages: {[f'{msg.from_node}->{msg.to_node}' for msg in history[-100:]]}")

        # Check if we need to trigger background summarization
        self._check_and_trigger_summarization()

        # Check if we need to trigger preference extraction
        if self._persist and self.llm_config:
            await self._preference_extractor.maybe_extract(self._history, self.llm_config)

        _is_worker = self._worker_snapshot is not None
        system_prompt, preferences_block, personality_block = await self._build_system_prompt_for_llm(
            trigger_msg, _is_worker
        )

        # Clear real-time CC events list and create collector that updates it
        self._current_cc_events.clear()

        # Create activity callback to push tool events to trigger sender
        async def push_cc_activity(event: CCToolEvent) -> None:
            """Push CC tool event to the user who triggered this turn."""
            activity_msg = make_tool_activity(
                from_node=self.node_id,
                to_node=trigger_msg.from_node,
                event_type=event.event_type,
                tool_name=event.tool_name,
                tool_source="cc",
                data={
                    "args": event.data if event.event_type == "tool_call" else None,
                    "result": event.data if event.event_type == "tool_result" else None,
                    "call_id": event.call_id,
                },
                in_reply_to=trigger_msg.id,
            )
            await self._conn.send(activity_msg)

        cc_collector = CCToolCollector(
            realtime_list=self._current_cc_events,
            activity_callback=push_cc_activity,
        )

        # Track messages sent via send_message tool during this request.
        # Also includes messages sent via capturing_send (RouterV2 worker mode),
        # where CC/other backends call send_message internally and the messages
        # are delivered via the monkey-patched self.send().
        messages_sent = False

        # Track how many times we've rejected plain text without send_message
        plain_text_rejections = 0

        # LLM loop: handle tool calls internally
        iteration = 0
        # Accumulate token usage across all LLM calls in this processing run
        self._cumulative_usage = {"input_tokens": 0, "output_tokens": 0, "cache_creation_tokens": 0, "cache_read_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0, "llm_calls": 0}
        cumulative_usage = self._cumulative_usage

        # Track side-effectful tool calls to prevent duplicates across iterations.
        # Key: (tool_name, dedup_key) where dedup_key varies by tool.
        self._sent_email_dedup: set[tuple[str, str]] = set()

        _is_worker = self._worker_snapshot is not None
        _max_iters = self.MAX_ITERATIONS
        self._in_flight_override = None
        # Always use high effort for CC calls
        if hasattr(self.llm_client, 'cc_effort'):
            self.llm_client.cc_effort = 'high'

        # Expose trigger_msg to socket handler so mesh-tool send_message works
        _cc_use_mcp = getattr(self.llm_config, 'cc_use_mcp', False) if self.llm_config else False
        _mcp_config: str | None = None
        if self._tool_socket_path:
            self._current_trigger_msg = trigger_msg
        if _cc_use_mcp and self._tool_socket_path:
            _mcp_config = self._build_mcp_config(self._tool_socket_path)

        # Mesh-harness backend: pass agent socket so subprocess can call agent-local tools
        if (self.llm_config and self.llm_config.backend == "mesh-harness"
                and self._tool_socket_path):
            self.llm_client.config.harness_agent_socket = self._tool_socket_path

        while iteration < _max_iters:
            iteration += 1

            # Check if we've been asked to abort (e.g., by reset_context)
            if self._abort_processing:
                logger.info(f"Aborting LLM processing at iteration {iteration} due to reset_context")
                return

            # Note: Messages that arrive during processing are queued and handled
            # as fresh triggers after this processing completes (post-completion loop).
            # This provides a static context snapshot during processing, which is
            # simpler and avoids duplicate response bugs.

            try:
                # Clear collector for this iteration
                cc_collector.clear()

                # Manage in-flight context: prune old tool results if over threshold
                history = self._manage_in_flight_context(history)

                # Track in-flight history reference for worker trace.
                # Updated each iteration because _manage_in_flight_context may
                # return a NEW list when pruning occurs.
                if hasattr(self, '_worker_all_cc_events'):
                    self._worker_in_flight_history = history

                # Call LLM with tool support
                # OpenAI backend will use native function calling
                # Other backends will use XML tools in prompt
                logger.debug(f"LLM iteration {iteration}")
                logger.debug(f"Calling LLM (iteration {iteration})")
                _instructions, _cc_system_prompt, _slim_prompt = await self._build_worker_instructions(
                    trigger_msg, _is_worker, _cc_use_mcp, controller_addendum,
                    preferences_block, personality_block, _mcp_config,
                    len(history), iteration,
                )

                response, tool_calls = await self.llm_client.complete_with_tools(
                    history=history,
                    node_id=self.node_id,
                    system_prompt=system_prompt,
                    tool_registry=self.tool_registry,
                    tool_names=self.enabled_tools if self.enabled_tools else None,
                    callback=cc_collector,
                    instructions=_instructions,
                    trigger_msg=trigger_msg,
                    mcp_config=_mcp_config,
                    cc_system_prompt=_cc_system_prompt,
                    cc_user_prompt=_slim_prompt,
                )
                logger.debug(f"LLM response ({len(response)} chars): {response[:200]!r}...")

                # Accumulate token usage from this LLM call
                if self.llm_client._last_usage:
                    u = self.llm_client._last_usage
                    for key in ("input_tokens", "output_tokens", "cache_creation_tokens", "cache_read_tokens", "reasoning_tokens", "total_tokens"):
                        cumulative_usage[key] += u.get(key, 0)
                    cumulative_usage["llm_calls"] += 1
                    # Preserve backend/model from last call
                    cumulative_usage["backend"] = u.get("backend", "")
                    cumulative_usage["model"] = u.get("model", "")

                # Check if capturing_send delivered messages during the LLM call.
                # This happens when the CC backend handles send_message internally
                # (the messages bypass our _execute_send_message but go through
                # the monkey-patched self.send → capturing_send → original_send).
                if getattr(self, '_capturing_send_count', 0) > 0:
                    messages_sent = True

                # Check abort flag again after LLM call (it may have taken time)
                if self._abort_processing:
                    logger.info(f"Aborting after LLM response due to reset_context")
                    return

                # Store CC tool events if any were collected
                if cc_collector.events:
                    # Accumulate full CC events for worker trace (before they're
                    # cleared next iteration via cc_collector.clear())
                    if hasattr(self, '_worker_all_cc_events'):
                        self._worker_all_cc_events.extend(cc_collector.events)
                    await self._store_cc_tool_context(cc_collector.events, trigger_msg)

                # Controller decision
                ctrl_action, controller_addendum = await self._handle_controller_llm_response(
                    response, tool_calls, history, trigger_msg,
                    messages_sent, controller_addendum,
                )
                if ctrl_action == "return":
                    return
                if ctrl_action == "continue":
                    continue

                # Tool execution
                if tool_calls:
                    tool_action, messages_sent = await self._process_tool_calls_in_loop(
                        response, tool_calls, trigger_msg, history,
                        messages_sent, cc_collector, iteration,
                    )
                    if tool_action == "return":
                        return
                    if tool_action == "continue":
                        continue

                # No tool calls - check if there's plain text to auto-route
                plain_text = response.strip()

                # v0.2: Strip internal controller XML before sending to user
                if self._is_v02_controller:
                    plain_text = strip_controller_xml(plain_text)

                if plain_text and not messages_sent:
                    # LLM produced text but didn't call send_message - auto-route it!
                    destination = self._infer_destination_from_trigger(trigger_msg)
                    logger.debug(f"📤 Auto-routing plaintext response to {destination}")
                    logger.info(
                        f"Auto-routing plaintext response to {destination} ({len(plain_text)} chars)"
                    )

                    await self.send(destination, plain_text, in_reply_to=trigger_msg.id)

                    logger.debug(f"✅ Plaintext auto-routed successfully")
                    logger.info(f"Request complete - plaintext auto-routed to {destination}")
                    return

                # Either no text, or we've already sent messages - we're done
                if messages_sent:
                    logger.debug(f"✅ Messages sent via send_message tool")
                    logger.info(f"Request complete - messages were sent via send_message")
                else:
                    logger.debug(f"✅ LLM completed (no messages to send)")
                    logger.info(f"Request complete - no messages sent")
                return

            except Exception as e:
                logger.exception(f"LLM processing error: {e}")
                # Send error response back to sender
                error_msg = f"[{self.node_id}] Error processing message: {e}"
                await self.send(trigger_msg.from_node, error_msg, in_reply_to=trigger_msg.id)
                return

        # If we get here, we hit the iteration limit
        logger.error(f"Hit max iterations ({self.MAX_ITERATIONS}) without completing")
        error_msg = f"[{self.node_id}] Error: Request processing exceeded maximum iterations"
        await self.send(trigger_msg.from_node, error_msg, in_reply_to=trigger_msg.id)

    # =========================================================================
    # Extracted helpers for _process_with_llm
    # =========================================================================

    async def _setup_controller_for_message(
        self, trigger_msg: Message,
    ) -> tuple[str | None, bool]:
        """Run controller on_message and set up streaming observer.

        Returns (controller_addendum, handled) where handled=True means
        the controller already sent a response and the caller should return.
        """
        from .controller.base import ControllerContext
        controller_addendum = None

        if isinstance(self.controller, PhaseFlowController):
            async def stream_phase_update(message: str) -> None:
                """Send phase update to the user who triggered this request."""
                status_msg = Message(
                    type=MessageType.STATUS,
                    from_node=self.node_id,
                    to_node=trigger_msg.from_node,
                    content=message,
                    in_reply_to=trigger_msg.id,
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                await self.send_message(status_msg)
            self.controller.set_observer(StreamingObserver(callback=stream_phase_update))

        if self.controller and hasattr(self.controller, 'on_message'):
            try:
                ctx = ControllerContext(
                    cwd=getattr(self, '_working_directory', ''),
                    history=self._history[-10:] if hasattr(self, '_history') else [],
                    agent_id=self.node_id,
                    message=trigger_msg,
                )
                decision = await self.controller.on_message(trigger_msg, ctx)
                logger.info(f"Controller decision: {decision.action}")

                if decision.system_addendum:
                    controller_addendum = decision.system_addendum
                    logger.debug(f"Controller provided system_addendum ({len(controller_addendum)} chars)")

                if decision.action == "PROCESS_WITH_LLM":
                    pass
                elif decision.action == "EXECUTE_TOOLS":
                    pass
                elif decision.action == "DONE":
                    response = decision.payload.get("response", "")
                    if response:
                        await self.send(trigger_msg.from_node, response, in_reply_to=trigger_msg.id)
                    return controller_addendum, True
                elif decision.action == "WAITING_APPROVAL":
                    message = decision.payload.get("message", "Edits require approval.")
                    await self.send(trigger_msg.from_node, message, in_reply_to=trigger_msg.id)
                    return controller_addendum, True
            except Exception as e:
                logger.error(f"Controller on_message failed: {e}", exc_info=True)

        return controller_addendum, False

    async def _build_system_prompt_for_llm(
        self, trigger_msg: Message, is_worker: bool,
    ) -> tuple[str, str, str]:
        """Build system prompt with preferences, memory, and personality.

        Returns (system_prompt, preferences_block, personality_block).
        """
        preferences_block = self._preference_extractor.get_preference_block() or ""
        memory_block = ""
        if self._memory_system:
            try:
                if isinstance(self._memory_system, MemorySystemV2):
                    v2_parts = []
                    rep_block = await self._memory_system.render_representative_block()
                    if rep_block:
                        v2_parts.append(rep_block)
                    map_block = await self._memory_system.render_maps_block()
                    if map_block:
                        v2_parts.append(map_block)
                    log_block = await self._memory_system.render_recent_log_block()
                    if log_block:
                        v2_parts.append(log_block)
                    summary_block = await self._memory_system.render_summary_block()
                    if summary_block:
                        v2_parts.append(summary_block)
                    _router_injected = ""
                    if is_worker and hasattr(self, '_router_v2') and self._router_v2:
                        _router_injected = getattr(
                            self._router_v2, '_injected_memory_context', ''
                        ) or ''
                    if _router_injected:
                        v2_parts.append(_router_injected)
                    else:
                        trigger_text = trigger_msg.content if isinstance(trigger_msg.content, str) else str(trigger_msg.content)
                        try:
                            retrieved_block = await self._memory_system.render_retrieved_context(
                                query=trigger_text, budget_tokens=2000,
                            )
                            if retrieved_block:
                                v2_parts.append(
                                    f"<retrieved_context>\n{retrieved_block}\n</retrieved_context>"
                                )
                        except Exception as e:
                            logger.warning(f"Automatic retrieval failed: {e}")
                    memory_block = "\n\n".join(v2_parts)
                else:
                    trigger_text = trigger_msg.content if isinstance(trigger_msg.content, str) else str(trigger_msg.content)
                    _mem_profile = self._memory_system.deep_profile
                    memory_block = await self._memory_system.render(
                        _mem_profile,
                        query=trigger_text,
                    ) or ""
            except Exception as e:
                logger.error(f"Memory query injection failed: {e}", exc_info=True)

        personality_block = ""
        if self._memory_system:
            personality_text = self._memory_system.get_personality()
            if personality_text:
                personality_block = f"<personality>\n{personality_text}\n</personality>"

        parts = [p for p in [preferences_block, personality_block, memory_block, self.system_prompt] if p]

        if getattr(self.config, "trace_as_history_enabled", False):
            parts.append(TRACE_HISTORY_FRAMING)

        system_prompt = "\n\n".join(parts)
        return system_prompt, preferences_block, personality_block

    async def _build_worker_instructions(
        self,
        trigger_msg: Message,
        is_worker: bool,
        cc_use_mcp: bool,
        controller_addendum: str | None,
        preferences_block: str,
        personality_block: str,
        mcp_config: str | None,
        history_len: int,
        iteration: int,
    ) -> tuple[str, str | None, str | None]:
        """Build worker instructions and optional briefing prompts.

        Returns (instructions, cc_system_prompt, slim_prompt).
        """
        _cc_system_prompt = None
        _slim_prompt = None

        _cc_worker_briefing = (
            is_worker
            and self.llm_config
            and getattr(self.llm_config, 'cc_worker_briefing', False)
            and self.llm_config.backend in ("claude-code", "zai")
        )
        _briefing_fell_back = False

        if is_worker:
            _trigger_to = getattr(trigger_msg, 'to_node', '') or ''
            if _trigger_to.startswith('channel:'):
                _routing_ctx = (
                    f"\nRouting: This task was triggered by an @mention in {_trigger_to}.\n"
                    f"Send your final response to {_trigger_to} using send_message.\n"
                )
            else:
                _routing_ctx = (
                    "\nRouting: This is a direct message task. Do NOT call send_message.\n"
                    "Your output will be synthesized and delivered automatically.\n"
                )

            _task_desc = ""
            if hasattr(self, '_router_v2') and self._router_v2:
                _task_desc = getattr(
                    self._router_v2, '_current_task_description', ''
                ) or ''
            if _task_desc:
                _routing_ctx += f"\nTask: {_task_desc}\n"

            if _cc_worker_briefing:
                _instructions = (
                    WORKER_BRIEFING_INSTRUCTIONS_MCP if cc_use_mcp
                    else WORKER_BRIEFING_INSTRUCTIONS
                )
                _instructions = _instructions.format(routing_context=_routing_ctx)
                if controller_addendum:
                    _instructions = f"{_instructions}\n\n{controller_addendum}"

                try:
                    briefing = await self._ensure_briefing(trigger_msg)
                except Exception as e:
                    logger.error(f"Briefing generation failed, falling back to legacy: {e}")
                    briefing = None
                    _cc_worker_briefing = False
                    _briefing_fell_back = True

                if _cc_worker_briefing and briefing:
                    _cc_sys_parts = [_instructions]
                    if preferences_block:
                        _cc_sys_parts.append(preferences_block)
                    if personality_block:
                        _cc_sys_parts.append(personality_block)
                    if self._memory_system and isinstance(self._memory_system, MemorySystemV2):
                        map_block = await self._memory_system.render_maps_block()
                        if map_block:
                            _cc_sys_parts.append(map_block)
                    _cc_sys_parts.append(f"<briefing>\n{briefing}\n</briefing>")
                    _cc_system_prompt = "\n\n".join(_cc_sys_parts)

                    _user_parts = []
                    if _task_desc:
                        _user_parts.append(f"Task: {_task_desc}")
                    if hasattr(self, '_router_v2') and self._router_v2:
                        recent = list(self._router_v2._history.window)[-3:]
                        for turn in recent:
                            role = getattr(turn, 'role', 'unknown')
                            content = getattr(turn, 'content', '')
                            if isinstance(content, str) and content.strip():
                                _user_parts.append(f"[{role}]: {content[:1000]}")
                    _slim_prompt = "\n\n".join(_user_parts) if _user_parts else trigger_msg.content

            if not _cc_worker_briefing:
                _instructions = WORKER_INSTRUCTIONS_MCP if cc_use_mcp else WORKER_INSTRUCTIONS
                _instructions = _instructions.format(routing_context=_routing_ctx)
                if controller_addendum:
                    _instructions = f"{_instructions}\n\n{controller_addendum}"
        else:
            _instructions = controller_addendum or ""
            _task_desc = ""
            briefing = None

        if iteration == 1 and is_worker and _cc_worker_briefing and _cc_system_prompt and _slim_prompt:
            self._log_worker_dispatch(
                trigger_msg=trigger_msg,
                task_desc=_task_desc,
                briefing=briefing,
                cc_system_prompt=_cc_system_prompt,
                slim_prompt=_slim_prompt,
                mcp_config=mcp_config,
                history_len=history_len,
                cc_use_mcp=cc_use_mcp,
            )
        elif iteration == 1 and is_worker and _briefing_fell_back:
            self._log_worker_dispatch_fallback(trigger_msg)

        return _instructions, _cc_system_prompt, _slim_prompt

    async def _handle_controller_llm_response(
        self,
        response: str,
        tool_calls: list | None,
        history: list,
        trigger_msg: Message,
        messages_sent: bool,
        controller_addendum: str | None,
    ) -> tuple[str, str | None]:
        """Process controller on_llm_response decision.

        Returns (action, updated_addendum) where action is:
        - "proceed": continue to tool execution
        - "continue": go to next LLM iteration
        - "return": exit the processing method
        """
        if not (self.controller and hasattr(self.controller, 'on_llm_response')):
            return "proceed", controller_addendum

        from .controller.base import ControllerContext
        try:
            ctx = ControllerContext(
                cwd=getattr(self, '_working_directory', ''),
                history=history[-10:] if history else [],
                agent_id=self.node_id,
                message=trigger_msg,
            )
            llm_decision = await self.controller.on_llm_response(
                response=response,
                tool_calls=tool_calls or [],
                context=ctx,
            )
            logger.debug(f"Controller on_llm_response: {llm_decision.action}")
            if llm_decision.phase:
                logger.info(f"Task phase: {llm_decision.phase}")

            if llm_decision.system_addendum:
                controller_addendum = llm_decision.system_addendum
                logger.debug(f"Updated controller_addendum for next phase ({len(controller_addendum)} chars)")

            if llm_decision.action == "DONE":
                logger.info(f"v0.2 flow complete, phase={llm_decision.phase}")
                done_response = strip_controller_xml(response.strip())
                destination = self._infer_destination_from_trigger(trigger_msg)

                sent_destinations = getattr(
                    self, '_worker_sent_destinations', set()
                ) or set()
                already_sent_to_trigger = destination in sent_destinations

                if done_response.strip() and not already_sent_to_trigger:
                    await self.send(destination, done_response.strip(), in_reply_to=trigger_msg.id)
                    logger.info(f"v0.2 DONE - sent response to {destination}")
                elif done_response.strip() and already_sent_to_trigger:
                    logger.info(
                        f"v0.2 DONE - suppressing done_response "
                        f"({len(done_response.strip())} chars); "
                        f"destination {destination} already received a message"
                    )
                elif not messages_sent:
                    await self.send(destination, "Done.", in_reply_to=trigger_msg.id)
                    logger.info(f"v0.2 DONE - sent minimal confirmation to {destination}")
                else:
                    logger.info(f"v0.2 DONE - messages already sent via send_message")
                return "return", controller_addendum

            elif llm_decision.action == "EXECUTE_TOOLS":
                pass

            elif llm_decision.action == "WAITING_APPROVAL":
                message = llm_decision.payload.get("message", "Edits require approval.")
                await self.send(trigger_msg.from_node, message, in_reply_to=trigger_msg.id)
                return "return", controller_addendum

            elif llm_decision.action == "PROCESS_WITH_LLM":
                logger.info(f"v0.2 phase transition: continuing to {llm_decision.phase}")
                return "continue", controller_addendum

        except Exception as e:
            logger.error(f"Controller on_llm_response failed: {e}", exc_info=True)

        return "proceed", controller_addendum

    async def _dispatch_special_tool_calls(
        self, tool_calls: list, trigger_msg: Message,
    ) -> tuple[list[str], list, bool]:
        """Execute agent-handled special tools.

        Returns (tool_results_parts, other_tool_calls, send_message_succeeded).
        """
        special_tool_names = {
            "send_message", "attach_file", "channel_list", "channel_members",
            "schedule_wake", "schedule_list", "schedule_cancel",
            "agent_shutdown", "mesh_status", "agent_status",
        } | self._TODO_TOOL_NAMES
        send_message_calls = [c for c in tool_calls if c.name == "send_message"]
        attach_file_calls = [c for c in tool_calls if c.name == "attach_file"]
        channel_list_calls = [c for c in tool_calls if c.name == "channel_list"]
        channel_members_calls = [c for c in tool_calls if c.name == "channel_members"]
        schedule_wake_calls = [c for c in tool_calls if c.name == "schedule_wake"]
        schedule_list_calls = [c for c in tool_calls if c.name == "schedule_list"]
        schedule_cancel_calls = [c for c in tool_calls if c.name == "schedule_cancel"]
        agent_shutdown_calls = [c for c in tool_calls if c.name == "agent_shutdown"]
        mesh_status_calls = [c for c in tool_calls if c.name == "mesh_status"]
        agent_status_calls = [c for c in tool_calls if c.name == "agent_status"]
        todo_calls = [c for c in tool_calls if c.name in self._TODO_TOOL_NAMES]
        other_tool_calls = [c for c in tool_calls if c.name not in special_tool_names]

        tool_results_parts = []
        send_message_succeeded = False

        if send_message_calls:
            logger.debug(f"📤 Sending {len(send_message_calls)} message(s)")
            for call in send_message_calls:
                result = await self._execute_send_message(
                    call.arguments, trigger_msg
                )
                tool_results_parts.append(
                    f'<mesh_result name="send_message">\n{result}\n</mesh_result>'
                )
                if "successfully" in result.lower() or "sent" in result.lower():
                    send_message_succeeded = True
            logger.info(f"Executed {len(send_message_calls)} send_message call(s)")

        if attach_file_calls:
            logger.debug(f"📎 Uploading {len(attach_file_calls)} attachment(s)")
            for call in attach_file_calls:
                result = await self._execute_attach_file(call.arguments)
                tool_results_parts.append(
                    f'<mesh_result name="attach_file">\n{result}\n</mesh_result>'
                )
            logger.info(f"Executed {len(attach_file_calls)} attach_file call(s)")

        if channel_list_calls:
            logger.debug(f"📋 Listing channels")
            for call in channel_list_calls:
                result = await self._execute_channel_list()
                tool_results_parts.append(
                    f'<mesh_result name="channel_list">\n{result}\n</mesh_result>'
                )
            logger.info(f"Executed {len(channel_list_calls)} channel_list call(s)")

        if channel_members_calls:
            logger.debug(f"👥 Querying channel members")
            for call in channel_members_calls:
                result = await self._execute_channel_members(call.arguments)
                tool_results_parts.append(
                    f'<mesh_result name="channel_members">\n{result}\n</mesh_result>'
                )
            logger.info(f"Executed {len(channel_members_calls)} channel_members call(s)")

        if schedule_wake_calls:
            logger.debug(f"⏰ Scheduling {len(schedule_wake_calls)} wake(s)")
            for call in schedule_wake_calls:
                result = self._execute_schedule_wake(
                    call.arguments,
                    requested_by=trigger_msg.from_node,
                )
                tool_results_parts.append(
                    f'<mesh_result name="schedule_wake">\n{result}\n</mesh_result>'
                )
            logger.info(f"Executed {len(schedule_wake_calls)} schedule_wake call(s)")

        if schedule_list_calls:
            logger.debug(f"📋 Listing scheduled wakes")
            for call in schedule_list_calls:
                result = self._execute_schedule_list()
                tool_results_parts.append(
                    f'<mesh_result name="schedule_list">\n{result}\n</mesh_result>'
                )
            logger.info(f"Executed {len(schedule_list_calls)} schedule_list call(s)")

        if schedule_cancel_calls:
            logger.debug(f"❌ Cancelling scheduled wake(s)")
            for call in schedule_cancel_calls:
                result = self._execute_schedule_cancel(call.arguments)
                tool_results_parts.append(
                    f'<mesh_result name="schedule_cancel">\n{result}\n</mesh_result>'
                )
            logger.info(f"Executed {len(schedule_cancel_calls)} schedule_cancel call(s)")

        if agent_shutdown_calls:
            logger.debug(f"🛑 Sending {len(agent_shutdown_calls)} shutdown request(s)")
            for call in agent_shutdown_calls:
                result = await self._execute_agent_shutdown(call.arguments)
                tool_results_parts.append(
                    f'<mesh_result name="agent_shutdown">\n{result}\n</mesh_result>'
                )
            logger.info(f"Executed {len(agent_shutdown_calls)} agent_shutdown call(s)")

        if mesh_status_calls:
            logger.debug(f"Querying mesh status")
            for call in mesh_status_calls:
                result = await self._execute_mesh_status()
                tool_results_parts.append(
                    f'<mesh_result name="mesh_status">\n{result}\n</mesh_result>'
                )
            logger.info(f"Executed {len(mesh_status_calls)} mesh_status call(s)")

        if agent_status_calls:
            logger.debug(f"Querying agent status")
            for call in agent_status_calls:
                result = await self._execute_agent_status(call.arguments)
                tool_results_parts.append(
                    f'<mesh_result name="agent_status">\n{result}\n</mesh_result>'
                )
            logger.info(f"Executed {len(agent_status_calls)} agent_status call(s)")

        if todo_calls:
            logger.debug(f"Updating/querying conversation todos")
            for call in todo_calls:
                result = await self._execute_todo_tool_safe(call.name, call.arguments, trigger_msg)
                tool_results_parts.append(
                    f'<mesh_result name="{call.name}">\n{result}\n</mesh_result>'
                )
            logger.info(f"Executed {len(todo_calls)} todo tool call(s)")

        return tool_results_parts, other_tool_calls, send_message_succeeded

    async def _process_tool_calls_in_loop(
        self,
        response: str,
        tool_calls: list,
        trigger_msg: Message,
        history: list,
        messages_sent: bool,
        cc_collector,
        iteration: int,
    ) -> tuple[str, bool]:
        """Process tool calls within the LLM loop iteration.

        Returns (action, updated_messages_sent) where action is:
        - "continue": continue the while loop
        - "return": exit _process_with_llm
        """
        tool_results_parts, other_tool_calls, tool_sent = await self._dispatch_special_tool_calls(
            tool_calls, trigger_msg,
        )
        if tool_sent:
            messages_sent = True

        _is_cc_backend = (
            self.llm_config and self.llm_config.backend in ("claude-code", "zai")
        )
        _cc_used_internal_tools = bool(cc_collector.events)
        only_query_tools = (
            not other_tool_calls
            and any(c for c in tool_calls if c.name in {
                "send_message", "attach_file", "channel_list", "channel_members",
                "schedule_wake", "schedule_list", "schedule_cancel",
                "agent_shutdown", "mesh_status", "agent_status",
            } | self._TODO_TOOL_NAMES)
        )
        if only_query_tools and _is_cc_backend and _cc_used_internal_tools and iteration == 1 and not messages_sent:
            logger.info(
                "CC backend used internal tools on iteration 1 but only emitted "
                "messaging/query mesh tools — continuing to give CC another turn "
                f"(CC events: {len(cc_collector.events)})"
            )
            tool_results_str = "\n".join(tool_results_parts)
            history.append(HistoryMessage(
                from_node=self.node_id,
                content=response,
                timestamp=trigger_msg.timestamp,
                source="in_flight",
            ))
            history.append(HistoryMessage(
                from_node="system",
                content=f"[Tool Results]\n{tool_results_str}\n\n"
                        f"[IMPORTANT: Your mesh tool calls above were executed. "
                        f"But the original task may not be complete yet. "
                        f"Review the original request and ensure ALL steps are done "
                        f"before stopping.]",
                timestamp=trigger_msg.timestamp,
                source="in_flight",
            ))
            return "continue", messages_sent
        elif only_query_tools:
            plain_text = response
            import re
            plain_text = re.sub(
                r'<mesh_call\s+name="[^"]*">\s*.*?</mesh_call>',
                '',
                plain_text,
                flags=re.DOTALL
            ).strip()

            if plain_text and not messages_sent:
                destination = self._infer_destination_from_trigger(trigger_msg)
                logger.debug(f"📤 Auto-routing plaintext response to {destination}")
                logger.info(
                    f"Auto-routing plaintext response to {destination} ({len(plain_text)} chars)"
                )
                await self.send(destination, plain_text, in_reply_to=trigger_msg.id)
                logger.debug(f"✅ Plaintext auto-routed successfully")
                logger.info(f"Request complete - plaintext auto-routed to {destination}")
            else:
                logger.info(
                    "Only query/messaging tools were called this iteration; "
                    "ending processing for this trigger."
                )
            return "return", messages_sent

        if other_tool_calls:
            logger.info(f"Executing {len(other_tool_calls)} tool call(s): {[c.name for c in other_tool_calls]}")
            other_results = await self._execute_tool_calls_with_confirmation(
                other_tool_calls, trigger_msg.from_node, trigger_msg.id
            )
            tool_results_parts.append(other_results)

            only_sleep = (
                len(tool_calls) == 1
                and other_tool_calls
                and len(other_tool_calls) == 1
                and other_tool_calls[0].name == "sleep"
            )
            if only_sleep:
                logger.info(
                    "Only sleep tool was called this iteration; "
                    "ending processing for this trigger."
                )
                return "return", messages_sent

        tool_results = "\n\n".join(tool_results_parts)
        tool_results = self._truncate_extreme_result(tool_results)

        response_for_history = response
        if not response and tool_calls:
            response_for_history = "\n".join(tc.raw_xml for tc in tool_calls)

        reasoning = getattr(self.llm_client, '_last_reasoning_content', None)
        if reasoning:
            response_for_history = f"<reasoning>\n{reasoning}\n</reasoning>\n{response_for_history}"

        history.append(HistoryMessage(
            from_node=self.node_id,
            content=response_for_history,
            timestamp=trigger_msg.timestamp,
            source="in_flight",
        ))

        history.append(HistoryMessage(
            from_node="system",
            content=f"Tool execution results:\n{tool_results}",
            timestamp=trigger_msg.timestamp,
            source="in_flight",
        ))

        await self._store_tool_context(tool_calls, tool_results, trigger_msg)
        return "continue", messages_sent

    def _infer_destination_from_trigger(self, trigger_msg: Message) -> str:
        """
        Infer the destination for a plaintext reply based on the trigger message.

        - If trigger was sent to a channel, reply to that channel
        - Otherwise, reply to the original sender
        """
        # Check if the trigger was addressed to a channel
        if trigger_msg.to_node and trigger_msg.to_node.startswith("channel:"):
            return trigger_msg.to_node

        # Default: reply to sender
        return trigger_msg.from_node

    async def _execute_send_message(
        self,
        args: dict[str, Any],
        trigger_msg: Message,
    ) -> str:
        """
        Execute a send_message tool call by routing the message.

        Args:
            args: Tool arguments with 'to' and 'content' fields
            trigger_msg: The original message that triggered this processing

        Returns:
            Result string indicating success or failure
        """
        to_node = args.get("to")
        content = args.get("content", "")
        raw_attachments = args.get("attachments") or []

        if not to_node:
            return "Error: 'to' parameter is required for send_message"

        if not content:
            return "Error: 'content' parameter is required for send_message"

        # Coerce content to string — LLMs occasionally pass non-string types
        content = str(content)

        # Bug 4: run tool-driven sends through the same outbound sanitizer used
        # by RouterV2._send_and_store. The CC monitor relays results via the
        # send_message TOOL (this path), not the router's free-text path, so
        # without this the most common path for CC results to reach the user is
        # the one path the XML-leakage fix didn't cover.
        try:
            from .router_v2 import RouterV2
            _sanitized = RouterV2._sanitize_outbound(content)
            if _sanitized:
                content = _sanitized.strip()
                if not content:
                    logger.info("send_message: content whitespace-only after sanitization, skipping send")
                    return "Message had no deliverable content after sanitization; nothing was sent."
            else:
                # Sanitized to nothing (e.g. pure <thinking>) — don't send an
                # empty message; report back so the loop doesn't retry blindly.
                logger.info("send_message: content empty after sanitization, skipping send")
                return "Message had no deliverable content after sanitization; nothing was sent."
        except Exception as e:
            logger.debug(f"send_message sanitization skipped: {e}")

        attachments: list[Attachment] = []
        if raw_attachments:
            if not isinstance(raw_attachments, list):
                return "Error: 'attachments' must be a list"
            for item in raw_attachments:
                if not isinstance(item, dict):
                    return "Error: each attachment must be an object"
                attachments.append(Attachment.from_dict(item))

        try:
            logger.info(
                f"send_message tool: self.node_id={self.node_id} "
                f"self.send={self.send.__name__ if hasattr(self.send, '__name__') else type(self.send).__name__} "
                f"to={to_node}"
            )
            await self.send(
                to_node,
                content,
                in_reply_to=trigger_msg.id if trigger_msg else None,
                attachments=attachments,
            )
            logger.info(f"send_message: sent to {to_node} ({len(content)} chars)")
            return f"Message sent successfully to {to_node}"
        except Exception as e:
            logger.exception(f"send_message failed: {e}")
            return f"Error sending message to {to_node}: {e}"

    def _router_http_base(self) -> str:
        """Derive the router HTTP base URL from node config."""
        if self.config.ws_url:
            url = self.config.ws_url
            url = url.replace("wss://", "https://", 1).replace("ws://", "http://", 1)
            return re.sub(r"/ws/?$", "", url)
        return f"http://{self.config.router_host}:{self.config.router_ws_port}"

    async def _execute_attach_file(self, args: dict[str, Any]) -> str:
        """Upload a local file to the router attachment store."""
        import aiohttp
        import json
        import mimetypes
        import urllib.parse

        path = args.get("path", "")
        if not path:
            return "Error: 'path' parameter is required for attach_file"
        from .paths import resolve_path as _resolve_home
        file_path = Path(_resolve_home(path))
        if not file_path.exists() or not file_path.is_file():
            return f"Error: file not found: {file_path}"
        if not self.config.auth_token:
            return "Error: attach_file requires an auth_token in node config"

        mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        headers = {
            "Authorization": f"Bearer {self.config.auth_token}",
            "X-Node-ID": self.node_id,
            "X-Filename": urllib.parse.quote(file_path.name),
            "Content-Type": mime,
        }
        url = f"{self._router_http_base()}/attachments"
        try:
            async with aiohttp.ClientSession() as session:
                with file_path.open("rb") as fh:
                    async with session.post(url, headers=headers, data=fh) as resp:
                        text = await resp.text()
                        if resp.status >= 400:
                            return f"Error uploading attachment ({resp.status}): {text}"
                        data = json.loads(text)
            return json.dumps(data)
        except Exception as e:
            logger.exception(f"attach_file failed: {e}")
            return f"Error uploading attachment: {e}"

    async def _execute_channel_list(self) -> str:
        """
        Execute a channel_list tool call by querying the router.

        Returns:
            Formatted string listing channels the agent is a member of.
        """
        try:
            channels = await self.request_channel_list()

            if not channels:
                return "You are not a member of any channels."

            # Filter to only channels where we are a member
            my_channels = [ch for ch in channels if ch.get("is_member", False)]

            if not my_channels:
                return "You are not a member of any channels."

            lines = ["## Your Channels\n"]
            for ch in my_channels:
                name = ch.get("name", "unknown")
                desc = ch.get("description", "")
                count = ch.get("member_count", 0)
                lines.append(f"- **{name}** ({count} members)")
                if desc:
                    lines.append(f"  {desc}")
            return "\n".join(lines)

        except Exception as e:
            logger.exception(f"channel_list failed: {e}")
            return f"Error listing channels: {e}"

    async def _execute_channel_members(self, args: dict[str, Any]) -> str:
        """
        Execute a channel_members tool call by querying the router.

        Args:
            args: Tool arguments with 'channel_name' field

        Returns:
            Formatted string listing members of the channel.
        """
        channel_name = args.get("channel_name", "").strip()

        if not channel_name:
            return "Error: 'channel_name' parameter is required"

        try:
            members = await self.request_channel_members(channel_name)

            if not members:
                return f"No members found for channel '{channel_name}' (or channel does not exist)."

            lines = [f"## Members of #{channel_name}\n"]
            for m in members:
                node_id = m.get("node_id", "unknown")
                online = m.get("online", False)
                status = "🟢 online" if online else "⚪ offline"
                lines.append(f"- {node_id} ({status})")

            return "\n".join(lines)

        except Exception as e:
            logger.exception(f"channel_members failed: {e}")
            return f"Error listing members for channel '{channel_name}': {e}"

    async def _execute_agent_shutdown(self, args: dict[str, Any]) -> str:
        """
        Execute an agent_shutdown tool call by sending a shutdown control message.

        Args:
            args: Tool arguments with 'target' and optional 'reason' fields

        Returns:
            Status message about the shutdown request.
        """
        from .protocol import make_shutdown_request

        target = args.get("target", "").strip()
        reason = args.get("reason", "")

        if not target:
            return "Error: 'target' parameter is required (e.g., 'agent:assistant:alice')"

        # Validate target format
        if not target.startswith("agent:"):
            return f"Error: target must be an agent (got '{target}'). Use format 'agent:{{type}}:{{nickname}}'"

        # Get auth token from config
        auth_token = self._auth_token
        if not auth_token:
            return "Error: No auth token available. Cannot send shutdown request without authentication."

        try:
            # Create and send shutdown request
            shutdown_msg = make_shutdown_request(
                from_node=self.node_id,
                target_node=target,
                auth_token=auth_token,
                reason=reason,
            )

            await self.send(shutdown_msg)
            logger.info(f"Sent shutdown request to {target}" +
                       (f" (reason: {reason})" if reason else ""))

            return f"Shutdown request sent to {target}. The agent should acknowledge and shut down gracefully."

        except Exception as e:
            logger.exception(f"Failed to send shutdown request to {target}: {e}")
            return f"Error sending shutdown request to {target}: {e}"

    async def _send_and_wait(
        self, msg: Message, timeout: float = 10.0
    ) -> Message | None:
        """Send a message and wait for a response keyed by in_reply_to.

        Used by mesh_status and agent_status tools for request-response patterns.
        Returns the response Message, or None on timeout.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Message] = loop.create_future()
        request_key = f"_send_and_wait_{msg.id}"
        self._pending_requests[request_key] = future

        try:
            await self._conn.send(msg)
            self.mark_activity()
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"_send_and_wait timed out for {msg.id}")
            return None
        except Exception as e:
            logger.error(f"_send_and_wait failed: {e}")
            return None
        finally:
            self._pending_requests.pop(request_key, None)

    async def _execute_mesh_status(self) -> str:
        """Execute mesh_status tool: query router for live agent status dashboard."""
        try:
            # Send LIST_AGENTS control message to router
            request = Message(
                from_node=self.node_id,
                to_node="router",
                type=MessageType.CONTROL,
                content={"action": ControlAction.LIST_AGENTS.value},
            )
            response = await self._send_and_wait(request, timeout=10.0)
            if response is None:
                return "Error: No response from router (timeout)"

            content = response.content if isinstance(response.content, dict) else {}
            connected = content.get("connected", [])
            connected_users = content.get("connected_users", [])
            status = content.get("status", {})
            cc_usage = content.get("cc_usage", "")

            lines = ["=== Mesh Agent Status ===", ""]

            # Add CC usage summary at the top if available
            if cc_usage:
                lines.append(cc_usage)
                lines.append("")

            # Format each connected agent with status
            for node_id in sorted(connected):
                s = status.get(node_id, {})
                state = s.get("state", "?").upper()
                worker_elapsed = s.get("worker_elapsed_s")
                ctx_tokens = s.get("context_tokens", 0)
                hist_turns = s.get("history_turns", 0)
                hist_pct = s.get("history_pct", 0)
                mem_pool = s.get("memory_pool", 0)
                mem_active = s.get("memory_active", 0)
                uptime_s = s.get("uptime_s", 0)

                # Format state with worker elapsed
                if state == "BUSY" and worker_elapsed is not None:
                    state_str = f"BUSY ({int(worker_elapsed)}s)"
                else:
                    state_str = state

                # Format context tokens
                if ctx_tokens >= 1000:
                    ctx_str = f"{ctx_tokens // 1000}k ctx"
                else:
                    ctx_str = f"{ctx_tokens} ctx"

                # Format uptime
                uptime_str = _format_uptime(uptime_s)

                # Active map
                active_map = s.get("active_map")
                map_str = f"map:{active_map}" if active_map else "map:none"

                lines.append(
                    f"{node_id:<35s} {state_str:<14s} {ctx_str:<10s} "
                    f"{hist_turns} turns ({hist_pct:.0f}%)   "
                    f"mem {mem_pool}/{mem_active}   {map_str}   up {uptime_str}"
                )

            # Include online users
            for user_id in sorted(connected_users):
                lines.append(f"{user_id:<35s} online")

            if not connected and not connected_users:
                lines.append("(no agents or users connected)")

            return "\n".join(lines)

        except Exception as e:
            logger.exception(f"mesh_status failed: {e}")
            return f"Error querying mesh status: {e}"

    def _build_diagnostic_report(self, section_filter: str | None = None) -> dict:
        """Build full diagnostic report for status responses."""
        import time as _time
        import os

        sections: dict = {}

        # identity
        try:
            from .tool_implementations import _bash_working_directory
            working_dir = _bash_working_directory or os.getcwd()
        except Exception:
            working_dir = os.getcwd()

        sections["identity"] = {
            "node_id": self.node_id,
            "nickname": self.config.nickname if self.config else None,
            "agent_type": self.config.agent_type if self.config else None,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "uptime_seconds": round(_time.monotonic() - self._start_time, 1) if hasattr(self, '_start_time') else 0,
            "working_directory": working_dir,
        }

        # llm
        sections["llm"] = {
            "backend": self.llm_config.backend if self.llm_config else None,
            "model": self.llm_config.model if self.llm_config else None,
            "router_llm_backend": (
                self._router_v2_llm_config.backend
                if self._router_v2_llm_config else None
            ),
            "router_llm_model": (
                self._router_v2_llm_config.model
                if self._router_v2_llm_config else None
            ),
        }

        # router (delegate to RouterV2.get_diagnostics())
        if self._router_v2:
            sections["router"] = self._router_v2.get_diagnostics()
        else:
            sections["router"] = {"state": "no_router", "detail": "RouterV2 not initialized"}

        # history (from RouterV2's ConversationHistory)
        if self._router_v2 and self._router_v2.history:
            h = self._router_v2.history
            est_tokens = h.estimate_tokens()
            hard = h._hard_limit
            soft = h._soft_limit
            sections["history"] = {
                "window_turns": len(h),
                "estimated_tokens": est_tokens,
                "soft_limit_tokens": soft,
                "hard_limit_tokens": hard,
                "utilization_pct": round(est_tokens / hard * 100, 1) if hard else 0,
                "summary_present": h._summary is not None,
                "summarization_enabled": h._summarization_enabled,
                "oldest_turn_timestamp": str(h.window[0].timestamp) if h.window else None,
                "newest_turn_timestamp": str(h.window[-1].timestamp) if h.window else None,
                "persist_path": str(h._persist_path) if h._persist_path else None,
            }
        else:
            sections["history"] = {"window_turns": 0, "detail": "no history instance"}

        # memory
        if self._memory_system:
            sections["memory"] = self._memory_system.get_diagnostics()
            sections["memory"]["enabled"] = True
        else:
            sections["memory"] = {"enabled": False, "detail": "MemorySystem not initialized"}

        # context_health — simple boolean sanity checks
        sections["context_health"] = self._run_health_checks(sections)

        if section_filter:
            return {section_filter: sections.get(section_filter, {"error": "unknown section"})}
        return sections

    def _run_health_checks(self, sections: dict) -> dict:
        """Run simple yes/no health checks on wiring."""
        checks = []

        # Router has history?
        hist = sections.get("history", {})
        turns = hist.get("window_turns", 0)
        checks.append({
            "name": "router_has_history",
            "ok": turns > 0,
            "detail": f"{turns} turns in window" if turns > 0 else "empty window",
        })

        # Worker snapshot mechanism active?
        checks.append({
            "name": "worker_gets_snapshot",
            "ok": self._router_v2 is not None,
            "detail": "snapshot mechanism active" if self._router_v2 else "RouterV2 not initialized",
        })

        # Memory initialized?
        mem = sections.get("memory", {})
        mem_enabled = mem.get("enabled", False)
        pool_size = mem.get("pool_size", 0)
        active_size = mem.get("active_set_size", 0)
        checks.append({
            "name": "memory_initialized",
            "ok": mem_enabled and pool_size > 0,
            "detail": f"{pool_size} pool / {active_size} active" if mem_enabled else "not initialized",
        })

        # Summarization disabled (rolling window)?
        summ_enabled = hist.get("summarization_enabled", True)
        checks.append({
            "name": "summarization_disabled",
            "ok": not summ_enabled,
            "detail": "rolling window active" if not summ_enabled else "summarization still enabled",
        })

        return {"checks": checks}

    async def _execute_agent_status(self, args: dict[str, Any]) -> str:
        """Execute agent_status tool: query full diagnostics from a specific agent."""
        from .protocol import make_status_request

        target = args.get("target", "self").strip()
        section = args.get("section")

        if target == "self" or target == self.node_id:
            report = self._build_diagnostic_report(section_filter=section)
            return _format_status_report(report, self.node_id)

        # Remote: send STATUS_REQUEST with diagnostics=True
        request = make_status_request(
            from_node=self.node_id,
            to_node=target,
            num_messages=0,
            diagnostics=True,
        )
        response = await self._send_and_wait(request, timeout=10.0)
        if response is None:
            return f"Error: No response from {target} (timeout or offline)"

        content = response.content if isinstance(response.content, dict) else {}
        diagnostics = content.get("diagnostics", {})
        if section:
            diagnostics = {section: diagnostics.get(section, {"error": "unknown section"})}
        return _format_status_report(diagnostics, target)

    def _execute_schedule_wake(self, args: dict[str, Any],
                              requested_by: str = "") -> str:
        """
        Execute a schedule_wake tool call.

        Args:
            args: Tool arguments with 'wake_time' and 'prompt' fields
            requested_by: Node ID of the user who triggered this (for response routing)

        Returns:
            JSON string with result status.
        """
        import json as json_module

        wake_time = args.get("wake_time", "").strip()
        prompt = args.get("prompt", "").strip()
        recurrence = args.get("recurrence", "").strip() or None

        if not wake_time:
            return json_module.dumps({"status": "error", "error": "'wake_time' parameter is required"})

        if not prompt:
            return json_module.dumps({"status": "error", "error": "'prompt' parameter is required"})

        result = self.schedule_wake(wake_time, prompt, requested_by=requested_by,
                                    recurrence=recurrence)
        return json_module.dumps(result)

    def _execute_schedule_list(self) -> str:
        """
        Execute a schedule_list tool call.

        Returns:
            Formatted string listing pending wakes.
        """
        wakes = self.list_scheduled_wakes()

        if not wakes:
            return "No scheduled wakes pending."

        lines = ["## Scheduled Wakes\n"]
        for w in wakes:
            lines.append(f"- **{w['id']}**: {w['wake_time_local']}")
            lines.append(f"  Prompt: {w['prompt_preview']}")
        return "\n".join(lines)

    def _execute_schedule_cancel(self, args: dict[str, Any]) -> str:
        """
        Execute a schedule_cancel tool call.

        Args:
            args: Tool arguments with 'wake_id' field

        Returns:
            JSON string with result status.
        """
        import json as json_module

        wake_id = args.get("wake_id", "").strip()

        if not wake_id:
            return json_module.dumps({"status": "error", "error": "'wake_id' parameter is required"})

        result = self.cancel_scheduled_wake(wake_id)
        return json_module.dumps(result)

    async def _add_to_history(self, msg: Message, direction: str) -> None:
        """
        Add a message to persistent history.

        Args:
            msg: The message to add
            direction: "incoming" or "outgoing"
        """
        from .node import HistoryEntry
        self._history.append(HistoryEntry(message=msg, direction=direction))
        if self._persist:
            self.save_history()
            if self._router_v2:
                try:
                    self._router_v2.save_history()
                except Exception as e:
                    logger.warning(f"Failed to save RouterV2 history: {e}")

        # Also archive to SQLite if we loaded from a conversation
        if self._message_store and self._loaded_conversation_id:
            try:
                self._message_store.archive_message(
                    msg, conversation_id=self._loaded_conversation_id
                )
            except Exception as e:
                logger.warning(f"Failed to archive message to SQLite: {e}")

    async def _store_tool_context(
        self,
        tool_calls: list[ToolCall],
        tool_results: str,
        trigger_msg: Message,
    ) -> None:
        """
        Store tool calls and results in persistent history for /status visibility.

        Creates internal MESSAGE entries so that tool calls appear in status responses.
        """
        from .node import HistoryEntry

        # Build a readable summary of tool calls
        tool_call_lines = []
        for tc in tool_calls:
            args_summary = ", ".join(f"{k}={v!r}" for k, v in list(tc.arguments.items())[:3])
            if len(tc.arguments) > 3:
                args_summary += ", ..."
            tool_call_lines.append(f"[Tool: {tc.name}({args_summary})]")
        tool_call_summary = "\n".join(tool_call_lines)

        # Create a MESSAGE for the tool calls
        tool_call_msg = Message(
            type=MessageType.MESSAGE,
            from_node=self.node_id,
            to_node="internal",
            content=tool_call_summary,
            timestamp=trigger_msg.timestamp,
            metadata={"tool_calls": True},
        )
        self._history.append(HistoryEntry(message=tool_call_msg, direction="outgoing"))

        # Create a MESSAGE for the tool results (abbreviated)
        # Truncate results to avoid bloating history
        results_preview = tool_results[:1000]
        if len(tool_results) > 1000:
            results_preview += f"\n... ({len(tool_results)} chars total)"

        tool_result_msg = Message(
            type=MessageType.MESSAGE,
            from_node="system",
            to_node=self.node_id,
            content=f"[Tool Results]\n{results_preview}",
            timestamp=trigger_msg.timestamp,
            metadata={"tool_results": True},
        )
        self._history.append(HistoryEntry(message=tool_result_msg, direction="incoming"))

        # Mirror tool activity to worker snapshot (if running under RouterV2 worker).
        # This allows the router to see tool call progress during busy mode.
        snapshot = self._worker_snapshot
        if snapshot is not None:
            from .conversation_history import Turn
            from datetime import datetime as _dt, timezone as _tz
            now = _dt.now(_tz.utc)
            snapshot.append(Turn(
                role="outgoing",
                content=tool_call_summary,
                timestamp=now,
                from_node=self.node_id,
                to_node="internal",
                meta={"tool_calls": True},
            ))
            snapshot.append(Turn(
                role="incoming",
                content=f"[Tool Results]\n{results_preview}",
                timestamp=now,
                from_node="system",
                to_node=self.node_id,
                meta={"tool_results": True},
            ))

        # Persist if enabled
        if self._persist:
            self.save_history()
            if self._router_v2:
                try:
                    self._router_v2.save_history()
                except Exception as e:
                    logger.warning(f"Failed to save RouterV2 history: {e}")

    async def _store_cc_tool_context(
        self,
        events: list[CCToolEvent],
        trigger_msg: Message,
    ) -> None:
        """
        Store CC (Claude Code) tool events in persistent history for /status visibility.

        These are tools that Claude Code's internal LLM calls use (Read, Edit, Bash, etc.)
        that we now have visibility into via streaming callbacks.
        """
        from .node import HistoryEntry

        # Group events by call_id to pair calls with results
        call_events = [e for e in events if e.event_type == "tool_call"]
        result_events = {e.call_id: e for e in events if e.event_type == "tool_result"}

        # Build readable summary
        cc_tool_lines = []
        for call in call_events:
            # Format the tool call
            if isinstance(call.data, dict):
                # Summarize arguments
                args = call.data
                if call.tool_name == "cc:Read":
                    summary = args.get("file_path", "")[:80]
                elif call.tool_name == "cc:Edit":
                    summary = f"{args.get('file_path', '')} ({len(args.get('old_string', ''))} -> {len(args.get('new_string', ''))} chars)"
                elif call.tool_name == "cc:Bash":
                    cmd = args.get("command", "")
                    summary = cmd[:80] + ("..." if len(cmd) > 80 else "")
                elif call.tool_name == "cc:Grep":
                    summary = f"pattern={args.get('pattern', '')!r}"
                elif call.tool_name == "cc:Glob":
                    summary = f"pattern={args.get('pattern', '')!r}"
                else:
                    # Generic summary
                    items = list(args.items())[:2]
                    summary = ", ".join(f"{k}={str(v)[:30]}" for k, v in items)
            else:
                summary = str(call.data)[:80]

            # Check for result
            result = result_events.get(call.call_id)
            if result:
                result_preview = str(result.data)[:100]
                if len(str(result.data)) > 100:
                    result_preview += "..."
                cc_tool_lines.append(f"[{call.tool_name}] {summary}\n  → {result_preview}")
            else:
                cc_tool_lines.append(f"[{call.tool_name}] {summary}")

        if not cc_tool_lines:
            return

        cc_summary = "\n".join(cc_tool_lines)

        # Create a MESSAGE for CC tool events
        cc_msg = Message(
            type=MessageType.MESSAGE,
            from_node=self.node_id,
            to_node="internal",
            content=f"[CC Tool Activity]\n{cc_summary}",
            timestamp=trigger_msg.timestamp,
            metadata={"cc_tool_events": True, "cc_tool_calls": len(call_events)},
        )
        self._history.append(HistoryEntry(message=cc_msg, direction="outgoing"))

        # Mirror CC tool activity to worker snapshot (if running under RouterV2 worker)
        snapshot = self._worker_snapshot
        if snapshot is not None:
            from .conversation_history import Turn
            from datetime import datetime as _dt, timezone as _tz
            snapshot.append(Turn(
                role="outgoing",
                content=f"[CC Tool Activity]\n{cc_summary}",
                timestamp=_dt.now(_tz.utc),
                from_node=self.node_id,
                to_node="internal",
                meta={"cc_tool_events": True, "cc_tool_calls": len(call_events)},
            ))

        logger.info(f"Stored {len(events)} CC tool events in history")

        # Persist if enabled
        if self._persist:
            self.save_history()
            if self._router_v2:
                try:
                    self._router_v2.save_history()
                except Exception as e:
                    logger.warning(f"Failed to save RouterV2 history: {e}")

    # =========================================================================
    # Tool execution with confirmation support
    # =========================================================================

    async def _execute_tool_calls_with_confirmation(
        self,
        calls: list[ToolCall],
        original_sender: str,
        trigger_msg_id: str | None = None,
    ) -> str:
        """
        Execute tool calls, requesting user confirmation when needed.

        For tools with `requires_confirmation=True`, sends a CONFIRM_REQUEST
        to the original sender and waits for their response before executing.
        """
        results = []

        for call in calls:
            result = await self._execute_single_tool_with_confirmation(
                call, original_sender, trigger_msg_id
            )
            results.append(f'<mesh_result name="{call.name}">\n{result}\n</mesh_result>')

        return "\n\n".join(results)

    async def _execute_single_tool_with_confirmation(
        self,
        call: ToolCall,
        original_sender: str,
        trigger_msg_id: str | None = None,
        skip_confirmation: bool = False,
    ) -> str:
        """Execute a single tool, requesting confirmation if required."""
        tool_def = self.tool_registry.get(call.name)

        if tool_def is None:
            return f"Error: Unknown tool '{call.name}'"

        if tool_def.handler is None:
            return f"Error: Tool '{call.name}' has no handler"

        # Dedup guard: prevent duplicate side-effectful sends within a single
        # processing run. The LLM sometimes re-calls gmail_send_message on a
        # subsequent iteration (especially after context pruning loses sight of
        # the earlier successful send).
        _dedup_set = getattr(self, '_sent_email_dedup', None)
        if _dedup_set is not None and call.name in ("gmail_send_message", "gmail_reply_to"):
            if call.name == "gmail_send_message":
                dedup_key = (call.name, (call.arguments.get("to") or "").strip().lower())
            else:  # gmail_reply_to
                dedup_key = (call.name, (call.arguments.get("message_id") or "").strip())
            if dedup_key in _dedup_set:
                skip_msg = (
                    f"Skipped duplicate {call.name}: already executed successfully "
                    f"earlier in this processing run."
                )
                logger.warning(skip_msg)
                return skip_msg
            # Will be added to the set after successful execution (below).

        # Check if confirmation is required
        if tool_def.requires_confirmation and not skip_confirmation:
            # Skip confirmation if this tool is in the agent's auto_confirm list
            if call.name in getattr(self.config, 'auto_confirm_tools', []):
                logger.info(f"Auto-confirming tool '{call.name}' (in auto_confirm_tools)")
            else:
                confirmed = await self._request_confirmation(
                    call.name, call.arguments, original_sender
                )
                if not confirmed:
                    return f"Tool '{call.name}' aborted: User rejected confirmation or timeout"

        # Push TOOL_ACTIVITY for tool_call
        await self._push_mesh_tool_activity(
            to_node=original_sender,
            event_type="tool_call",
            tool_name=call.name,
            data={"args": call.arguments},
            in_reply_to=trigger_msg_id,
        )

        # Execute the tool
        try:
            logger.debug(f"Executing tool {call.name} with args: {call.arguments}")

            if asyncio.iscoroutinefunction(tool_def.handler):
                result = await tool_def.handler(**call.arguments)
            else:
                # Run synchronous tool handlers in a thread to avoid
                # blocking the event loop (e.g., bash_exec with long commands)
                result = await asyncio.to_thread(tool_def.handler, **call.arguments)

            logger.debug(f"Tool {call.name} result: {str(result)[:200]}...")
            result_str = str(result)

            # Record successful side-effectful sends for dedup
            if _dedup_set is not None and call.name in ("gmail_send_message", "gmail_reply_to"):
                if call.name == "gmail_send_message":
                    _dk = (call.name, (call.arguments.get("to") or "").strip().lower())
                else:
                    _dk = (call.name, (call.arguments.get("message_id") or "").strip())
                _dedup_set.add(_dk)
                logger.info(f"Dedup: recorded successful {call.name}, key={_dk}")

            # Push TOOL_ACTIVITY for tool_result
            await self._push_mesh_tool_activity(
                to_node=original_sender,
                event_type="tool_result",
                tool_name=call.name,
                data={"result": result_str[:1000], "success": True},
                in_reply_to=trigger_msg_id,
            )

            return result_str

        except TypeError as e:
            logger.error(f"Tool {call.name} argument error: {e}")
            error_msg = f"Error: Invalid arguments for '{call.name}': {e}"
            await self._push_mesh_tool_activity(
                to_node=original_sender,
                event_type="tool_result",
                tool_name=call.name,
                data={"result": error_msg, "success": False, "error": str(e)},
                in_reply_to=trigger_msg_id,
            )
            return error_msg
        except Exception as e:
            logger.exception(f"Tool {call.name} execution failed: {e}")
            error_msg = f"Error executing '{call.name}': {e}"
            await self._push_mesh_tool_activity(
                to_node=original_sender,
                event_type="tool_result",
                tool_name=call.name,
                data={"result": error_msg, "success": False, "error": str(e)},
                in_reply_to=trigger_msg_id,
            )
            return error_msg

    async def _push_mesh_tool_activity(
        self,
        to_node: str,
        event_type: str,
        tool_name: str,
        data: dict[str, Any],
        in_reply_to: str | None = None,
        tool_source: str = "mesh",
    ) -> None:
        """Push a TOOL_ACTIVITY message for a mesh tool event."""
        activity_msg = make_tool_activity(
            from_node=self.node_id,
            to_node=to_node,
            event_type=event_type,
            tool_name=tool_name,
            tool_source=tool_source,
            data=data,
            in_reply_to=in_reply_to,
        )
        await self._conn.send(activity_msg)

    async def _request_confirmation(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        original_sender: str,
    ) -> bool:
        """
        Request user confirmation for a tool action.

        Sends CONFIRM_REQUEST to the original sender and waits for response.
        Returns True if confirmed, False if rejected or timeout.
        """
        # Build human-readable preview
        preview = self._format_tool_preview(tool_name, tool_args)

        # Create and send confirmation request
        confirm_msg = make_confirm_request(
            from_node=self.node_id,
            to_node=original_sender,
            tool_name=tool_name,
            tool_args=tool_args,
            preview=preview,
        )

        # Register pending confirmation
        event = asyncio.Event()
        self._pending_confirms[confirm_msg.id] = event

        logger.info(f"Requesting confirmation from {original_sender} for {tool_name}")
        await self._conn.send(confirm_msg)

        # Wait for response with timeout
        try:
            await asyncio.wait_for(event.wait(), timeout=self.CONFIRM_TIMEOUT)
            confirmed = self._confirm_results.pop(confirm_msg.id, False)
            logger.info(f"Confirmation result for {tool_name}: {confirmed}")
            return confirmed
        except asyncio.TimeoutError:
            logger.warning(f"Confirmation timeout for {tool_name}")
            return False
        finally:
            # Cleanup
            self._pending_confirms.pop(confirm_msg.id, None)
            self._confirm_results.pop(confirm_msg.id, None)

    def _format_tool_preview(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Format a human-readable preview of a tool action."""
        # Tool-specific formatting for common tools
        if tool_name == "gmail_send_message":
            to = tool_args.get("to", "?")
            subject = tool_args.get("subject", "(no subject)")
            body = tool_args.get("body", "")[:100]
            return f"Send email to {to}\nSubject: {subject}\nBody: {body}..."

        if tool_name == "gmail_reply_to":
            msg_id = tool_args.get("message_id", "?")
            body = tool_args.get("body", "")[:100]
            return f"Reply to message {msg_id[:20]}...\nBody: {body}..."

        if tool_name == "calendar_create_event":
            summary = tool_args.get("summary", "?")
            start = tool_args.get("start", "?")
            end = tool_args.get("end", "?")
            return f"Create event: {summary}\nFrom {start} to {end}"

        if tool_name == "calendar_delete_event":
            event_id = tool_args.get("event_id", "?")
            return f"Delete calendar event: {event_id}"

        # Generic fallback
        args_str = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
        return f"{tool_name}({args_str})"

    # =========================================================================
    # Summarization
    # =========================================================================

    def load_summary_from_disk(self) -> bool:
        """Load saved summary from disk if available."""
        summary = self.load_summary()
        if summary:
            self._summary = summary
            # Also sync to ConversationHistory
            self._conv_history.summary = summary
            return True
        return False

    def _manage_in_flight_context(
        self,
        history: list[HistoryMessage],
        in_flight_threshold: float = 0.8,
    ) -> list[HistoryMessage]:
        """
        Prune in-flight tool results if context grows too large during tool loop.

        During a single request, tool results accumulate and can balloon the
        context far beyond what persisted history summarization manages. This
        method trims older in-flight results to keep context manageable.

        Args:
            history: Current history including in-flight tool results
            in_flight_threshold: Fraction of soft_limit to trigger pruning (default 0.8)

        Returns:
            Pruned history list with [previous tool results omitted] markers
        """
        # Use dedicated in-flight limit from config (decoupled from rolling window)
        # Simple workers get a lower limit to prevent over-processing
        _in_flight_override = getattr(self, '_in_flight_override', None)
        if _in_flight_override:
            threshold_tokens = _in_flight_override
        elif getattr(self.config, 'worker_in_flight_token_limit', None):
            threshold_tokens = self.config.worker_in_flight_token_limit
        else:
            threshold_tokens = int(self._soft_limit * in_flight_threshold)
        estimated = estimate_history_tokens(history)

        if estimated <= threshold_tokens:
            return history  # No pruning needed

        logger.info(
            f"In-flight context exceeds threshold ({estimated} > {threshold_tokens}), "
            f"pruning older tool results (keeping last {self._keep_recent_results})"
        )

        # Identify in-flight entries (tool results from current loop)
        # These have source="in_flight"
        in_flight_indices = [
            i for i, msg in enumerate(history)
            if getattr(msg, 'source', 'persisted') == 'in_flight'
        ]

        if len(in_flight_indices) <= self._keep_recent_results:
            # Not enough in-flight results to prune
            logger.debug("Not enough in-flight results to prune, skipping")
            return history

        # Find indices to prune (all except the last N in-flight entries)
        indices_to_prune = in_flight_indices[:-self._keep_recent_results]

        # Build pruned history
        pruned = []
        pruned_count = 0

        for i, msg in enumerate(history):
            if i in indices_to_prune:
                pruned_count += 1
            else:
                pruned.append(msg)

        # Insert marker after persisted history, before remaining in-flight
        if pruned_count > 0:
            # Find where to insert the marker (after last persisted, before first remaining in-flight)
            insert_idx = None
            for i, msg in enumerate(pruned):
                if getattr(msg, 'source', 'persisted') == 'in_flight':
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

            new_estimate = estimate_history_tokens(pruned)
            logger.info(
                f"Pruned {pruned_count} in-flight entries: "
                f"{estimated} -> {new_estimate} tokens"
            )

        return pruned

    def _truncate_extreme_result(self, result: str, max_chars: int | None = None) -> str:
        """
        Truncate a tool result that exceeds the context limit on its own.

        This is a safety valve for extreme cases where a single tool result
        (e.g., a huge file read or web fetch) would exceed the soft limit.

        Args:
            result: The tool result string
            max_chars: Max characters to allow. Defaults to soft_limit * 3 (chars/token ratio)

        Returns:
            Original result if under limit, truncated result with marker if over
        """
        # Estimate: ~3 chars per token for mixed content
        if max_chars is None:
            max_chars = self._soft_limit * 3

        if len(result) <= max_chars:
            return result

        original_size = len(result)
        # Keep first portion, truncate rest
        truncated = result[:max_chars]
        marker = f"\n\n[TRUNCATED: Original size {original_size:,} chars, kept first {max_chars:,} chars]"
        logger.warning(
            f"Extreme result truncation: {original_size:,} -> {max_chars:,} chars"
        )
        return truncated + marker

    async def _maybe_summarize_on_startup(self) -> None:
        """
        Check if context is bloated at startup and run summarization synchronously.

        This prevents sending huge contexts to the LLM on first message after
        a restart with a large history file but no summary.
        """
        if self.llm_client is None:
            return
        if not getattr(self.config, 'history_summarization_enabled', False):
            return

        # Sync conv_history and check token estimate
        self._sync_conv_history()
        window_tokens = self._conv_history.estimate_window_tokens()

        # Use rolling window trigger: 2× window budget (same as normal trigger)
        W = self._conv_history._window_budget
        STARTUP_THRESHOLD = 2 * W

        if window_tokens > STARTUP_THRESHOLD:
            logger.info(
                f"Startup: window bloated ({window_tokens} tokens > 2×W={STARTUP_THRESHOLD}), "
                f"running synchronous summarization before accepting messages"
            )
            self._summarizing = True
            try:
                await self._run_summarization()
            finally:
                self._summarizing = False

    def _check_and_trigger_summarization(self) -> None:
        """
        Check if summarization is needed and trigger it in the background.

        Delegates to ConversationHistory for token estimation and triggering.
        """
        if self._summarizing:
            return

        if self.llm_client is None:
            return
        if not getattr(self.config, 'history_summarization_enabled', False):
            return

        # Sync conv_history from _history so token estimation is accurate
        self._sync_conv_history()

        if self._conv_history.needs_summarization():
            W = self._conv_history._window_budget
            window_tokens = self._conv_history.estimate_window_tokens()
            logger.info(
                f"Rolling window trigger: window={window_tokens} tokens >= 2×W={2 * W}, "
                f"triggering background summarization"
            )
            self._summarizing = True
            self._summarization_task = asyncio.create_task(
                self._run_summarization()
            )

    async def _run_summarization(self) -> None:
        """
        Run summarization in the background.

        Delegates to ConversationHistory.summarize() which handles:
        - Rolling window partition at W tokens
        - Bootstrap truncation for huge histories
        - LLM call via pluggable client
        - SummaryState creation and persistence

        DUAL-STORE SYNC NOTE: AgentNode has two history stores:
        - self._history: canonical append-only list (never pruned)
        - self._conv_history: ConversationHistory with summary+window

        After ConversationHistory.summarize() trims the window, this method
        overrides messages_summarized with the _history-indexed value
        (len(self._history) - len(window)). This ensures _sync_conv_history()
        correctly skips summarized entries on the next startup. The rolling
        window change does not affect this reconciliation.
        """
        try:
            # Sync conv_history from _history
            self._sync_conv_history()

            # Use ConversationHistory's summarize method
            await self._conv_history.summarize(
                llm_client=self.llm_client,
            )

            # Sync the summary back to AgentNode state
            if self._conv_history.summary:
                # Adjust messages_summarized to reflect position in _history
                # ConversationHistory.summarize() folds window turns; we need
                # to translate that back to _history offset
                new_summary = self._conv_history.summary
                # The window now contains only the kept (recent) turns.
                # messages_summarized = total _history entries - window size
                kept_window_size = len(self._conv_history.window)
                messages_summarized = len(self._history) - kept_window_size

                self._summary = SummaryState(
                    summary_text=new_summary.summary_text,
                    messages_summarized=messages_summarized,
                    created_at=new_summary.created_at,
                    token_estimate=new_summary.token_estimate,
                )

                # Keep _conv_history.summary aligned with self._summary
                # so both use the _history-indexed messages_summarized.
                self._conv_history.summary = self._summary

                # Persist to file
                if self._persist:
                    self.save_summary(self._summary)

                # Archive to SQLite if we have a message store
                if self._message_store:
                    try:
                        conv_id = f"agent:{self.node_id}"
                        self._message_store.save_summary(
                            conversation_id=conv_id,
                            summary_text=self._summary.summary_text,
                            messages_summarized=self._summary.messages_summarized,
                            token_estimate=self._summary.token_estimate,
                            created_at=self._summary.created_at,
                        )
                        logger.debug(f"Archived summary to SQLite for {conv_id}")
                    except Exception as e:
                        logger.warning(f"Failed to archive summary to SQLite: {e}")

                logger.info(
                    f"Summarization complete: {messages_summarized} messages -> "
                    f"~{self._summary.token_estimate} token summary"
                )

        except Exception as e:
            logger.exception(f"Summarization failed: {e}")
        finally:
            self._summarizing = False
            self._summarization_task = None

    # =========================================================================
    # Scheduled Wakes (agent-local timer management)
    # =========================================================================

    async def _scheduler_loop(self) -> None:
        """
        Background loop that checks for due scheduled wakes and delivers them.

        Runs every _scheduler_check_interval seconds (default 10s).
        """
        logger.debug("Scheduler loop starting")
        while True:
            try:
                await asyncio.sleep(self._scheduler_check_interval)

                # Don't fire wakes while agent is processing (extended processing)
                # This prevents interrupting the agent mid-thought
                if self._processing:
                    continue

                now = datetime.now(timezone.utc)
                due_wakes = [
                    w for w in self._scheduled_wakes.values()
                    if w.wake_time <= now
                ]

                for wake in due_wakes:
                    # Remove from in-memory dict BEFORE delivering to prevent
                    # duplicate delivery if the loop re-enters during await
                    del self._scheduled_wakes[wake.id]

                    # Handle recurrence: compute next time and re-insert
                    if wake.recurrence:
                        next_time = compute_next_recurrence(wake.wake_time, wake.recurrence)
                        if next_time:
                            # Same wake ID — advance the time, keep everything else
                            next_wake = ScheduledWake(
                                id=wake.id,
                                wake_time=next_time,
                                prompt=wake.prompt,
                                requested_by=wake.requested_by,
                                created_at=wake.created_at,
                                recurrence=wake.recurrence,
                            )
                            self._scheduled_wakes[wake.id] = next_wake
                            # Update SQLite with the new wake time
                            if self._memory_system and self._memory_system._store:
                                try:
                                    self._memory_system._store.save_wake(
                                        wake_id=wake.id,
                                        wake_time=next_time.isoformat(),
                                        prompt=wake.prompt,
                                        requested_by=wake.requested_by,
                                        created_at=wake.created_at.isoformat(),
                                        recurrence=wake.recurrence,
                                    )
                                except Exception as e:
                                    logger.warning(f"Failed to persist recurring wake {wake.id}: {e}")
                            logger.info(f"Recurring wake {wake.id} rescheduled for {next_time.isoformat()} "
                                        f"(rule={wake.recurrence})")
                        else:
                            # Invalid recurrence — delete from SQLite, don't reschedule
                            if self._memory_system and self._memory_system._store:
                                try:
                                    self._memory_system._store.delete_wake(wake.id)
                                except Exception as e:
                                    logger.warning(f"Failed to delete wake {wake.id} from SQLite: {e}")
                            logger.warning(f"Recurring wake {wake.id} has invalid recurrence '{wake.recurrence}', not rescheduling")
                    else:
                        # One-shot wake — delete from SQLite
                        if self._memory_system and self._memory_system._store:
                            try:
                                self._memory_system._store.delete_wake(wake.id)
                            except Exception as e:
                                logger.warning(f"Failed to delete wake {wake.id} from SQLite: {e}")

                    logger.info(f"Delivering scheduled wake: {wake.id}")
                    await self._deliver_wake(wake)

            except asyncio.CancelledError:
                logger.debug("Scheduler loop cancelled")
                break
            except Exception as e:
                logger.exception(f"Scheduler loop error: {e}")
                # Continue running despite errors

    async def _deliver_wake(self, wake: ScheduledWake) -> None:
        """
        Deliver a scheduled wake by routing it through on_message().

        The message is framed as coming from the user who scheduled it
        (or a default user), so the LLM treats it as a real request and
        sends its response back to that user. Routes through the full
        RouterV2 pipeline when available.
        """
        # Format the wake time for display
        import time as time_module
        local_offset = timedelta(seconds=-time_module.timezone if time_module.daylight == 0 else -time_module.altzone)
        local_tz = timezone(local_offset)
        wake_time_local = wake.wake_time.astimezone(local_tz)
        scheduled_at_local = wake.created_at.astimezone(local_tz)

        # Use the requesting user as from_node so the response routes back to them.
        # Fall back to user:yourname if no requester recorded (legacy wakes).
        from_node = wake.requested_by or "user:yourname"

        recurrence_note = ""
        if wake.recurrence:
            recurrence_note = f"\nThis is a recurring wake ({wake.recurrence}). It will fire again automatically.\n"

        synthetic_msg = Message(
            type=MessageType.MESSAGE,
            from_node=from_node,
            to_node=self.node_id,
            content=(
                f"[Scheduled Wake — {wake.id}]\n"
                f"You scheduled this wake at {scheduled_at_local.strftime('%Y-%m-%d %H:%M %Z')} "
                f"to fire at {wake_time_local.strftime('%Y-%m-%d %H:%M %Z')}.\n"
                f"{recurrence_note}"
                f"The prompt you left yourself:\n\n"
                f"{wake.prompt}\n\n"
                f"Please act on this prompt now and send your response to {from_node}."
            ),
        )

        # Route through on_message() so it goes through RouterV2/processing lock/etc.
        try:
            await self.on_message(synthetic_msg)
        except Exception as e:
            logger.exception(f"Error processing scheduled wake {wake.id}: {e}")

    def schedule_wake(self, wake_time: str, prompt: str,
                      requested_by: str = "",
                      recurrence: str | None = None) -> dict:
        """
        Schedule a wake-up at the specified time with the given prompt.

        Args:
            wake_time: When to wake. ISO 8601, relative ("in 30 minutes"), or
                      natural time ("5pm").
            prompt: The prompt to deliver at wake time.
            requested_by: Node ID of the user who triggered the schedule.
            recurrence: Optional recurrence rule (e.g. "daily", "every 2 hours").

        Returns:
            dict with status, wake_id, and scheduled time (or error).
        """
        try:
            parsed_time = parse_wake_time(wake_time)

            # Validate it's in the future
            now = datetime.now(timezone.utc)
            if parsed_time <= now:
                return {
                    "status": "error",
                    "error": f"Wake time must be in the future. Parsed: {parsed_time.isoformat()}, now: {now.isoformat()}"
                }

            # Validate recurrence rule if provided
            if recurrence:
                test_next = compute_next_recurrence(parsed_time, recurrence)
                if test_next is None:
                    return {
                        "status": "error",
                        "error": f"Invalid recurrence rule: '{recurrence}'. "
                                 f"Supported: daily, weekly, weekdays, hourly, every N minutes/hours/days"
                    }

            # Create the wake
            wake_id = f"wake-{uuid.uuid4().hex[:8]}"
            wake = ScheduledWake(
                id=wake_id,
                wake_time=parsed_time,
                prompt=prompt,
                requested_by=requested_by,
                recurrence=recurrence,
            )

            self._scheduled_wakes[wake_id] = wake

            # Persist to SQLite if memory system is available
            if self._memory_system and self._memory_system._store:
                try:
                    self._memory_system._store.save_wake(
                        wake_id=wake_id,
                        wake_time=parsed_time.isoformat(),
                        prompt=prompt,
                        requested_by=requested_by,
                        created_at=wake.created_at.isoformat(),
                        recurrence=recurrence,
                    )
                except Exception as e:
                    logger.warning(f"Failed to persist wake {wake_id} to SQLite: {e}")

            # Format local time for confirmation
            import time as time_module
            local_offset = timedelta(seconds=-time_module.timezone if time_module.daylight == 0 else -time_module.altzone)
            local_tz = timezone(local_offset)
            local_time = parsed_time.astimezone(local_tz)

            logger.info(f"Scheduled wake {wake_id} for {local_time.strftime('%Y-%m-%d %H:%M %Z')} "
                        f"(requested_by={requested_by}, recurrence={recurrence})")

            result = {
                "status": "ok",
                "wake_id": wake_id,
                "wake_time_utc": parsed_time.isoformat(),
                "wake_time_local": local_time.strftime("%Y-%m-%d %H:%M %Z"),
                "prompt_preview": prompt[:100] + ("..." if len(prompt) > 100 else ""),
            }
            if recurrence:
                result["recurrence"] = recurrence
            return result

        except ValueError as e:
            return {"status": "error", "error": str(e)}

    def list_scheduled_wakes(self) -> list[dict]:
        """
        List all pending scheduled wakes.

        Returns:
            List of wake info dicts with id, time, and prompt preview.
        """
        import time as time_module
        local_offset = timedelta(seconds=-time_module.timezone if time_module.daylight == 0 else -time_module.altzone)
        local_tz = timezone(local_offset)

        wakes = []
        for wake in sorted(self._scheduled_wakes.values(), key=lambda w: w.wake_time):
            local_time = wake.wake_time.astimezone(local_tz)
            entry = {
                "id": wake.id,
                "wake_time_utc": wake.wake_time.isoformat(),
                "wake_time_local": local_time.strftime("%Y-%m-%d %H:%M %Z"),
                "prompt_preview": wake.prompt[:100] + ("..." if len(wake.prompt) > 100 else ""),
                "created_at": wake.created_at.isoformat(),
            }
            if wake.recurrence:
                entry["recurrence"] = wake.recurrence
            wakes.append(entry)
        return wakes

    def cancel_scheduled_wake(self, wake_id: str) -> dict:
        """
        Cancel a scheduled wake by ID.

        Args:
            wake_id: The ID of the wake to cancel.

        Returns:
            dict with status (and error if failed).
        """
        if wake_id not in self._scheduled_wakes:
            return {"status": "error", "error": f"No scheduled wake with ID: {wake_id}"}

        wake = self._scheduled_wakes.pop(wake_id)

        # Remove from SQLite
        if self._memory_system and self._memory_system._store:
            try:
                self._memory_system._store.delete_wake(wake_id)
            except Exception as e:
                logger.warning(f"Failed to delete wake {wake_id} from SQLite: {e}")

        logger.info(f"Cancelled scheduled wake: {wake_id}")

        return {
            "status": "ok",
            "cancelled_id": wake_id,
            "was_scheduled_for": wake.wake_time.isoformat(),
        }

    # =========================================================================
    # Router V2 Full Mode — Tool Execution + LLM Loop
    # =========================================================================

    async def _execute_all_tools(
        self,
        tool_calls: list["ToolCall"],
        trigger_msg: Message,
        allowed_tools: set[str] | None = None,
        per_call_results: dict[str, str] | None = None,
    ) -> str:
        """Execute tool calls — handles both mesh special tools and registry tools.

        Same execution paths as _process_with_llm() but without:
        - messages_sent / capturing_send tracking
        - "only query tools" early termination
        - sleep-as-terminal logic

        Args:
            per_call_results: When provided, populated with {call_id: result_text}
                for each tool call.  Used by the native multi-turn reasoning path
                to build per-tool-call ``role: tool`` messages for DeepSeek.

        Bug 5: when ``allowed_tools`` is provided (restricted CC monitor mode),
        any tool call whose name is not in that set is rejected — not executed —
        with a warning. This is the execution-time enforcement that the
        prompt-level ``tool_filter`` alone did not provide: the XML-fallback
        parser will happily surface a ``<tool_call name="bash_exec">`` emitted
        as text, and without this gate it would run.
        """
        SPECIAL_TOOLS = {
            "send_message", "attach_file", "channel_list", "channel_members",
            "schedule_wake", "schedule_list", "schedule_cancel",
            "agent_shutdown", "mesh_status", "agent_status",
            "sleep",
        } | self._TODO_TOOL_NAMES

        results = []

        def _track(call: "ToolCall", result_text: str) -> None:
            if per_call_results is not None and call.call_id:
                per_call_results[call.call_id] = result_text

        # Bug 5: enforce the offered allowlist at execution time.
        if allowed_tools is not None:
            rejected = [c for c in tool_calls if c.name not in allowed_tools]
            tool_calls = [c for c in tool_calls if c.name in allowed_tools]
            for call in rejected:
                logger.warning(
                    "[TOOL-GUARD] Rejected out-of-scope tool '%s' (allowed: %s)",
                    call.name, sorted(allowed_tools),
                )
                err_text = (
                    f"Error: tool '{call.name}' is not available in this mode. "
                    f"Only these tools may be used: {', '.join(sorted(allowed_tools))}."
                )
                results.append(
                    f'<mesh_result name="{call.name}">\n{err_text}\n</mesh_result>'
                )
                _track(call, err_text)
        special = [c for c in tool_calls if c.name in SPECIAL_TOOLS]
        other = [c for c in tool_calls if c.name not in SPECIAL_TOOLS]

        # Route worker tools (worker_launch, worker_status) through
        # RouterV2's per-instance handlers BEFORE the global registry.
        # These tools need RouterV2 instance state and can't be static.
        router = getattr(self, '_router_v2', None)
        worker_handlers = getattr(router, '_worker_tool_handlers', {}) if router else {}
        worker_tool_calls = [c for c in other if c.name in worker_handlers]
        other = [c for c in other if c.name not in worker_handlers]

        for call in worker_tool_calls:
            handler = worker_handlers[call.name]
            try:
                result = await handler(**call.arguments)
            except Exception as e:
                result = f"Error: {call.name} failed: {e}"
                logger.exception(f"Worker tool {call.name} raised: {e}")
            results.append(f'<mesh_result name="{call.name}">\n{result}\n</mesh_result>')
            _track(call, str(result))

        # Execute mesh-specific tools via dedicated handlers
        for call in special:
            if call.name == "send_message":
                result = await self._execute_send_message(call.arguments, trigger_msg)
                if self._router_v2:
                    self._router_v2._last_router_call_sent_message = True
            elif call.name == "attach_file":
                result = await self._execute_attach_file(call.arguments)
            elif call.name == "channel_list":
                result = await self._execute_channel_list()
            elif call.name == "channel_members":
                result = await self._execute_channel_members(call.arguments)
            elif call.name == "schedule_wake":
                result = self._execute_schedule_wake(
                    call.arguments, requested_by=trigger_msg.from_node,
                )
            elif call.name == "schedule_list":
                result = self._execute_schedule_list()
            elif call.name == "schedule_cancel":
                result = self._execute_schedule_cancel(call.arguments)
            elif call.name == "agent_shutdown":
                result = await self._execute_agent_shutdown(call.arguments)
            elif call.name == "mesh_status":
                result = await self._execute_mesh_status()
            elif call.name == "agent_status":
                result = await self._execute_agent_status(call.arguments)
            elif call.name in self._TODO_TOOL_NAMES:
                result = await self._execute_todo_tool_safe(call.name, call.arguments, trigger_msg)
            elif call.name == "sleep":
                reason = call.arguments.get("reason", "No reason given")
                result = f"Sleep recorded: {reason}"
                logger.info(f"Router tool loop: sleep called — {reason}")
            else:
                result = f"Error: Unknown special tool '{call.name}'"
            results.append(f'<mesh_result name="{call.name}">\n{result}\n</mesh_result>')
            _track(call, str(result))

        # Execute registry tools via standard confirmation path
        if other:
            for call in other:
                result = await self._execute_single_tool_with_confirmation(
                    call, trigger_msg.from_node, trigger_msg.id
                )
                results.append(f'<mesh_result name="{call.name}">\n{result}\n</mesh_result>')
                _track(call, result)

        return "\n\n".join(results)

    async def _router_process_with_llm(
        self,
        trigger_msg: Message,
        system_prompt: str,
        llm_client: "LLMClient",
        tool_names: list[str] | None = None,
        max_iters: int = 10,
        router_history: "ConversationHistory | None" = None,
        instructions: str = "",
        monitor_mode: bool = False,
    ) -> str:
        """Simplified LLM tool loop for the full router.

        monitor_mode: set by the CC session monitor's delivery path. When True,
        (a) the offered tool_names are enforced as a hard allowlist (Bug 5), and
        (b) ``sleep`` is terminal and ``send_message`` ends the loop unless
        accompanied by another actionable tool call (Bug 6) — preventing the
        sleep-loops and duplicate deliveries seen in monitor events.

        Handles both mesh-specific tools (send_message, schedule_*, mesh_status)
        and registry tools (file_read, exa_search, etc.).

        Retains all backend robustness from _process_with_llm():
        - CC event collection + activity streaming
        - Token usage accumulation
        - Reasoning content preservation
        - Extreme result truncation
        - In-flight context management
        - Per-iteration error handling
        """
        from .llm import HistoryMessage
        from .conversation_history import Turn
        from datetime import datetime, timezone

        # Build history from router's ConversationHistory (required)
        if router_history:
            history = router_history.build_context_for_llm()
        else:
            logger.warning("_router_process_with_llm called without router_history, using empty history")
            history = []

        # ── CC Event Collection ──
        # Router-originated CC events go into a SEPARATE list (_router_cc_events)
        # to prevent leaking into worker activity monitoring. The watchdog reads
        # _current_cc_events (worker's list) via _cc_events_fn; if we shared the
        # list, router tool calls during BUSY handling would masquerade as worker
        # progress and confuse the watchdog into thinking the worker has drifted.
        self._router_cc_events.clear()

        async def push_cc_activity(event: CCToolEvent) -> None:
            """Push CC tool event to the user who triggered this turn."""
            activity_msg = make_tool_activity(
                from_node=self.node_id,
                to_node=trigger_msg.from_node,
                event_type=event.event_type,
                tool_name=event.tool_name,
                tool_source="cc",
                data={
                    "args": event.data if event.event_type == "tool_call" else None,
                    "result": event.data if event.event_type == "tool_result" else None,
                    "call_id": event.call_id,
                },
                in_reply_to=trigger_msg.id,
            )
            await self._conn.send(activity_msg)

        cc_collector = CCToolCollector(
            realtime_list=self._router_cc_events,
            activity_callback=push_cc_activity,
        )

        # ── Usage Tracking ──
        cumulative_usage = {
            "input_tokens": 0, "output_tokens": 0,
            "cache_creation_tokens": 0, "cache_read_tokens": 0,
            "reasoning_tokens": 0, "total_tokens": 0, "llm_calls": 0,
        }

        # ── MCP Config ──
        # Expose trigger_msg to socket handler so mesh-tool send_message works
        _cc_use_mcp = getattr(self.llm_config, 'cc_use_mcp', False) if self.llm_config else False
        _mcp_config: str | None = None
        if self._tool_socket_path:
            self._current_trigger_msg = trigger_msg
        if _cc_use_mcp and self._tool_socket_path:
            _mcp_config = self._build_mcp_config(self._tool_socket_path)

        # Mesh-harness backend: pass agent socket so subprocess can call agent-local tools
        if (self.llm_config and self.llm_config.backend == "mesh-harness"
                and self._tool_socket_path):
            self.llm_client.config.harness_agent_socket = self._tool_socket_path

        response_text = ""
        _intermediate_text: list[str] = []
        # Signature of the previous iteration's tool calls, used to detect a
        # non-convergent loop (the model re-issuing identical calls every turn).
        _prev_tool_sig: tuple | None = None
        _worker_launched_via_tool = False
        _base_instructions = instructions

        # ── Native Multi-Turn Reasoning (DeepSeek) ──
        # DeepSeek v4-pro with thinking requires reasoning_content passback in
        # native assistant messages between tool iterations.  Without this, the
        # model restarts its reasoning chain from scratch every iteration.
        _use_native_reasoning = llm_client.supports_native_reasoning_multiturn
        _native_messages: list[dict] | None = None
        _openai_tools: list[dict] | None = None

        for iteration in range(max_iters):
            try:
                # Clear CC collector for this iteration
                cc_collector.clear()

                # ── Turn Counter ──
                remaining = max_iters - iteration - 1
                turn_hint = (
                    f"\n\n[Turn {iteration + 1} of {max_iters}. "
                    f"{remaining} turn{'s' if remaining != 1 else ''} remaining. "
                )
                if remaining <= 3 and remaining > 0:
                    turn_hint += (
                        "You are running low on turns. If the task needs more "
                        "work than you can accomplish, consider launching a CC "
                        "session or worker now.]"
                    )
                elif remaining == 0:
                    turn_hint += (
                        "This is your FINAL turn — no further actions after this.]"
                    )
                else:
                    turn_hint += "Budget your actions accordingly.]"
                if not _worker_launched_via_tool:
                    instructions = (_base_instructions or "") + turn_hint

                # ── In-Flight Context Management ──
                # Prevents context from ballooning during multi-iteration tool loops.
                history = self._manage_in_flight_context(history)

                # ── LLM Call ──
                if _native_messages is not None:
                    # Native multi-turn path: DeepSeek with reasoning.
                    # Inject turn hint as a user message so the model sees budget info.
                    if not _worker_launched_via_tool:
                        _native_messages.append({
                            "role": "user",
                            "content": turn_hint.strip("[] \n"),
                        })
                    # Strip tools when empty (post-worker_launch)
                    _mt_tools = _openai_tools if tool_names else None
                    response_text, tool_calls, _mt_usage = (
                        await llm_client.complete_multi_turn(
                            _native_messages, tools=_mt_tools,
                        )
                    )
                    # Remove the turn-hint user message we injected — it was
                    # consumed by the API and shouldn't persist across iterations.
                    if (not _worker_launched_via_tool
                            and _native_messages
                            and _native_messages[-1].get("role") == "user"):
                        _native_messages.pop()
                    logger.info(
                        "[NATIVE-MT] iteration %d: content=%d chars, "
                        "tool_calls=%d, reasoning=%s",
                        iteration + 1, len(response_text or ""),
                        len(tool_calls),
                        bool(getattr(llm_client, '_last_reasoning_content', None)),
                    )
                else:
                    response_text, tool_calls = await llm_client.complete_with_tools(
                        history=history,
                        node_id=self.node_id,
                        system_prompt=system_prompt,
                        tool_registry=self.tool_registry,
                        tool_names=tool_names,
                        callback=cc_collector,
                        instructions=instructions,
                        trigger_msg=trigger_msg,
                        mcp_config=_mcp_config,
                    )

                # ── Usage Accumulation ──
                if llm_client._last_usage:
                    u = llm_client._last_usage
                    for key in ("input_tokens", "output_tokens",
                                "cache_creation_tokens", "cache_read_tokens",
                                "reasoning_tokens", "total_tokens"):
                        cumulative_usage[key] += u.get(key, 0)
                    cumulative_usage["llm_calls"] += 1
                    cumulative_usage["backend"] = u.get("backend", "")
                    cumulative_usage["model"] = u.get("model", "")

                # ── CC Event Storage ──
                # Persists CC tool activity into history for future context.
                if cc_collector.events:
                    await self._store_cc_tool_context(cc_collector.events, trigger_msg)

                if not tool_calls:
                    # XML fallback: some models (DeepSeek) emit tool calls as
                    # XML text instead of using the API's function calling.
                    # Parse them and continue the tool loop.
                    if response_text:
                        from .router_v2 import extract_xml_tool_calls, strip_xml_tool_calls
                        xml_calls = extract_xml_tool_calls(response_text)
                        if xml_calls:
                            logger.info(
                                f"[XML-FALLBACK] Parsed {len(xml_calls)} tool call(s) "
                                f"from response text: {[c.name for c in xml_calls]}"
                            )
                            tool_calls = xml_calls
                            response_text = strip_xml_tool_calls(response_text)
                    if not tool_calls:
                        # Final text response — do NOT append to router_history here.
                        # The caller (_handle_idle_with_llm / _handle_busy_with_llm)
                        # stores the response via _send_and_store(), which appends a
                        # Turn(role="outgoing") to the same ConversationHistory.
                        # Appending here too would double-store the response, causing
                        # the next LLM call to see the response twice in history and
                        # potentially re-answer already-answered questions.
                        if _intermediate_text:
                            prefix = "\n\n".join(_intermediate_text)
                            response_text = (
                                f"{prefix}\n\n{response_text}"
                                if response_text else prefix
                            )
                        if _worker_launched_via_tool and response_text:
                            response_text = response_text.rstrip() + "\n\n[Worker launched]"
                        return response_text

                # ── Duplicate tool-call guard ──
                # If the model re-issues the exact same tool calls as the
                # previous iteration, it has failed to converge — e.g. a daily
                # briefing wake re-running file_read + 3x exa_search on every
                # turn. Re-executing identical calls only pollutes the context
                # (the gathered results and narration pile up in history), and
                # the subsequent final/synthesis turn then mis-reads that pile
                # as "this already happened", producing a spurious duplicate /
                # no-op message (observed on Alice's daily wake, 2026-06-21/22).
                # Break out and force a synthesis from the results already
                # gathered instead of re-executing the calls.
                import json as _json
                _sig = tuple(sorted(
                    (tc.name, _json.dumps(tc.arguments, sort_keys=True))
                    for tc in tool_calls
                ))
                if _sig == _prev_tool_sig:
                    logger.info(
                        f"Router tool loop: iteration {iteration + 1} repeats "
                        f"the previous iteration's tool calls "
                        f"({[tc.name for tc in tool_calls]}) — breaking to "
                        f"force synthesis instead of re-executing"
                    )
                    break
                _prev_tool_sig = _sig

                # ── Accumulate intermediate narration ──
                # Text produced alongside tool calls is lost when response_text
                # is overwritten on the next iteration.  Capture it so the final
                # return includes the full investigative trail.
                if response_text and response_text.strip():
                    _intermediate_text.append(response_text.strip())

                # ── Execute ALL tools — mesh specials + registry ──
                # Bug 5: in monitor mode the offered tool_names are an enforced
                # allowlist (the restricted _CC_SESSION_TOOLS set).
                _allowed = set(tool_names) if (monitor_mode and tool_names) else None
                _per_call: dict[str, str] | None = (
                    {} if _use_native_reasoning else None
                )
                tool_results = await self._execute_all_tools(
                    tool_calls, trigger_msg, allowed_tools=_allowed,
                    per_call_results=_per_call,
                )

                # ── Track tool calls for observability ──
                router = getattr(self, '_router_v2', None)
                if router and hasattr(router, '_last_router_call_tools'):
                    for tc in tool_calls:
                        arg_brief = ""
                        if tc.name == "cc_send_input":
                            raw = tc.arguments.get("text", "")
                            arg_brief = (raw[:50] + "…") if len(raw) > 50 else raw
                        elif tc.name == "send_message":
                            arg_brief = tc.arguments.get("to", "")
                        router._last_router_call_tools.append((tc.name, arg_brief))

                # ── Extreme Result Truncation ──
                tool_results = self._truncate_extreme_result(tool_results)

                # ── Build History Entry ──
                # For OpenAI native tools, response might be empty — synthesize from calls.
                response_for_history = response_text
                if not response_text and tool_calls:
                    response_for_history = "\n".join(
                        tc.raw_xml for tc in tool_calls if hasattr(tc, "raw_xml")
                    )

                # Prepend reasoning content if available (reasoning models)
                reasoning = getattr(llm_client, '_last_reasoning_content', None)
                if reasoning:
                    response_for_history = (
                        f"<reasoning>\n{reasoning}\n</reasoning>\n"
                        f"{response_for_history}"
                    )

                ts = datetime.now(timezone.utc)

                # Append to LOCAL history (for this call's growing context)
                history.append(HistoryMessage(
                    from_node=self.node_id, content=response_for_history,
                    timestamp=ts.isoformat(), source="in_flight",
                ))
                history.append(HistoryMessage(
                    from_node="system",
                    content=f"Tool execution results:\n{tool_results}",
                    timestamp=ts.isoformat(), source="in_flight",
                ))

                # Persist to router's ConversationHistory (M2 fix)
                if router_history:
                    router_history.append(Turn(
                        role="assistant", content=response_for_history,
                        timestamp=ts, from_node=self.node_id,
                    ))
                    router_history.append(Turn(
                        role="tool",
                        content=f"Tool execution results:\n{tool_results}",
                        timestamp=ts,
                    ))

                # Store tool calls in persistent history for /status visibility
                await self._store_tool_context(tool_calls, tool_results, trigger_msg)

                # ── Native Multi-Turn Reasoning Update ──
                # Build/extend the native messages array so the next iteration
                # can call complete_multi_turn with reasoning_content preserved.
                if _use_native_reasoning and _per_call is not None:
                    raw_msg = getattr(llm_client, '_last_raw_message', None)
                    if raw_msg:
                        if _native_messages is None:
                            # First tool-calling iteration: seed from the XML prompt
                            initial_prompt = getattr(llm_client, '_last_prompt', None)
                            if initial_prompt:
                                _native_messages = [
                                    {"role": "user", "content": initial_prompt}
                                ]
                                _openai_tools = self.tool_registry.get_openai_tools(
                                    tool_names
                                )
                                logger.info(
                                    "[NATIVE-MT] Initialized native multi-turn "
                                    "reasoning with %d tools", len(_openai_tools or [])
                                )
                        if _native_messages is not None:
                            # Append assistant message preserving reasoning_content
                            asst_msg: dict[str, Any] = {
                                "role": "assistant",
                            }
                            if raw_msg.get("content"):
                                asst_msg["content"] = raw_msg["content"]
                            else:
                                asst_msg["content"] = None
                            if raw_msg.get("reasoning_content"):
                                asst_msg["reasoning_content"] = raw_msg["reasoning_content"]
                            if raw_msg.get("tool_calls"):
                                asst_msg["tool_calls"] = raw_msg["tool_calls"]
                            _native_messages.append(asst_msg)

                            # Append per-tool results as native tool messages
                            for tc in tool_calls:
                                _native_messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc.call_id,
                                    "content": _per_call.get(tc.call_id, ""),
                                })

                # ── Bug 6: terminal tools in CC monitor mode ──
                # A monitor event should resolve in a couple of iterations.
                # Without this, the loop keeps going after a delivery: the
                # "sent successfully" result invites another iteration where
                # the model sends again (duplicate deliveries), and `sleep`
                # returns "Sleep recorded" and is re-called repeatedly (sleep
                # loops, 30-iteration burn). Treat `sleep` as terminal, and end
                # the loop after `send_message` unless another actionable tool
                # accompanied it.
                if monitor_mode:
                    _names = {tc.name for tc in tool_calls}
                    if "sleep" in _names:
                        logger.info("[CC-MONITOR] sleep is terminal — ending loop")
                        return response_text or ""
                    _actionable = _names - {"send_message", "sleep"}
                    if "send_message" in _names and not _actionable:
                        logger.info(
                            "[CC-MONITOR] send_message with no further action — "
                            "ending loop"
                        )
                        return response_text or ""

                # ── worker_launch is terminal ──
                # After dispatching a worker, the router gets one more toolless
                # turn to describe what it launched, then the loop exits.
                if any(tc.name == "worker_launch" for tc in tool_calls):
                    tool_names = []
                    max_iters = iteration + 2
                    _worker_launched_via_tool = True
                    instructions = (
                        (instructions + "\n\n" if instructions else "")
                        + "This is your final response. The worker has been dispatched. "
                        "Summarize what you did and what the worker will do. "
                        "Do not describe actions you will take next — the loop is ending after this response. "
                        "Note: the prior turns in this response were internal tool-gathering — the user "
                        "has NOT seen them yet. Do not claim work was 'already delivered' or that this "
                        "message is a duplicate; deliver the substantive answer (e.g. the briefing) as "
                        "part of this response."
                    )
                    logger.info("worker_launch detected — stripping tools, one final turn")

                logger.debug(
                    f"Router tool loop iteration {iteration + 1}: "
                    f"{len(tool_calls)} tool call(s)"
                )

            except Exception as e:
                logger.exception(f"Router LLM processing error (iter {iteration + 1}): {e}")
                return f"[{self.node_id}] Error processing message: {e}"

        # ── Forced Synthesis ──
        # The loop exhausted max_iters while the model was still making tool
        # calls. Make one final call with no tools to force a text-only response
        # that synthesizes the tool results gathered so far.
        logger.warning(
            f"Router tool loop hit max iterations ({max_iters}), "
            "forcing synthesis call"
        )
        instructions = (
            "You have run out of router iterations. You are no longer executing — "
            "the loop is ending NOW.\n"
            "Summarize what you attempted to do and what progress you made.\n"
            "Do NOT describe future actions you plan to take — you will not get another turn."
        )
        try:
            history = self._manage_in_flight_context(history)
            cc_collector.clear()
            if _native_messages is not None:
                _native_messages.append({
                    "role": "user", "content": instructions,
                })
                synthesis_text, _, _ = await llm_client.complete_multi_turn(
                    _native_messages, tools=None,
                )
            else:
                synthesis_text, _ = await llm_client.complete_with_tools(
                    history=history,
                    node_id=self.node_id,
                    system_prompt=system_prompt,
                    tool_registry=self.tool_registry,
                    tool_names=[],
                    callback=cc_collector,
                    instructions=instructions,
                    trigger_msg=trigger_msg,
                    mcp_config=_mcp_config,
                )
            if synthesis_text.strip():
                response_text = synthesis_text
        except Exception as e:
            logger.exception(f"Router forced synthesis failed: {e}")
        if _worker_launched_via_tool and response_text:
            response_text = response_text.rstrip() + "\n\n[Worker launched]"
        return response_text

    # =========================================================================
    # Router V2 Integration
    # =========================================================================

    def _resolve_cc_interactive_binary(self) -> str:
        """Resolve the claude binary path for CC interactive sessions.

        Checks per-agent config, then worker LLM config, then router LLM config.
        """
        if getattr(self.config, 'cc_interactive_binary', ''):
            return self.config.cc_interactive_binary
        for cfg in (self.llm_config, self._router_v2_llm_config):
            if cfg and getattr(cfg, 'cc_binary', ''):
                return cfg.cc_binary
        return ""

    def _resolve_cc_interactive_effort(self) -> str:
        """Resolve the CC effort level for interactive sessions."""
        if getattr(self.config, 'cc_interactive_effort', ''):
            return self.config.cc_interactive_effort
        for cfg in (self.llm_config, self._router_v2_llm_config):
            if cfg and getattr(cfg, 'cc_effort', ''):
                return cfg.cc_effort
        return ""

    def _resolve_cc_interactive_model(self) -> str:
        """Resolve the model for CC interactive sessions."""
        return getattr(self.config, 'cc_interactive_model', '') or ""

    def _resolve_cc_interactive_fallback_homes(self) -> list[str]:
        """Resolve fallback HOME dirs for CC account rotation.

        Checks worker LLM config, then router LLM config, then auto-discovers
        ~/.claude-acct* directories on the filesystem.
        """
        for cfg in (self.llm_config, self._router_v2_llm_config):
            if cfg and getattr(cfg, 'cc_fallback_homes', []):
                return list(cfg.cc_fallback_homes)
        # Auto-discover: scan for ~/.claude-acct* directories
        import glob
        import pwd
        real_home = pwd.getpwuid(os.getuid()).pw_dir
        discovered = sorted(glob.glob(os.path.join(real_home, ".claude-acct*")))
        if discovered:
            logger.info(f"CC interactive: auto-discovered {len(discovered)} fallback accounts")
        return discovered

    def _init_router_v2(self) -> None:
        """
        Initialize Router V2 for mediating I/O and LLM processing.

        Called from connect() after LLM client is ready.
        """
        if self._router_v2_config is None:
            return

        # Cancel any in-flight worker from a previous router instance (reconnect path).
        # Without this, the old worker's CC subprocess becomes an orphan.
        old_router = getattr(self, '_router_v2', None)
        if old_router is not None:
            logger.info("_init_router_v2: cancelling old router worker before replacement")
            asyncio.ensure_future(old_router.cancel_worker())

        # Build identity block for router context
        nick = self._nickname or ''
        identity_block = f"""<identity>
You are {self.node_id}.
Your agent type is "{self._agent_type or 'agent'}".
Your nickname is "{nick}" (how users will address you).
When you see @{nick} in a message, that message is addressed to YOU — it is not from {nick} or about a third party.
</identity>"""

        # Get tool prompt if available
        tools_block = getattr(self, '_tool_prompt', '') or ''

        # Use separate router LLM client if configured (avoids sharing with worker)
        router_llm_client = self.llm_client
        if self._router_v2_llm_config:
            self._router_v2_llm_config.node_id = self.node_id
            router_llm_client = LLMClient(self._router_v2_llm_config)
            logger.info(
                f"RouterV2 using separate LLM: backend={self._router_v2_llm_config.backend}, "
                f"model={self._router_v2_llm_config.model}"
            )

        # Summarization uses the router's LLM client (consolidated).
        if getattr(self.config, 'history_summarization_enabled', False):
            logger.info("RouterV2 summarization enabled (uses router LLM client)")
        else:
            logger.info("RouterV2 summarization disabled (rolling window mode)")

        # Build full-router tool loop callback (closes over the router's own LLM client).
        # self._router_v2 is read at call time (late binding), so it will be set by then.
        async def _router_process(trigger_msg, system_prompt, tool_names, max_iters,
                                  instructions="", monitor_mode=False):
            router_hist = getattr(self._router_v2, '_history', None) if self._router_v2 else None
            return await self._router_process_with_llm(
                trigger_msg=trigger_msg,
                system_prompt=system_prompt,
                llm_client=router_llm_client,
                tool_names=tool_names,
                max_iters=max_iters,
                router_history=router_hist,
                instructions=instructions,
                monitor_mode=monitor_mode,
            )

        # ── CC-session mode: use RouterCC + CCSession instead of RouterV2 ──
        if getattr(self.config, 'context_mode', 'rolling-window') == 'cc-session':
            from .cc_session import CCSession
            from .router_cc import RouterCC

            cc_config = self.config.cc_session
            cc_session = CCSession(
                nickname=self._nickname or self.node_id,
                agent_type=self._agent_type or "agent",
                node_id=self.node_id,
                config=cc_config,
                llm_config=self.llm_config,
                memory_system=self._memory_system,
                identity_block=identity_block,
                personality_block=self._memory_system.get_personality() if self._memory_system else "",
                mesh_protocol_block=self.system_prompt or "",
                router_host=self.config.router_host,
                router_port=getattr(self.config, 'router_ws_port', 8765),  # MCP server needs WS port
                auth_token=self.config.auth_token,
            )
            # Start the session (loads persisted session ID)
            async def _safe_cc_start():
                try:
                    await cc_session.start()
                except Exception as e:
                    logger.error(f"[{self.node_id}] CCSession.start() failed: {e}", exc_info=True)
            asyncio.ensure_future(_safe_cc_start())

            self._cc_session = cc_session
            self._router_v2 = RouterCC(
                cc_session=cc_session,
                send_fn=self._router_v2_send,
                config=self._router_v2_config,
                node_id=self.node_id,
                nickname=self.nickname,
                agent_type=self.config.agent_type,
                llm_client=router_llm_client,
                system_prompt=self.system_prompt or "",
                identity_block=identity_block,
                memory_system=self._memory_system,
                raw_send_fn=self._conn.send if self._conn else None,
            )
            logger.info(
                f"CC-session mode enabled for {self.node_id}: "
                f"model={cc_config.cc_model or (self.llm_config.model if self.llm_config else 'default')}"
            )

            # Load persisted history
            loaded = self._router_v2.load_history()
            if loaded > 0:
                logger.info(f"RouterCC loaded {loaded} persisted history entries")
            return

        # ── Rolling-window mode (default): RouterV2 or RouterV3 ──
        router_kwargs = dict(
            worker_fn=self._router_v2_worker,
            send_fn=self._router_v2_send,
            config=self._router_v2_config,
            node_id=self.node_id,
            nickname=self.nickname,
            agent_type=self.config.agent_type,
            llm_client=router_llm_client,
            system_prompt=self.system_prompt or "",
            identity_block=identity_block,
            tools_block=tools_block,
            cc_events_fn=self._get_cc_live_events,
            memory_system=self._memory_system,
            session_gap_secs=self.config.memory_reflection_session_gap_secs,
            flush_interval_tools=self.config.memory_reflection_flush_interval_tools,
            worker_llm_client=self.llm_client,
            router_process_fn=_router_process,
            cc_interactive_tools=getattr(self.config, 'cc_interactive_tools', False),
            cc_binary=self._resolve_cc_interactive_binary(),
            cc_effort=self._resolve_cc_interactive_effort(),
            cc_model=self._resolve_cc_interactive_model(),
            cc_fallback_homes=self._resolve_cc_interactive_fallback_homes(),
            harness_session_tools=getattr(self.config, 'harness_session_tools', False),
            harness_session_llm_config=getattr(self, '_harness_session_llm_config', None),
            todo_store_path=getattr(self.config, 'storage_path', None),
        )

        if getattr(self.config, 'use_router_v3', False):
            from .router_v3 import RouterV3

            async def _plan_execute_tools(tool_calls):
                """Execute tool calls for planning phases (wraps AgentNode infra)."""
                return await self._execute_tool_calls_with_confirmation(
                    tool_calls, self.node_id
                )

            self._router_v2 = RouterV3(
                **router_kwargs,
                tool_registry=self.tool_registry,
                execute_tool_fn=_plan_execute_tools,
            )
            logger.info("RouterV3 (planning pipeline) enabled")
        else:
            self._router_v2 = RouterV2(**router_kwargs)

        # Try to load persisted router history first
        loaded = self._router_v2.load_history()
        if loaded > 0:
            logger.info(
                f"RouterV2 loaded {loaded} persisted history entries "
                f"(with summary support)"
            )
        elif self._history:
            # Fallback: seed router context from worker's persisted history
            # (for first run after upgrade, or if router history doesn't exist yet)
            max_ctx = self._router_v2_config.max_context_messages
            recent = list(self._history[-max_ctx:])
            self._router_v2.set_context(recent)
            logger.info(
                f"RouterV2 seeded with {len(recent)} worker history entries "
                f"(of {len(self._history)} total)"
            )

    def _get_cc_live_events(self) -> list[Any]:
        """Return synthetic entries for in-progress CC tool calls.

        Used as the cc_events_fn callback for RouterV2, so the router
        can see what CC tools are currently executing during busy mode.
        """
        from .node import HistoryEntry

        if not self._current_cc_events:
            return []

        call_events = [e for e in self._current_cc_events if e.event_type == "tool_call"]
        result_events = {e.call_id: e for e in self._current_cc_events if e.event_type == "tool_result"}

        cc_tool_lines = []
        for call in call_events:
            args = call.data if isinstance(call.data, dict) else {}
            if call.tool_name == "cc:Read":
                summary = args.get("file_path", "")[:80]
            elif call.tool_name == "cc:Bash":
                cmd = args.get("command", "")
                summary = cmd[:80] + ("..." if len(cmd) > 80 else "")
            elif call.tool_name == "cc:Edit":
                summary = f"{args.get('file_path', '')} ({len(args.get('old_string', ''))} -> {len(args.get('new_string', ''))} chars)"
            elif call.tool_name in ("cc:Grep", "cc:Glob"):
                summary = f"pattern={args.get('pattern', '')!r}"
            else:
                items = list(args.items())[:2]
                summary = ", ".join(f"{k}={str(v)[:30]}" for k, v in items)

            result = result_events.get(call.call_id)
            if result:
                preview = str(result.data)[:100]
                if len(str(result.data)) > 100:
                    preview += "..."
                cc_tool_lines.append(f"[{call.tool_name}] {summary}\n  → {preview}")
            else:
                cc_tool_lines.append(f"[{call.tool_name}] {summary} (in progress)")

        if not cc_tool_lines:
            return []

        cc_msg = Message(
            type=MessageType.MESSAGE,
            from_node=self.node_id,
            to_node="internal",
            content=f"[CC Tool Activity (live)]\n" + "\n".join(cc_tool_lines),
            metadata={"cc_tool_events": True, "live": True},
        )
        return [HistoryEntry(message=cc_msg, direction="outgoing")]

    async def _router_v2_worker(
        self,
        context: list[Any],
        trigger: Message
    ) -> WorkerResult:
        """
        Worker function for Router V2.

        Wraps _process_with_llm() to execute the full LLM processing flow.
        Returns the worker result with response and updated context.

        Context unification: `context` is a mutable list[Turn] snapshot of the
        router's ConversationHistory. The worker appends Turn objects to it so
        the router can see live progress. On completion, the router merges the
        delta back into its canonical ConversationHistory.

        The worker intercepts ALL outgoing messages via self.send() to capture
        them as drafts — nothing is sent directly. The router's completion
        handler is the single point of outgoing communication, preventing
        duplicate messages.
        """
        from .conversation_history import Turn
        from datetime import datetime, timezone

        response_text = ""
        error = None

        # Store reference to the snapshot for appending Turn objects
        self._worker_snapshot = context

        # Track whether capturing_send has delivered any messages.
        # _process_with_llm() checks this to avoid sending redundant
        # "Done." stubs after real content has already been delivered.
        self._capturing_send_count = 0

        # Worker synthesis: initialize accumulation fields
        self._worker_all_cc_events: list = []
        self._worker_in_flight_history = None
        self._worker_buffered_messages: list[tuple[str, str]] = []
        # Latest non-empty content from capturing_send. Read by RouterV2's
        # cancel-flush helper when the buffer is empty (e.g. passthrough mode
        # produced output but synthesize-mode buffered nothing). Reset on
        # worker start and in the cleanup block below.
        self._worker_response_text: str = ""
        # Track ALL destinations the worker has sent to (buffered DMs + direct
        # channel sends). Used by the v0.2 DONE handler to decide whether the
        # LLM's wrap-up commentary would double-post to the trigger's destination.
        self._worker_sent_destinations: set[str] = set()

        # Check if synthesis is enabled (buffer instead of sending immediately)
        _synthesize = getattr(
            self._router_v2, '_config', None
        ) and getattr(self._router_v2._config, 'synthesize_enabled', False)

        # Synthesis only needs to capture messages that flow BACK to the
        # dispatch origin (the trigger's sender) or to this agent itself —
        # those are what the router's completion handler synthesizes and
        # relays. A DM to any third party (e.g. tasking another agent) is a
        # side-effect that must land LIVE: buffering it until the dispatch
        # completes means the recipient may be gone by flush time, and a
        # round-trip with it can never complete inside this dispatch.
        _buffer_dests = {
            d for d in (getattr(trigger, "from_node", None), self.node_id) if d
        }

        def _legacy_history_has_id(msg_id: str | None) -> bool:
            return bool(msg_id) and any(
                getattr(entry.message, "id", None) == msg_id
                for entry in self._history
            )

        def _legacy_history_has_outgoing(content: str, from_node: str | None) -> bool:
            return any(
                getattr(entry, "direction", "") == "outgoing"
                and getattr(entry.message, "from_node", None) == from_node
                and getattr(entry.message, "content", None) == content
                for entry in self._history
            )

        # Intercept send() to capture worker messages.
        # When synthesis is enabled, messages are BUFFERED (not sent).
        # When synthesis is disabled, messages are sent immediately (passthrough).
        original_send = self.send

        async def capturing_send(to_node, content, in_reply_to=None, attachments=None):
            nonlocal response_text
            from .node import HistoryEntry

            # Coerce content to string — LLMs occasionally pass non-string types
            content = str(content) if not isinstance(content, str) else content

            # Canonicalize attachments — stored refs must have url=None per the
            # invariant from docs/plans/mesh-attachments.md (router signs at relay).
            canonical_attachments = [a.canonical() for a in (attachments or [])]

            if isinstance(content, str) and content:
                response_text = content
                self._worker_response_text = content
                self._capturing_send_count += 1
                self._worker_sent_destinations.add(to_node)

                if _synthesize and to_node in _buffer_dests:
                    # BUFFER mode: store for synthesis, don't send yet.
                    # Only replies to the dispatch origin (or this agent) are
                    # buffered; messages to channels and third-party agents
                    # bypass buffering (sent immediately below).
                    self._worker_buffered_messages.append((to_node, content))
                    logger.info(
                        f"capturing_send: buffered message to={to_node} "
                        f"content_len={len(content)} attachments={len(canonical_attachments)}"
                    )
                    # Create a synthetic Message for history tracking
                    msg = Message(
                        type=MessageType.MESSAGE,
                        from_node=self.node_id,
                        to_node=to_node,
                        content=content,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        attachments=canonical_attachments,
                    )
                else:
                    # PASSTHROUGH mode: send immediately (legacy behavior)
                    logger.info(
                        f"capturing_send: self.node_id={self.node_id} to={to_node} "
                        f"content_len={len(content)} attachments={len(canonical_attachments)}"
                    )
                    msg = await original_send(
                        to_node,
                        content,
                        in_reply_to=in_reply_to,
                        attachments=canonical_attachments,
                    )
                    logger.info(
                        f"capturing_send: msg.from_node={msg.from_node} msg.to_node={msg.to_node} "
                        f"msg.id={msg.id}"
                    )

                # Append buffered synthetic messages to legacy _history. Messages
                # sent through original_send() have already been appended by
                # Node._send_with_retry(), so guard by id to avoid duplicate JSONL
                # entries in agent-{nickname}.json.
                if not _legacy_history_has_id(msg.id):
                    self._history.append(HistoryEntry(message=msg, direction="outgoing"))
                # Also append to snapshot as Turn (for router visibility + merge-back)
                if self._worker_snapshot is not None:
                    self._worker_snapshot.append(Turn(
                        role="outgoing",
                        content=content,
                        timestamp=datetime.now(timezone.utc),
                        from_node=self.node_id,
                        to_node=to_node,
                    ))
                logger.debug(f"RouterV2 worker {'buffered' if _synthesize else 'sent'} response ({len(content)} chars)")
                return msg
            else:
                # Non-string or empty content — send normally (rare edge case)
                msg = await original_send(
                    to_node,
                    content,
                    in_reply_to=in_reply_to,
                    attachments=canonical_attachments,
                )
                if not _legacy_history_has_id(msg.id):
                    self._history.append(HistoryEntry(message=msg, direction="outgoing"))
                return msg

        try:
            # Store the trigger in history before processing
            if not _legacy_history_has_id(trigger.id):
                await self._add_to_history(trigger, "incoming")

            # Inject router's pre-dispatch responses (e.g., "Working on it...")
            # into worker history so the worker LLM knows what was already
            # said to the user and doesn't repeat it.
            from .node import HistoryEntry as _HE
            _router_acks = []
            for _t in reversed(context):
                if _t.role == "incoming":
                    break  # Reached the trigger message — stop
                if (_t.role == "outgoing"
                        and isinstance(getattr(_t, 'meta', None), dict)
                        and _t.meta.get("router_response")):
                    _router_acks.append(_t)
            for _t in reversed(_router_acks):  # Chronological order
                if _legacy_history_has_outgoing(_t.content, _t.from_node or self.node_id):
                    continue
                _ts = _t.timestamp
                if hasattr(_ts, 'isoformat'):
                    _ts = _ts.isoformat()
                _ack_msg = Message(
                    type=MessageType.MESSAGE,
                    from_node=_t.from_node or self.node_id,
                    to_node=_t.to_node or trigger.from_node,
                    content=_t.content,
                    timestamp=_ts,
                    id=f"router-ack-{id(_t):x}",
                )
                self._history.append(_HE(message=_ack_msg, direction="outgoing"))
                logger.debug(f"Injected router ack into worker history: {_t.content[:80]!r}")

            # Temporarily replace send with capturing version
            self.send = capturing_send

            # Use existing LLM processing flow
            # This handles tool calls, controller logic, etc.
            await self._process_with_llm(trigger)

        except Exception as e:
            logger.error(f"RouterV2 worker failed: {e}", exc_info=True)
            error = e
            response_text = f"I encountered an error while processing your request: {e}"
        finally:
            # Restore original send and clean up worker state
            self.send = original_send
            # Note: _worker_snapshot, _worker_in_flight_history, _worker_all_cc_events,
            # and _worker_buffered_messages are NOT cleared here — they are consumed
            # by the WorkerResult and cleaned up after the router processes them.

        # Capture synthesis fields before cleanup
        in_flight_history = getattr(self, '_worker_in_flight_history', None)
        buffered_msgs = getattr(self, '_worker_buffered_messages', None)
        cc_events = getattr(self, '_worker_all_cc_events', None)

        # Clean up synthesis-related fields on self
        self._worker_in_flight_history = None
        self._worker_all_cc_events = []
        self._worker_buffered_messages = []
        self._worker_response_text = ""
        self._worker_sent_destinations = set()
        self._worker_snapshot = None
        self._in_flight_override = None
        self._capturing_send_count = 0

        # Return result with the snapshot (which the router also holds a reference to)
        usage = getattr(self, '_cumulative_usage', None)
        return WorkerResult(
            response=response_text,
            context=context,  # The mutable snapshot — router uses it for merge
            error=error,
            usage=usage if usage and usage.get("llm_calls", 0) > 0 else None,
            worker_in_flight_history=in_flight_history if not error else None,
            buffered_messages=buffered_msgs if buffered_msgs else None,
            worker_cc_events=cc_events if cc_events else None,
        )

    async def _router_v2_send(
        self,
        content: str,
        in_reply_to: Message | None
    ) -> None:
        """
        Send function for Router V2.

        Uses _original_send (not self.send) to bypass the worker's
        capturing_send monkey-patch. The router must always send directly.
        """
        if in_reply_to:
            target = self._infer_destination_from_trigger(in_reply_to)
            await self._original_send(target, content, in_reply_to=in_reply_to.id)
        elif self._last_user_node:
            # F5: Fallback to last known user node instead of dropping
            logger.warning(
                f"RouterV2 send called without in_reply_to, "
                f"falling back to last user: {self._last_user_node}"
            )
            await self._original_send(self._last_user_node, content)
        else:
            logger.error(
                "RouterV2 send called without in_reply_to and no last_user_node — "
                "dropping message (no valid destination)"
            )


class SimpleAgentNode(AgentNode):
    """
    Simplified agent for testing without full LLM integration.

    Echoes messages back with a prefix.
    """

    async def _process_with_llm(self, trigger_msg: Message) -> None:
        """Simple echo behavior for testing."""
        content = trigger_msg.content if isinstance(trigger_msg.content, str) else str(trigger_msg.content)
        response = f"[{self.node_id}] Received: {content}"
        await self.send(trigger_msg.from_node, response, in_reply_to=trigger_msg.id)
