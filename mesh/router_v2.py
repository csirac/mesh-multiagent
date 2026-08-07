"""
Router V2 - Thin classifier + direct worker passthrough.

Architecture:
1. Classifies messages (needs_response, needs_worker) via LLM
2. For simple messages: responds directly with short acks/greetings
3. For substantive requests: dispatches to worker, passes response directly to user
4. Handles status queries while worker is busy (with live context peek)
5. Merges worker context back with origin attribution

State machine:
- IDLE: No worker running, ready for new request
- BUSY: Worker running, router handles incoming messages

LLM Integration:
- Router LLM is used ONLY for classification and busy-state responses
- Worker's response goes directly to user (no re-summarization)
- All responses (acks, busy, worker passthrough) stored in history

Context model:
- Router's ConversationHistory is the single source of truth
- Worker receives a snapshot (mutable list[Turn]) at dispatch time
- Worker appends to the snapshot; router sees live progress via reference
- On completion, worker's delta is merged back with worker_origin attribution
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import html
import inspect
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Awaitable, Any, TYPE_CHECKING, Sequence

from pathlib import Path

from datetime import datetime, timezone

from .protocol import Message, MessageType
from .conversation_history import ConversationHistory, Turn, ROUTER_SUMMARY_PROMPT
from .llm import estimate_tokens
from .memory.system import EpisodeStats
from .memory.system_v2 import MemorySystemV2
from .storage import MessageStore
from .tool_visibility import append_tools_called_block, strip_tools_called_block
from .procedural_memory import (
    SkillCardError,
    SkillStore,
    build_skill_draft_package,
    persist_completed_worker_trace,
)
from .config import (
    BACKEND_ACCESS_CLOSED,
    BACKEND_ACCESS_OPEN,
    PevTaskConfig,
    TaskPromptConfig,
    classify_backend_access,
    normalize_entity_resolution_mode,
    normalize_pev_task_config,
    normalize_worker_task_types,
)

if TYPE_CHECKING:
    from .config import FixedToolConfig
    from .llm import LLMClient

logger = logging.getLogger(__name__)

# Compatibility alias name for "whatever backend this agent falls back to".
# It is normally also a concrete entry in backends.yaml.
_DEFAULT_BACKEND_SENTINEL = "default"

# =============================================================================
# Router Tool Set — restricted to read-only + single-action tools
# =============================================================================

# Tools the full router is allowed to use. Worker-only tools (bash_exec, file_edit,
# file_write, file_create, file_diff, agent_shutdown, browser_*, plaid mutations)
# are excluded. The router can inspect and query but not mutate the filesystem.
ROUTER_TOOL_NAMES: set[str] = {
    # Information retrieval (read-only)
    "channel_list", "channel_members", "tool_help", "mesh_list",
    "worker_launch", "worker_list", "worker_status", "worker_cancel",
    "skill_draft",
    "solicitation_scout",
    "mesh_status", "agent_status", "current_time", "get_working_directory",
    "file_read", "list_dir", "grep", "get_context", "count_words",
    "style_filter", "math_thinking",
    "exa_search", "exa_fetch_full", "extract_url",
    "literature_search", "literature_fulltext",
    "arxiv_search", "arxiv_get", "arxiv_fulltext",
    "pubmed_search", "pubmed_get", "pubmed_fulltext", "pubmed_related",
    "plaid_link_status", "plaid_accounts", "plaid_transactions",
    "synthetic_quota", "claude_code_usage",

    # Single-action (quick mutations)
    "account_get_current", "account_list", "account_set_current",
    "schedule_wake", "schedule_list", "schedule_cancel",
    "set_working_directory", "write_lines",
    "gmail_list_from_date", "gmail_get_email", "gmail_search_emails",
    "gmail_send_message", "gmail_reply_to",
    "gmail_create_draft", "gmail_draft_reply",
    "calendar_list_on_date", "calendar_create_event", "calendar_delete_event",
    "notes_search", "notes_get", "notes_list", "notes_read",
    "notes_add", "notes_delete", "notes_edit",
    "remember", "memory_list", "memory_get", "memory_search",
    "memory_add", "memory_delete", "memory_edit",
    "digest_get", "digest_edit",
    # Autonomous agent mode — project state, immutable reports, worker budget.
    "dossier_read", "dossier_edit", "dossier_write_report",
    "dossier_check_budget", "dossier_spend_budget",
    "entity_link_correct",
    "history_search",
    "essay_list", "essay_get", "essay_edit",
    "personality_get", "personality_set",
    "todo_list", "todo_add", "todo_update", "todo_toggle", "todo_remove",
    "todo_reorder", "todo_set_section_order",
    "conversation_notes_get", "conversation_notes_set",

    # Map tools (v2 memory — router can edit maps inline)
    "map_list", "map_get", "map_edit", "map_create", "set_project_context",

    # Messaging
    "send_message", "sleep",
}

# Backend types that have their own internal tool loops (ReAct / TAOR).
# Router-native worker tools are not exposed to these backends; they dispatch
# through the backend-neutral <dispatch_worker> response block instead.
HARNESS_BACKENDS: frozenset[str] = frozenset({"codex", "claude-code", "mesh-harness"})

# ``curation_status()`` surfaces the most recent turn's refusals only, capped
# so agent_status stays small even if a turn refuses on every call.
CURATION_STATUS_REJECTION_CAP = 20

# ``@deep`` is recognized anywhere in the message, not only as a prefix. The
# ``(?:^|(?<=\s))`` head requires start-of-string or a preceding whitespace
# character so address-like text (``bob@deepblue.com``, ``word@deep``) never
# reads as a directive.
_ROUTER_DEEP_DIRECTIVE_RE = re.compile(
    r"(?:^|(?<=\s))@deep(?=$|[\s:])(?:\s*:\s*|\s*)",
    re.IGNORECASE,
)


def strip_router_deep_directive(content: str) -> tuple[str, bool]:
    """Remove one ``@deep`` routing directive from string content.

    The directive may appear anywhere in the message. Only the first match is
    removed. The seam left behind is normalized so the caller never has to
    care where the directive sat: a leading directive drops the whitespace
    that preceded it, and a trailing directive drops the whitespace that
    separated it from the preceding word.
    """
    match = _ROUTER_DEEP_DIRECTIVE_RE.search(content)
    if match is None:
        return content, False
    prefix = content[:match.start()]
    suffix = content[match.end():]
    if not prefix.strip():
        # Leading directive (optionally indented): legacy behaviour.
        return suffix, True
    if not suffix:
        # Trailing directive: the separating whitespace goes with it.
        return prefix.rstrip(), True
    return prefix + suffix, True


def prepare_router_deep_override(msg: Message) -> tuple[Message, bool]:
    """Return an execution-only trigger clone for a human ``@deep`` request."""
    if (
        msg.type != MessageType.MESSAGE
        or not str(msg.from_node or "").startswith("user:")
        or not isinstance(msg.content, str)
    ):
        return msg, False
    stripped, overridden = strip_router_deep_directive(msg.content)
    if not overridden:
        return msg, False
    metadata = dict(msg.metadata) if isinstance(msg.metadata, dict) else {}
    metadata["router_deep_override"] = True
    return replace(msg, content=stripped, metadata=metadata), True


def normalize_router_deep_prompt_history(
    history: list[Any],
    trigger_msg: Message,
) -> list[Any]:
    """Normalize only the copied prompt-history trigger for exact matching.

    Persistent RouterV2 history retains the original protocol text. The LLM
    formatter identifies ``<message_received>`` by exact sender/content, so a
    transient view must carry the stripped trigger content as well.
    """
    metadata = (
        trigger_msg.metadata if isinstance(trigger_msg.metadata, dict) else {}
    )
    if not metadata.get("router_deep_override"):
        return history
    trigger_content = trigger_msg.content
    if not isinstance(trigger_content, str):
        return history

    for index in range(len(history) - 1, -1, -1):
        entry = history[index]
        if getattr(entry, "from_node", None) != trigger_msg.from_node:
            continue
        entry_to = getattr(entry, "to_node", None)
        if entry_to and trigger_msg.to_node and entry_to != trigger_msg.to_node:
            continue
        entry_content = getattr(entry, "content", None)
        if not isinstance(entry_content, str):
            continue
        normalized, overridden = strip_router_deep_directive(entry_content)
        if overridden and normalized == trigger_content:
            prompt_history = list(history)
            prompt_history[index] = replace(entry, content=trigger_content)
            return prompt_history
    return history

# ReAct loop cap for direct (non-harness) backends.
# Raised to 30 for CC-session turns (start→trust→send→monitor cycle needs many iterations).
REACT_MAX_ITERS: int = 100

# Tools that are only available to direct (non-harness) router backends.
# Harness backends have their own internal tool loops and cannot use these.
# Workers never see these tools (they use their own tool_names).
WORKER_ROUTER_TOOLS: frozenset[str] = frozenset({
    "worker_launch", "worker_list", "worker_status", "worker_cancel",
})

# Managed worker tools remain callable from ordinary harness routers, unlike
# worker_launch (which harness routers express with <dispatch_worker>).  They
# still occupy the same worker slot and are hidden when a persistent CC or
# harness session is the configured execution mechanism.
MANAGED_WORKER_ROUTER_TOOLS: frozenset[str] = frozenset({"skill_draft"})

# Typed external pipelines use the same single execution slot as workers.
# They remain available to normal harness-backed routers, but are hidden when
# a persistent CC/harness session is the configured execution mechanism.
FIXED_TOOL_ROUTER_TOOLS: frozenset[str] = frozenset({"solicitation_scout"})

# CC interactive tools: gated by NodeConfig.cc_interactive_tools.
# Added to the router tool list only when enabled per-agent.
CC_INTERACTIVE_TOOLS: frozenset[str] = frozenset({
    "cc_start_session", "cc_get_screen", "cc_send_input",
})

# Native harness session tools: gated by NodeConfig.harness_session_tools.
# The idle router gets start/send_input/get_status; the manager's full tool set
# (incl. stop) is enforced during event-delivery (monitor) mode.
HARNESS_SESSION_INTERACTIVE_TOOLS: frozenset[str] = frozenset({
    "harness_start_session", "harness_send_input", "harness_get_status",
})


CONVERSATION_NOTES_GUIDANCE = """\
<conversation_notes> may contain concise pointers to files relevant to this
conversation. Common examples include a worklog file tracking task progress
and an operations file documenting code entrypoints, launch commands, running
procedures, or host-specific constraints. Before dispatching a worker for
operational or code work, check any referenced operations/worklog files and
include the relevant constraints in your dispatch. After making operational
changes, keep the referenced files and notes pointer up to date.\
"""

SKILL_DRAFT_ROUTER_GUIDANCE = """\
ON-DEMAND SKILL FORMATION:
When the user explicitly asks you to capture the just-completed recurring
procedure as a skill/card, call `skill_draft` directly. Summarize the narrow
procedure in `task_summary` and include known authoritative absolute paths in
`source_files`. The tool automatically uses the latest completed substantive
worker trace; set the optional absolute `trace_path` only when the user means
an older worker episode. Do not substitute a generic worker delegation call: `skill_draft`
provides the validated proposal-only persistence boundary. It never activates
the card; human review remains mandatory. Do not call it speculatively without
the user's request.
"""


# =============================================================================
# Map curation staleness threshold (hours)
# =============================================================================
MAP_CURATION_STALE_HOURS = 12  # Trigger passive curation if map not updated in this many hours


# =============================================================================
# Tool Call Fallback Parser
# =============================================================================


def extract_xml_tool_calls(
    text: str,
    offered_tool_names: set[str] | list[str] | tuple[str, ...] | None = None,
    tool_registry: Any | None = None,
) -> list:
    """Parse fallback tool calls embedded in LLM response text.

    Backward-compatible wrapper for the original XML-only fallback.  The shared
    parser also handles DeepSeek DSML, markdown ``Calling:`` pseudo-calls, and
    ``<mesh_call>`` blocks, validating synthesized calls before execution.
    """
    from .tool_call_salvage import extract_salvaged_tool_calls

    return extract_salvaged_tool_calls(
        text,
        offered_tool_names=offered_tool_names,
        tool_registry=tool_registry,
    )


def strip_xml_tool_calls(text: str) -> str:
    """Remove fallback tool-call markup from response text."""
    from .tool_call_salvage import strip_salvaged_tool_calls

    return strip_salvaged_tool_calls(text)


# =============================================================================
# Router Instructions Templates
# =============================================================================

ROUTER_INSTRUCTIONS_IDLE = """\
You are {nickname} ({agent_type}). Classify this message and output a single JSON object.

STATE: IDLE — no worker is running. If the user asks to stop, cancel, or check on
a running worker, set needs_worker=false and tell them nothing is running.

IMPORTANT: Your ENTIRE response must be a single JSON object. No reasoning, no explanation, no markdown fences. Just the JSON.

─── STEP 1: SHOULD YOU RESPOND? ───

Check the to= header FIRST — this is the most important check:
• to="agent:{agent_type}:{nickname}" → ALWAYS needs_response=true. This is a DM to you. Full stop.
• to="agent:...:OTHER_NAME" or to="user:..." → needs_response=false (not for you)
• to="channel:..." → needs_response=true ONLY if message text contains @{nickname} or @{agent_type} (case-insensitive)

CRITICAL RULES:
• A DM to you (to="agent:{agent_type}:{nickname}") = ALWAYS respond. No exceptions. Do NOT evaluate the content.
• "{nickname}" without the @ prefix is NOT an @mention. "what {nickname} said" → false.
• @{nickname} or @{agent_type} with the @ prefix IS a mention, case-insensitive.

─── STEP 2: WORKER NEEDED? ───

needs_worker=false for:
• Greetings: "hello", "hey", "good morning", "how are you"
• Thanks: "thanks", "got it", "that fixed it"
• Farewells: "good night", "bye"
• A lone "?" (just means "you there?")
• **Ambiguous requests where you'd ask a clarifying question before acting.**
  If your response would be "Should I...?", "Want me to...?", "Do you mean X or Y?" —
  that IS the response. Set needs_worker=false and ask the question.
  The user's next message will confirm, and THAT message triggers the worker.

needs_worker=true ONLY when you are confident the user wants action taken NOW.
Do not dispatch a worker and ask for permission in the same breath — pick one.

─── STEP 3: TASK COMPLEXITY (only when needs_worker=true) ───

Classify based on the INHERENT COMPLEXITY of the task, not on what context is available.

"simple" — the task itself is straightforward and well-defined:
  • Operational: check status, restart, tail logs, disk space, start a known service
  • Data lookup: search email, read a note, check calendar, look up a value
  • Single-step actions: send an email, create a note, delete a file, write a known script
  • Quick recall: "what was X?", "which branch?", "what port?"

  Examples: "restart alice", "check nginx status", "what's our CUDA version?",
  "search email for Anthropic", "create a note about X", "write a launch script"

"complex" — the task requires significant investigation, judgment, or multi-step work:
  • Debugging/diagnosis: finding root causes, tracing errors through systems
  • Implementation: writing or modifying code, building features
  • Multi-file changes: refactoring, coordinated config + code + test changes
  • Planning/design: architecture decisions, spec writing, code review
  • Research: understanding how something works, comparing approaches
  • Setup/infrastructure: installing, configuring, deploying new systems

  Examples: "fix the auth bug", "implement dark mode", "why did the router crash?",
  "set up Roundcube", "review the PR", "plan the migration"

─── STEP 4: RESPONSE TEXT ───

For needs_worker=false (social):
  Be warm, personable, use their name. Show personality — you're a friendly {agent_type}, not a robot. 1-2 sentences.

For needs_worker=true (brief ack while worker starts):
  This is an acknowledgment, NOT a full response. A worker with full context and tools
  will handle the real work. You can write a sentence or short paragraph — just keep it
  clearly an ack, not a full answer. Do NOT write a lengthy essay or detailed analysis.

  BANNED: "On it." / "Let me check." / "Looking into that." / "I'll take a look."

  Your ack should:
  1. Name the specific task
  2. Signal that work is starting
  3. Optionally add brief context (a sentence or two is fine)

  Good examples:
  • "Pulling up note #42 for you."
  • "Searching your inbox for Anthropic emails — one sec."
  • "Checking the nginx logs now."
  • "Tracing the message flow through the router logs now."
  • "Kicking off the vLLM setup. I'll check the tunnel config and port forwarding."

  Bad examples (these are full worker-length responses, not acks):
  • A multi-paragraph analysis with findings, tables, or recommendations
  • Answering the question in detail before the worker has a chance to investigate
  • Writing 200+ words of explanation — that's the worker's job

─── OUTPUT FORMAT ───

Output EXACTLY one JSON object. Nothing else before or after it.

{{"needs_response": false}}
{{"needs_response": true, "needs_worker": false, "response": "Hey Project Owner, doing well! What's on your mind?"}}
{{"needs_response": true, "needs_worker": true, "response": "Checking nginx status now."}}
{{"needs_response": true, "needs_worker": true, "response": "Digging into the 502 errors now."}}
"""

ROUTER_INSTRUCTIONS_BUSY = """
MODE: BUSY — A worker ({worker_id}) is processing a request.

Original request: "{pending_task_summary}"
Elapsed time: {elapsed:.0f}s

A new message just arrived. The conversation history shows the full thread —
respond to the MOST RECENT message (the last one before these instructions).
Earlier messages are context, not requests to address.

Your history includes worker activity entries showing what the worker is doing.

Give a DETAILED status update. Specifically:
- What tools/commands the worker has used so far
- What files or resources it is working with
- Any interim results or progress indicators
- What stage of the task it appears to be in (starting, mid-way, wrapping up)

Ground your description in the actual worker activity entries visible in
your history — do not guess or fabricate progress. If the worker has just
started and there is little activity, say so.

Also acknowledge the new message and let them know you'll handle it
when the current task completes.

Respond with plain text (no JSON).
"""

_WORKER_DISPATCH_TOOL_INSTRUCTIONS = """\
Dispatch by calling the Mesh worker_launch tool:
<mesh_call name="worker_launch"><task>...</task><task_type>...</task_type><reason>...</reason></mesh_call>
task_type is REQUIRED. A dispatch with no task_type is refused and no worker
runs. Add <backend>...</backend> only for a verbatim user backend override,
and only alongside a task type that has no configured Plan-Execute-Verify
workflow; naming a backend on a PEV type is refused.
Do not use Codex/Claude's own internal task delegation, team-worker launcher,
shell commands, or project-editing surface to do router dispatch work.

The worker does NOT see this conversation. It receives your task text, its
standing digest, and nothing else. Anything it needs — file paths, prior
findings, constraints, what "it" refers to — has to be written into task, in
full, as if to someone who just walked in.

Call worker_launch BEFORE any send_message announcement. The system renders
and delivers the launch receipt (including any `type=... (reason: ...)` line);
never write that receipt yourself. Do not claim a worker was launched unless
worker_launch returned status="dispatched". Do not issue more than one
worker_launch in the same router turn.

Worker backend control:
{worker_backend_instructions}
"""

_WORKER_DISPATCH_BLOCK_INSTRUCTIONS = """\
Dispatch by including this block at the END of your response, after any
conversational text:

<dispatch_worker>
task: Complete, self-contained description of what the worker should do
task_type: REQUIRED configured task type; a dispatch without one is refused
backend: HARD OVERRIDE ONLY when the user explicitly named one
reason: Required one-line reason for the task_type
</dispatch_worker>

Use this block directly. Do not use Codex/Claude's internal delegation,
team-worker launcher, shell commands, or project-editing surface for router
dispatch work.

The worker does NOT see this conversation. It receives your task text, its
standing digest, and nothing else. Anything it needs — file paths, prior
findings, constraints, what "it" refers to — has to be written into task, in
full, as if to someone who just walked in.

You MAY include conversational text before the dispatch block. For example:
  "Okay, let me check on that now.
   <dispatch_worker>
   task: Read /var/log/nginx/error.log and report every 502 in the last hour,
     grouped by upstream, with the request path and timestamp for each.
   task_type: simple-code
   reason: Bounded single-file log inspection with a fixed output shape.
   </dispatch_worker>"

The system renders any `type=... (reason: ...)` receipt after admission.
Never write that receipt yourself.

Worker backend control:
{worker_backend_instructions}
"""

_HARNESS_DISPATCH_RECENCY_REMINDER = """\
FINAL ROUTER-DISPATCH RULE — follow this instead of any launcher behavior
described in prior history: if the current request needs worker execution, do
not call or narrate Codex/Claude collaboration, subagent, team-worker, or shell
tools. The system renders the launch receipt; never write a
`type=... (reason: ...)` receipt yourself. End your response with exactly one
Mesh dispatch block:

<dispatch_worker>
task: Complete, self-contained description of what the worker should do —
  the worker does not see this conversation, so write it in full
task_type: REQUIRED configured task type; a dispatch without one is refused
backend: HARD OVERRIDE ONLY when the user explicitly named one
reason: Required one-line reason for the task_type
</dispatch_worker>
"""


ROUTER_INSTRUCTIONS_FULL = """\
You are {nickname} ({agent_type}).

STATE: IDLE — no worker is running. If the user asks to stop, cancel, or check on
a running worker, tell them nothing is running. Do not hallucinate a cancellation.

The conversation history above shows the full thread. Your job is to respond to
the MOST RECENT message — the last one before these instructions. Earlier messages
are context for continuity, not requests to address.

You respond to messages, handle discussion,
answer questions, and dispatch workers when a task requires
extended autonomous work.

You respond by writing plain text. Your text output is automatically delivered
to the user or channel. Use tools (file_read, worker_launch, memory_search,
etc.) when you need to take action or look something up. When you have
nothing more to do, simply produce your final answer as text — no tool call
is needed to deliver it. Your turn ends when you produce a response with no
tool calls.

Use send_message only when you need to route a message to a DIFFERENT
destination than the default reply target (e.g., sending to another agent or
a different channel). For normal replies, just write your answer.

For worker_launch, call the launch tool first. The system renders the
acknowledgment; never write the `type=... (reason: ...)` receipt yourself.

Use the tool syntax shown in the tools block. If tools are presented as
<mesh_call name="..."> XML, emit exact <mesh_call> blocks. If tools are
presented as native function calls, use the native interface.

Never emit obsolete raw tool syntaxes like <bash_exec>, <file_read>, <invoke>,
or <thinking> tags. For XML-backed tools, only <mesh_call name="..."> is
executable.

If a message does NOT require your response (it's addressed to someone else,
or it's a channel message that doesn't mention you), call the sleep tool with a
brief reason. If sleep is not available in the current tool set, output ONLY:

<no_response/>

─── HANDLING MESSAGES ───

Guidelines:
• Discussion, questions, opinions, decisions → respond with your thoughts
• Factual questions about code, config, or state → look it up, don't guess
• Recalls from memory → verify against source when a tool call can confirm it
• Status checks, greetings, thanks → respond naturally
• Clarifying questions → ask them directly before dispatching work

Your tools are read-only — use them freely to ground your answers. Check files,
search notes, read logs, verify configs before stating facts. Dispatch a worker
if the task requires WRITE operations (file edits, shell commands, restarts)
or sustained multi-step execution.

─── DISPATCHING A WORKER ───

Dispatch a worker when the task requires:
  - File modifications or shell commands
  - Multi-step autonomous execution (build, deploy, debug cycles)
  - Work that produces artifacts (scripts, configs, commits)

CRITICAL: Never dispatch a worker while asking the user for confirmation.
If your response asks "Should I...?", "Want me to...?", or "Do you want..." —
do NOT include a <dispatch_worker> block. Ask the question, wait for their
answer, and dispatch on the NEXT message when you have a clear go-ahead.
Dispatching while asking makes the question dishonest.

{worker_dispatch_instructions}

IMPORTANT — Write rich task descriptions. The worker starts with NO conversation
context except what you put in the task field. Include:
  - What the user wants and why (the full intent, not just the action)
  - Relevant prior decisions, constraints, and justifications from the conversation
  - Specific file paths, error messages, or details the worker will need
  - Any context about what was already tried or ruled out
A thin "fix the bug" dispatch produces thin results. A detailed dispatch with
context, constraints, and rationale lets the worker plan and execute effectively.

─── GUIDELINES ───

• You have a personality section — try to follow it.
• When you have relevant memories, reference them naturally.
• If memory contains conflicting information about a topic, surface it and ask.
• If you ({nickname}) have already addressed an ambiguity or clarification in the
  conversation history, the user has seen it — there's no need to restate it
  unless they ask again.
• If the user asks to review, refresh, update, or check the project map, dispatch
  a worker with the task: "Call the map_review tool." Do NOT describe the manual
  process — map_review handles filesystem reconciliation automatically.
"""

ROUTER_INSTRUCTIONS_FULL_HARNESS = """\
You are {nickname} ({agent_type}).

STATE: IDLE — no worker is running. If the user asks to stop, cancel, or check on
a running worker, tell them nothing is running. Do not hallucinate a cancellation.

The conversation history above shows the full thread. Your job is to respond to
the MOST RECENT message — the last one before these instructions. Earlier messages
are context for continuity, not requests to address.

You respond to messages, handle discussion,
answer questions, and dispatch workers when a task requires
extended autonomous work.

Write your response as natural text. It will be sent directly to the user.
Use the tool syntax shown in the tools block. If tools are presented as
<mesh_call name="..."> XML, emit exact <mesh_call> blocks. If tools are
presented as native function calls, use the native interface.

Never emit obsolete raw tool syntaxes like <bash_exec>, <file_read>, <invoke>,
or <thinking> tags. For XML-backed tools, only <mesh_call name="..."> is
executable.

If a message does NOT require your response (it's addressed to someone else,
or it's a channel message that doesn't mention you), output ONLY:

<no_response/>

─── HANDLING MESSAGES ───

Guidelines:
• Discussion, questions, opinions, decisions → respond with your thoughts
• Factual questions about code, config, or state → look it up, don't guess
• Recalls from memory → verify against source when a tool call can confirm it
• Status checks, greetings, thanks → respond naturally
• Clarifying questions → ask them directly before dispatching work

Your tools are read-only — use them freely to ground your answers. Check files,
search notes, read logs, verify configs before stating facts. Dispatch a worker
if the task requires WRITE operations (file edits, shell commands, restarts)
or sustained multi-step execution.

─── DISPATCHING A WORKER ───

Dispatch a worker when the task requires:
  - File modifications or shell commands
  - Multi-step autonomous execution (build, deploy, debug cycles)
  - Work that produces artifacts (scripts, configs, commits)

CRITICAL: Never dispatch a worker while asking the user for confirmation.
If your response asks "Should I...?", "Want me to...?", or "Do you want..." —
do NOT include a <dispatch_worker> block. Ask the question, wait for their
answer, and dispatch on the NEXT message when you have a clear go-ahead.
Dispatching while asking makes the question dishonest.

{worker_dispatch_instructions}

IMPORTANT — Write rich task descriptions. The worker starts with NO conversation
context except what you put in the task field. Include:
  - What the user wants and why (the full intent, not just the action)
  - Relevant prior decisions, constraints, and justifications from the conversation
  - Specific file paths, error messages, or details the worker will need
  - Any context about what was already tried or ruled out
A thin "fix the bug" dispatch produces thin results. A detailed dispatch with
context, constraints, and rationale lets the worker plan and execute effectively.

─── GUIDELINES ───

• You have a personality section — try to follow it.
• When you have relevant memories, reference them naturally.
• If memory contains conflicting information about a topic, surface it and ask.
• If you ({nickname}) have already addressed an ambiguity or clarification in the
  conversation history, the user has seen it — there's no need to restate it
  unless they ask again.
• If the user asks to review, refresh, update, or check the project map, dispatch
  a worker with the task: "Call the map_review tool." Do NOT describe the manual
  process — map_review handles filesystem reconciliation automatically.
"""

ROUTER_INSTRUCTIONS_BUSY_FULL = """\
You are {nickname} ({agent_type}).

One or more ordinary worker slots are active. The primary compatibility view is:
  Worker: {worker_id}
  Task: "{pending_task_summary}"
  Elapsed: {elapsed:.0f}s

The fresh <worker_slots> block is authoritative. It lists every fixed slot,
including empty slots, and is regenerated before each LLM call. Use worker_list
for a compact view, worker_status with a worker ID for detail, and worker_cancel
only with an exact worker ID or cancel_all=true.

The conversation history shows the full thread — respond to the MOST RECENT
message (the last one before these instructions). Earlier messages are context,
not requests to address.

Your history includes worker activity entries showing what the worker is doing
in real time.

─── RESPONDING ───

Your default behavior is to RESPOND. Write your answer as plain text — it is
automatically delivered. Use send_message only to route to a different
destination. Your turn ends when you produce a response with no tool calls.

If a message does NOT require your response (addressed to someone else,
channel message without @mention), call the sleep tool with a brief reason. If
sleep is not available, output ONLY:

<no_response/>

Rules:
• to="agent:{agent_type}:{nickname}" → ALWAYS respond
• to="agent:...:OTHER_NAME" or to="user:..." → <no_response/>
• to="channel:..." → respond ONLY if @{nickname} or @{agent_type} appears

─── HANDLING MESSAGES WHILE BUSY ───

You can still handle read-only tasks and answer questions while the worker runs:
• Status queries → report what the worker is doing based on the activity entries
  in your history. Ground your description in actual activity — don't guess.
• Lookups → use your read-only tools (file reads, email search, notes, etc.)
• Discussion / opinions → respond conversationally from memory and context
• Scheduling, emails, notes → use these tools normally

Do not cancel or replace a running worker from this prompt. Explicit stop/cancel
requests are handled by a deterministic pre-router path. If the user gives a
new task while capacity remains, worker_launch may start an additional worker.
If capacity is full, say that no new worker was launched and name the running
worker(s). Don't cancel a worker just because a new question comes in — handle
the question yourself and let the worker continue.

─── GUIDELINES ───

• When asked about progress, give specifics from the worker activity entries.
• If the new message is about the running task (e.g., "actually, use Sonnet instead"),
  note it in your response. The worker will not see this mid-execution.
"""

ROUTER_INSTRUCTIONS_BUSY_FULL_HARNESS = """\
You are {nickname} ({agent_type}).

One or more ordinary worker slots are active. The primary compatibility view is:
  Worker: {worker_id}
  Task: "{pending_task_summary}"
  Elapsed: {elapsed:.0f}s

The fresh <worker_slots> block is authoritative. It lists every fixed slot,
including empty slots, and is regenerated before each LLM call. Use worker_list
for a compact view, worker_status with a worker ID for detail, and worker_cancel
only with an exact worker ID or cancel_all=true.

The conversation history shows the full thread — respond to the MOST RECENT
message (the last one before these instructions). Earlier messages are context,
not requests to address.

Your history includes worker activity entries showing what the worker is doing
in real time.

─── RESPONDING ───

Your default behavior is to RESPOND. Write your response as natural text.

If a message does NOT require your response (addressed to someone else,
channel message without @mention), output ONLY:

<no_response/>

Rules:
• to="agent:{agent_type}:{nickname}" → ALWAYS respond
• to="agent:...:OTHER_NAME" or to="user:..." → <no_response/>
• to="channel:..." → respond ONLY if @{nickname} or @{agent_type} appears

─── HANDLING MESSAGES WHILE BUSY ───

You can still handle read-only tasks and answer questions while the worker runs:
• Status queries → report what the worker is doing based on the activity entries
  in your history. Ground your description in actual activity — don't guess.
• Lookups → use your read-only tools (file reads, email search, notes, etc.)
• Discussion / opinions → respond conversationally from memory and context
• Scheduling, emails, notes → use these tools normally

Do not cancel or replace a running worker from this prompt. Explicit stop/cancel
requests are handled by a deterministic pre-router path. If the user gives a
new task while capacity remains, worker_launch may start an additional worker.
If capacity is full, say that no new worker was launched and name the running
worker(s). Don't cancel a worker just because a new question comes in — handle
the question yourself and let the worker continue.

─── GUIDELINES ───

• When asked about progress, give specifics from the worker activity entries.
• If the new message is about the running task (e.g., "actually, use Sonnet instead"),
  note it in your response. The worker will not see this mid-execution.
"""

ROUTER_INSTRUCTIONS_BUSY_CC = """\
You are {nickname} ({agent_type}).

A Claude Code session is currently running a task on your behalf:
  Task: "{cc_task}"
  Session: {cc_session}
  Elapsed: {elapsed:.0f}s

A background monitor watches this session and will deliver its results to you
automatically when it finishes — you do NOT need to poll it.

The conversation history shows the full thread — respond to the MOST RECENT
message (the last one before these instructions). Your history includes
[CC Tool Activity] entries showing what the session is doing in real time.

─── RESPONDING ───

Your default behavior is to RESPOND. Write your answer as plain text — it is
automatically delivered. Use send_message only to route to a different
destination. Your turn ends when you produce a response with no tool calls.

If a message does NOT require your response (addressed to someone else,
channel message without @mention), call the sleep tool with a brief reason. If
sleep is not available, output ONLY:

<no_response/>

Rules:
• to="agent:{agent_type}:{nickname}" → ALWAYS respond
• to="agent:...:OTHER_NAME" or to="user:..." → <no_response/>
• to="channel:..." → respond ONLY if @{nickname} or @{agent_type} appears

─── HANDLING MESSAGES WHILE THE SESSION RUNS ───

• Status queries → describe what the session is doing, grounded in the
  [CC Tool Activity] entries in your history. Don't guess.
• Questions, lookups, discussion → answer normally from memory and context.
• A Claude Code session is ALREADY running this work. Do NOT launch a worker
  and do NOT emit a <dispatch_worker> block — there is no worker to cancel,
  and a worker would run in parallel with the live session.

─── STOPPING THE SESSION ───

If the user explicitly asks to stop, cancel, or abort the running session,
say so plainly — the system will stop the session for you. Do not claim the
work is finished when it is still running.
"""

ROUTER_INSTRUCTIONS_BUSY_CC_HARNESS = """\
You are {nickname} ({agent_type}).

A Claude Code session is currently running a task on your behalf:
  Task: "{cc_task}"
  Session: {cc_session}
  Elapsed: {elapsed:.0f}s

A background monitor watches this session and will deliver its results to you
automatically when it finishes — you do NOT need to poll it.

The conversation history shows the full thread — respond to the MOST RECENT
message (the last one before these instructions). Your history includes
[CC Tool Activity] entries showing what the session is doing in real time.

─── RESPONDING ───

Your default behavior is to RESPOND. Write your response as natural text.

If a message does NOT require your response (addressed to someone else,
channel message without @mention), output ONLY:

<no_response/>

Rules:
• to="agent:{agent_type}:{nickname}" → ALWAYS respond
• to="agent:...:OTHER_NAME" or to="user:..." → <no_response/>
• to="channel:..." → respond ONLY if @{nickname} or @{agent_type} appears

─── HANDLING MESSAGES WHILE THE SESSION RUNS ───

• Status queries → describe what the session is doing, grounded in the
  [CC Tool Activity] entries in your history. Don't guess.
• Questions, lookups, discussion → answer normally from memory and context.
• A Claude Code session is ALREADY running this work. Do NOT launch a worker
  and do NOT emit a <dispatch_worker> block — there is no worker to cancel,
  and a worker would run in parallel with the live session.

─── STOPPING THE SESSION ───

If the user explicitly asks to stop, cancel, or abort the running session,
say so plainly — the system will stop the session for you. Do not claim the
work is finished when it is still running.
"""

SYNTHESIZE_INSTRUCTIONS = """\
You have just finished working on a task. Below is your execution log
followed by conversation context for reference.

Your job is to summarize the EXECUTION LOG into a response to the user.
The CONVERSATION CONTEXT is background only — do NOT summarize it.

Write a response that:

1. Provides a thorough account of what was accomplished in the execution log
2. Includes relevant details (file paths, command outputs, key findings, metrics)
3. Is well-structured (headings, bullets, code blocks as appropriate)
4. Does not repeat information already present in the conversation context
5. Omits internal reasoning, false starts, or abandoned approaches

IMPORTANT: any send_message calls to the user in the execution log were
captured, NOT delivered. Your response is the ONLY message the user will
receive. If the log contains a message composed for the user, reproduce
its content in full — never describe it as already sent.

If you encountered an error, report it clearly with what was attempted.
Write naturally as yourself — this is your response to the user.

═══ EXECUTION LOG (summarize this) ═══
{worker_trace}

═══ CONVERSATION CONTEXT (for reference only — do NOT summarize) ═══
{context_block}
"""


ROUTER_INSTRUCTIONS_WATCHDOG = """\
You are {nickname} ({agent_type}).

Your worker ({worker_id}) has been running for {elapsed:.0f}s on the following task:
  "{pending_task_summary}"

Your conversation history includes worker activity entries showing what
the worker has done so far.

Review the worker's activity and assess whether it has encountered anything
unusual that would warrant notifying the user. This could be positive
(unexpectedly good results, an interesting finding) or negative (appears
stuck, looping, drifted significantly from the original task, error
accumulation).

Respond with plain text only — do not call tools or send messages.
If everything is progressing within expected parameters, simply respond:
Nothing to report.

Otherwise, describe what you've observed concisely — the user will see
your response directly.
"""


# =============================================================================
# v2 Memory Classification Additions (appended conditionally)
# =============================================================================

_V2_CLASSIFIER_ADDITIONS = """

─── DISPATCH CRITERIA ───

Use needs_worker=true when the task requires:
- Writing or editing code, config files, or documents
- Running shell commands, tests, or deployments
- Multi-step investigation that needs tool access
- Any work that would take more than a quick answer

If you find yourself describing what *should be done* rather than doing
it — that's a dispatch, not a direct response. Don't explain the task;
hand it to a worker.
"""

_V2_FULL_ADDITIONS = """

─── DISPATCH CRITERIA ───

Dispatch when the task requires:
- Writing or editing code, config files, or documents
- Running shell commands, tests, or deployments
- Multi-step investigation that needs tool access
- Any work that would take more than a quick answer

If you find yourself describing what *should be done* rather than doing
it — that's a dispatch, not a direct response.
"""



# Per-async-task trigger context. _call_router_full sets this so concurrent
# router invocations (e.g. the CC monitor delivering results while a BUSY
# handler answers a new channel message) don't clobber each other's reply
# destination through shared instance attributes (Bug 9). contextvars are
# isolated per asyncio Task, so each in-flight call sees its own value.
_CC_TRIGGER_CTX: contextvars.ContextVar = contextvars.ContextVar(
    "cc_trigger_ctx", default=None
)


@dataclass
class RouterCallState:
    """Everything one ``_call_router_full`` invocation mutates about itself.

    These fields used to live on the ``RouterV2`` instance and were reset
    unconditionally at the top of every call, which is exactly why curation
    turns had to be serialised behind ``_router_turn_lock``: a curation turn
    starting mid-message-turn would wipe the message turn's tool ledger,
    worker-launch guards and delivery flag.  Holding that lock across a 1-4
    minute LLM call starved message processing.

    Carrying the state on a contextvar instead makes each call's view private
    to its own asyncio task, so curation and message turns can run at the same
    time without either observing the other's bookkeeping.
    """

    router_identity: int
    sent_message: bool = False
    tools: list[tuple[str, str]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    failure_class: str = ""
    tool_visibility_pending: bool = True
    last_worker_launch: dict[str, Any] | None = None
    worker_launches: list[str] = field(default_factory=list)
    worker_task_keys: set[str] = field(default_factory=set)
    curation_rejections: list[dict[str, Any]] = field(default_factory=list)
    trigger_from_node: str | None = None
    trigger_to_node: str | None = None
    # Trusted autonomous-session scope of the message that opened this call.
    # The native worker_launch tool builds a synthetic trigger, so without
    # carrying the scope here it would be dropped between the wake and the
    # dispatch seam.  Router-minted at wake delivery; never model-supplied.
    autonomous_scope: dict[str, Any] = field(default_factory=dict)
    router_cc_events: list[Any] = field(default_factory=list)


# Per-async-task per-call router state.  A completed state stays bound long
# enough for its caller to inspect the delivery/tool summary; ``None`` means no
# state has been installed (or it was explicitly cleared), so readers use the
# durable per-instance fallback.
_CTX_ROUTER_CALL_STATE: contextvars.ContextVar[RouterCallState | None] = (
    contextvars.ContextVar("mesh_router_call_state", default=None)
)


class _CallStateField:
    """Proxy a legacy per-call instance attribute onto the task-local state.

    Every historic reader and writer keeps using ``self._last_router_call_tools``
    and friends; the descriptor routes each access to the current task's
    :class:`RouterCallState`.  Migrating ~145 call sites by hand would risk
    missing one — and a missed site is a silent cross-turn data leak — so the
    indirection is centralised here instead.
    """

    def __init__(self, field_name: str, doc: str = ""):
        self._field = field_name
        self.__doc__ = doc or f"Task-local RouterCallState.{field_name}."

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj._get_call_state(), self._field)

    def __set__(self, obj, value):
        setattr(obj._get_call_state(), self._field, value)


class RouterState(Enum):
    """Router state machine states."""
    IDLE = "idle"
    BUSY = "busy"
    PLANNING = "planning"  # Used only by RouterV3
    AUTO = "auto"


class WorkerLifecycle(Enum):
    """Authoritative lifecycle for one stable execution slot."""

    EMPTY = "empty"
    STARTING = "starting"
    RUNNING = "running"
    CANCELLING = "cancelling"
    REPORTING = "reporting"
    FAILED = "failed"


@dataclass
class WorkerResult:
    """Result from worker processing."""
    response: str           # Final response text (sent directly to user)
    context: list[Any]      # Full worker context (list of HistoryEntry from node._history)
    error: Exception | None = None  # If worker failed
    usage: dict | None = None  # Cumulative token usage from worker's LLM calls
    # Worker synthesis: full in-flight history for trace (uncapped, ephemeral)
    worker_in_flight_history: list[Any] | None = None
    # Worker synthesis: buffered send_message calls (not yet delivered)
    buffered_messages: list[tuple[str, str]] | None = None
    # Worker synthesis: cumulative CC tool events across all iterations
    worker_cc_events: list[Any] | None = None
    # True when the worker terminated via send_report (contract §4)
    report_sent: bool = False


@dataclass
class WorkerSlot:
    """One stable RouterV2 worker slot.

    ``index`` never changes for the lifetime of the router.  All other fields
    are per-run and are reset in-place when the tombstone retention window
    expires or a later dispatch reuses the slot.
    """

    index: int = 0
    lifecycle: WorkerLifecycle = WorkerLifecycle.EMPTY
    worker_id: str | None = None
    origin_message_id: str = ""
    router_turn_id: str = ""
    dispatch_key: str = ""
    task_fingerprint: str = ""
    task_description: str = ""
    selection_metadata: dict[str, Any] = field(default_factory=dict)
    trigger: Message | None = None
    task: asyncio.Task | None = None
    started_event: asyncio.Event | None = None
    start_time: float = 0.0
    snapshot: list[Turn] = field(default_factory=list)
    snapshot_start: int = 0
    backend: str | None = None
    pev: PevTaskConfig | None = None
    kind: str = "worker"
    fixed_tool_name: str | None = None
    skill_card_ids: tuple[str, ...] = ()
    skill_selected_at: str | None = None
    execution_context: Any | None = None
    watchdog_task: asyncio.Task | None = None
    flush_task: asyncio.Task | None = None
    flush_snapshot_cursor: int = 0
    flush_tools_since_last: int = 0
    flush_tools_already_flushed: int = 0
    report_accepted: bool = False
    cleanup_complete: bool = False
    latest_activity: str = ""
    failure: str | None = None
    completed_at: float | None = None
    # Set when cancel-flush already appended this worker's partial trace, so
    # its own completion can skip _complete_via_trace without silencing any
    # other slot's trace.
    trace_appended_on_cancel: bool = False

    @property
    def active(self) -> bool:
        return self.lifecycle in {
            WorkerLifecycle.STARTING,
            WorkerLifecycle.RUNNING,
            WorkerLifecycle.CANCELLING,
            WorkerLifecycle.REPORTING,
        }


class DispatchBriefTier(str, Enum):
    """Ordered provenance tiers for a worker's authoritative brief."""

    METADATA = "metadata"
    ROUTER_STATE = "router_state"
    TRIGGER_CONTENT = "trigger_content"


@dataclass(frozen=True)
class ResolvedDispatchBrief:
    """One resolved brief plus the provenance that made it trustworthy."""

    text: str
    tier: DispatchBriefTier


def resolve_dispatch_brief(
    trigger: Message,
    router_task_description: str = "",
) -> ResolvedDispatchBrief:
    """Resolve a worker brief without hiding which fallback supplied it."""
    metadata = getattr(trigger, "metadata", None)
    if isinstance(metadata, dict):
        brief = str(metadata.get("worker_task_description") or "").strip()
        if brief:
            return ResolvedDispatchBrief(brief, DispatchBriefTier.METADATA)
    router_brief = str(router_task_description or "").strip()
    if router_brief:
        return ResolvedDispatchBrief(
            router_brief,
            DispatchBriefTier.ROUTER_STATE,
        )
    return ResolvedDispatchBrief(
        str(getattr(trigger, "content", "") or "").strip(),
        DispatchBriefTier.TRIGGER_CONTENT,
    )


@dataclass(frozen=True)
class DispatchReceipt:
    """Immutable, door-neutral worker dispatch outcome and audit record."""

    dispatch_key: str
    status: str
    worker_id: str | None
    slot_index: int | None
    origin_message_id: str
    router_turn_id: str
    task_description: str
    backend: str | None = None
    message: str = ""
    source: str = ""
    brief_tier: str = ""
    task_type: str = ""
    reason: str = ""
    acknowledgment: str = ""
    acknowledgment_source: str = "system"
    request_record: str = ""
    # Autonomous-session scope resolved at the admission seam (§8.4/§17).
    # Empty/False on every ordinary interactive dispatch.
    autonomous_session: bool = False
    project_key: str = ""
    session_id: str = ""


@dataclass(frozen=True)
class WorkerSelection:
    """One resolved per-launch worker selection and its audit metadata."""

    backend: str = ""
    task_type: str = ""
    reason: str = ""
    user_override: bool = False
    pev: PevTaskConfig | None = None
    prompts: TaskPromptConfig | None = None
    warning: str | None = None
    # A refusal is not a fallback.  ``warning`` means "the request was
    # unusable, so the configured default ran instead"; ``refusal`` means no
    # worker starts at all, because running *anything* would silently change
    # the shape of the requested work.
    refusal: str | None = None


@dataclass
class RouterV2Config:
    """Configuration for RouterV2."""
    # Max context messages to keep (legacy simple truncation — used as fallback
    # when ConversationHistory is not enabled)
    max_context_messages: int = 100

    # Status query patterns. RouterV2 accepts only a short, complete query
    # matching one of these patterns; substantive messages go to the router.
    status_patterns: list[str] = field(default_factory=lambda: [
        "status", "what's happening", "you there", "working on",
        "still there", "hello?", "hey?", "update?"
    ])

    # Cancel request patterns (exact phrase matching for worker cancellation)
    cancel_patterns: list[str] = field(default_factory=lambda: [
        "stop the worker", "cancel the worker",
    ])

    # LLM integration
    llm_enabled: bool = True  # Use LLM for classification/responses

    # Worker context peek settings
    worker_peek_max_lines: int = 20  # Max lines per tool output in peek

    # Memory v2 retrieval
    memory_retrieve_max_rounds: int = 2      # max retrieval round-trips per classification
    memory_retrieve_budget_tokens: int = 6000  # token budget for each retrieval

    # Worker synthesis settings
    synthesize_enabled: bool = True           # Enable synthesis step on worker completion
    worker_digest_max_tokens: int = 15_000    # Token cap for worker digest (persistent)
    synthesis_max_tokens: int = 150_000       # Total token cap for synthesis prompt
    # Deliver worker messages buffered for the dispatch origin verbatim
    # (concatenated into ONE message) instead of synthesizing a description
    # of them; synthesis still covers the empty-buffer case. Default off.
    deliver_buffered_verbatim: bool = False
    synthesis_trace_max_lines: int = 200      # Per-result line cap in worker trace for synthesis
    synthesis_context_turns: int = 40         # Recent conversation turns injected into synthesis
    # Direct RouterV2 constructions (especially tests) opt in explicitly.
    # AgentNode enables this from NodeConfig in production.
    worker_trace_persist: bool = False

    # Trace-as-history (docs/plans/trace-as-history-2026-04-27.md)
    # When enabled, worker trace turns are appended to history and synthesis is skipped.
    trace_as_history_enabled: bool = False    # OFF by default; canary on hypatia first
    tool_result_max_lines: int = 80           # Per-tool-result line cap when persisted
    tool_result_max_chars: int = 6400         # Per-tool-result char cap (single-line payloads)

    # History settings (ConversationHistory-based summarization + persistence)
    history_window_tokens: int | None = None  # rolling window budget (W); default: soft_limit // 2
    history_soft_limit_tokens: int = 70_000   # backward compat; trigger = 2W (derived from this if window_tokens not set)
    history_hard_limit_tokens: int = 105_000  # hard cap (raised for summary growth headroom)
    history_target_ratio: float = 0.25        # deprecated — kept for backward compat only
    history_persist: bool = True              # persist router history to disk
    history_persist_path: str | None = None   # custom path (auto-derived if None)
    history_summarization_enabled: bool = False  # off = rolling window only

    # Router mode: "full" (conversational agent), "classifier" (legacy thin
    # classifier), or "pipeline" (typed router pipeline).
    # Default "classifier" for backward compatibility; opt-in to "full" per-agent via config
    router_mode: str = "classifier"
    pipeline_backend: str = "deepseek"
    pipeline_plan_path: str = ""

    # Max tool-loop iterations for the full router (safety cap)
    router_max_iters: int = 30

    # Periodic map curation: runs every N minutes if ≥ min_turns new turns (0 = disabled)
    map_curation_interval_minutes: int = 120
    map_curation_min_turns: int = 10

    # Worker watchdog: periodic check-in while BUSY (0 = disabled)
    watchdog_interval_minutes: int = 0

    # Worker context: smaller window for workers
    worker_context_window_tokens: int = 25_000  # Token budget for worker context snapshot
    max_concurrent_workers: int = 1             # 1 preserves legacy single-worker behavior
    min_worker_brief_chars: int = 120            # backstop after provenance validation
    worker_backend_override_enabled: bool = False
    # Deprecated compatibility field; stale YAML is tolerated but ignored.
    worker_backends_allowed: list[str] = field(default_factory=list)
    worker_task_types: dict[str, dict[str, str]] = field(default_factory=dict)

    # Memory retrieval redesign (C3): TOC-based injection
    memory_retrieval_redesign_enabled: bool = False
    memory_toc_size: int = 30

    # Rev-10 standing-digest read pathway: digest replaces the memory TOC
    # in prompt composition when enabled (alongside-deploy, default off).
    standing_digest_enabled: bool = False
    standing_digest_path: str = ""

    # Entity/group/digest self-curation.  AgentNode copies these from
    # NodeConfig so the queued RouterV2 turn uses the enrolled agent's actual
    # limits and Phase 2 gate rather than getattr() fallbacks.
    entity_self_curation_mode: str = "off"
    entity_self_curation_groups_enabled: bool = False
    standing_digest_budget_tokens: int = 32000
    essay_token_budget: int = 4000
    entity_activation_window_threshold: int = 3
    entity_registry_injection_cap: int = 1000
    curation_stale_group_batches: int = 50
    curation_failure_alert_threshold: int = 5
    # Phase 3 backfill bounds (§9, "Phase 3").
    entity_self_curation_backfill_on_startup: bool = True
    entity_self_curation_backfill_max_batches: int = 50
    entity_self_curation_backfill_slice_size: int = 10
    # Essay generation folded into the curation turn (default OFF).
    entity_self_curation_essays_enabled: bool = False
    entity_self_curation_essays_max_per_turn: int = 1

    # Autonomous agent mode (docs/plans/autonomous-agent-mode.md §8, §17).
    # AgentNode copies these from NodeConfig.  The hard admission guard in
    # _dispatch_worker() is a no-op unless autonomous_agent_mode_enabled is
    # True, so an unenrolled agent keeps byte-identical dispatch behavior.
    autonomous_agent_mode_enabled: bool = False
    autonomous_projects: list[str] = field(default_factory=list)
    autonomous_max_workers_per_session: int = 2
    # The controller mandate text (mesh/prompts/autonomous_controller.txt),
    # loaded by AgentNode at enrollment.  Held here rather than concatenated
    # into system_prompt so it can be injected per-turn (plan §10.1).
    autonomous_mandate_prompt: str = ""
    # Execute-only mandate for report-as-trigger continuations.  Empty falls
    # back to autonomous_mandate_prompt so existing deployments stay intact.
    autonomous_continuation_mandate_prompt: str = ""
    # The initial autonomous PLAN wake may select the deep router.  Worker
    # report continuations always remain on the light router.
    autonomous_plan_backend: str = "light"

    def __post_init__(self):
        if self.router_mode not in ("classifier", "full", "pipeline"):
            raise ValueError(
                f"Invalid router_mode: {self.router_mode!r}, "
                "must be 'classifier', 'full', or 'pipeline'"
            )
        requested_plan_backend = str(
            self.autonomous_plan_backend or ""
        ).strip().lower()
        self.autonomous_plan_backend = (
            requested_plan_backend
            if requested_plan_backend in {"light", "deep"}
            else "light"
        )


# The autonomous controller tags every dispatch brief it owns with the
# project it is spending against (plan §8.7 step 3).  The tag is the scope
# carrier; the key inside it is validated against project_dossier's own
# grammar before any path or ledger is derived from it.
_AUTONOMOUS_PROJECT_TAG_RE = re.compile(
    r"\[PROJECT:\s*(project:[^\]\s]+)\s*\]",
    re.IGNORECASE,
)

# A SESSION PLAN is controller-authored state, not ordinary conversational
# history.  Keep only the compact schema required by the mandate so a model
# cannot turn the continuation carrier into an unbounded prompt side channel.
_SESSION_PLAN_START_RE = re.compile(r"^\s*SESSION PLAN\s*$", re.IGNORECASE)
_SESSION_PLAN_FIELD_RE = re.compile(
    r"^\s*(GOAL|TASKS|EVIDENCE|FIRST)\s*=\s*\S.*$", re.IGNORECASE
)
_SESSION_PLAN_REQUIRED_FIELDS = frozenset({"GOAL", "TASKS", "EVIDENCE", "FIRST"})
_SESSION_PLAN_MAX_LINES = 10
_SESSION_PLAN_MAX_CHARS = 4_000


def _tail_file(path: str, max_lines: int = 50) -> list[str]:
    """Read the last N lines of a file. Returns empty list on any error."""
    try:
        from collections import deque
        with open(path, "r", errors="replace") as f:
            lines = deque(f, maxlen=max_lines)
        return [line.rstrip("\n") for line in lines]
    except Exception:
        return []


class RouterV2:
    """
    Thin classifier + direct worker passthrough router.

    Key responsibilities:
    - Classifies messages via LLM (respond? dispatch to worker?)
    - Dispatches work to a worker coroutine
    - Handles status queries while worker is busy (with live context peek)
    - Merges worker context back with origin attribution
    - Passes worker's response directly to user (no re-summarization)

    Usage:
        router = RouterV2(
            worker_fn=my_worker_function,
            send_fn=my_send_function,
            llm_client=my_llm_client,
        )

        # On each incoming message:
        await router.on_message(msg)
    """

    # ── Per-router-call state ────────────────────────────────────────────
    # These names are unchanged from when they were plain instance attributes;
    # each now reads and writes the current asyncio task's RouterCallState so
    # that a curation turn and a message turn running concurrently keep
    # separate bookkeeping.  See RouterCallState for why.
    _last_router_call_sent_message = _CallStateField(
        "sent_message",
        "Whether send_message was called during this router call.",
    )
    _last_router_call_tools = _CallStateField(
        "tools",
        "(tool_name, brief_args) executed during this router call.",
    )
    _last_router_call_usage = _CallStateField("usage")
    _last_router_failure_class = _CallStateField("failure_class")
    _tool_visibility_pending = _CallStateField("tool_visibility_pending")
    _last_worker_launch = _CallStateField("last_worker_launch")
    _router_call_worker_launches = _CallStateField("worker_launches")
    _router_call_worker_task_keys = _CallStateField("worker_task_keys")
    _last_curation_rejections = _CallStateField("curation_rejections")
    # Captured from _call_router_full so worker tools can build accurate
    # synthetic triggers (M1 fix — avoids self-referential from_node).
    _current_trigger_from_node = _CallStateField("trigger_from_node")
    _current_trigger_to_node = _CallStateField("trigger_to_node")
    _current_autonomous_scope = _CallStateField("autonomous_scope")

    def _get_call_state(self) -> RouterCallState:
        """This task's active/recent call state, or a durable fallback.

        The completed state remains bound until the next call because several
        callers inspect its delivery/tool summary immediately after
        ``_call_router_full`` returns.  Tasks that have never run a call use a
        single per-instance fallback, preserving construction-time and
        out-of-band legacy attribute access.
        """
        state = _CTX_ROUTER_CALL_STATE.get()
        # The ContextVar is module-global, so the same task can briefly touch
        # two RouterV2 instances during replacement/tests.  A state belongs
        # only to the router that created it; otherwise the second router's
        # descriptor writes would corrupt the first router's live call.
        if state is not None and state.router_identity == id(self):
            return state
        # __dict__ directly: the descriptors live on the class, so ordinary
        # attribute access here would be fine, but this keeps the fallback
        # lookup free of any descriptor interaction.
        fallback = self.__dict__.get("_fallback_call_state")
        if fallback is None:
            fallback = RouterCallState(
                router_identity=id(self),
                tool_visibility_pending=False,
            )
            self.__dict__["_fallback_call_state"] = fallback
        return fallback

    def _init_call_state(self, msg: Message | None = None) -> RouterCallState:
        """Install a fresh per-call state on the current task.

        Replaces the block of unconditional instance-attribute resets that used
        to open every ``_call_router_full``.  The contextvar assignment is
        visible to everything this task awaits, including the tool loop, but
        invisible to any other task.
        """
        self._clear_call_state()
        state = RouterCallState(
            router_identity=id(self),
            tool_visibility_pending=True,
            trigger_from_node=msg.from_node if msg is not None else None,
            trigger_to_node=msg.to_node if msg is not None else None,
            autonomous_scope=self.autonomous_completion_metadata(msg),
        )
        _CTX_ROUTER_CALL_STATE.set(state)
        return state

    def _clear_call_state(self) -> None:
        """Drop this router's prior call state before reusing the task.

        Callers intentionally inspect the completed call state immediately
        after ``_call_router_full`` returns, so cleanup cannot happen at that
        boundary.  The next call on the same task clears the completed state
        before installing its fresh replacement.
        """
        state = _CTX_ROUTER_CALL_STATE.get()
        if state is not None and state.router_identity == id(self):
            _CTX_ROUTER_CALL_STATE.set(None)

    def __init__(
        self,
        worker_fn: Callable[[list[Any], Message], Awaitable[WorkerResult]],
        send_fn: Callable[[str, Message | None], Awaitable[None]],
        config: RouterV2Config | None = None,
        status_push_fn: Callable[[], Awaitable[None]] | None = None,
        node_id: str = "",
        nickname: str = "",
        agent_type: str = "",
        llm_client: "LLMClient | None" = None,
        deep_llm_client: "LLMClient | None" = None,
        deep_backend_name: str = "",
        router_deep_enabled: bool = False,
        system_prompt: str = "",
        identity_block: str = "",
        tools_block: str = "",
        worker_context_fn: Callable[[], list[Any]] | None = None,
        cc_events_fn: Callable[[], list[Any]] | None = None,
        memory_system: Any | None = None,
        session_gap_secs: int = 900,
        flush_interval_tools: int = 0,
        worker_llm_client: "LLMClient | None" = None,
        router_process_fn: "Callable[..., Awaitable[str]] | None" = None,
        cc_interactive_tools: bool = False,
        cc_binary: str = "",
        cc_effort: str = "",
        cc_model: str = "",
        cc_fallback_homes: list[str] | None = None,
        harness_session_tools: bool = False,
        harness_session_llm_config: Any | None = None,
        worker_backend_names: set[str] | list[str] | tuple[str, ...] | None = None,
        # Per-backend open/closed classification, keyed by backend name.  A
        # name absent from this mapping is UNCLASSIFIED and counts as closed.
        worker_backend_access: dict[str, str] | None = None,
        # False makes this agent structurally unable to dispatch a closed
        # model.  True (the default) is today's behaviour: the gate no-ops.
        worker_closed_models: bool = True,
        worker_task_types: dict[str, Any] | None = None,
        # Deprecated compatibility parameter. Stale callers may still pass it,
        # but backend reachability is defined by worker_task_types instead.
        worker_backends_allowed: set[str] | list[str] | tuple[str, ...] | None = None,
        default_worker_backend: str | None = None,
        fixed_tool_configs: dict[str, "FixedToolConfig"] | None = None,
        todo_store_path: str | Path | None = None,
        entity_resolution_enabled: bool = False,
        entity_resolution_mode: str = "off",
        # Scoped agent state. ``None`` keeps every root on the legacy
        # mesh.paths globals, which is the unisolated behaviour.
        state_paths: Any | None = None,
        # Normalized per-agent isolation policy (Phase 2A).  ``None`` and a
        # disabled policy both mean "offer and execute exactly today's tools".
        isolation_policy: Any | None = None,
    ):
        self._worker_fn = worker_fn
        # Explicit reference to the agent that owns _worker_buffered_messages
        # and _worker_response_text. Read by _flush_worker_buffer_on_cancel.
        # In production, worker_fn is a bound method (self._router_v2_worker)
        # so __self__ is the AgentNode. In tests that pass a bare function,
        # this is None and the cancel-flush helper no-ops gracefully.
        self._worker_agent = getattr(worker_fn, "__self__", None)
        self._worker_backend_names = {
            name for name in (worker_backend_names or []) if name
        }
        self._worker_backend_access = {
            str(name): str(access or "")
            for name, access in (worker_backend_access or {}).items()
            if name
        }
        self._worker_closed_models = bool(worker_closed_models)
        self._worker_task_types = normalize_worker_task_types(worker_task_types)
        # Retained only for introspection/backward compatibility. It has no
        # authority over dispatch resolution.
        self._worker_backends_allowed = {
            name for name in (worker_backends_allowed or []) if name
        }
        self._default_worker_backend = default_worker_backend or ""
        self._worker_backend_override: str | None = None
        self._fixed_tools = dict(fixed_tool_configs or {})
        self._send_fn = send_fn
        self._status_push_fn = status_push_fn
        self._config = config or RouterV2Config()
        self._node_id = node_id
        self._nickname = nickname or "agent"
        self._agent_type = agent_type or "assistant"
        self._state_paths = state_paths
        self._isolation_policy = isolation_policy
        self._skill_store = SkillStore(
            self._nickname,
            root=(state_paths.skills_dir if state_paths is not None else None),
        )

        # Validate configured bindings at router construction, before the first
        # dispatch. Invalid entries remain unavailable and fall back safely.
        self._configured_worker_task_types()

        # LLM integration
        self._llm_client = llm_client
        self._deep_llm_client = deep_llm_client
        self._deep_backend_name = deep_backend_name
        self._router_deep_enabled = router_deep_enabled
        self._worker_llm_client = worker_llm_client  # kept for backward compat; synthesis uses router LLM
        self._system_prompt = system_prompt
        self._identity_block = identity_block
        self._tools_block = tools_block

        # Memory system (optional)
        self._memory = memory_system
        self._session_gap_secs = session_gap_secs
        self._v2_drop_in_progress = False  # Guard against concurrent v2 window drops
        self._v2_drop_task: asyncio.Task | None = None  # prevent GC of fire-and-forget task
        self._v2_curate_in_progress = False  # Guard against concurrent passive curations
        self._v2_curate_task: asyncio.Task | None = None
        self._v2_curation_timer_task: asyncio.Task | None = None  # Periodic curation timer
        self._v2_turns_at_last_curation: int = 0  # Window turn count at last curation
        self._flush_interval_tools = flush_interval_tools

        # Latest user message for memory Relevant slice query
        self._latest_user_message: str | None = None

        # v2: static relevant memory (top-5 cosine similarity per message)
        self._relevant_context: str = ""

        # Legacy worker context peek (kept for backward compat but unused by snapshot architecture)
        self._worker_context_fn = worker_context_fn

        # CC live events callback — returns synthetic entries for in-progress CC tool calls
        self._cc_events_fn = cc_events_fn

        # Harness session events callback — returns formatted strings from _event_tail
        self._harness_events_fn: Callable[[], list[str]] | None = None

        # State — serialized via _state_lock to prevent races
        self._state = RouterState.IDLE
        # The autonomous controller may dispatch ordinary workers, whose
        # lifecycle otherwise overwrites _state with BUSY/IDLE mid-session.
        self._autonomous_controller_active = False
        # Direct router LLM turns do not own a worker slot. Track them
        # separately so heartbeat/status consumers never mistake a live turn
        # for IDLE. Autonomous wake metadata promotes the same turn to AUTO.
        self._llm_turn_active = False
        self._autonomous_session_active = False
        self._state_lock = asyncio.Lock()
        # Router decisions are serialized independently from state mutation.
        # Workers never acquire this lock and therefore continue concurrently.
        self._router_turn_lock = asyncio.Lock()
        # ── Entity/group/digest self-curation (§4.3) ──
        # Unbounded lossless FIFO: one item per successful formation batch, one
        # internal router turn per item.  Batches are never coalesced or
        # dropped.  The single drain task serializes curation batches with each
        # other but deliberately does not take _router_turn_lock, so message
        # turns remain responsive during the long curation LLM calls.
        self._curation_queue: "asyncio.Queue[Any]" = asyncio.Queue()
        self._curation_drain_task: asyncio.Task | None = None
        self._curation_idle = asyncio.Event()
        self._curation_idle.set()
        self._curation_batches_seen = 0
        self._curation_turn_sequence = 0
        self._consecutive_curation_failures = 0
        self._last_curation_at: str = ""
        self._last_failed_curation_memory_ids: tuple[str, ...] = ()
        self._curation_recovery_ids: list[str] = []
        # IDs accepted by the FIFO but not yet finished.  Phase 3 planning
        # treats these as covered so a startup/manual trigger cannot enqueue a
        # second curation turn for work that is already queued or in flight.
        # Counts (rather than a set) keep direct duplicate enqueue calls safe.
        self._curation_scheduled_memory_counts: dict[str, int] = {}
        # pending group -> (largest observed bridge-window count, turn sequence
        # when that count last increased).  Staleness is "no new evidence for
        # N processed batches", not "has zero evidence forever".
        self._curation_group_bridge_state: dict[str, tuple[int, int]] = {}
        self._curation_turn_hook: Callable[[Any], Any] | None = None
        # Phase 3 backfill counters (§9, "Phase 3").
        self._curation_backfill_runs = 0
        self._curation_backfill_slices_queued = 0
        # Admission is a separate short critical section.  No preparation,
        # memory retrieval, client construction, or worker execution occurs
        # while this lock is held.
        self._dispatch_lock = asyncio.Lock()

        # ConversationHistory: durable conversation entries with summarization
        persist_path = None
        if self._config.history_persist:
            if self._config.history_persist_path:
                persist_path = Path(self._config.history_persist_path)
            else:
                # Default: ~/.mesh/history/router-{nickname}.json, or the
                # scoped history_dir when an isolation policy supplied one.
                safe_nick = (nickname or "router").replace(":", "-")
                if state_paths is not None:
                    persist_path = state_paths.router_history_file(safe_nick)
                else:
                    from .paths import HISTORY_DIR

                    persist_path = HISTORY_DIR / f"router-{safe_nick}.json"

        self._history = ConversationHistory(
            soft_token_limit=self._config.history_soft_limit_tokens,
            hard_token_limit=self._config.history_hard_limit_tokens,
            target_ratio=self._config.history_target_ratio,
            window_budget=self._config.history_window_tokens,
            summarization_prompt=ROUTER_SUMMARY_PROMPT,
            summarization_enabled=self._config.history_summarization_enabled,
            persist_path=persist_path,
        )

        # Ephemeral peeks: planning activity snapshots (used by RouterV3 planning peeks only)
        self._ephemeral_peeks: list[dict] = []

        # Worker tracking.  _slot_table is authoritative; the dict/list below
        # are derived compatibility views for diagnostics and older tests.
        self._worker_task: asyncio.Task | None = None
        self._pending_trigger: Message | None = None
        self._worker_start_time: float | None = None
        self._worker_id_counter = 0
        self._slot_table: list[WorkerSlot] = [
            WorkerSlot(index=index)
            for index in range(1, self._configured_worker_capacity() + 1)
        ]
        self._worker_slots: dict[str, WorkerSlot] = {}
        self._worker_slot_order: list[str] = []
        self._slot_revision = 0
        self._dispatch_receipts: dict[str, DispatchReceipt] = {}
        self._last_dispatch_receipt: DispatchReceipt | None = None
        self._router_turn_counter = 0
        self._current_origin_message_id = ""
        self._current_router_turn_id = ""
        self._current_launch_ordinal = 0
        self._report_wake_queue: asyncio.Queue[Message] = asyncio.Queue()
        self._report_wake_task: asyncio.Task | None = None
        # One-shot autonomous-controller leaf dispatches use the normal worker
        # lifecycle but capture completion locally instead of delivering it to
        # the user before the controller's mandatory session report.
        self._controller_worker_waiters: dict[str, asyncio.Future] = {}

        # Worker snapshot: mutable list[Turn] the worker appends to during execution.
        # Router holds a reference to see live worker progress.
        self._worker_snapshot: list[Turn] | None = None
        self._worker_snapshot_start: int = 0  # len(snapshot) at dispatch — entries after this are worker's

        # Current worker ID (e.g., "bob-worker1")
        self._current_worker_id: str | None = None
        self._current_worker_backend: str | None = None
        self._current_worker_kind: str = "worker"
        self._current_fixed_tool_name: str | None = None

        # Trace-as-history (C3): set by _flush_worker_buffer_on_cancel when it
        # appends partial trace, so _complete_via_trace can skip duplicates if
        # completion races into delivery after cancel.
        #
        # The authoritative record is per-worker: WorkerSlot.trace_appended_on_cancel
        # while the slot lives, and this worker-id ledger afterwards, because
        # cancel resets the slot before the cancelled worker's completion task
        # reaches delivery.  This bool survives only as a compatibility view for
        # the legacy single-worker path; concurrent completions must never read
        # it, or one worker's cancel would silence another worker's trace.
        self._trace_appended_on_cancel: bool = False
        # Insertion-ordered, bounded set of worker ids whose partial trace was
        # already appended by cancel-flush.
        self._trace_appended_workers: dict[str, None] = {}

        # Router-level memory retrieval: IDs injected into the current worker dispatch
        self._injected_memory_ids: set[str] = set()
        # Rendered XML block of the most recent selection.  Diagnostics and
        # router-prompt dedup only — worker prompt assembly reads the block off
        # its own WorkerExecutionContext, never from here.
        self._injected_memory_context: str = ""

        # Task description from full router dispatch (H3 fix)
        self._current_task_description: str = ""

        # Per-call state (send_message flag, tool ledger, usage, worker-launch
        # guards, trigger nodes) now lives on the task-local RouterCallState
        # reached through the _CallStateField descriptors declared on the
        # class.  Nothing to initialise here: _get_call_state() materialises a
        # durable fallback on first access, and _call_router_full() installs a
        # fresh state per call.
        self._last_curation_tokens_in: int = 0
        self._last_curation_tokens_out: int = 0
        self._last_router_deep_override_event: dict[str, Any] = {}
        # Rejections from the most recently *completed* curation turn.  The
        # live per-call list is task-local, so status queries — which run on a
        # different task and after the turn ends — read this durable copy.
        self._last_completed_curation_rejections: list[dict[str, Any]] = []
        #: Turn-level write-attempt roll-up handed over by the agent node when
        #: a curation turn ends (G-004).  Per-attempt detail lives in the
        #: ``curation_write_attempt`` event trail; this is the summary.
        self._last_curation_write_summary: dict[str, Any] = {}

        # Full router tool loop callback (set by _init_router_v2 in agent_node.py)
        self._router_process_fn = router_process_fn
        self._router_tool_names = sorted(ROUTER_TOOL_NAMES)
        resolved_entity_mode = normalize_entity_resolution_mode(
            entity_resolution_mode,
            legacy_enabled=entity_resolution_enabled,
        )
        if resolved_entity_mode != "write":
            self._router_tool_names = [
                name
                for name in self._router_tool_names
                if name != "entity_link_correct"
            ]
        self._router_tool_names = [
            name for name in self._router_tool_names
            if name not in FIXED_TOOL_ROUTER_TOOLS or name in self._fixed_tools
        ]
        # Phase 2A offer-time filter for the static router list.  A disabled
        # (or absent) policy returns the list unchanged.
        self._router_tool_names = self._filter_router_tools(self._router_tool_names)
        self._pipeline_router = None
        self._todo_store_path = Path(todo_store_path).expanduser() if todo_store_path else None
        self._todo_store: MessageStore | None = None

        # CC interactive session tools (gated by config) — must be set before
        # _init_worker_tool_handlers() which checks it.
        self._cc_interactive_enabled = cc_interactive_tools
        # Native harness session tools (gated by config) — same "interactive
        # session is the only execution route" contract as cc_interactive.
        self._harness_session_enabled = harness_session_tools

        # Per-instance worker tool handlers (worker_launch, worker_status).
        # These tools are router-only — they read/write RouterV2 instance state
        # (snapshot, worker ID, _start_worker machinery) and cannot be global.
        # Dispatched by agent_node.py's _execute_all_tools BEFORE the registry.
        self._worker_tool_handlers: dict[str, Callable[..., Awaitable[str]]] = {}
        self._init_worker_tool_handlers()
        if cc_interactive_tools or harness_session_tools:
            self._router_tool_names = [
                n for n in self._router_tool_names
                if n not in (
                    WORKER_ROUTER_TOOLS
                    | MANAGED_WORKER_ROUTER_TOOLS
                    | FIXED_TOOL_ROUTER_TOOLS
                )
            ]
        # Interactive Claude Code session subsystem. All CC session state and
        # lifecycle logic lives in CCSessionManager (mesh/cc_session_manager.py);
        # the manager holds a back-reference to this router as ``self.r``. The
        # manager is always constructed (so external accessors like the lazy
        # reaper can read self._cc_mgr._cc_tmux_session unconditionally); only
        # tool registration and tool-name exposure are gated on the config flag.
        # Imported here (deferred) rather than at module top to avoid a circular
        # import — cc_session_manager imports RouterState from this module.
        from .cc_session_manager import CCSessionManager
        self._cc_mgr = CCSessionManager(
            self,
            cc_binary=cc_binary,
            cc_effort=cc_effort,
            cc_model=cc_model,
            cc_fallback_homes=cc_fallback_homes,
        )
        if cc_interactive_tools:
            self._router_tool_names = self._filter_router_tools(
                sorted(set(self._router_tool_names) | CC_INTERACTIVE_TOOLS)
            )
            self._init_cc_interactive_handlers()

        # Native interactive harness session subsystem. Same pattern as the CC
        # manager: always constructed (so external accessors are safe), but tool
        # registration and tool-name exposure are gated on the config flag. The
        # manager holds a back-reference to this router as ``self.r``.
        from .harness_session_manager import HarnessSessionManager
        self._harness_session_mgr = HarnessSessionManager(
            self, session_llm_config=harness_session_llm_config,
        )
        self._harness_events_fn = self._harness_session_mgr.get_recent_event_strings
        if harness_session_tools:
            self._router_tool_names = self._filter_router_tools(
                sorted(set(self._router_tool_names) | HARNESS_SESSION_INTERACTIVE_TOOLS)
            )
            self._init_harness_session_handlers()

        # Session-level stats accumulation for memory reflection.
        # Stats are accumulated across completions within a session.
        # A new session starts when the gap between completions exceeds
        # the configured threshold. When a gap fires, the *previous*
        # session's accumulated stats are evaluated for reflection.
        self._session_stats: EpisodeStats | None = None
        self._session_last_completion_time: float = 0.0  # monotonic
        self._session_trigger_text: str = ""  # first trigger of the session
        self._session_last_result: WorkerResult | None = None  # last result for reflection
        self._session_last_worker_id: str | None = None

        # Intra-worker periodic flush: monitors the live worker snapshot
        # and fires reflections every _flush_interval_tools tool calls.
        self._flush_monitor_task: asyncio.Task | None = None
        self._flush_snapshot_cursor: int = 0  # how far into snapshot we've counted
        self._flush_tools_since_last: int = 0  # tool calls counted since last flush
        self._flush_tools_already_flushed: int = 0  # total tools flushed mid-worker (to subtract at completion)

        # Worker watchdog: periodic check-in on worker progress
        self._watchdog_task: asyncio.Task | None = None

        # Token budget tracking — cached after each _build_router_prompt() call
        self._last_prompt_tokens: int = 0
        # Static portion: system_prompt + identity + tools (doesn't change per-call)
        self._static_prompt_tokens: int = (
            estimate_tokens(system_prompt) +
            estimate_tokens(identity_block) +
            estimate_tokens(tools_block)
        )

    def _capture_autonomous_session_plan_metadata(
        self, content: str,
    ) -> dict[str, str]:
        """Validate a SESSION PLAN and bind it to this trusted router call."""
        plan = self._extract_autonomous_session_plan(content)
        scope = self._current_autonomous_scope
        if not plan or not isinstance(scope, dict):
            return {}
        session_id = str(scope.get("autonomous_session_id") or "").strip()
        if not session_id:
            return {}

        # worker_launch builds its synthetic trigger from this scope before the
        # assistant/tool turns are appended.  Updating it here keeps the plan
        # attached across that production ordering seam.
        scope["autonomous_session_plan"] = plan
        return {
            "autonomous_session_id": session_id,
            "autonomous_session_plan": plan,
        }

    def _append_turn(self, turn: Turn) -> None:
        """Append a turn to the router history."""
        # The wake mandate requires the controller to write its SESSION PLAN
        # before the first worker admission.  A later continuation cannot
        # safely recover that plan from rolling history: it may have been
        # summarized or dropped while the worker was running.  Stamp a bounded
        # copy onto this outgoing turn while its RouterCallState still names
        # the trusted session; _dispatch_worker() then carries it forward on
        # the router-minted worker trigger.
        if turn.role == "outgoing":
            captured = self._capture_autonomous_session_plan_metadata(
                turn.content
            )
            if captured:
                metadata = dict(turn.meta) if isinstance(turn.meta, dict) else {}
                metadata.update(captured)
                turn.meta = metadata
        self._history.append(turn)
        # Memory Formation v3: notify agent node so the token-pressure trigger
        # can fire (no-op if v3 disabled).
        agent = getattr(self, "_worker_agent", None)
        hook = getattr(agent, "_v3_on_turn_appended", None) if agent else None
        if hook is not None:
            try:
                hook(turn)
            except Exception as e:
                logger.warning("v3 token-pressure hook raised: %s", e)

    @property
    def state(self) -> RouterState:
        """Derive observable state from autonomous and worker execution."""
        active_slots = self._active_worker_slots()
        if (
            self._autonomous_controller_active
            or self._autonomous_session_active
            or any(
                isinstance(slot.selection_metadata, dict)
                and slot.selection_metadata.get("autonomous_session")
                for slot in active_slots
            )
        ):
            return RouterState.AUTO
        if active_slots or self._llm_turn_active:
            return RouterState.BUSY
        return self._state

    @property
    def is_busy(self) -> bool:
        """True if worker is currently processing."""
        return (
            self._active_worker_count() > 0
            or self._llm_turn_active
            or self._state == RouterState.BUSY
        )

    async def _status_push(self) -> None:
        """Publish a state transition without making router work depend on it."""
        if self._status_push_fn is None:
            return
        try:
            await self._status_push_fn()
        except Exception as exc:
            logger.debug("Router status push failed: %s", exc)

    def _configured_worker_capacity(self) -> int:
        """Return the configured worker capacity, clamped to at least one."""
        try:
            config = getattr(self, "_config", None)
            return max(1, int(getattr(config, "max_concurrent_workers", 1) or 1))
        except (TypeError, ValueError):
            return 1

    def _ensure_slot_table(self) -> list[WorkerSlot]:
        """Create or resize the fixed table without disturbing active slots."""
        capacity = self._configured_worker_capacity()
        table = getattr(self, "_slot_table", None)
        if table is None:
            table = []
            # Adopt pre-existing compatibility slots used by narrow tests.
            for index, slot in enumerate(
                getattr(self, "_worker_slots", {}).values(), start=1
            ):
                slot.index = index
                if (
                    slot.lifecycle == WorkerLifecycle.EMPTY
                    and slot.task is not None
                    and not slot.task.done()
                ):
                    slot.lifecycle = WorkerLifecycle.RUNNING
                table.append(slot)
        while len(table) < capacity:
            table.append(WorkerSlot(index=len(table) + 1))
        if len(table) > capacity:
            overflow = table[capacity:]
            if any(slot.active for slot in overflow):
                # Runtime capacity reductions take effect after overflow work
                # drains; never orphan a live worker.
                capacity = len(table)
            else:
                table = table[:capacity]
        self._slot_table = table
        return table

    def _bump_slot_revision(self) -> None:
        self._slot_revision = int(getattr(self, "_slot_revision", 0)) + 1

    def _sync_worker_compat_views(self) -> None:
        """Refresh legacy worker-id views from the authoritative table."""
        table = self._ensure_slot_table()
        self._worker_slots = {
            slot.worker_id: slot
            for slot in table
            if slot.worker_id and slot.active
        }
        self._worker_slot_order = [
            slot.worker_id
            for slot in sorted(table, key=lambda item: item.index)
            if slot.worker_id and slot.active
        ]

    def _slot_for_worker(self, worker_id: str | None) -> WorkerSlot | None:
        if not worker_id:
            return None
        for slot in self._ensure_slot_table():
            if slot.worker_id == worker_id:
                return slot
        return None

    def _trace_appended_ledger(self) -> dict[str, None]:
        """Worker ids whose partial trace cancel-flush already appended."""
        ledger = getattr(self, "_trace_appended_workers", None)
        if ledger is None:
            ledger = {}
            self._trace_appended_workers = ledger
        return ledger

    def _mark_trace_appended_on_cancel(
        self,
        slot: WorkerSlot | None,
        worker_id: str | None,
    ) -> None:
        """Record that cancel-flush already appended one worker's trace."""
        if slot is not None:
            slot.trace_appended_on_cancel = True
        if worker_id:
            ledger = self._trace_appended_ledger()
            ledger[worker_id] = None
            while len(ledger) > 64:
                ledger.pop(next(iter(ledger)))
        # Compatibility view for the legacy single-worker path only.
        self._trace_appended_on_cancel = True

    def _trace_already_appended_on_cancel(
        self,
        slot: WorkerSlot | None,
        worker_id: str | None,
    ) -> bool:
        """Answer the suppression question for exactly one worker.

        A live slot is authoritative.  Once cancel has reset the slot, the
        worker-id ledger answers for that same worker.  The router-global bool
        is consulted only when no per-worker record exists anywhere, which is
        the legacy single-worker path — never when some *other* worker's cancel
        set it.
        """
        if slot is not None:
            return bool(getattr(slot, "trace_appended_on_cancel", False))
        ledger = self._trace_appended_ledger()
        if worker_id and worker_id in ledger:
            return True
        if ledger:
            return False
        return bool(getattr(self, "_trace_appended_on_cancel", False))

    def _clear_trace_appended_on_cancel(
        self,
        slot: WorkerSlot | None,
        worker_id: str | None,
    ) -> None:
        """Clear only the targeted worker's suppression record."""
        if slot is not None:
            slot.trace_appended_on_cancel = False
        ledger = self._trace_appended_ledger()
        if worker_id:
            ledger.pop(worker_id, None)
        if not ledger:
            self._trace_appended_on_cancel = False

    @staticmethod
    def _bounded_slot_text(value: Any, limit: int = 240) -> str:
        text = " ".join(str(value or "").split())
        return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."

    def _slot_latest_activity(self, slot: WorkerSlot) -> str:
        if slot.latest_activity:
            return self._bounded_slot_text(slot.latest_activity, 240)
        if slot.snapshot and slot.snapshot_start < len(slot.snapshot):
            turn = slot.snapshot[-1]
            return self._bounded_slot_text(getattr(turn, "content", ""), 240)
        if slot.failure:
            return self._bounded_slot_text(slot.failure, 240)
        return ""

    def _worker_slots_xml(self) -> str:
        """Render fresh ephemeral status for one router LLM call."""
        table = self._ensure_slot_table()
        active = sum(1 for slot in table if slot.active)
        lines = [
            (
                f'<worker_slots revision="{int(getattr(self, "_slot_revision", 0))}" '
                f'capacity="{len(table)}" active="{active}">'
            )
        ]
        now = time.monotonic()
        for slot in table:
            attrs = [
                f'index="{slot.index}"',
                f'state="{slot.lifecycle.value}"',
            ]
            if slot.worker_id:
                attrs.append(f'worker_id="{html.escape(slot.worker_id)}"')
            if slot.origin_message_id:
                attrs.append(
                    f'origin_message_id="{html.escape(slot.origin_message_id)}"'
                )
            if slot.start_time:
                attrs.append(
                    f'elapsed_seconds="{max(0, int(now - slot.start_time))}"'
                )
            if slot.lifecycle == WorkerLifecycle.EMPTY:
                lines.append(f"  <slot {' '.join(attrs)}/>")
                continue
            lines.append(f"  <slot {' '.join(attrs)}>")
            if slot.task_description:
                lines.append(
                    f"    <task>{html.escape(self._bounded_slot_text(slot.task_description, 500))}</task>"
                )
            activity = self._slot_latest_activity(slot)
            if activity:
                lines.append(
                    f"    <latest_activity>{html.escape(activity)}</latest_activity>"
                )
            lines.append("  </slot>")
        receipt = getattr(self, "_last_dispatch_receipt", None)
        if receipt is not None:
            lines.append(
                "  <latest_dispatch "
                f'status="{html.escape(receipt.status)}" '
                f'dispatch_key="{html.escape(receipt.dispatch_key)}" '
                f'worker_id="{html.escape(receipt.worker_id or "")}" '
                f'slot_index="{receipt.slot_index or ""}"/>'
            )
        lines.append("</worker_slots>")
        return "\n".join(lines)

    def _slot_summary(self, slot: WorkerSlot) -> dict[str, Any]:
        elapsed = (
            round(max(0.0, time.monotonic() - slot.start_time), 1)
            if slot.start_time else 0.0
        )
        return {
            "slot_index": slot.index,
            "state": slot.lifecycle.value,
            "worker_id": slot.worker_id,
            "origin_message_id": slot.origin_message_id or None,
            "elapsed_seconds": elapsed,
            "task_description": self._bounded_slot_text(
                slot.task_description, 240
            ),
            "latest_activity": self._slot_latest_activity(slot),
            "worker_backend": slot.backend,
            "worker_kind": slot.kind if slot.worker_id else None,
        }

    def _reset_slot(self, slot: WorkerSlot) -> None:
        """Release one stable slot while preserving its index."""
        index = slot.index
        slot.__dict__.clear()
        slot.__dict__.update(WorkerSlot(index=index).__dict__)
        self._bump_slot_revision()
        self._sync_worker_compat_views()

    @staticmethod
    def _worker_task_key(task: str) -> str:
        """Normalize a task description for duplicate-dispatch detection."""
        return re.sub(r"\s+", " ", (task or "").strip().lower())

    def _active_worker_slots(self) -> list[WorkerSlot]:
        """Return active slots in stable index order."""
        slots = [
            slot for slot in self._ensure_slot_table()
            if slot.active
        ]
        self._sync_worker_compat_views()
        if (
            not slots
            and getattr(self, "_worker_task", None) is not None
            and not self._worker_task.done()
        ):
            trigger = getattr(self, "_pending_trigger", None) or Message(
                type=MessageType.MESSAGE,
                from_node="",
                to_node=getattr(self, "_node_id", ""),
                content=getattr(self, "_current_task_description", "") or "",
            )
            slots.append(WorkerSlot(
                index=1,
                lifecycle=WorkerLifecycle.RUNNING,
                worker_id=getattr(self, "_current_worker_id", None) or "worker",
                task_description=getattr(self, "_current_task_description", "") or "",
                trigger=trigger,
                task=self._worker_task,
                start_time=getattr(self, "_worker_start_time", None) or time.monotonic(),
                snapshot=getattr(self, "_worker_snapshot", None) or [],
                snapshot_start=getattr(self, "_worker_snapshot_start", 0) or 0,
                backend=getattr(self, "_current_worker_backend", None),
                kind=getattr(self, "_current_worker_kind", "worker"),
                fixed_tool_name=getattr(self, "_current_fixed_tool_name", None),
            ))
        return slots

    def _active_worker_count(self) -> int:
        """Return the number of currently live workers."""
        return len(self._active_worker_slots())

    def _set_current_worker_slot(self, slot: WorkerSlot | None) -> None:
        """Refresh legacy singleton fields to point at a selected worker slot."""
        if slot is None:
            self._worker_task = None
            self._pending_trigger = None
            self._worker_start_time = None
            self._current_worker_id = None
            self._current_worker_backend = None
            self._current_worker_kind = "worker"
            self._current_fixed_tool_name = None
            self._current_task_description = ""
            self._worker_snapshot = None
            self._worker_snapshot_start = 0
            return
        self._worker_task = slot.task
        self._pending_trigger = slot.trigger
        self._worker_start_time = slot.start_time
        self._current_worker_id = slot.worker_id
        self._current_worker_backend = slot.backend
        self._current_worker_kind = slot.kind
        self._current_fixed_tool_name = slot.fixed_tool_name
        self._current_task_description = slot.task_description
        self._worker_snapshot = slot.snapshot
        self._worker_snapshot_start = slot.snapshot_start

    def _select_primary_worker_slot(self) -> WorkerSlot | None:
        """Choose the worker slot used by legacy status/watchdog paths."""
        active = self._active_worker_slots()
        if not active:
            return None
        current = getattr(self, "_current_worker_id", None)
        if current:
            for slot in active:
                if slot.worker_id == current:
                    return slot
        return active[-1]

    def _refresh_primary_worker_slot(self) -> WorkerSlot | None:
        """Point legacy singleton fields at the current primary live worker."""
        slot = self._select_primary_worker_slot()
        self._set_current_worker_slot(slot)
        if slot is None:
            self._state = RouterState.IDLE
        else:
            self._state = RouterState.BUSY
        return slot

    @property
    def context(self) -> list[Any]:
        """Current conversation context (read-only view).

        Returns Turn objects from the ConversationHistory window.
        """
        return list(self._history.window)

    @property
    def history(self) -> ConversationHistory:
        """The router's ConversationHistory instance."""
        return self._history

    def _get_todo_store(self) -> MessageStore | None:
        """Return the configured message store for read-only todo context."""
        if not self._todo_store_path:
            return None
        if not self._todo_store_path.exists():
            return None
        if self._todo_store is None:
            try:
                self._todo_store = MessageStore(self._todo_store_path)
            except Exception as e:
                logger.debug("Todo context store unavailable: %s", e)
                return None
        return self._todo_store

    @staticmethod
    def _conversation_id_from_message(msg: Message | None) -> str | None:
        """Resolve the stable conversation id for a trigger message."""
        if msg is None:
            return None
        if not msg.from_node or not msg.to_node:
            return None
        return MessageStore.compute_conversation_id(msg.from_node, msg.to_node)

    def _render_todo_context(self, conversation_id: str | None) -> str:
        """Render compact per-conversation todos for router/worker context."""
        if not conversation_id:
            return ""
        store = self._get_todo_store()
        if store is None:
            return ""
        try:
            todos = store.list_todos(conversation_id, include_done=True, limit=200)
        except Exception as e:
            logger.debug("Todo context render failed: %s", e)
            return ""

        live = [t for t in todos if not t.get("deleted_at")]
        visible = [t for t in live if t.get("status") in {"open", "in_progress"}]
        completed = [t for t in live if t.get("status") == "done"]
        cancelled_count = sum(1 for t in live if t.get("status") == "cancelled")
        if not visible and not completed and not cancelled_count:
            return ""

        visible.sort(key=lambda t: (int(t.get("position", 0)), t.get("created_at", ""), t.get("id", "")))
        completed.sort(key=lambda t: (t.get("completed_at") or t.get("updated_at") or ""), reverse=True)

        open_count = sum(1 for t in visible if t.get("status") == "open")
        progress_count = sum(1 for t in visible if t.get("status") == "in_progress")
        done_count = len(completed)
        attrs = (
            f'conversation_id="{html.escape(conversation_id, quote=True)}" '
            f'open="{open_count}" in_progress="{progress_count}" done="{done_count}"'
        )
        if cancelled_count:
            attrs += f' cancelled="{cancelled_count}"'

        lines = [f"<conversation_todos {attrs}>"]
        item_count = 0
        truncated = False

        for todo in visible:
            if item_count >= 10:
                truncated = True
                break
            text = html.escape(str(todo.get("text", "")))
            status = html.escape(str(todo.get("status", "")))
            todo_id = html.escape(str(todo.get("id", "")))
            section = html.escape(str(todo.get("section") or ""))
            by = html.escape(str(todo.get("updated_by") or todo.get("created_by") or ""))
            section_attr = f", section={section}" if section else ""
            lines.append(f"[{item_count + 1}] {status}: {text} (id={todo_id}{section_attr}, by={by})")
            item_count += 1
            if sum(len(line) + 1 for line in lines) > 1200:
                truncated = True
                break

        remaining_visible = max(0, len(visible) - item_count)
        if truncated or remaining_visible:
            lines.append(
                f"... {remaining_visible} more open/in-progress — use todo_list for the full list"
            )

        recent_done = completed[:3]
        if recent_done:
            latest = html.escape(str(recent_done[0].get("text", "")))
            lines.append(f"recently done: {len(recent_done)} shown of {done_count} (latest: \"{latest}\")")
        elif done_count:
            lines.append(f"done: {done_count}")

        lines.append("</conversation_todos>")
        block = "\n".join(lines)
        if len(block) > 1400:
            block = block[:1360].rstrip() + "\n... truncated — use todo_list for the full list\n</conversation_todos>"
        return block

    def _render_conversation_notes(self, conversation_id: str | None) -> str:
        """Render compact per-conversation pinned notes for router context."""
        if not conversation_id:
            return ""
        store = self._get_todo_store()
        if store is None:
            return ""
        try:
            notes = store.get_conversation_notes(conversation_id)
        except Exception as e:
            logger.debug("Conversation notes context render failed: %s", e)
            return ""
        if not notes or not str(notes.get("content", "")).strip():
            return ""

        content = str(notes["content"]).strip()
        if len(content) > 500:
            content = content[:497].rstrip() + "..."
        return f"<conversation_notes>\n{html.escape(content)}\n</conversation_notes>"

    def get_diagnostics(self) -> dict:
        """Return structured diagnostic data for status reporting."""
        import time as _time
        active_slots = self._active_worker_slots()
        primary = self._select_primary_worker_slot()
        if primary is not None:
            self._set_current_worker_slot(primary)
        result = {
            "state": self.state.value,
            "worker_active": bool(active_slots),
            "worker_count": len(active_slots),
            "max_concurrent_workers": self._configured_worker_capacity(),
            "workers": [
                {
                    "worker_id": slot.worker_id,
                    "worker_backend": slot.backend,
                    "worker_kind": slot.kind,
                    "fixed_tool": slot.fixed_tool_name,
                    "elapsed_seconds": round(_time.monotonic() - slot.start_time, 1),
                    "task_description": slot.task_description,
                    "pending_trigger_from": slot.trigger.from_node if slot.trigger else None,
                    "pending_trigger_preview": (
                        str(slot.trigger.content)[:100] if slot.trigger else None
                    ),
                    "worker_snapshot_turns": len(slot.snapshot) if slot.snapshot else None,
                }
                for slot in active_slots
            ],
            "worker_id": self._current_worker_id,
            "worker_elapsed_seconds": (
                round(_time.monotonic() - self._worker_start_time, 1)
                if self._worker_start_time else None
            ),
            "pending_trigger_from": (
                self._pending_trigger.from_node if self._pending_trigger else None
            ),
            "pending_trigger_preview": (
                str(self._pending_trigger.content)[:100]
                if self._pending_trigger else None
            ),
            "worker_snapshot_turns": (
                len(self._worker_snapshot) if self._worker_snapshot else None
            ),
        }
        if self._session_stats:
            result["session_stats"] = {
                "tool_calls": self._session_stats.tool_calls,
                "user_turns": self._session_stats.num_user_visible_turns,
                "total_chars": self._session_stats.total_user_visible_chars,
                "agent_response_chars": self._session_stats.agent_response_chars,
                "has_errors": self._session_stats.has_errors,
            }
        result["curation"] = self.curation_status()
        return result

    def set_context(self, context: list[Any]) -> None:
        """Seed the router from legacy history (list of HistoryEntry).

        Converts HistoryEntry objects into Turn objects and appends to
        the ConversationHistory. Used when loading from persisted worker history.
        """
        from datetime import datetime, timezone

        for entry in context:
            if hasattr(entry, 'message'):
                msg = entry.message
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                ts = msg.timestamp
                if isinstance(ts, str) and ts:
                    try:
                        ts = datetime.fromisoformat(ts)
                    except ValueError:
                        ts = datetime.now(timezone.utc)
                elif not ts:
                    ts = datetime.now(timezone.utc)
                self._append_turn(Turn(
                    role=entry.direction,
                    content=content,
                    timestamp=ts,
                    from_node=msg.from_node or "",
                    to_node=msg.to_node,
                ))
            elif isinstance(entry, Message):
                content = entry.content if isinstance(entry.content, str) else str(entry.content)
                ts = entry.timestamp
                if isinstance(ts, str) and ts:
                    try:
                        ts = datetime.fromisoformat(ts)
                    except ValueError:
                        ts = datetime.now(timezone.utc)
                elif not ts:
                    ts = datetime.now(timezone.utc)
                self._append_turn(Turn(
                    role="incoming",
                    content=content,
                    timestamp=ts,
                    from_node=entry.from_node or "",
                    to_node=entry.to_node,
                ))

    def clear_context(self) -> None:
        """Clear the conversation context."""
        self._history._window.clear()
        self._history._summary = None
        self._history._next_seq_id = 1
        self._ephemeral_peeks.clear()

    def load_history(self) -> int:
        """Load persisted router history from disk. Returns count of entries loaded."""
        return self._history.load()

    def save_history(self) -> None:
        """Persist router history to disk."""
        self._history.save()

    def _append_to_history(self, msg: Message) -> None:
        """Append a message to the router history (no lock, caller must hold _state_lock)."""
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        from datetime import datetime, timezone
        ts = msg.timestamp
        if isinstance(ts, str) and ts:
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                ts = datetime.now(timezone.utc)
        elif not ts:
            ts = datetime.now(timezone.utc)
        self._append_turn(Turn(
            role="incoming",
            content=content,
            timestamp=ts,
            from_node=msg.from_node or "",
            to_node=msg.to_node,
        ))
        if msg.from_node and msg.from_node.startswith("user:"):
            self._latest_user_message = content

    async def add_to_history_only(self, msg: Message) -> None:
        """Add a message to history for passive context awareness, without triggering LLM classification."""
        async with self._state_lock:
            self._append_to_history(msg)
            self._check_and_trigger_summarization()
            try:
                self.save_history()
            except Exception as e:
                logger.warning(f"Failed to save history after passive add: {e}")
            logger.debug(f"RouterV2 add_to_history_only: from={msg.from_node}, to={msg.to_node}")

    async def on_message(self, msg: Message) -> None:
        """Serialize router decisions while preserving deterministic fast paths."""
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if self._is_cancel_request(content):
            await self._on_message_serialized(msg)
            return
        if self._is_user_status_query(msg, content):
            # Read-only diagnostic fast path: never wait behind an unrelated
            # router LLM turn. Preserve both the incoming user turn and the
            # generated diagnostic in history for subsequent router turns.
            async with self._state_lock:
                self._append_to_history(msg)
                self._check_and_trigger_summarization()
            await self._send_and_store(
                self._render_user_worker_status(),
                msg,
                meta={"status_query_fast_path": True},
                include_tool_visibility=False,
                mark_router_turn_sent=False,
            )
            self.save_history()
            return
        if not hasattr(self, "_router_turn_lock"):
            self._router_turn_lock = asyncio.Lock()
        async with self._router_turn_lock:
            self._router_turn_counter = (
                int(getattr(self, "_router_turn_counter", 0)) + 1
            )
            self._current_origin_message_id = str(
                msg.id or f"message-{id(msg):x}"
            )
            self._current_router_turn_id = (
                f"{self._node_id}-turn{self._router_turn_counter}"
            )
            self._current_launch_ordinal = 0
            await self._on_message_serialized(msg)

    async def _on_message_serialized(self, msg: Message) -> None:
        """
        Handle an incoming message.

        Acquires _state_lock to prevent races between:
        - Two rapid messages both seeing IDLE and double-dispatching workers
        - A busy handler reading state while worker completion clears it

        - IDLE: Classify message, respond, optionally start worker
        - BUSY: Generate contextual busy response with worker peek
        """
        # Fast-path: cancel requests bypass _state_lock entirely.
        # If a hung LLM call holds the lock, cancel must still work.
        content_raw = msg.content if isinstance(msg.content, str) else str(msg.content)
        if self._is_cancel_request(content_raw) and self.state == RouterState.BUSY:
            self._append_to_history(msg)
            # Bug 10: CC-aware cancel. If a CC session (not a worker) is what's
            # BUSY, route the cancel to cc_stop_session. cancel_worker knows
            # nothing about sessions and would report "nothing to cancel" while
            # the session keeps running.
            _worker_active = (
                self._worker_task is not None and not self._worker_task.done()
            )
            if self._cc_mgr._cc_tmux_session and not _worker_active:
                logger.info("Fast-path cancel → stopping CC session %s",
                            self._cc_mgr._cc_tmux_session)
                stop_raw = await self._cc_mgr._tool_cc_stop_session(
                    rationale=f"user requested cancel: {content_raw[:80]}",
                    force=True,
                )
                try:
                    stop = json.loads(stop_raw)
                except (ValueError, TypeError):
                    stop = {}
                if stop.get("status") == "stopped":
                    killed = stop.get("force_killed_children") or []
                    note = (
                        f" ({len(killed)} child process(es) also terminated)"
                        if killed else ""
                    )
                    await self._send_and_store(
                        f"Stopped the running Claude Code session{note}.",
                        msg, meta={"cc_session_stopped": True},
                    )
                else:
                    await self._send_and_store(
                        "The session had already ended — nothing to stop.", msg,
                    )
                return
            logger.info(
                "Fast-path cancel from %s (bypassing state lock)", msg.from_node
            )
            target, cancel_all = self._cancel_target_from_content(content_raw)
            active = self._active_worker_slots()
            if len(active) > 1 and target is None and not cancel_all:
                await self._send_and_store(
                    (
                        "Multiple workers are active. Specify a worker ID or "
                        "say \"stop all\":\n"
                        + "\n".join(
                            f"- {slot.worker_id}: {slot.task_description[:120]}"
                            for slot in active
                        )
                    ),
                    msg,
                )
                return
            cancelled = await self.cancel_worker(
                msg,
                worker_id=target,
                cancel_all=cancel_all,
            )
            if cancelled:
                await self._send_and_store(
                    "Cancelled the current task. What would you like me to do instead?",
                    msg, meta={"worker_cancelled": True},
                )
            else:
                await self._send_and_store(
                    "The task just finished — nothing to cancel.", msg,
                )
            return

        # Determine state and snapshot under the lock, but release
        # the lock before BUSY-path LLM calls so cancel_worker() isn't
        # blocked behind a multi-minute CC subprocess.
        busy_snapshot = None
        async with self._state_lock:
            # Always add to history (router sees everything)
            self._append_to_history(msg)

            # Lazy reap: auto-stop warm CC sessions idle >30 minutes
            if (
                self._cc_mgr._cc_tmux_session
                and self._cc_mgr._cc_session_warm
                and self._cc_mgr._cc_last_task_time
                and time.time() - self._cc_mgr._cc_last_task_time > 1800
            ):
                logger.info(
                    f"[CC-INTERACTIVE] Lazy reap: warm session "
                    f"{self._cc_mgr._cc_tmux_session} idle >30 min — stopping"
                )
                # Bug 2: don't ignore a "blocked" result. If the idle session
                # still has live children (e.g. a non-nohup background job),
                # do NOT force-kill it — that would silently destroy the user's
                # work, exactly the failure cc_stop_session's guard exists to
                # prevent. Log it and defer the reap by pushing the idle window
                # forward, so we don't re-block on every subsequent message.
                _reap_raw = await self._cc_mgr._tool_cc_stop_session()
                try:
                    _reap = json.loads(_reap_raw)
                except (ValueError, TypeError):
                    _reap = {}
                if _reap.get("status") == "blocked":
                    logger.warning(
                        "[CC-INTERACTIVE] Lazy reap BLOCKED — session has "
                        f"active children {_reap.get('child_pids')}; leaving "
                        "it running and deferring re-check ~30 min"
                    )
                    self._cc_mgr._cc_last_task_time = time.time()

            # Check if summarization is needed
            self._check_and_trigger_summarization()

            # Persist incoming message immediately — downstream paths
            # (needs_response=false, busy ack, direct response) may exit
            # without saving, losing the message on crash.
            try:
                self.save_history()
            except Exception as e:
                logger.warning(f"Failed to save history after incoming message: {e}")

            logger.debug(f"RouterV2 on_message: state={self.state}, from={msg.from_node}")

            # Pre-router intercept: "set context to <path>" — bypass classification
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            ctx_path = self._extract_set_context_path(content)
            if ctx_path is not None:
                await self._handle_set_context_request(msg, ctx_path)
                return

            # Pre-router intercept: "review map" — trigger map review
            if self._is_review_map_request(content):
                await self._handle_review_map_request(msg)
                return

            if self.state == RouterState.IDLE:
                busy_snapshot = ("idle", None, None)
            else:
                # Snapshot busy state; LLM call happens outside the lock
                busy_snapshot = (
                    self._current_worker_id,
                    self._pending_trigger,
                    self._worker_start_time,
                )

        # -- BUSY path: LLM call runs outside _state_lock so that
        #    cancel_worker() can acquire the lock immediately.
        if busy_snapshot is not None and busy_snapshot[0] == "idle":
            if self._config.llm_enabled and self._llm_client:
                await self._handle_idle_with_llm(msg)
            else:
                launched = await self._start_worker(msg)
                if launched:
                    receipt = getattr(self, "_last_dispatch_receipt", None)
                    slot = self._slot_for_worker(
                        receipt.worker_id if receipt else None
                    )
                    await self._send_and_store(
                        self._format_worker_launch_ack(
                            worker_id=receipt.worker_id if receipt else None,
                            task=(
                                slot.task_description
                                if slot is not None
                                else str(msg.content)
                            ),
                            backend=slot.backend if slot is not None else None,
                        ),
                        msg,
                    )
                else:
                    receipt = getattr(self, "_last_dispatch_receipt", None)
                    if receipt is not None and receipt.message:
                        await self._send_and_store(receipt.message, msg)
            return
        if busy_snapshot is not None:
            worker_id, pending_trigger, worker_start_time = busy_snapshot
            if self._config.llm_enabled and self._llm_client:
                await self._handle_busy_with_llm(
                    msg, worker_id, pending_trigger, worker_start_time
                )
            else:
                await self._handle_busy(
                    msg, worker_id, pending_trigger, worker_start_time
                )

    def _enqueue_report_wake(self, msg: Message) -> None:
        """Accept one report trigger and ensure an ordered drain task exists."""
        if not hasattr(self, "_report_wake_queue"):
            self._report_wake_queue = asyncio.Queue()
        self._report_wake_queue.put_nowait(msg)
        task = getattr(self, "_report_wake_task", None)
        if task is None or task.done():
            self._report_wake_task = asyncio.create_task(
                self._drain_report_wakes()
            )

    async def _drain_report_wakes(self) -> None:
        """Deliver queued report triggers through the serialized router turn."""
        while not self._report_wake_queue.empty():
            msg = await self._report_wake_queue.get()
            metadata = msg.metadata if isinstance(msg.metadata, dict) else {}
            worker_id = str(metadata.get("worker_id") or "worker")
            report_content = str(
                metadata.get("report_fallback_content") or msg.content or ""
            )
            self._last_router_call_sent_message = False
            try:
                await self.on_message(msg)
            except Exception as exc:
                logger.exception(
                    "RouterV2 report-as-trigger processing failed for %s: %s",
                    worker_id,
                    exc,
                )
                await self._send_and_store(
                    self._format_unsynthesized_worker_report(
                        worker_id,
                        report_content,
                        reason="failed",
                    ),
                    msg,
                    meta={
                        "unsynthesized_worker_report": True,
                        "worker_id": worker_id,
                    },
                )
            else:
                if not getattr(self, "_last_router_call_sent_message", False):
                    await self._send_and_store(
                        self._format_unsynthesized_worker_report(
                            worker_id,
                            report_content,
                            reason="timed out or produced no visible response",
                        ),
                        msg,
                        meta={
                            "unsynthesized_worker_report": True,
                            "worker_id": worker_id,
                        },
                    )
            finally:
                self._report_wake_queue.task_done()

    # =========================================================================
    # LLM-enabled handlers
    # =========================================================================

    async def _load_relevant_context(self, msg: Message) -> None:
        """Pre-load top-5 relevant memories based on the incoming message."""
        if not self._memory or not isinstance(self._memory, MemorySystemV2):
            return
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if not content.strip():
            return
        try:
            budget = self._config.memory_retrieve_budget_tokens
            self._relevant_context = await self._memory.render_retrieved_context(
                content, budget_tokens=budget,
            )
            if self._relevant_context:
                logger.info(
                    "Loaded relevant memories: %d tokens",
                    estimate_tokens(self._relevant_context),
                )
        except Exception as e:
            logger.warning("Failed to load relevant memories: %s", e)
            self._relevant_context = ""

    def _record_router_deep_override(
        self,
        msg: Message,
        *,
        started_at: float,
        failure_class: str = "",
        source: str = "@deep",
    ) -> None:
        """Emit metadata-only telemetry for one forced deep router turn."""
        client = self._deep_llm_client
        usage_source = getattr(self, "_last_router_call_usage", {})
        usage_keys = (
            "input_tokens",
            "output_tokens",
            "cache_creation_tokens",
            "cache_read_tokens",
            "reasoning_tokens",
            "total_tokens",
            "llm_calls",
        )
        usage = {
            key: int(usage_source.get(key, 0) or 0)
            for key in usage_keys
            if usage_source.get(key) is not None
        }
        turn_id = str(
            getattr(self, "_current_router_turn_id", "") or msg.id or ""
        )
        dispatch_outcomes = sorted({
            receipt.status
            for receipt in getattr(self, "_dispatch_receipts", {}).values()
            if receipt.router_turn_id == turn_id
        })
        last_receipt = getattr(self, "_last_dispatch_receipt", None)
        if (
            last_receipt is not None
            and last_receipt.router_turn_id == turn_id
            and last_receipt.status not in dispatch_outcomes
        ):
            dispatch_outcomes.append(last_receipt.status)
        event = {
            "event": "router_deep_override",
            "source": source,
            "turn_id": turn_id,
            "origin_message_id": str(msg.id or ""),
            "agent": self._node_id,
            "selected_backend": self._deep_backend_name,
            "selected_model": (
                str(client.config.model or "") if client is not None else ""
            ),
            "latency_ms": max(
                0,
                int(round((time.monotonic() - started_at) * 1000)),
            ),
            "usage": usage,
            "tool_count": len(getattr(self, "_last_router_call_tools", [])),
            "dispatch_outcomes": dispatch_outcomes,
            "failure_class": (
                failure_class
                or getattr(self, "_last_router_failure_class", "")
                or ""
            ),
        }
        self._last_router_deep_override_event = event
        logger.info(
            "router_deep_override %s",
            json.dumps(event, sort_keys=True, separators=(",", ":")),
        )

    def _ensure_pipeline_router(self) -> Any | None:
        """Lazily construct the pipeline router bridge."""
        if self._pipeline_router is not None:
            return self._pipeline_router
        try:
            from .router_pipeline import PipelineRouter

            self._pipeline_router = PipelineRouter(
                llm_backend_config=self._config.pipeline_backend,
                agent_name=f"{self._agent_type}:{self._nickname}",
                nickname=self._nickname,
                history_dir=(
                    self._state_paths.history_dir
                    if self._state_paths is not None
                    else None
                ),
                plan_path=self._config.pipeline_plan_path,
            )
            return self._pipeline_router
        except Exception:
            logger.exception("Failed to initialize pipeline router")
            return None

    async def _build_pipeline_router_context(
        self, msg: Message, busy_context: bool = False,
    ) -> dict[str, Any]:
        """Build the context object passed to PipelineRouter.process().

        When ``busy_context`` is True, includes a ``worker_status`` block
        describing the currently running worker — task, elapsed time, and
        recent activity lines — so the pipeline can make informed
        cancel/relaunch/continue decisions in BUSY state.
        """
        personality = ""
        project_context_parts: list[str] = []
        project_maps: list[str] = []

        if self._memory:
            try:
                personality = self._memory.get_personality() or ""
            except Exception as e:
                logger.debug("Pipeline router personality lookup failed: %s", e)

        memory_toc = ""
        conversation_summary = ""

        if self._memory and isinstance(self._memory, MemorySystemV2):
            map_context = self._get_last_n_turns_text(5)
            try:
                map_block = await self._memory.render_relevant_maps_block(map_context)
                if map_block:
                    project_context_parts.append(map_block)
                    project_maps.append(map_block)
            except Exception as e:
                logger.debug("Pipeline router map context failed: %s", e)
            try:
                log_block = await self._memory.render_recent_log_block()
                if log_block:
                    project_context_parts.append(log_block)
            except Exception as e:
                logger.debug("Pipeline router recent log context failed: %s", e)
            try:
                digest_block = self._standing_digest_block()
                use_toc = getattr(self._config, "memory_retrieval_redesign_enabled", False)
                if digest_block:
                    memory_toc = digest_block
                elif use_toc:
                    query_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                    toc = await self._memory.build_toc(
                        query_text=query_text,
                        k=getattr(self._config, "memory_toc_size", 30),
                        context_text=map_context,
                    )
                    toc = self._memory.dedup_toc_against_window(toc, self._history)
                    toc_block = self._memory.render_toc_block(
                        toc, injected_ids=self._injected_memory_ids,
                    )
                    if toc_block:
                        memory_toc = toc_block
            except Exception as e:
                logger.debug("Pipeline router memory TOC failed: %s", e)
            try:
                summary_block = await self._memory.render_summary_block()
                if summary_block:
                    conversation_summary = summary_block
            except Exception as e:
                logger.debug("Pipeline router conversation summary failed: %s", e)
        elif self._memory:
            try:
                memory_block = await self._memory.render(
                    self._memory.light_profile,
                    query=msg.content if isinstance(msg.content, str) else str(msg.content),
                )
                if memory_block:
                    project_context_parts.append(memory_block)
            except Exception as e:
                logger.debug("Pipeline router legacy memory context failed: %s", e)

        skill_index = self._skill_index_block()
        if skill_index:
            project_context_parts.append(skill_index)

        worker_status = ""
        if busy_context:
            worker_status = self._render_pipeline_worker_status()
        todo_context = self._render_todo_context(
            self._conversation_id_from_message(msg)
        )
        notes_context = self._render_conversation_notes(
            self._conversation_id_from_message(msg)
        )

        return {
            "personality": personality,
            "relevant_memories": self._relevant_context,
            "project_context": "\n\n".join(project_context_parts),
            "todo_context": todo_context,
            "notes_context": notes_context,
            "project_maps": project_maps,
            "agent_tools": ", ".join(self._router_tool_names),
            "agent_name": f"{self._agent_type}:{self._nickname}",
            "nickname": self._nickname,
            "node_id": self._node_id,
            "system_prompt": self._system_prompt or "",
            "memory_toc": memory_toc,
            "conversation_summary": conversation_summary,
            "worker_status": worker_status,
        }

    def _standing_digest_block(self) -> str:
        """Rev-10 read pathway: the published standing digest replaces the
        memory TOC in prompt composition.

        Returns "" unless standing_digest_enabled with a readable,
        non-empty digest file — callers fall back to the TOC branch on "",
        so a missing/unreadable digest degrades to the old pathway instead
        of leaving the agent memoryless.
        """
        if not getattr(self._config, "standing_digest_enabled", False):
            return ""
        path = os.path.expanduser(
            getattr(self._config, "standing_digest_path", "") or "")
        if not path:
            return ""
        from .digest_io import read_digest

        try:
            # Shared lock: never observe a half-written digest while another
            # task is mid-replacement.
            content = read_digest(path)
        except OSError as e:
            logger.warning(
                "standing digest unreadable (%s); falling back to memory TOC",
                e)
            return ""
        if not content.strip():
            return ""
        header = (
            "Your standing digest is a maintained, compressed summary of your "
            "entire history — who you are, what you've done, and what matters. "
            "Tokens like [m_xxxx] are memory references: call memory_get(id=\"m_xxxx\") "
            "to retrieve the full record. Use the digest to ground your answers; "
            "when the user asks for specifics, exact wording, or deeper detail, "
            "follow the references with memory_get or use memory_search to find "
            "what the digest summarizes."
        )
        return (
            f"<standing_digest>\n{header}\n\n"
            f"{content.strip()}\n</standing_digest>"
        )

    def _skill_index_block(self) -> str:
        """Render the active-card index, failing closed on malformed state."""
        store = getattr(self, "_skill_store", None)
        if store is None:
            return ""
        try:
            return store.render_index_block()
        except (OSError, SkillCardError) as exc:
            logger.warning("governed skill index unavailable: %s", exc)
            return ""

    def _render_pipeline_worker_status(self) -> str:
        """Render the authoritative fixed-slot table for pipeline prompts."""
        if not self._active_worker_slots():
            return ""
        return self._worker_slots_xml()

    async def _call_router_pipeline(
        self, msg: Message, busy_context: bool = False,
    ) -> dict[str, Any]:
        router = self._ensure_pipeline_router()
        if router is None:
            raise RuntimeError("pipeline router is not available")

        # Match the full-router ledger lifecycle so a pipeline response never
        # inherits tools from a prior turn and its final delivery can add the
        # human-only visibility footer.
        self._last_router_call_sent_message = False
        self._last_router_call_tools = []
        self._tool_visibility_pending = True
        self._last_worker_launch = None
        self._router_call_worker_launches = []
        self._router_call_worker_task_keys = set()

        context = await self._build_pipeline_router_context(msg, busy_context=busy_context)
        parsed = await router.process(msg, context)
        executed_tool_names = parsed.get("executed_tool_names", [])
        if not isinstance(executed_tool_names, list):
            executed_tool_names = []
        for tool_name in executed_tool_names:
            if not isinstance(tool_name, str) or tool_name in {
                "send_message", "send_report", "worker_launch",
            }:
                continue
            self._last_router_call_tools.append((tool_name, ""))
        return parsed

    async def _run_llm_turn_with_status(
        self,
        msg: Message,
        handler: Callable[..., Awaitable[None]],
        *handler_args: Any,
    ) -> None:
        """Run one direct LLM handler while publishing its live router state.

        This deliberately starts after ``_on_message_serialized`` chooses its
        IDLE/BUSY branch.  Setting the flag before that decision would make a
        new direct turn look BUSY to its own dispatcher.
        """
        metadata = msg.metadata if isinstance(msg.metadata, dict) else {}
        previous_llm_active = self._llm_turn_active
        previous_autonomous_active = self._autonomous_session_active
        self._llm_turn_active = True
        self._autonomous_session_active = (
            previous_autonomous_active
            or bool(metadata.get("autonomous_session"))
        )
        await self._status_push()
        try:
            await handler(msg, *handler_args)
        finally:
            self._llm_turn_active = previous_llm_active
            self._autonomous_session_active = previous_autonomous_active
            await self._status_push()

    async def _handle_idle_with_llm(self, msg: Message) -> None:
        await self._run_llm_turn_with_status(
            msg, self._handle_idle_with_llm_impl
        )

    async def _handle_idle_with_llm_impl(self, msg: Message) -> None:
        """Handle message in IDLE state — dispatches to full or classifier mode."""
        prompt_msg = msg
        deep_override = False
        deep_selection_source = ""
        deep_started_at = 0.0
        selected_llm_client = None

        # ``@deep`` requires a direct light full router; the deep backend may
        # be direct or harness (``_call_router_full`` picks the code path from
        # the selected client). Detect and select the deep client before
        # memory retrieval or either router loop.
        if self._config.router_mode == "full":
            light_config = getattr(self._llm_client, "config", None)
            light_backend = getattr(light_config, "backend", "")
        else:
            light_backend = ""
        if (
            self._config.router_mode == "full"
            and light_backend not in HARNESS_BACKENDS
        ):
            prompt_msg, deep_override = prepare_router_deep_override(msg)
            if deep_override:
                deep_started_at = time.monotonic()
                self._last_router_call_tools = []
                self._last_router_call_usage = {}
                self._last_router_failure_class = ""
                if (
                    self._router_deep_enabled
                    and self._deep_llm_client is not None
                ):
                    selected_llm_client = self._deep_llm_client
                    deep_selection_source = "@deep"
                else:
                    self._last_router_failure_class = "DeepRouterUnavailable"
                    await self._send_and_store(
                        "The deep router is unavailable for this request.",
                        msg,
                        meta={"router_deep_unavailable": True},
                    )
                    self._record_router_deep_override(
                        msg,
                        started_at=deep_started_at,
                        failure_class="DeepRouterUnavailable",
                    )
                    return
            else:
                metadata = (
                    msg.metadata if isinstance(msg.metadata, dict) else {}
                )
                if (
                    self._config.autonomous_plan_backend == "deep"
                    and metadata.get("autonomous_session")
                    and not metadata.get("worker_report")
                ):
                    deep_started_at = time.monotonic()
                    self._last_router_call_tools = []
                    self._last_router_call_usage = {}
                    self._last_router_failure_class = ""
                    if (
                        self._router_deep_enabled
                        and self._deep_llm_client is not None
                    ):
                        selected_llm_client = self._deep_llm_client
                        deep_selection_source = "autonomous_plan"
                    else:
                        logger.warning(
                            "Autonomous PLAN requested deep router for %s, "
                            "but it is unavailable; falling back to light",
                            self._node_id,
                        )

        # Pre-load relevant memories for this message
        await self._load_relevant_context(prompt_msg)

        if self._config.router_mode == "classifier":
            return await self._handle_idle_classifier(msg)

        if self._config.router_mode == "pipeline":
            try:
                parsed = await self._call_router_pipeline(msg)
                logger.debug("RouterV2 pipeline response: %s", str(parsed)[:500])

                if parsed.get("no_response"):
                    logger.info("RouterV2 pipeline: no_response, staying silent")
                    return

                response_text = parsed.get("response", "")
                dispatch_outcome = None
                history_content = None
                if parsed.get("dispatch_worker"):
                    dispatch_outcome = await self._dispatch_worker(
                        msg,
                        parsed,
                        source="xml",
                    )
                    history_content = self._dispatch_history_record(
                        response_text,
                        dispatch_outcome,
                    )
                    if dispatch_outcome.status == "running":
                        response_text = self._ensure_worker_launch_ack(
                            response_text,
                            receipt=dispatch_outcome,
                        )
                    else:
                        response_text = dispatch_outcome.message
                else:
                    logger.info(
                        f"[WORKER] PIPELINE DIRECT RESPONSE: {self._nickname} "
                        f"responding directly for message from {msg.from_node}"
                    )
                if response_text:
                    await self._send_and_store(
                        response_text,
                        msg,
                        history_content=history_content,
                    )

                if response_text or not parsed.get("dispatch_worker"):
                    self.save_history()
                return
            except Exception as e:
                to_node = getattr(msg, 'to_node', '') or ''
                if to_node.startswith("channel:"):
                    logger.warning(
                        "RouterV2 pipeline failed for channel msg: %s, staying silent",
                        e,
                    )
                    return
                error_notice = f"Router pipeline error: {e}. Falling back to worker dispatch."
                logger.error(error_notice)
                await self._send_and_store(error_notice, msg)
                launched = await self._start_worker(msg)
                if launched:
                    await self._send_and_store(
                        self._format_worker_launch_ack(
                            worker_id=self._current_worker_id,
                            task=self._current_task_description or str(msg.content),
                            backend=self._current_worker_backend,
                        ),
                        msg,
                    )
                else:
                    receipt = getattr(self, "_last_dispatch_receipt", None)
                    if receipt is not None and receipt.message:
                        await self._send_and_store(receipt.message, msg)
                return

        # Full conversational router mode
        if not self._router_process_fn:
            logger.error("RouterV2 full mode: router_process_fn not set, falling back to classifier")
            return await self._handle_idle_classifier(msg)

        try:
            raw_response = await self._call_router_full(
                prompt_msg,
                llm_client=selected_llm_client,
            )
            logger.debug(f"RouterV2 full raw response: {raw_response[:500]}")
            parsed = self._parse_router_response(raw_response)

            if parsed["no_response"]:
                logger.info("RouterV2 full: <no_response>, staying silent")
                return

            # If the tool loop already launched a worker, suppress any legacy
            # <dispatch_worker> block in the final text. This closes the
            # same-turn double-dispatch path even when max_concurrent_workers > 1.
            if parsed["dispatch_worker"] and self._last_worker_launch:
                logger.warning(
                    "RouterV2 full: suppressing dispatch_worker because "
                    "worker_launch already dispatched %s in this turn",
                    self._last_worker_launch.get("worker_id"),
                )
                parsed["dispatch_worker"] = False
                if not parsed["response"]:
                    parsed["response"] = self._format_worker_launch_ack()

            # Send conversational response (if any)
            response_text = parsed["response"]
            dispatch_outcome = None
            history_content = None

            # Dispatch worker if requested
            if parsed["dispatch_worker"]:
                # Harness routers carry worker dispatches in text, so their
                # narrated SESSION PLAN reaches this seam before the outgoing
                # turn is stored.  Bind the validated plan to the original
                # wake trigger now; _dispatch_worker() will then preserve it on
                # the detached worker trigger and its report continuation.
                # Native worker_launch performs the same capture in AgentNode
                # before it builds its synthetic trigger.
                plan_metadata = self._capture_autonomous_session_plan_metadata(
                    raw_response
                )
                if plan_metadata and isinstance(msg.metadata, dict):
                    msg.metadata.update(plan_metadata)
                dispatch_outcome = await self._dispatch_worker(
                    msg,
                    parsed,
                    source="xml",
                )
                history_content = self._dispatch_history_record(
                    response_text,
                    dispatch_outcome,
                )
                if dispatch_outcome.status == "running":
                    response_text = self._ensure_worker_launch_ack(
                        response_text,
                        receipt=dispatch_outcome,
                    )
                else:
                    response_text = dispatch_outcome.message
            else:
                logger.info(
                    f"[WORKER] DIRECT RESPONSE: {self._nickname} responding directly "
                    f"(no worker dispatch) for message from {msg.from_node}"
                )

            if response_text:
                await self._send_and_store(
                    response_text,
                    msg,
                    history_content=history_content,
                )

            # Persist history after every response (dispatch or not)
            if response_text or not parsed["dispatch_worker"]:
                self.save_history()

        except Exception as e:
            if deep_override:
                failure_class = type(e).__name__
                self._last_router_failure_class = failure_class
                logger.error(
                    "Forced deep router turn failed for %s: %s",
                    self._node_id,
                    failure_class,
                )
                await self._send_and_store(
                    (
                        "The deep router is unavailable for this request "
                        f"({failure_class})."
                    ),
                    msg,
                    meta={"router_deep_unavailable": True},
                )
                return
            to_node = getattr(msg, 'to_node', '') or ''
            if to_node.startswith("channel:"):
                logger.warning(f"RouterV2 full classification failed for channel msg: {e}, staying silent")
                return
            error_notice = f"Router error: {e}. Falling back to worker dispatch."
            logger.error(error_notice)
            await self._send_and_store(error_notice, msg)
            launched = await self._start_worker(msg)
            if launched:
                await self._send_and_store(
                    self._format_worker_launch_ack(
                        worker_id=self._current_worker_id,
                        task=self._current_task_description or str(msg.content),
                        backend=self._current_worker_backend,
                    ),
                    msg,
                )
            else:
                receipt = getattr(self, "_last_dispatch_receipt", None)
                if receipt is not None and receipt.message:
                    await self._send_and_store(receipt.message, msg)
        finally:
            if deep_selection_source:
                self._record_router_deep_override(
                    msg,
                    started_at=deep_started_at,
                    source=deep_selection_source,
                )

    async def _handle_idle_classifier(self, msg: Message) -> None:
        """Legacy classifier path — thin JSON classification + worker dispatch."""
        try:
            classification = await self._classify_message(msg)

            if not classification.get("needs_response", True):
                logger.info(f"RouterV2 LLM classified as needs_response=false, no response sent")
                return

            if classification.get("needs_worker", True):
                logger.info("RouterV2 LLM classified as needs_worker=true, starting worker")
                # Send the router's ack and store in history
                ack_response = classification.get("response", "Looking into that now...")
                launched = await self._start_worker(msg)
                if launched:
                    ack_response = self._ensure_worker_launch_ack(
                        ack_response,
                        worker_id=self._current_worker_id,
                        task=self._current_task_description or str(msg.content),
                        backend=self._current_worker_backend,
                    )
                else:
                    receipt = getattr(self, "_last_dispatch_receipt", None)
                    ack_response = (
                        receipt.message
                        if receipt is not None and receipt.message
                        else (
                            "I could not launch a worker because worker "
                            "capacity is full. No new worker was started."
                        )
                    )
                if ack_response:
                    await self._send_and_store(ack_response, msg)
            else:
                response = classification.get("response", "")
                if response:
                    await self._send_and_store(response, msg)
                logger.info(f"RouterV2 LLM classified as needs_worker=false, staying IDLE")

        except Exception as e:
            to_node = getattr(msg, 'to_node', '') or ''
            if to_node.startswith("channel:"):
                logger.warning(f"RouterV2 LLM classification failed for channel msg: {e}, staying silent")
                return
            # Surface the error to the user
            error_notice = f"Router LLM error during classification: {e}. Falling back to worker dispatch."
            logger.error(error_notice)
            await self._send_and_store(error_notice, msg)
            launched = await self._start_worker(msg)
            if launched:
                await self._send_and_store(
                    self._format_worker_launch_ack(
                        worker_id=self._current_worker_id,
                        task=self._current_task_description or str(msg.content),
                        backend=self._current_worker_backend,
                    ),
                    msg,
                )
            else:
                receipt = getattr(self, "_last_dispatch_receipt", None)
                if receipt is not None and receipt.message:
                    await self._send_and_store(receipt.message, msg)

    async def _handle_busy_with_llm(
        self,
        msg: Message,
        worker_id: str | None,
        pending_trigger: Message | None,
        worker_start_time: float | None,
    ) -> None:
        await self._run_llm_turn_with_status(
            msg,
            self._handle_busy_with_llm_impl,
            worker_id,
            pending_trigger,
            worker_start_time,
        )

    async def _handle_busy_with_llm_impl(
        self,
        msg: Message,
        worker_id: str | None,
        pending_trigger: Message | None,
        worker_start_time: float | None,
    ) -> None:
        """Handle message in BUSY state — dispatches to full or classifier mode."""
        # Check for cancel request before anything else
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if self._is_cancel_request(content):
            await self._handle_cancel_request(msg, worker_id)
            return

        # Pre-load relevant memories for this message
        await self._load_relevant_context(msg)

        if self._config.router_mode == "classifier":
            return await self._handle_busy_classifier(
                msg, worker_id, pending_trigger, worker_start_time
            )

        if self._config.router_mode == "pipeline":
            try:
                parsed = await self._call_router_pipeline(msg, busy_context=True)
                logger.debug("RouterV2 pipeline BUSY response: %s", str(parsed)[:500])

                if parsed.get("no_response"):
                    logger.info("RouterV2 pipeline BUSY: no_response, staying silent")
                    return

                _worker_active = (
                    self._worker_task is not None and not self._worker_task.done()
                )
                if parsed.get("dispatch_worker") and self._cc_mgr._cc_tmux_session and not _worker_active:
                    logger.info(
                        "[CC-INTERACTIVE] Suppressing pipeline dispatch_worker during active "
                        "CC session — would run a worker in parallel with the session"
                    )
                    parsed["dispatch_worker"] = False

                if parsed.get("dispatch_worker"):
                    original_response = parsed.get("response", "")
                    dispatch_outcome = await self._dispatch_worker(
                        msg,
                        parsed,
                        source="xml",
                    )
                    history_content = self._dispatch_history_record(
                        original_response,
                        dispatch_outcome,
                    )
                    busy_response = (
                        self._ensure_worker_launch_ack(
                            original_response,
                            receipt=dispatch_outcome,
                        )
                        if dispatch_outcome.status == "running"
                        else dispatch_outcome.message
                    )
                else:
                    busy_response = parsed.get("response", "")
                    history_content = None
                if busy_response:
                    await self._send_and_store(
                        busy_response,
                        msg,
                        history_content=history_content,
                    )

                if busy_response or not parsed.get("dispatch_worker"):
                    self.save_history()
                return
            except Exception as e:
                to_node = getattr(msg, 'to_node', '') or ''
                if to_node.startswith("channel:"):
                    logger.warning(
                        "RouterV2 pipeline BUSY failed for channel msg: %s, staying silent",
                        e,
                    )
                    return
                error_notice = f"Router pipeline error during busy response: {e}"
                logger.error(error_notice)
                await self._send_and_store(error_notice, msg)
                return

        # Full conversational router mode (BUSY)
        if not self._router_process_fn:
            logger.error("RouterV2 full BUSY: router_process_fn not set, falling back to classifier")
            return await self._handle_busy_classifier(
                msg, worker_id, pending_trigger, worker_start_time
            )

        # Acquire the CC router lock when a CC session is active to
        # serialise against concurrent monitor deliveries.
        _cc_lock = (
            self._cc_mgr._cc_router_lock
            if self._cc_mgr._cc_tmux_session
            else None
        )

        try:
            if _cc_lock:
                await _cc_lock.acquire()
            try:
                raw_response = await self._call_router_full(
                    msg, busy=True,
                    worker_id=worker_id,
                    pending_trigger=pending_trigger,
                    worker_start_time=worker_start_time,
                )
            finally:
                if _cc_lock:
                    _cc_lock.release()
            logger.debug(f"RouterV2 full BUSY raw response: {raw_response[:500]}")
            parsed = self._parse_router_response(raw_response)

            if parsed["no_response"]:
                logger.info("RouterV2 full BUSY: <no_response>, staying silent")
                return

            # Bug 10: if BUSY is caused by a live CC session (not a worker),
            # a <dispatch_worker> block would launch a worker IN PARALLEL with
            # the session. Suppress it — the session is the work in progress.
            _worker_active = (
                self._worker_task is not None and not self._worker_task.done()
            )
            if parsed.get("dispatch_worker") and self._cc_mgr._cc_tmux_session and not _worker_active:
                logger.info(
                    "[CC-INTERACTIVE] Suppressing dispatch_worker during active "
                    "CC session — would run a worker in parallel with the session"
                )
                parsed["dispatch_worker"] = False

            busy_response = parsed["response"]
            history_content = None

            # Additional dispatch while BUSY: start another worker only when
            # capacity remains. Never cancel/relaunch from LLM output.
            if parsed.get("dispatch_worker"):
                original_response = busy_response
                dispatch_outcome = await self._dispatch_worker(
                    msg,
                    parsed,
                    source="xml",
                )
                history_content = self._dispatch_history_record(
                    original_response,
                    dispatch_outcome,
                )
                if dispatch_outcome.status == "running":
                    busy_response = self._ensure_worker_launch_ack(
                        original_response,
                        receipt=dispatch_outcome,
                    )
                else:
                    busy_response = dispatch_outcome.message
            if busy_response:
                await self._send_and_store(
                    busy_response,
                    msg,
                    history_content=history_content,
                )

            # Persist request-shaped history after dispatch as well as ordinary
            # BUSY responses; a crash must not lose the exemplar we just sent.
            if busy_response or not parsed.get("dispatch_worker"):
                self.save_history()

        except Exception as e:
            to_node = getattr(msg, 'to_node', '') or ''
            if to_node.startswith("channel:"):
                logger.warning(f"RouterV2 full BUSY failed for channel msg: {e}, staying silent")
                return
            error_notice = f"Router error during busy response: {e}"
            logger.error(error_notice)
            await self._send_and_store(error_notice, msg)

    async def _handle_busy_classifier(
        self,
        msg: Message,
        worker_id: str | None,
        pending_trigger: Message | None,
        worker_start_time: float | None,
    ) -> None:
        """Legacy classifier path for BUSY state."""
        try:
            classification = await self._classify_message(msg)
            if not classification.get("needs_response", True):
                logger.info("RouterV2 BUSY: message doesn't need response, staying silent")
                return

            # Generate busy response (LLM will see worker progress in history)
            response = await self._generate_busy_response(
                msg, worker_id, pending_trigger, worker_start_time
            )
            await self._send_and_store(response, msg)

        except Exception as e:
            to_node = getattr(msg, 'to_node', '') or ''
            if to_node.startswith("channel:"):
                logger.warning(f"RouterV2 LLM busy classification failed for channel msg: {e}, staying silent")
                return
            # Surface the error
            error_notice = f"Router LLM error during busy response: {e}"
            logger.error(error_notice)
            await self._send_and_store(error_notice, msg)

    async def _classify_message(self, msg: Message) -> dict:
        """
        Use LLM to classify the message and generate a response.

        Returns:
            dict with keys: needs_response, needs_worker, response
        """
        instructions = ROUTER_INSTRUCTIONS_IDLE.format(
            nickname=self._nickname,
            agent_type=self._agent_type,
        )
        # v2: append retrieval + dispatch criteria + self-check sections
        if self._memory and isinstance(self._memory, MemorySystemV2):
            instructions += _V2_CLASSIFIER_ADDITIONS
            mem_profile = None
        elif self._memory:
            # v1 uses classifier_profile for lightweight memory in classification
            mem_profile = self._memory.classifier_profile
        else:
            mem_profile = None
        prompt = await self._build_router_prompt(
            instructions,
            memory_profile=mem_profile,
            include_tools=False,
            max_history_turns=30,
            trigger_msg=msg,
        )

        logger.debug(f"RouterV2 calling LLM for classification")
        try:
            raw_response = await asyncio.wait_for(
                self._llm_client.complete(prompt),
                timeout=120,
            )
        except asyncio.TimeoutError:
            logger.error("RouterV2 classification LLM call timed out after 120s")
            return {"needs_response": True, "needs_worker": True, "response": ""}
        logger.debug(f"RouterV2 raw classification: {raw_response[:300]}")

        return self._parse_classification_response(raw_response)

    def _parse_classification_response(self, raw_response: str) -> dict:
        """Parse the LLM's classification response.

        Handles both v1 JSON format (needs_response/needs_worker) and
        v2 key-value format (action: direct/dispatch).
        """
        try:
            # Strip markdown fences if present
            text = raw_response.strip()
            if text.startswith("```"):
                # Remove opening fence (```json or ```)
                text = re.sub(r'^```\w*\s*\n?', '', text)
                text = re.sub(r'\n?```\s*$', '', text)
                text = text.strip()

            # v2 key-value format: action: direct/dispatch
            action_match = re.search(
                r'action:\s*(direct|dispatch)\b', text, re.IGNORECASE
            )
            if action_match and '{' not in text[:action_match.start()]:
                action = action_match.group(1).lower()
                if action == "dispatch":
                    task_match = re.search(r'task_summary:\s*(.+)', text, re.IGNORECASE)
                    return {
                        "needs_response": True,
                        "needs_worker": True,
                        "response": "",
                        "task_summary": task_match.group(1).strip().strip('"') if task_match else "",
                    }
                elif action == "direct":
                    response_match = re.search(r'response:\s*"?(.*?)"?\s*$', text[action_match.end():], re.DOTALL)
                    return {
                        "needs_response": True,
                        "needs_worker": False,
                        "response": response_match.group(1).strip() if response_match else "",
                    }

            # Try direct JSON parse (handles most cases including v2 JSON with action field)
            try:
                result = json.loads(text)
                if isinstance(result, dict) and "needs_response" in result or "needs_worker" in result:
                    if "needs_response" not in result:
                        result["needs_response"] = True
                    return result
            except (json.JSONDecodeError, ValueError):
                pass

            # Fallback: extract JSON with balanced braces
            start = text.find('{')
            if start >= 0:
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == '{':
                        depth += 1
                    elif text[i] == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:i + 1]
                            try:
                                result = json.loads(candidate)
                                if isinstance(result, dict):
                                    if "needs_response" not in result:
                                        result["needs_response"] = True
                                    return result
                            except (json.JSONDecodeError, ValueError):
                                pass
                            break

            # Nothing parsed — default to worker dispatch
            logger.warning(f"RouterV2 failed to parse classification JSON, defaulting to needs_worker=true")
            return {
                "needs_response": True,
                "needs_worker": True,
                "response": "Let me look into that..."
            }

        except Exception:
            logger.warning(f"RouterV2 classification parse error, defaulting to needs_worker=true")
            return {
                "needs_response": True,
                "needs_worker": True,
                "response": "Let me look into that..."
            }

    def _parse_router_response(self, raw_response: str) -> dict:
        """Parse the full router's natural-language response.

        Returns:
            dict with keys:
            - no_response: bool (True if router opted out)
            - response: str (conversational text, empty if no_response)
            - dispatch_worker: bool (True if worker block present)
            - task: str (worker task description)
            - task_type: str (optional configured task-shape choice)
            - backend: str (hard override only; user must name it explicitly)
            - backend_reason: str (required for a task_type choice)
        """
        text = raw_response.strip()

        # Strip <reasoning>...</reasoning> blocks that DeepSeek sometimes
        # embeds inline in the content body (distinct from the API-level
        # reasoning_content field which is already handled separately).
        text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL).strip()

        # Check for <no_response/>
        if re.search(r'<no_response\s*/?\s*>', text):
            return {"no_response": True, "response": "", "dispatch_worker": False}

        # Check for <dispatch_worker> block
        dispatch_match = re.search(
            r'<dispatch_worker>(.*?)</dispatch_worker>',
            text, re.DOTALL
        )

        if dispatch_match:
            # Hard gate: agents with cc_interactive_tools enabled must route ALL
            # code execution through a CC session (cc_start_session), never a
            # traditional worker. A <dispatch_worker> block would launch a worker
            # with full write access, bypassing the CC route. Suppress it here —
            # this parser is the single chokepoint for both the IDLE and BUSY
            # dispatch paths — keeping only the conversational text that preceded
            # the block.
            if self._cc_interactive_enabled or self._harness_session_enabled:
                route = ("cc_start_session" if self._cc_interactive_enabled
                         else "harness_start_session")
                logger.warning(
                    "[INTERACTIVE-SESSION] dispatch_worker blocked — an interactive "
                    "session is enabled; use %s instead of a traditional worker", route,
                )
                return {
                    "no_response": False,
                    "response": text[:dispatch_match.start()].strip(),
                    "dispatch_worker": False,
                }
            block = dispatch_match.group(1)
            # Extract task description from the block.
            # The LLM may format as separate lines or pipe-separated.
            # "complexity:" lines are ignored (legacy, no longer used).
            task = ""
            task_lines: list[str] = []
            collecting_task = False
            task_type = ""
            backend = ""
            backend_reason = ""
            for line in block.strip().split('\n'):
                line = line.strip().strip('|').strip()
                low = line.lower() if line else ""
                if low.startswith('task:'):
                    collecting_task = True
                    raw_task = line[5:].strip()
                    # Handle pipe-separated: "task: X | complexity: Y"
                    if '|' in raw_task:
                        parts = raw_task.split('|')
                        for part in parts:
                            part = part.strip()
                            # Skip legacy complexity fields
                            if part.lower().startswith('complexity:'):
                                continue
                            elif part and any(ch.isalnum() for ch in part):
                                task_lines.append(part)
                    else:
                        if raw_task:
                            task_lines.append(raw_task)
                elif low.startswith('complexity:'):
                    # Legacy field — ignore but stop collecting task lines
                    collecting_task = False
                elif low.startswith('task_type:'):
                    collecting_task = False
                    task_type = line[len('task_type:'):].strip()
                elif low.startswith('backend:'):
                    collecting_task = False
                    backend = line[len('backend:'):].strip()
                elif low.startswith('reason:'):
                    collecting_task = False
                    backend_reason = line[len('reason:'):].strip()
                elif collecting_task and line:
                    # Continuation line of a multi-line task
                    task_lines.append(line)
            task = "\n".join(task_lines)
            # Validate task — reject punctuation-only garbage
            if task and not any(c.isalnum() for c in task):
                logger.warning(f"RouterV2: dispatch block has non-alphanumeric task={task!r}, clearing")
                task = ""

            # Response text is everything before the dispatch block
            response = text[:dispatch_match.start()].strip()

            return {
                "no_response": False,
                "response": response,
                "dispatch_worker": True,
                "task": task,
                "task_type": task_type,
                "backend": backend,
                "backend_reason": backend_reason,
            }

        # Plain response — no dispatch, no opt-out
        return {
            "no_response": False,
            "response": text,
            "dispatch_worker": False,
        }

    # ── Entity/group/digest self-curation (§4.3) ──────────────────

    def enqueue_curation_batch(self, batch: Any) -> bool:
        """Accept one post-commit formation batch.  Never awaits, never drops.

        Called synchronously from ``MemorySystemV2`` while
        ``_formation_lock`` is held, so it must only enqueue and return.
        """
        try:
            self._curation_idle.clear()
            self._curation_queue.put_nowait(batch)
            for memory_id in tuple(getattr(batch, "memory_ids", ()) or ()):
                key = str(memory_id)
                self._curation_scheduled_memory_counts[key] = (
                    self._curation_scheduled_memory_counts.get(key, 0) + 1
                )
            self._curation_batches_seen += 1
        except Exception:
            logger.exception("failed to enqueue curation batch")
            return False
        self._ensure_curation_drain_task()
        return True

    def _ensure_curation_drain_task(self) -> None:
        task = self._curation_drain_task
        if task is not None and not task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop yet (synchronous construction/test context).  The next
            # enqueue from inside the loop starts the drain; nothing is lost
            # because the queue is unbounded.
            return
        self._curation_drain_task = loop.create_task(
            self._curation_drain_loop(),
            name=f"{self._node_id}-curation-drain",
        )

    async def _curation_drain_loop(self) -> None:
        """Drain curation batches, one turn at a time, off the message path.

        This loop deliberately does **not** take ``_router_turn_lock``.  It used
        to; because a curation turn is a full 1-4 minute LLM call, that made
        every incoming message wait behind curation — the starvation this whole
        change exists to fix.  Concurrency is safe because each
        ``_call_router_full`` now carries its per-call state on a contextvar
        (see :class:`RouterCallState`) rather than on shared instance
        attributes, so a curation turn and a message turn cannot overwrite each
        other's tool ledger, delivery flag or worker-launch guards.

        Curation turns remain serialised *with each other* — there is exactly
        one drain task, and it awaits each turn before taking the next batch.
        """
        # create_task() copies its creator's context.  Formation commonly
        # starts this drain from inside a live message router call, so discard
        # that inherited binding before touching any descriptor-backed state;
        # otherwise the drain could mutate the parent message's ledger before
        # its own _call_router_full() installs a fresh state.
        self._clear_call_state()
        while True:
            batch = await self._curation_queue.get()
            try:
                try:
                    await self._run_curation_turn(batch)
                    self._consecutive_curation_failures = 0
                    # Every recovery ID is inserted into this turn's batch
                    # evidence block, so a successful turn covered all of
                    # them, not only the newly formed IDs.
                    self._curation_recovery_ids.clear()
                    self._last_curation_at = datetime.now(
                        timezone.utc
                    ).isoformat()
                    # Essays are written here, after the turn has committed and
                    # outside its transaction: composition is a minutes-long
                    # LLM call, and holding a SQLite write lock across it would
                    # reintroduce exactly the starvation this loop exists to
                    # avoid.  A failure here is logged, never escalated to a
                    # curation-turn failure — the turn itself already succeeded.
                    await self._generate_curation_essays(batch)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception("curation turn failed")
                    self._record_curation_failure(batch, exc)
            except asyncio.CancelledError:
                raise
            finally:
                for memory_id in tuple(getattr(batch, "memory_ids", ()) or ()):
                    key = str(memory_id)
                    remaining = (
                        self._curation_scheduled_memory_counts.get(key, 0) - 1
                    )
                    if remaining > 0:
                        self._curation_scheduled_memory_counts[key] = remaining
                    else:
                        self._curation_scheduled_memory_counts.pop(key, None)
                self._curation_queue.task_done()
                if self._curation_queue.empty():
                    self._curation_idle.set()

    async def _generate_curation_essays(self, batch: Any = None) -> dict:
        """Write essays for entities this agent activated but never dossiered.

        Runs after a curation turn commits.  The entity set is *queried*, not
        tracked from the turn's activation results, which makes the hook
        idempotent and self-healing: an essay lost to a crashed run, a refused
        write, or a turn that predates this feature is picked up by the next
        turn rather than needing a separate repair batch.

        Bounded to ``entity_self_curation_essays_max_per_turn`` so a large
        backlog (a from-scratch backfill activating dozens of entities) drains
        across turns instead of stalling one turn behind N LLM calls.
        """
        if not getattr(
            self._config, "entity_self_curation_essays_enabled", False
        ):
            return {"skipped": "disabled"}
        mode = self._curation_mode()
        if mode == "off":
            return {"skipped": "curation off"}
        service = self._curation_entity_service()
        if service is None:
            return {"skipped": "no entity service"}
        client = getattr(self._memory, "_llm_client", None) if self._memory else None
        if client is None:
            return {"skipped": "no memory llm client"}
        store = getattr(self._memory, "_store", None)
        con = getattr(store, "_conn", None)
        if con is None:
            return {"skipped": "no store connection"}

        from .llm import estimate_tokens
        from .memory.entity_essays import generate_missing_essays
        from .memory.entities import EntityExecutionContext

        limit = max(
            1,
            int(getattr(
                self._config, "entity_self_curation_essays_max_per_turn", 1,
            ) or 1),
        )
        turn_id = None
        if batch is not None and hasattr(batch, "turn_id"):
            try:
                turn_id = batch.turn_id(self._node_id)
            except Exception:
                turn_id = None
        context = EntityExecutionContext(
            actor_node=self._node_id,
            source_author=self._node_id,
            curation_turn_id=turn_id,
        )
        # "shadow" runs every validator and rolls back, so a dry run can never
        # diverge from the authoritative path.
        validate_only = mode == "shadow"
        try:
            summary = await generate_missing_essays(
                client,
                service,
                con,
                node_id=self._node_id,
                context=context,
                measure=estimate_tokens,
                limit=limit,
                token_budget=int(
                    getattr(self._config, "essay_token_budget", 4000) or 4000
                ),
                reason="essay generated during self-curation for newly "
                       "active entity",
                validate_only=validate_only,
            )
        except Exception:
            logger.exception("curation essay generation failed")
            return {"error": "exception"}

        for item in summary.get("failed", ()):
            logger.warning(
                "curation essay refused for %s: %s",
                item.get("entity_key"), item.get("error"),
            )
        if summary.get("written"):
            logger.info(
                "curation wrote %d essay(s)%s: %s",
                len(summary["written"]),
                " (shadow, rolled back)" if validate_only else "",
                ", ".join(summary["written"]),
            )
        try:
            if summary.get("written") or summary.get("failed"):
                service.record_curation_event(
                    "curation_essays",
                    reason="essay generation during curation turn",
                    run_key=turn_id,
                    details={
                        "written": list(summary.get("written", ())),
                        "failed": [
                            {"entity_key": f.get("entity_key"),
                             "error": f.get("error")}
                            for f in summary.get("failed", ())
                        ],
                        "shadow": validate_only,
                    },
                )
        except Exception:
            logger.debug("could not record curation_essays event", exc_info=True)
        return summary

    def _record_curation_failure(self, batch: Any, exc: BaseException) -> None:
        """Log the failure, keep the memory IDs for the next turn's evidence."""
        memory_ids = tuple(getattr(batch, "memory_ids", ()) or ())
        self._consecutive_curation_failures += 1
        self._last_failed_curation_memory_ids = memory_ids
        for item in memory_ids:
            if item not in self._curation_recovery_ids:
                self._curation_recovery_ids.append(item)
        threshold = int(
            getattr(self._config, "curation_failure_alert_threshold", 5) or 5
        )
        if self._consecutive_curation_failures >= threshold:
            logger.error(
                "self-curation has failed on %d consecutive batches "
                "(latest: %s: %s) — the digest is no longer being maintained",
                self._consecutive_curation_failures,
                type(exc).__name__,
                exc,
            )
        store = getattr(self._memory, "_store", None) if self._memory else None
        if store is None:
            return
        try:
            from .memory.entities import EntityService

            service = EntityService(
                store._conn,
                actor_node=self._node_id,
                mutations_enabled=True,
            )
            service.record_curation_event(
                "curation_turn_failed",
                reason=f"curation turn failed: {type(exc).__name__}: {exc}"[:500],
                run_key=batch.turn_id(self._node_id)
                if hasattr(batch, "turn_id")
                else None,
                details={
                    "memory_ids": list(memory_ids),
                    "consecutive_failures": self._consecutive_curation_failures,
                    "exception_class": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
        except Exception:
            logger.debug("could not record curation failure event", exc_info=True)

    async def wait_for_curation_idle(
        self, timeout: float | None = 60.0,
    ) -> bool:
        """Await an empty queue with no turn in flight.  True when drained."""
        if self._curation_queue.empty() and self._curation_idle.is_set():
            return True
        self._ensure_curation_drain_task()
        try:
            if timeout is None:
                await self._curation_idle.wait()
            else:
                await asyncio.wait_for(
                    self._curation_idle.wait(), timeout=timeout
                )
        except asyncio.TimeoutError:
            logger.warning(
                "curation drain timed out after %.1fs with %d batch(es) queued",
                timeout,
                self._curation_queue.qsize(),
            )
            return False
        return True

    async def shutdown_curation(
        self, timeout: float | None = 60.0,
    ) -> bool:
        """Drain then cancel the drain task; used by AgentNode.disconnect()."""
        drained = await self.wait_for_curation_idle(timeout=timeout)
        if not drained:
            # Do not cancel an accepted batch merely because an optional
            # caller timeout elapsed.  Graceful AgentNode shutdown passes
            # ``None`` and therefore always drains before dependencies close.
            return False
        task = self._curation_drain_task
        self._curation_drain_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        return True

    # ── Phase 3 agent-driven backfill (§9, "Phase 3") ─────────────

    def _curation_mode(self) -> str:
        return str(
            getattr(self._config, "entity_self_curation_mode", "off") or "off"
        )

    def _curated_memory_ids(self) -> set[str]:
        """Every memory ID a completed curation turn has already covered.

        Read from the ``curation_turn`` audit rows rather than a second cursor:
        the events table is the artifact the turn already writes, so there is
        no new state to keep consistent.  ``curation_turn_failed`` rows are
        deliberately excluded — a failed turn leaves its memories uncurated,
        and backfill is the specified repair path for them (§10.2).
        """
        store = getattr(self._memory, "_store", None) if self._memory else None
        conn = getattr(store, "_conn", None) if store is not None else None
        if conn is None:
            return set()
        covered: set[str] = set()
        try:
            rows = conn.execute(
                "SELECT details_json FROM entity_events "
                "WHERE event_type = 'curation_turn'"
            ).fetchall()
        except Exception:
            logger.debug("curated-id scan failed", exc_info=True)
            return covered
        for (payload,) in rows:
            try:
                details = json.loads(payload or "{}")
            except (TypeError, ValueError):
                continue
            # A shadow turn only records what it *would* mutate.  It is a
            # completed shadow audit pass, but it must not prevent a later
            # write-mode backfill from applying those mutations for real.
            # Rows from before the mode field existed are treated as write for
            # compatibility with already-deployed curation audit records.
            event_mode = str((details or {}).get("mode") or "write")
            if self._curation_mode() == "write" and event_mode == "shadow":
                continue
            for memory_id in (details or {}).get("memory_ids") or ():
                if memory_id:
                    covered.add(str(memory_id))
        return covered

    def _backfill_candidate_rows(self) -> list[tuple[str, str]]:
        """Every curatable memory, oldest-first, as ``(id, created_at)``."""
        store = getattr(self._memory, "_store", None) if self._memory else None
        conn = getattr(store, "_conn", None) if store is not None else None
        if conn is None:
            return []
        from .memory.curation import BACKFILL_EXCLUDED_FORMATION_SOURCES

        excluded = tuple(sorted(BACKFILL_EXCLUDED_FORMATION_SOURCES))
        placeholders = ", ".join("?" for _ in excluded) or "''"
        try:
            rows = conn.execute(
                "SELECT id, created_at FROM memories "
                f"WHERE COALESCE(formation_source, '') NOT IN ({placeholders}) "
                "ORDER BY created_at ASC, id ASC",
                excluded,
            ).fetchall()
        except Exception:
            logger.debug("backfill candidate scan failed", exc_info=True)
            return []
        return [(str(row[0]), str(row[1] or "")) for row in rows]

    def plan_curation_backfill(
        self, max_batches: int | None = None,
    ) -> list[Any]:
        """Return the bounded oldest-first slices this invocation would run."""
        from .memory.curation import slice_backfill_batches

        ceiling = int(
            getattr(
                self._config,
                "entity_self_curation_backfill_max_batches",
                50,
            ) or 50
        )
        requested = ceiling if max_batches is None else int(max_batches)
        if requested < 1:
            return []
        return slice_backfill_batches(
            self._backfill_candidate_rows(),
            curated_ids=(
                self._curated_memory_ids()
                | set(self._curation_scheduled_memory_counts)
            ),
            slice_size=int(
                getattr(
                    self._config,
                    "entity_self_curation_backfill_slice_size",
                    10,
                ) or 10
            ),
            max_batches=min(requested, ceiling),
        )

    def enqueue_curation_backfill(
        self, max_batches: int | None = None,
    ) -> dict[str, Any]:
        """Queue bounded backfill slices on the existing curation FIFO.

        Deliberately synchronous and non-awaiting, exactly like
        ``enqueue_curation_batch``.  The caller is normally *inside* a router
        turn holding ``_router_turn_lock``; it must enqueue and return promptly
        rather than turn a message into a backfill wait.  Each slice then runs
        as an ordinary internal curation turn alongside message processing —
        backfill inherits the drain loop's FIFO ordering, failure recording,
        shutdown drain, and shadow/write authority without adding machinery.
        """
        mode = self._curation_mode()
        ceiling = int(
            getattr(
                self._config, "entity_self_curation_backfill_max_batches", 50,
            ) or 50
        )
        result: dict[str, Any] = {
            "status": "empty",
            "mode": mode,
            "queued": 0,
            "memory_ids": 0,
            "turn_ids": [],
            "slice_size": int(
                getattr(
                    self._config,
                    "entity_self_curation_backfill_slice_size",
                    10,
                ) or 10
            ),
            "max_batches": ceiling if max_batches is None else min(
                int(max_batches), ceiling,
            ),
        }
        if mode == "off":
            result["status"] = "disabled"
            result["detail"] = (
                "entity self-curation is off; backfill has no pipeline to run on"
            )
            return result
        # Counts every invocation that got past the enrolment gate, including
        # the no-op that finds the backlog already drained.
        self._curation_backfill_runs += 1
        batches = self.plan_curation_backfill(max_batches)
        if not batches:
            result["detail"] = "no uncurated formation batches found"
            return result
        queued = 0
        for batch in batches:
            if not self.enqueue_curation_batch(batch):
                break
            queued += 1
            result["turn_ids"].append(batch.turn_id(self._node_id))
            result["memory_ids"] += len(batch.memory_ids)
        self._curation_backfill_slices_queued += queued
        result["queued"] = queued
        result["status"] = "queued" if queued else "empty"
        logger.info(
            "curation backfill queued %d slice(s)/%d memories for %s (%s mode)",
            queued,
            result["memory_ids"],
            self._node_id,
            mode,
        )
        return result

    def curation_status(self) -> dict[str, Any]:
        """Expose curation health for ``agent_status`` (§10.3)."""
        return {
            "curation_mode": self._curation_mode(),
            "curation_backfill_runs": self._curation_backfill_runs,
            "curation_backfill_slices_queued": (
                self._curation_backfill_slices_queued
            ),
            "curation_queue_depth": self._curation_queue.qsize(),
            "curation_batches_seen": self._curation_batches_seen,
            "curation_turns_started": self._curation_turn_sequence,
            "last_curation_at": self._last_curation_at,
            "consecutive_curation_failures": self._consecutive_curation_failures,
            "last_failed_curation_memory_ids": list(
                self._last_failed_curation_memory_ids
            ),
            "pending_curation_recovery_ids": list(self._curation_recovery_ids),
            "last_rejections": [
                dict(item) for item in self._last_completed_curation_rejections[
                    :CURATION_STATUS_REJECTION_CAP
                ]
            ],
            "last_tokens_in": self._last_curation_tokens_in,
            "last_tokens_out": self._last_curation_tokens_out,
            # Over-ceiling additions still owed (T-001).  A number that only
            # grows is the signal that an artifact has no headroom left and
            # compaction itself needs attention, not the write path.
            "pending_additions": self._pending_addition_count(),
        }

    def _pending_addition_count(self) -> int:
        """Depth of the durable pending-additions queue, or ``0`` (T-001)."""
        store = getattr(self._memory, "_store", None) if self._memory else None
        connection = getattr(store, "_conn", None)
        if connection is None:
            return 0
        try:
            from .memory.pending_additions import PendingAdditionLedger

            return PendingAdditionLedger(connection).pending_count()
        except Exception:
            logger.debug("pending-addition count unavailable", exc_info=True)
            return 0

    def _curation_groups_enabled(self) -> bool:
        return bool(
            getattr(self._config, "entity_self_curation_groups_enabled", False)
        )

    def _curation_entity_service(self):
        store = getattr(self._memory, "_store", None) if self._memory else None
        if store is None:
            return None
        from .memory.entities import EntityService

        return EntityService(
            store._conn,
            actor_node=self._node_id,
            activation_window_threshold=int(
                getattr(self._config, "entity_activation_window_threshold", 3) or 3
            ),
            active_entity_cap=int(
                getattr(self._config, "entity_registry_injection_cap", 1000) or 1000
            ),
            mutations_enabled=True,
        )

    def _render_curation_instruction(self, batch: Any) -> tuple[str, str]:
        """Return ``(instruction, batch_block)`` for one curation turn."""
        from .llm import estimate_tokens, _encoder
        from .memory import curation as curation_mod

        service = self._curation_entity_service()
        store = getattr(self._memory, "_store", None) if self._memory else None

        rows: list[dict[str, Any]] = []
        for memory_id in getattr(batch, "memory_ids", ()) or ():
            row: dict[str, Any] = {"id": memory_id}
            if store is not None:
                try:
                    record = store._conn.execute(
                        "SELECT retrieval_key, digest_candidate FROM memories "
                        "WHERE id = ?",
                        (memory_id,),
                    ).fetchone()
                except Exception:
                    record = None
                if record is not None:
                    row["retrieval_key"] = record[0] or ""
                    row["digest_candidate"] = bool(record[1])
            if service is not None:
                try:
                    row["entities"] = [
                        link["entity_key"]
                        for link in service.links_for_memory(memory_id)
                    ]
                except Exception:
                    row["entities"] = []
            rows.append(row)

        batch_block = curation_mod.render_batch_block(
            batch, rows, prior_failed_ids=self._curation_recovery_ids,
        )

        registry_block = "(entity registry unavailable)"
        if service is not None:
            try:
                injection = service.serialize_registry_for_injection(
                    int(getattr(self._config, "entity_registry_injection_cap", 1000)
                        or 1000)
                )
                registry_block = (
                    f"Active entity registry "
                    f"({injection.candidates_injected} entities):\n"
                    f"{injection.payload}"
                )
            except Exception:
                logger.debug("registry injection failed", exc_info=True)

        digest_text = self._standing_digest_block_raw()
        digest_ceiling = int(
            getattr(self._config, "standing_digest_budget_tokens", 32000) or 32000
        )
        dossier_ceiling = int(getattr(self._config, "essay_token_budget", 4000) or 4000)
        dossiers: list[tuple[str, int]] = []
        degraded: list[str] = []
        stale: list[str] = []
        if store is not None:
            try:
                for key, body in store._conn.execute(
                    "SELECT entity_key, body FROM essays"
                ).fetchall():
                    dossiers.append((key, estimate_tokens(body or "")))
            except Exception:
                logger.debug("dossier measurement failed", exc_info=True)
        if service is not None and self._curation_groups_enabled():
            degraded, stale = self._curation_group_reports(service)

        budgets_block = curation_mod.render_budgets_block(
            digest_tokens=estimate_tokens(digest_text),
            digest_ceiling=digest_ceiling,
            dossiers=dossiers,
            dossier_ceiling=dossier_ceiling,
            degraded_groups=degraded,
            stale_groups=stale,
            tokenizer_ok=_encoder is not None,
        )

        pending_block = self._render_curation_pending_block(store)

        template = curation_mod.load_update_template()
        instruction = curation_mod.render_update_instruction(
            template,
            batch_block=batch_block,
            registry_block=registry_block,
            budgets_block=budgets_block,
            pending_block=pending_block,
            groups_block=(
                curation_mod.GROUPS_INSTRUCTION_BLOCK
                if self._curation_groups_enabled()
                else ""
            ),
        )
        return instruction, batch_block

    def _render_curation_pending_block(self, store: Any) -> str:
        """Drain block for additions still owed from an earlier turn (T-001).

        Read-only and oldest-first.  Rendering is where a queued addition
        becomes visible again, so this is also where the offer counter is
        bumped: a row offered many times and still pending says something true
        about the artifact that no other signal carries.

        Returns an empty string on any failure.  A drain block that cannot be
        built must not take the curation turn down with it — the queue is
        durable, so the additions simply wait for the next turn.
        """
        connection = getattr(store, "_conn", None)
        if connection is None:
            return ""
        try:
            from .memory.pending_additions import (
                DRAIN_RENDER_CAP,
                PendingAdditionLedger,
                render_pending_block,
            )

            ledger = PendingAdditionLedger(connection, agent=self._node_id)
            total = ledger.pending_count()
            if not total:
                return ""
            additions = ledger.pending(limit=DRAIN_RENDER_CAP)
            block = render_pending_block(additions, total_pending=total)
            ledger.note_offered([item.rowid for item in additions])
            logger.info(
                "curation drain: offering %d of %d pending addition(s)",
                len(additions), total,
            )
            return block
        except Exception:
            logger.debug("pending-addition drain block failed", exc_info=True)
            return ""

    def _curation_group_reports(self, service) -> tuple[list[str], list[str]]:
        """Deterministic group reconciliation plus degraded/stale reporting."""
        degraded: list[str] = []
        stale: list[str] = []
        try:
            rows = service.connection.execute(
                "SELECT entity_key, status FROM entities "
                "WHERE entity_type = 'group' AND status <> 'retired' "
                "ORDER BY entity_key"
            ).fetchall()
        except Exception:
            return degraded, stale
        threshold = int(
            getattr(self._config, "curation_stale_group_batches", 50) or 50
        )
        for group_key, status in rows:
            try:
                write_mode = (
                    getattr(
                        self._config, "entity_self_curation_mode", "off"
                    )
                    == "write"
                )
                if status == "active":
                    self._curation_group_bridge_state.pop(group_key, None)
                    if write_mode:
                        report = service.reconcile_group_membership(
                            group_key,
                            reason="curation turn reconciliation",
                            token_budget=int(
                                getattr(
                                    self._config,
                                    "essay_token_budget",
                                    4000,
                                )
                                or 4000
                            ),
                            measure=estimate_tokens,
                        )
                        if report.get("roster_error"):
                            logger.warning(
                                "group roster reconciliation failed for %s: %s",
                                group_key,
                                report["roster_error"],
                            )
                        active_members = report["active_members"]
                    else:
                        active_members = service.active_group_member_count(
                            group_key
                        )
                    if active_members < 2:
                        degraded.append(group_key)
                else:
                    if write_mode:
                        activation = service.activate_group_if_eligible(
                            group_key,
                            reason="curation turn activation gate",
                        )
                    else:
                        activation = service.group_activation_report(group_key)
                        activation["activated"] = False
                    if write_mode and activation.get("activated"):
                        self._curation_group_bridge_state.pop(group_key, None)
                    else:
                        bridge_count = len(
                            activation.get("bridge_windows") or ()
                        )
                        previous = self._curation_group_bridge_state.get(
                            group_key
                        )
                        if previous is None or bridge_count > previous[0]:
                            last_growth = self._curation_turn_sequence
                        else:
                            last_growth = previous[1]
                        self._curation_group_bridge_state[group_key] = (
                            bridge_count,
                            last_growth,
                        )
                        if (
                            self._curation_turn_sequence - last_growth
                            >= threshold
                        ):
                            stale.append(group_key)
            except Exception:
                logger.debug(
                    "group reconciliation failed for %s", group_key, exc_info=True
                )
        return degraded, stale

    def _standing_digest_block_raw(self) -> str:
        """Read the digest file directly (no XML wrapper), or "" if absent."""
        path = os.path.expanduser(
            getattr(self._config, "standing_digest_path", "") or ""
        )
        from .digest_io import read_digest_or_empty

        return read_digest_or_empty(path)

    async def _run_curation_turn(self, batch: Any) -> str:
        """One internal maintenance turn on the agent's own router backend."""
        from .memory import curation as curation_mod

        self._curation_turn_sequence += 1
        turn_id = batch.turn_id(self._node_id)
        instruction, batch_block = self._render_curation_instruction(batch)
        tool_names = curation_mod.curation_tool_names(
            groups_enabled=self._curation_groups_enabled(),
        )
        synthetic = Message(
            type=MessageType.MESSAGE,
            from_node=self._node_id,
            to_node=self._node_id,
            content=batch_block,
            id=turn_id,
            metadata={
                "internal_curation": True,
                "curation_reason": getattr(batch, "reason", ""),
                "curation_memory_ids": list(getattr(batch, "memory_ids", ()) or ()),
            },
        )
        if self._curation_turn_hook is not None:
            # Test/instrumentation seam; never used in production paths.
            result = self._curation_turn_hook(batch)
            if inspect.isawaitable(result):
                await result
        response = await self._call_router_full(
            synthetic,
            tool_filter=tool_names,
            instructions_override=instruction,
            execution_scope_kind="curation",
            internal_turn=True,
        )
        self._record_curation_turn(batch, turn_id, response)
        return response or ""

    def _record_curation_turn(self, batch: Any, turn_id: str, response: str) -> None:
        # ``_last_router_call_usage`` is the live cumulative dict the router
        # tool loop accumulates across every iteration of this turn, so it is
        # the whole turn's cost, not the last LLM call's.
        usage = getattr(self, "_last_router_call_usage", None) or {}
        tokens_in = int(usage.get("input_tokens", 0) or 0)
        tokens_out = int(usage.get("output_tokens", 0) or 0)
        self._last_curation_tokens_in = tokens_in
        self._last_curation_tokens_out = tokens_out
        rejections = list(self._last_curation_rejections or [])
        # Publish to a durable instance field: the live list above is
        # task-local to this curation turn, but status queries run on another
        # task and after the turn has ended.
        self._last_completed_curation_rejections = rejections

        service = self._curation_entity_service()
        if service is None:
            return
        try:
            service.record_curation_event(
                "curation_turn",
                reason="curation turn completed",
                run_key=turn_id,
                details={
                    "formation_reason": getattr(batch, "reason", ""),
                    "memory_ids": list(getattr(batch, "memory_ids", ()) or ()),
                    "mode": self._curation_mode(),
                    "tool_calls": [
                        name for name, _ in (self._last_router_call_tools or [])
                    ],
                    "rejections": rejections,
                    "write_attempts": dict(
                        getattr(self, "_last_curation_write_summary", None) or {}
                    ),
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "llm_calls": int(usage.get("llm_calls", 0) or 0),
                    "response_chars": len(response or ""),
                },
            )
        except Exception:
            logger.debug("could not record curation_turn event", exc_info=True)

    async def _call_router_full(
        self,
        msg: Message,
        busy: bool = False,
        watchdog: bool = False,
        worker_id: str | None = None,
        pending_trigger: Message | None = None,
        worker_start_time: float | None = None,
        tool_filter: frozenset[str] | None = None,
        instructions_override: str | None = None,
        monitor_mode: bool = False,
        llm_client: "LLMClient | None" = None,
        execution_scope_kind: str = "router",
        internal_turn: bool = False,
    ) -> str:
        """Build the router prompt and delegate to the tool loop.

        IMPORTANT: We do NOT use _build_router_prompt() here because it produces
        a self-contained prompt (with history, tools, identity, instructions all
        embedded). The full router uses complete_with_tools() which adds those
        components via format_history_xml(). Using both would duplicate everything.

        Instead, we build only the system_prompt (agent prompt + memory) and pass
        instructions separately through the callback chain.

        When watchdog=True, the watchdog instruction template is used instead of
        the BUSY template, and tool_names=[] / max_iters=1 are passed to produce
        a single non-agentic assessment.
        """
        # Capture the original trigger's sender for use by worker-launch tools.
        # Without this, synthesised triggers use self._node_id as from_node,
        # which makes BUSY status messages misleading ("from agent:sysadmin:bob"
        # instead of "from user:operator").
        # Install this call's private state on the current asyncio task.  It
        # carries the trigger nodes, the tool ledger, usage, the delivery flag
        # and the worker-launch guards, so a curation turn running at the same
        # time on its own task cannot reset any of them out from under us.
        self._init_call_state(msg)
        # Bug 9: also stash on a per-task contextvar so a concurrent router
        # call can't clobber the destination this call will later read back.
        _CC_TRIGGER_CTX.set((msg.from_node, msg.to_node))

        # Resolve the effective backend/tool contract before rendering the
        # instructions.  The prompt must describe only dispatch mechanisms the
        # current call can actually execute.  Harness backends (including both
        # Claude Code and Codex) use the proven <dispatch_worker> response path;
        # their own internal subagents remain available inside leaf workers.
        selected_llm_client = llm_client or self._llm_client
        router_backend = (
            selected_llm_client.config.backend if selected_llm_client else ""
        )
        is_harness = router_backend in HARNESS_BACKENDS
        if tool_filter is not None:
            effective_filter = set(tool_filter)
            if is_harness:
                effective_filter.difference_update(WORKER_ROUTER_TOOLS)
            resolved_tool_names = sorted(effective_filter)
            policy = getattr(self, "_isolation_policy", None)
            if policy is not None and getattr(policy, "enabled", False):
                resolved_tool_names = self._filter_router_tools(
                    resolved_tool_names
                )
        elif watchdog:
            resolved_tool_names = []
        elif is_harness:
            resolved_tool_names = [
                n for n in self._router_tool_names
                if n not in WORKER_ROUTER_TOOLS
            ]
        else:
            resolved_tool_names = list(self._router_tool_names)
        worker_launch_offered = "worker_launch" in resolved_tool_names
        worker_dispatch_instructions = (
            _WORKER_DISPATCH_TOOL_INSTRUCTIONS.format(
                worker_backend_instructions=self._worker_backend_instructions()
            )
            if worker_launch_offered
            else _WORKER_DISPATCH_BLOCK_INSTRUCTIONS.format(
                worker_backend_instructions=self._worker_backend_instructions()
            )
        )
        fixed_tool_instructions = self._fixed_tool_instructions(resolved_tool_names)

        if instructions_override:
            instructions = instructions_override
        elif busy or watchdog:
            elapsed = 0.0
            if worker_start_time:
                elapsed = time.monotonic() - worker_start_time
            pending_task_summary = self._summarize_trigger(pending_trigger)

            # Bug 10: when BUSY is caused by a CC session (not a worker), use a
            # CC-aware template instead of the worker one — which otherwise
            # claims "A worker (worker) is executing" with a (none) task and
            # offers <dispatch_worker>, which would start a worker in parallel
            # with the live session.
            _worker_active = (
                self._worker_task is not None and not self._worker_task.done()
            )
            cc_busy = bool(self._cc_mgr._cc_tmux_session) and not _worker_active

            # Select instruction template based on mode
            if watchdog:
                instructions = ROUTER_INSTRUCTIONS_WATCHDOG.format(
                    nickname=self._nickname,
                    agent_type=self._agent_type,
                    worker_id=worker_id or "worker",
                    pending_task_summary=pending_task_summary,
                    elapsed=elapsed,
                )
            elif cc_busy:
                cc_elapsed = 0.0
                if self._cc_mgr._cc_last_task_time:
                    cc_elapsed = max(0.0, time.time() - self._cc_mgr._cc_last_task_time)
                busy_cc_template = (
                    ROUTER_INSTRUCTIONS_BUSY_CC_HARNESS
                    if is_harness else ROUTER_INSTRUCTIONS_BUSY_CC
                )
                instructions = busy_cc_template.format(
                    nickname=self._nickname,
                    agent_type=self._agent_type,
                    cc_task=self._cc_mgr._cc_session_task or "(no task description)",
                    cc_session=self._cc_mgr._cc_tmux_session or "(unknown)",
                    elapsed=cc_elapsed,
                )
            else:
                busy_template = (
                    ROUTER_INSTRUCTIONS_BUSY_FULL_HARNESS
                    if is_harness else ROUTER_INSTRUCTIONS_BUSY_FULL
                )
                instructions = busy_template.format(
                    nickname=self._nickname,
                    agent_type=self._agent_type,
                    worker_id=worker_id or "worker",
                    pending_task_summary=pending_task_summary,
                    elapsed=elapsed,
                )

            # Build worker activity (shared between BUSY and watchdog). Skipped
            # for the CC-busy case — its activity lives in [CC Tool Activity]
            # heartbeat turns already in history, and there is no worker.
            if not cc_busy:
                activity_lines = self._build_worker_activity_lines(worker_id=worker_id)
                if activity_lines:
                    instructions += (
                        "\n\n─── CURRENT WORKER ACTIVITY ───\n\n"
                        + "\n".join(activity_lines)
                    )
        else:
            full_template = (
                ROUTER_INSTRUCTIONS_FULL_HARNESS
                if is_harness else ROUTER_INSTRUCTIONS_FULL
            )
            instructions = full_template.format(
                nickname=self._nickname,
                agent_type=self._agent_type,
                worker_dispatch_instructions=worker_dispatch_instructions,
            )
            if fixed_tool_instructions:
                instructions += "\n\n" + fixed_tool_instructions
            if "skill_draft" in resolved_tool_names:
                instructions += "\n\n" + SKILL_DRAFT_ROUTER_GUIDANCE
            if not instructions_override:
                instructions += "\n\n" + CONVERSATION_NOTES_GUIDANCE
            # v2: append retrieval, dispatch criteria, and self-check sections
            if self._memory and isinstance(self._memory, MemorySystemV2):
                instructions += _V2_FULL_ADDITIONS

        # CC interactive session guidance (appended to both IDLE and BUSY)
        # Skip when using instructions_override (e.g., CC monitor mode has
        # its own scoped instructions and must not see the full CC guidance).
        if self._cc_interactive_enabled and not instructions_override:
            instructions += self._cc_mgr.INTERACTIVE_INSTRUCTIONS
        if self._harness_session_enabled and not instructions_override:
            instructions += self._harness_session_mgr.INTERACTIVE_INSTRUCTIONS

        # Harness routers receive long histories that may contain obsolete
        # failed-launcher narratives. Repeat the backend-neutral contract at
        # the very end of the router instructions so it is adjacent to the
        # current trigger in format_history_xml(). Native harness-session mode
        # intentionally hard-gates <dispatch_worker>, so do not advertise it
        # there.
        if (
            is_harness
            and not worker_launch_offered
            and not busy
            and not watchdog
            and not instructions_override
            and not self._harness_session_enabled
        ):
            instructions += "\n\n" + _HARNESS_DISPATCH_RECENCY_REMINDER

        # Keep immutable/slow-changing context in the system prefix and pass
        # query-dependent context separately.  complete_with_tools() positions
        # the tail after durable history, allowing vLLM to reuse the history
        # prefix on the next router call.
        stable_parts, dynamic_parts = await self._build_router_context_blocks(
            trigger_msg=msg,
        )
        system_prompt = "\n\n".join(stable_parts)
        context_tail = "\n\n".join(dynamic_parts)

        # Every report-as-trigger is a fresh reasoning task after worker cleanup,
        # not the tail of the turn that dispatched the worker.  Give it the full
        # configured router budget explicitly.  This also prevents a harness
        # router's normal one-outer-call optimization from presenting a worker
        # report under a misleading "Turn 1 of 1 / FINAL" instruction; harness
        # backends will still normally terminate naturally after their first
        # internally-agentic call.
        metadata = msg.metadata if isinstance(msg.metadata, dict) else {}
        is_report_trigger = bool(metadata.get("worker_report"))
        configured_max_iters = min(
            REACT_MAX_ITERS,
            self._config.router_max_iters,
        )

        # Harness backends (Codex, Claude Code, mesh-harness) normally finish
        # in their own ReAct/TAOR loop: a natural-text reply exits the outer
        # loop after its first call.  But when a harness response falls back to
        # a salvageable Mesh tool call, the outer loop must have the configured
        # budget to feed that result back.  Giving it the configured cap does
        # not add calls to the normal natural-text path; it only prevents a
        # one-call forced-synthesis truncation on that fallback path.
        # Direct backends use the same Mesh ReAct loop for native router tools.
        if watchdog:
            effective_max_iters = 1
        elif is_report_trigger:
            effective_max_iters = configured_max_iters
        elif is_harness:
            effective_max_iters = configured_max_iters
        else:
            effective_max_iters = configured_max_iters
            # Bug 6: a CC monitor delivery should resolve in a few iterations
            # (read screen → relay / continue / stop). Cap it low so a stuck
            # model can't burn the full 30-iteration budget on sleep loops.
            if monitor_mode:
                effective_max_iters = min(effective_max_iters, 5)

        kwargs = {
            "trigger_msg": msg,
            "system_prompt": system_prompt,
            "context_tail": context_tail,
            "tool_names": resolved_tool_names,
            "max_iters": effective_max_iters,
            "instructions": instructions,
            "monitor_mode": monitor_mode,
            "is_harness_router": is_harness,
            "dynamic_context_fn": self._worker_slots_xml,
        }
        if llm_client is not None:
            kwargs["llm_client"] = selected_llm_client
        # An internal maintenance turn needs three mechanical changes in the
        # tool loop (§4.3): no send_message synthesis on natural text, the
        # curation allowlist enforced on every iteration, and no persistent
        # history write.  A scope kind other than "router" changes the
        # capability the callback registers.
        optional_kwargs = ["dynamic_context_fn"]
        if internal_turn:
            kwargs["internal_turn"] = True
            optional_kwargs.insert(0, "internal_turn")
        if execution_scope_kind != "router":
            kwargs["execution_scope_kind"] = execution_scope_kind
            optional_kwargs.insert(0, "execution_scope_kind")
        while True:
            try:
                return await self._router_process_fn(**kwargs)
            except TypeError as exc:
                dropped = next(
                    (
                        name
                        for name in optional_kwargs
                        if name in kwargs and name in str(exc)
                    ),
                    None,
                )
                if dropped is None:
                    raise
                if dropped in {"internal_turn", "execution_scope_kind"}:
                    # Never silently downgrade a curation turn into an
                    # ordinary router turn: it would write to history and
                    # could synthesize a user-facing send_message.
                    raise
                kwargs.pop(dropped)

    async def _build_router_context_blocks(
        self,
        *,
        trigger_msg: "Any | None" = None,
        memory_profile: Any = None,
    ) -> tuple[list[str], list[str]]:
        """Render router context, split by change frequency.

        Each block keeps its existing XML and text exactly.  The caller places
        the stable prefix before durable history and the dynamic tail after it,
        so retrieval changes do not invalidate the history's cache prefix.
        """
        stable_parts: list[str] = []
        dynamic_parts: list[str] = []

        if self._system_prompt:
            stable_parts.append(self._system_prompt)

        if self._memory:
            personality = self._memory.get_personality()
            if personality:
                stable_parts.append(f"<personality>\n{personality}\n</personality>")

            if isinstance(self._memory, MemorySystemV2):
                use_toc = getattr(
                    self._config, "memory_retrieval_redesign_enabled", False
                )
                map_context = self._get_last_n_turns_text(5)
                digest_block = self._standing_digest_block()
                if digest_block:
                    # The published digest changes only at fold cadence.
                    stable_parts.append(digest_block)
                elif use_toc:
                    query_text = self._latest_user_message or ""
                    toc = await self._memory.build_toc(
                        query_text=query_text,
                        k=getattr(self._config, "memory_toc_size", 30),
                        context_text=map_context,
                    )
                    toc = self._memory.dedup_toc_against_window(toc, self._history)
                    toc_block = self._memory.render_toc_block(
                        toc, injected_ids=self._injected_memory_ids,
                    )
                    if toc_block:
                        dynamic_parts.append(toc_block)
                else:
                    rep_block = await self._memory.render_representative_block()
                    if rep_block:
                        dynamic_parts.append(rep_block)

                map_block = await self._memory.render_relevant_maps_block(map_context)
                if map_block:
                    dynamic_parts.append(map_block)
                log_block = await self._memory.render_recent_log_block()
                if log_block:
                    dynamic_parts.append(log_block)

                # Summary changes only at compaction, so it can extend the
                # cacheable prefix with the standing digest and history.
                summary_block = await self._memory.render_summary_block()
                if summary_block:
                    stable_parts.append(summary_block)
                if self._relevant_context:
                    dynamic_parts.append(
                        f"<relevant_memories>\n{self._relevant_context}\n</relevant_memories>"
                    )
            else:
                # v1 three-slice rendering depends on the latest query.
                profile = memory_profile or self._memory.light_profile
                memory_block = await self._memory.render(
                    profile, query=self._latest_user_message,
                )
                if memory_block:
                    dynamic_parts.append(memory_block)

        skill_index = self._skill_index_block()
        if skill_index:
            stable_parts.append(skill_index)

        conversation_id = self._conversation_id_from_message(trigger_msg)
        todo_context = self._render_todo_context(conversation_id)
        if todo_context:
            dynamic_parts.append(todo_context)
        notes_context = self._render_conversation_notes(conversation_id)
        if notes_context:
            dynamic_parts.append(notes_context)

        # Last in the dynamic tail: the operating mandate sits adjacent to the
        # trigger it governs, and toggling it between turns never invalidates
        # the stable prefix the history cache depends on.
        mandate = self._autonomous_mandate_block(trigger_msg)
        if mandate:
            dynamic_parts.append(mandate)

        return stable_parts, dynamic_parts

    def _autonomous_mandate_block(self, trigger_msg: "Any | None") -> str:
        """Return the controller mandate for an autonomous-session turn.

        Plan §10.1: the mandate is injected only when the agent is enrolled,
        the trigger carries trusted autonomous-session metadata, and the named
        project is one this agent controls.  Ordinary user messages, channel
        messages, and interactive worker reports get an empty string, so an
        enrolled agent still holds ordinary conversations.

        A worker completion inherits ``autonomous_session`` from its dispatch
        trigger (see ``autonomous_completion_metadata``), which is what makes
        the report turn a continuation of the same session rather than a new
        unmandated turn.
        """
        mandate = str(
            getattr(self._config, "autonomous_mandate_prompt", "") or ""
        )
        if not mandate or not self._autonomous_mode_enabled():
            return ""
        metadata = getattr(trigger_msg, "metadata", None)
        if not isinstance(metadata, dict) or not metadata.get(
            "autonomous_session"
        ):
            return ""
        key = str(metadata.get("autonomous_project_key") or "")
        configured = list(
            getattr(self._config, "autonomous_projects", []) or []
        )
        if configured and key not in configured:
            logger.warning(
                "[AUTONOMOUS] Trigger claims session scope %r, which is not in "
                "autonomous_projects %s; withholding the controller mandate.",
                key,
                configured,
            )
            return ""
        if metadata.get("worker_report"):
            continuation = str(
                getattr(
                    self._config,
                    "autonomous_continuation_mandate_prompt",
                    "",
                )
                or ""
            )
            return "\n\n".join(
                (
                    continuation or mandate,
                    self._autonomous_session_plan_block(metadata),
                )
            )
        return mandate

    @staticmethod
    def _extract_autonomous_session_plan(content: str) -> str:
        """Return one schema-valid, bounded SESSION PLAN block from text."""
        lines = str(content or "").splitlines()
        for start, line in enumerate(lines):
            if not _SESSION_PLAN_START_RE.fullmatch(line):
                continue
            plan_lines: list[str] = []
            fields: set[str] = set()
            for candidate in lines[start : start + _SESSION_PLAN_MAX_LINES]:
                plan_lines.append(candidate.strip())
                match = _SESSION_PLAN_FIELD_RE.fullmatch(candidate)
                if match is not None:
                    fields.add(match.group(1).upper())
                    if (
                        match.group(1).upper() == "FIRST"
                        and fields == _SESSION_PLAN_REQUIRED_FIELDS
                    ):
                        plan = "\n".join(plan_lines).strip()
                        return plan if len(plan) <= _SESSION_PLAN_MAX_CHARS else ""
        return ""

    def _session_plan_from_history(self, session_id: str) -> str:
        """Find the latest plan captured for one active autonomous session."""
        for turn in reversed(self._history.window):
            metadata = getattr(turn, "meta", None)
            if not isinstance(metadata, dict):
                continue
            if str(metadata.get("autonomous_session_id") or "").strip() != session_id:
                continue
            plan = self._extract_autonomous_session_plan(
                str(metadata.get("autonomous_session_plan") or "")
            )
            if plan:
                return plan
        return ""

    def _autonomous_session_plan_block(self, metadata: dict[str, Any]) -> str:
        """Render durable plan state for an execute-only continuation prompt."""
        plan = self._extract_autonomous_session_plan(
            str(metadata.get("autonomous_session_plan") or "")
        )
        if plan:
            return "[SESSION PLAN CARRY-FORWARD]\n" + plan
        return (
            "[SESSION PLAN UNAVAILABLE]\n"
            "The report trigger has no valid durable SESSION PLAN. Do not silently "
            "assume or invent the old plan: re-read the dossier, record a replacement "
            "plan in the session report before another dispatch, and explain the gap."
        )

    async def _generate_busy_response(
        self,
        msg: Message,
        worker_id: str | None,
        pending_trigger: Message | None,
        worker_start_time: float | None,
    ) -> str:
        """Use LLM to generate a contextual busy response."""
        elapsed = 0.0
        if worker_start_time:
            elapsed = time.monotonic() - worker_start_time

        pending_summary = self._summarize_trigger(pending_trigger)

        instructions = ROUTER_INSTRUCTIONS_BUSY.format(
            worker_id=worker_id or "worker",
            pending_task_summary=pending_summary,
            elapsed=elapsed,
        )

        prompt = await self._build_router_prompt(instructions, trigger_msg=msg)

        logger.debug(f"RouterV2 calling LLM for busy response")
        response = await self._llm_client.complete(prompt)

        return response.strip()

    async def _build_router_prompt(
        self,
        instructions: str,
        memory_profile: Any = None,
        preferences_block: str = "",
        history_entries: "list[Any] | None" = None,
        include_tools: bool = True,
        max_history_turns: int | None = None,
        trigger_msg: "Any | None" = None,
    ) -> str:
        """Build the full prompt for all modes (router, worker, planner, validator).

        Args:
            instructions: Mode-specific instructions for the <instructions> block.
            memory_profile: Memory rendering profile (defaults to light_profile).
            preferences_block: User preferences XML string (empty if none).
            history_entries: If provided, use these HistoryMessage objects instead
                             of the router's ConversationHistory. Used by the worker
                             to render its snapshot.
            include_tools: Whether to include the tools block (False for classifier).
            max_history_turns: If set, limit history to this many recent turns
                               (preserving summary at index 0 if present).
            trigger_msg: The message that triggered processing. When provided, it is
                         extracted from <history> and rendered as <message_received>
                         between </history> and <instructions>.
        """
        stable_parts, dynamic_parts = await self._build_router_context_blocks(
            trigger_msg=trigger_msg,
            memory_profile=memory_profile,
        )
        parts: list[str] = []

        # Keep the legacy standalone renderer's outer <system> wrapper while
        # reusing the same stability split as the production full-router path.
        if self._system_prompt:
            parts.append(f"<system>\n{self._system_prompt}\n</system>")
            stable_parts = stable_parts[1:]
        if self._identity_block:
            parts.append(self._identity_block)
        parts.extend(stable_parts)

        if include_tools and self._tools_block:
            parts.append(self._tools_block)

        history_xml = self._build_history_xml(
            history_entries=history_entries,
            max_history_turns=max_history_turns,
            trigger_msg=trigger_msg,
        )
        trigger_xml = ""
        if history_xml:
            trigger_marker = "\n\n<message_received "
            if trigger_marker in history_xml:
                history_xml, trigger_xml = history_xml.split(trigger_marker, 1)
                trigger_xml = "<message_received " + trigger_xml
            parts.append(history_xml)

        # Query-dependent context must remain near the actionable trigger, but
        # after the durable history so it cannot break the reusable prefix.
        parts.extend(dynamic_parts)
        if preferences_block:
            parts.append(preferences_block)
        if trigger_xml:
            parts.append(trigger_xml)

        parts.append(f"<instructions>\n{instructions.strip()}\n</instructions>")

        full_prompt = "\n\n".join(parts)
        self._last_prompt_tokens = estimate_tokens(full_prompt)
        return full_prompt

    def _build_history_xml(
        self,
        history_entries: "list[Any] | None" = None,
        max_history_turns: int | None = None,
        trigger_msg: "Any | None" = None,
    ) -> str:
        """Build XML-formatted history from ConversationHistory + worker snapshot + planning peeks.

        When trigger_msg is provided, the matching history entry is extracted from
        <history> and rendered as a separate <message_received> block after </history>.
        This structurally separates context from the actionable message.

        Args:
            history_entries: If provided, use these instead of self._history.build_context_for_llm().
                             Used by the worker to render its snapshot.
            max_history_turns: If set, limit durable history to this many recent turns,
                               preserving the summary message at index 0 if present.
            trigger_msg: The message that triggered processing (extracted and rendered
                         as <message_received> between </history> and <instructions>).
        """
        if history_entries is not None:
            durable = history_entries
        else:
            durable = self._history.build_context_for_llm()

        # Strip tool-call visibility blocks (Contract §5)
        for _msg in durable:
            if isinstance(getattr(_msg, 'content', None), str):
                _msg.content = strip_tools_called_block(_msg.content)

        # Optionally cap history length for slim contexts (e.g., classifier)
        if max_history_turns and len(durable) > max_history_turns:
            has_summary = durable and getattr(durable[0], 'from_node', '') == 'system'
            if has_summary:
                durable = [durable[0]] + durable[-(max_history_turns - 1):]
            else:
                durable = durable[-max_history_turns:]

        # Collect worker progress entries from snapshot (if worker is running)
        worker_progress = self._get_worker_progress()

        # Collect CC live events (in-progress tool calls not yet in any history)
        cc_events = self._cc_events_fn() if self._cc_events_fn and self._worker_snapshot else []

        # Collect harness session events (tool activity from harness workers)
        harness_event_lines = self._harness_events_fn(n=10, label="harness") if self._harness_events_fn else []

        if not durable and not self._ephemeral_peeks and not worker_progress and not cc_events and not harness_event_lines:
            return ""

        # Identify which durable entry is the trigger (scan from end)
        trigger_idx = -1
        if trigger_msg is not None:
            t_from = getattr(trigger_msg, 'from_node', None)
            t_content = getattr(trigger_msg, 'content', None)
            if t_from and t_content:
                for i in range(len(durable) - 1, -1, -1):
                    if getattr(durable[i], 'from_node', '') == t_from and getattr(durable[i], 'content', '') == t_content:
                        trigger_idx = i
                        break

        from .protocol import to_local_display
        lines = ["<history>"]

        # Format durable entries (summary + window) as XML, skipping trigger
        for idx, msg in enumerate(durable):
            if idx == trigger_idx:
                continue
            from_node = msg.from_node or "unknown"
            timestamp = to_local_display(msg.timestamp)
            to_node = msg.to_node or ""

            if to_node:
                lines.append(f'<message from="{from_node}" to="{to_node}" timestamp="{timestamp}">')
            else:
                lines.append(f'<message from="{from_node}" timestamp="{timestamp}">')
            lines.append(msg.content)
            lines.append("</message>")

        # Append worker progress from snapshot (entries the worker has added)
        if worker_progress:
            wid = self._current_worker_id or "worker"
            max_lines = self._config.worker_peek_max_lines
            activity_parts = []
            for turn in worker_progress:
                content = turn.content
                # Truncate long tool outputs
                content_lines = content.split('\n')
                if len(content_lines) > max_lines:
                    content = '\n'.join(content_lines[:max_lines])
                    content += f'\n[... truncated, {len(content_lines)} lines total]'
                label = turn.from_node or turn.role
                activity_parts.append(f"[{label}] {content}")

            if activity_parts:
                lines.append(f'<worker_activity worker="{wid}">')
                lines.append('\n'.join(activity_parts))
                lines.append("</worker_activity>")

        # Append CC live events (in-progress tool calls)
        if cc_events:
            wid = self._current_worker_id or "worker"
            for event_entry in cc_events:
                if hasattr(event_entry, 'message'):
                    content = event_entry.message.content if isinstance(event_entry.message.content, str) else str(event_entry.message.content)
                elif hasattr(event_entry, 'content'):
                    content = event_entry.content if isinstance(event_entry.content, str) else str(event_entry.content)
                else:
                    content = str(event_entry)
                lines.append(f'<worker_activity worker="{wid}" live="true">')
                lines.append(content)
                lines.append("</worker_activity>")

        # Append harness session events (tool activity from harness workers)
        if harness_event_lines:
            wid = self._current_worker_id or "worker"
            lines.append(f'<worker_activity worker="{wid}" live="true" source="harness">')
            lines.append('\n'.join(harness_event_lines))
            lines.append("</worker_activity>")

        # Append ephemeral planning peeks (used by RouterV3 planning pipeline only)
        for peek in self._ephemeral_peeks:
            worker_id = peek.get("worker_id", "worker")
            activity = peek["worker_activity"]
            lines.append(f'<worker_activity worker="{worker_id}">')
            lines.append(activity)
            lines.append("</worker_activity>")

        lines.append("</history>")

        # Render the trigger as <message_received> after </history>
        if trigger_idx >= 0:
            t = durable[trigger_idx]
            t_from = getattr(t, 'from_node', 'unknown') or 'unknown'
            t_ts = to_local_display(getattr(t, 'timestamp', '') or '')
            t_to = getattr(t, 'to_node', '') or ''
            attrs = f'from="{t_from}" timestamp="{t_ts}"'
            if t_to:
                attrs += f' to="{t_to}"'
            lines.append("")
            lines.append(f"<message_received {attrs}>")
            lines.append(t.content)
            lines.append("</message_received>")

        return "\n".join(lines)

    def _get_worker_progress(self, worker_id: str | None = None) -> list[Turn]:
        """Return worker's new entries from the snapshot (entries after dispatch point)."""
        slot = self._slot_for_worker(worker_id)
        snapshot = slot.snapshot if slot is not None else self._worker_snapshot
        snapshot_start = (
            slot.snapshot_start
            if slot is not None else self._worker_snapshot_start
        )
        if not snapshot:
            return []
        return snapshot[snapshot_start:]

    def _build_worker_activity_lines(self, worker_id: str | None = None) -> list[str]:
        """Build worker activity lines from snapshot + CC events + harness events.

        Extracted from the inline code formerly in _call_router_full(busy=True).
        Used by both the BUSY handler and the watchdog tick.
        """
        activity_lines: list[str] = []
        wid = worker_id or "worker"
        max_lines = self._config.worker_peek_max_lines

        slot = self._slot_for_worker(worker_id)
        worker_progress = self._get_worker_progress(worker_id)
        if worker_progress:
            for turn in worker_progress:
                content = turn.content
                content_split = content.split('\n')
                if len(content_split) > max_lines:
                    content = '\n'.join(content_split[:max_lines])
                    content += f'\n[... truncated, {len(content_split)} lines total]'
                label = turn.from_node or turn.role
                activity_lines.append(f"[{label}] {content}")

        if slot is not None and slot.execution_context is not None:
            cc_events = list(
                getattr(slot.execution_context, "current_cc_events", []) or []
            )
        else:
            cc_events = (
                self._cc_events_fn()
                if self._cc_events_fn and self._worker_snapshot else []
            )
        if cc_events:
            for event_entry in cc_events:
                if hasattr(event_entry, 'message'):
                    content = event_entry.message.content if isinstance(event_entry.message.content, str) else str(event_entry.message.content)
                elif hasattr(event_entry, 'content'):
                    content = event_entry.content if isinstance(event_entry.content, str) else str(event_entry.content)
                else:
                    content = str(event_entry)
                activity_lines.append(f"[{wid} live] {content}")

        if self._harness_events_fn and (
            slot is None or self._active_worker_count() <= 1
        ):
            harness_lines = self._harness_events_fn(n=10, label=f"{wid} live")
            activity_lines.extend(harness_lines)

        fixed_status = self._get_fixed_tool_status()
        fixed_status_active = bool(
            fixed_status
            and fixed_status.get("status") in {"starting", "running", "detached"}
        )
        if fixed_status and (
            fixed_status_active or self._current_worker_kind == "fixed_tool"
        ):
            phase = fixed_status.get("current_phase", 0)
            total = fixed_status.get("total_phases", 0)
            status = fixed_status.get("status", "running")
            run_dir = fixed_status.get("run_dir", "")
            activity_lines.append(
                f"[fixed:{fixed_status.get('tool_name', 'tool')}] "
                f"status={status} phase={phase}/{total} run_dir={run_dir}"
            )
            log_tail = str(fixed_status.get("log_tail") or "").strip()
            if log_tail:
                for line in log_tail.splitlines()[-10:]:
                    activity_lines.append(
                        f"[fixed:{fixed_status.get('tool_name', 'tool')} log] {line}"
                    )

        return activity_lines

    # =========================================================================
    # Worker tool handlers (router-instance tools)
    # =========================================================================

    def _init_worker_tool_handlers(self) -> None:
        """Set up per-instance handlers for worker_launch and worker_status.

        These are bound methods on the RouterV2 instance. agent_node.py's
        _execute_all_tools checks this dict BEFORE falling through to the
        global ToolRegistry, so two routers in the same process each see
        their own worker state.

        Skipped when cc_interactive_tools is enabled — CC sessions replace
        the worker concept entirely for these agents.
        """
        if self._cc_interactive_enabled:
            return
        self._worker_tool_handlers["worker_launch"] = self._tool_worker_launch
        self._worker_tool_handlers["worker_list"] = self._tool_worker_list
        self._worker_tool_handlers["worker_status"] = self._tool_worker_status
        self._worker_tool_handlers["worker_cancel"] = self._tool_worker_cancel
        self._worker_tool_handlers["skill_draft"] = self._tool_skill_draft
        if "solicitation_scout" in self._fixed_tools:
            self._worker_tool_handlers["solicitation_scout"] = (
                self._tool_solicitation_scout
            )

    # ── Phase 2A isolation guard ──────────────────────────────────────

    def _filter_router_tools(self, names: "list[str]") -> "list[str]":
        """Offer-time filter for the static router tool list.

        Returns ``names`` unchanged when no policy is installed or the policy
        is disabled, which is every live node today.
        """
        policy = getattr(self, "_isolation_policy", None)
        if policy is None or not getattr(policy, "enabled", False):
            return names
        from .tool_capabilities import filter_tool_names

        permitted = filter_tool_names(names, policy)
        withheld = sorted(set(names) - set(permitted))
        if withheld:
            logger.info(
                "[ISOLATION] router %s: withholding %s",
                self._node_id, ", ".join(withheld),
            )
        return permitted

    def _isolation_refusal(self, tool_name: str) -> "str | None":
        """Execution-time guard for router-owned tool handlers.

        Router-native tools (``worker_launch``, ``skill_draft``, the cc_*/
        harness_* session tools) never reach ``AgentNode``'s registry funnel,
        so they are re-checked here rather than relying on the offer-time
        filter alone.
        """
        policy = getattr(self, "_isolation_policy", None)
        if policy is None or not getattr(policy, "enabled", False):
            return None
        from .tool_capabilities import guard_tool

        refusal = guard_tool(policy, tool_name)
        if refusal is not None:
            logger.warning(
                "[ISOLATION] router %s denied tool '%s'", self._node_id, tool_name,
            )
        return refusal

    def _init_cc_interactive_handlers(self) -> None:
        """Register CC interactive tool handlers (gated by cc_interactive_tools).

        Thin wrappers: the four cc_* tools delegate to CCSessionManager, which
        owns the session state and lifecycle (mesh/cc_session_manager.py)."""
        self._worker_tool_handlers["cc_start_session"] = self._cc_mgr._tool_cc_start_session
        self._worker_tool_handlers["cc_get_screen"] = self._cc_mgr._tool_cc_get_screen
        self._worker_tool_handlers["cc_send_input"] = self._cc_mgr._tool_cc_send_input
        self._worker_tool_handlers["cc_stop_session"] = self._cc_mgr._tool_cc_stop_session

    def _init_harness_session_handlers(self) -> None:
        """Register native harness session tool handlers (gated by
        harness_session_tools). Thin wrappers delegating to HarnessSessionManager,
        which owns session state and lifecycle (mesh/harness_session_manager.py)."""
        m = self._harness_session_mgr
        self._worker_tool_handlers["harness_start_session"] = m._tool_harness_start_session
        self._worker_tool_handlers["harness_send_input"] = m._tool_harness_send_input
        self._worker_tool_handlers["harness_get_status"] = m._tool_harness_get_status
        self._worker_tool_handlers["harness_stop_session"] = m._tool_harness_stop_session

    def _known_worker_backends(self) -> list[str]:
        """Return worker backend names available to this router instance."""
        names = set(getattr(self, "_worker_backend_names", set()) or set())
        agent = getattr(self, "_worker_agent", None)
        getter = getattr(agent, "get_worker_backend_names", None)
        if callable(getter):
            try:
                names.update(getter())
            except Exception:
                logger.exception("Failed to read worker backend names from agent")
        return sorted(n for n in names if n)

    def _configured_worker_task_types(self) -> dict[str, dict[str, Any]]:
        """Return rich task-type definitions with available backends."""
        configured = normalize_worker_task_types(
            getattr(self, "_worker_task_types", {}) or {}
        )
        known = set(self._known_worker_backends())
        invalid = {
            task_type: definition["backend"]
            for task_type, definition in configured.items()
            if definition["backend"] not in known
        }
        if invalid:
            rendered = ", ".join(
                f"{task_type} -> {backend}"
                for task_type, backend in sorted(invalid.items())
            )
            logger.warning(
                "Configured worker_task_types backends are not present in "
                "llm_backends and will be ignored: %s",
                rendered,
            )
        valid: dict[str, dict[str, Any]] = {}
        for task_type, definition in configured.items():
            if definition["backend"] not in known:
                continue
            pev = definition.get("pev")
            if isinstance(pev, PevTaskConfig):
                missing = [
                    backend
                    for backend in (pev.plan, pev.execute, pev.verify)
                    if backend and backend not in known
                ]
                if missing:
                    logger.warning(
                        "Configured PEV backends for worker task type %s are not "
                        "present in llm_backends; PEV is disabled for this type: %s",
                        task_type,
                        ", ".join(missing),
                    )
                    definition = dict(definition)
                    definition.pop("pev", None)
            prompts = definition.get("prompts")
            if (
                isinstance(prompts, TaskPromptConfig)
                and prompts.sync_backend
                and prompts.sync_backend not in known
            ):
                logger.warning(
                    "Configured sync backend for worker task type %s is not "
                    "present in llm_backends; prompt bundle is disabled: %s",
                    task_type,
                    prompts.sync_backend,
                )
                definition = dict(definition)
                definition.pop("prompts", None)
            valid[task_type] = definition
        return valid

    def _fixed_tool_instructions(self, offered_tools: list[str]) -> str:
        """Render the typed fixed-tool handoff contract for the router."""
        offered = [
            self._fixed_tools[name]
            for name in offered_tools
            if name in self._fixed_tools
        ]
        if not offered:
            return ""

        lines = [
            "FIXED TOOLS — typed pipelines that occupy the worker slot:",
        ]
        for tool in offered:
            required = [p.name for p in tool.parameters if p.required]
            optional = [p.name for p in tool.parameters if not p.required]
            signature = ", ".join(
                required + [f"{name}?" for name in optional]
            )
            lines.append(f"- {tool.name}({signature}): {tool.description}")
        lines.extend([
            "Call the typed tool directly when it matches the user's request; "
            "do not launch a general worker to wrap it.",
            "The call returns immediately after launch. It owns the one worker "
            "slot until completion; worker_status reports phase/log progress, "
            "and final artifacts are delivered through the normal worker "
            "completion path.",
        ])
        return "\n".join(lines)

    def _resolve_fixed_tool_input(
        self,
        path_value: str,
        *,
        allow_run_state: bool = False,
    ) -> Path:
        """Resolve and authorize a fixed-tool filesystem argument."""
        from .tool_implementations import get_working_directory

        policy = getattr(self, "_isolation_policy", None)
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            if policy is not None and getattr(policy, "enabled", False):
                path = Path(policy.workspace) / path
            else:
                # Preserve the legacy resolver byte-for-byte while isolation
                # is disabled, including its existing working-directory shape.
                path = Path(get_working_directory()) / path
        resolved = path.resolve()

        if policy is None or not getattr(policy, "enabled", False):
            return resolved
        if not policy.contains(resolved):
            raise PermissionError(
                f"fixed-tool path {resolved} is outside this agent's isolation boundary"
            )
        if allow_run_state:
            state_paths = getattr(self, "_state_paths", None)
            runs_root = getattr(state_paths, "fixed_tool_runs_dir", None)
            from .isolation import is_path_contained

            if runs_root is None or not is_path_contained(resolved, (runs_root,)):
                raise PermissionError(
                    f"fixed-tool run directory {resolved} is not in this agent's run state"
                )
        elif policy.is_protected_state(resolved):
            raise PermissionError(
                f"fixed-tool input {resolved} is protected agent state"
            )
        return resolved

    def _scope_fixed_tool_papers(self, value: str) -> str:
        """Validate local-file members of solicitation-scout's mixed paper list.

        URLs, DOI/arXiv identifiers, and titles remain strings.  Absolute,
        explicitly relative, or existing workspace paths are canonicalized
        through the same boundary as the dedicated file arguments.
        """
        policy = getattr(self, "_isolation_policy", None)
        if policy is None or not getattr(policy, "enabled", False):
            return value

        scoped: list[str] = []
        for raw in (item.strip() for item in str(value or "").split(",")):
            if not raw:
                continue
            if re.match(r"https?://", raw, re.I):
                scoped.append(raw)
                continue
            doi = re.sub(
                r"^(?:doi:|https?://(?:dx\.)?doi\.org/)",
                "",
                raw,
                flags=re.I,
            )
            if re.fullmatch(r"10\.\d{4,9}/\S+", doi, flags=re.I):
                scoped.append(raw)
                continue

            path = Path(raw).expanduser()
            workspace_candidate = Path(policy.workspace) / path
            path_like = (
                path.is_absolute()
                or raw.startswith(("~", "."))
                or "/" in raw
                or "\\" in raw
                or workspace_candidate.exists()
            )
            if path_like:
                scoped.append(str(self._resolve_fixed_tool_input(raw)))
            else:
                scoped.append(raw)
        return ",".join(scoped)

    async def _tool_solicitation_scout(
        self,
        cv: str = "",
        research_threads: str = "",
        pi_papers: str = "",
        project_name: str = "",
        run_dir: str = "",
        dry_run: bool = False,
    ) -> str:
        """Plan, launch, or resume solicitation-scout."""
        tool_name = "solicitation_scout"
        tool = self._fixed_tools.get(tool_name)
        if tool is None:
            return json.dumps({
                "status": "error",
                "message": "solicitation_scout is not configured for this router.",
            })

        resume_value = str(run_dir or "").strip()
        if dry_run and resume_value:
            return json.dumps({
                "status": "error",
                "message": "dry_run and run_dir are mutually exclusive.",
            })

        scoped_pi_papers = str(pi_papers or "").strip()

        # Validate every model-supplied path before a dry-run subprocess or a
        # worker sees it.  Subsequent resolution below is retained for the
        # legacy path and existence/error reporting.
        if getattr(getattr(self, "_isolation_policy", None), "enabled", False):
            for parameter_name, value in (
                ("cv", cv),
                ("research_threads", research_threads),
            ):
                if not str(value or "").strip():
                    continue
                try:
                    self._resolve_fixed_tool_input(str(value))
                except PermissionError as exc:
                    return json.dumps({
                        "status": "error",
                        "parameter": parameter_name,
                        "message": str(exc),
                    })
            if scoped_pi_papers:
                try:
                    scoped_pi_papers = self._scope_fixed_tool_papers(scoped_pi_papers)
                except PermissionError as exc:
                    return json.dumps({
                        "status": "error",
                        "parameter": "pi_papers",
                        "message": str(exc),
                    })
            if resume_value:
                try:
                    self._resolve_fixed_tool_input(
                        resume_value,
                        allow_run_state=True,
                    )
                except PermissionError as exc:
                    return json.dumps({
                        "status": "error",
                        "parameter": "run_dir",
                        "message": str(exc),
                    })

        def append_optional_args(args: list[str]) -> None:
            if scoped_pi_papers:
                args.extend(["--pi-papers", scoped_pi_papers])
            if str(project_name or "").strip():
                args.extend(["--project-name", str(project_name).strip()])

        if dry_run:
            # solicitation-scout requires --output-dir even in dry-run mode,
            # but it deliberately does not create the directory or any files.
            state_paths = getattr(self, "_state_paths", None)
            isolated_tmp = (
                state_paths.tmp_dir
                if (
                    getattr(getattr(self, "_isolation_policy", None), "enabled", False)
                    and state_paths is not None
                )
                else Path(os.environ.get("TMPDIR", "/tmp"))
            )
            dry_run_dir = Path(isolated_tmp) / (
                f"mesh-solicitation-scout-dry-run-{uuid.uuid4().hex}"
            )
            dry_run_args = [
                "--cv",
                str(self._resolve_fixed_tool_input(str(cv))) if str(cv or "").strip() else "",
                "--research-threads",
                (
                    str(self._resolve_fixed_tool_input(str(research_threads)))
                    if str(research_threads or "").strip()
                    else ""
                ),
            ]
            append_optional_args(dry_run_args)
            dry_run_args.extend([
                "--output-dir",
                str(dry_run_dir),
                "--dry-run",
            ])
            command_path = Path(tool.command).expanduser().resolve()
            fixed_cwd = str(command_path.parent)
            fixed_env: dict[str, str] | None = None
            policy = getattr(self, "_isolation_policy", None)
            if getattr(policy, "enabled", False):
                from .isolation import WorkerIsolationScope, assert_cwd_in_scope
                from .llm import _build_subprocess_env

                fixed_scope = WorkerIsolationScope.from_policy(policy)
                fixed_cwd = assert_cwd_in_scope(
                    fixed_scope, fixed_scope.primary_workspace or ""
                )
                fixed_env = _build_subprocess_env()
                fixed_env.update(fixed_scope.to_env())
            try:
                process = await asyncio.create_subprocess_exec(
                    str(command_path),
                    *dry_run_args,
                    cwd=fixed_cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    **({"env": fixed_env} if fixed_env is not None else {}),
                )
                stdout, _ = await asyncio.wait_for(
                    process.communicate(),
                    timeout=120.0,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return json.dumps({
                    "status": "error",
                    "message": "solicitation-scout dry-run exceeded 120 seconds.",
                })
            except OSError as exc:
                return json.dumps({
                    "status": "error",
                    "message": f"Unable to run solicitation-scout dry-run: {exc}",
                })

            output = stdout.decode("utf-8", errors="replace")
            if process.returncode != 0:
                return json.dumps({
                    "status": "error",
                    "exit_code": process.returncode,
                    "message": "solicitation-scout dry-run failed.",
                    "output": output,
                })
            return json.dumps({
                "status": "dry_run",
                "output": output,
            })

        resolved_run_dir: Path | None = None
        resolved_inputs: dict[str, Path] = {}
        if resume_value:
            resolved_run_dir = self._resolve_fixed_tool_input(
                resume_value,
                allow_run_state=(
                    getattr(getattr(self, "_isolation_policy", None), "enabled", False)
                ),
            )
            if not resolved_run_dir.is_dir():
                return json.dumps({
                    "status": "error",
                    "parameter": "run_dir",
                    "path": str(resolved_run_dir),
                    "message": f"Resume directory does not exist: {resolved_run_dir}",
                })
            run_config = resolved_run_dir / "run-config.json"
            if not run_config.is_file():
                return json.dumps({
                    "status": "error",
                    "parameter": "run_dir",
                    "path": str(resolved_run_dir),
                    "message": (
                        "Resume directory has no run-config.json: "
                        f"{resolved_run_dir}"
                    ),
                })
            run_args = ["--resume", str(resolved_run_dir)]
            # The underlying CLI accepts supplied source inputs on resume and
            # checks their hashes against run-config.json.
            for parameter_name, value, flag in (
                ("cv", cv, "--cv"),
                ("research_threads", research_threads, "--research-threads"),
            ):
                if not str(value or "").strip():
                    continue
                resolved = self._resolve_fixed_tool_input(str(value))
                if not resolved.is_file():
                    return json.dumps({
                        "status": "error",
                        "parameter": parameter_name,
                        "path": str(resolved),
                        "message": f"Input file does not exist: {resolved}",
                    })
                run_args.extend([flag, str(resolved)])
        else:
            for parameter_name, value in (
                ("cv", cv),
                ("research_threads", research_threads),
            ):
                if not str(value or "").strip():
                    return json.dumps({
                        "status": "error",
                        "parameter": parameter_name,
                        "message": (
                            f"Required parameter {parameter_name!r} is empty "
                            "for a new run."
                        ),
                    })
                resolved = self._resolve_fixed_tool_input(str(value))
                if not resolved.is_file():
                    return json.dumps({
                        "status": "error",
                        "parameter": parameter_name,
                        "path": str(resolved),
                        "message": f"Input file does not exist: {resolved}",
                    })
                resolved_inputs[parameter_name] = resolved
            run_args = [
                "--cv", str(resolved_inputs["cv"]),
                "--research-threads", str(resolved_inputs["research_threads"]),
            ]
        append_optional_args(run_args)

        trigger_from, trigger_to = self._trigger_nodes()
        if resolved_run_dir is not None:
            trigger_content = f"Resume solicitation-scout run {resolved_run_dir}."
        else:
            trigger_content = (
                "Run solicitation-scout for CV "
                f"{resolved_inputs['cv']} and research threads "
                f"{resolved_inputs['research_threads']}."
            )
        trigger = Message(
            id=f"fixed-{uuid.uuid4().hex[:8]}",
            from_node=trigger_from or self._node_id,
            to_node=trigger_to or self._node_id,
            type=MessageType.MESSAGE,
            content=trigger_content,
            metadata={
                "fixed_tool": tool_name,
                "fixed_tool_args": run_args,
                "fixed_tool_project_name": str(project_name or "").strip(),
                "fixed_tool_resume": resolved_run_dir is not None,
                "fixed_tool_run_dir": (
                    str(resolved_run_dir) if resolved_run_dir is not None else ""
                ),
            },
        )

        previous_task = self._current_task_description
        self._current_task_description = trigger.content
        launched = await self._start_worker(trigger)
        if not launched:
            self._current_task_description = previous_task
            fixed_status = self._get_fixed_tool_status()
            return json.dumps({
                "status": "already_running",
                "running_worker_id": self._current_worker_id,
                "running_worker_kind": self._current_worker_kind,
                "running_fixed_tool": self._current_fixed_tool_name,
                "fixed_tool_status": fixed_status,
                "message": (
                    "The worker slot is already occupied; solicitation-scout "
                    "was not started."
                ),
            })

        # _start_worker schedules the worker task. Yield once so AgentNode can
        # create the run directory and manifest before we form the handoff.
        launched_worker_id = self._current_worker_id
        await asyncio.sleep(0)
        fixed_status = self._get_fixed_tool_status()
        return json.dumps({
            "status": "launched",
            "tool": tool_name,
            "worker_id": launched_worker_id,
            "run_dir": (
                str(resolved_run_dir)
                if resolved_run_dir is not None
                else fixed_status.get("run_dir")
                if fixed_status and fixed_status.get("tool_name") == tool_name
                else None
            ),
            "phases": len(tool.phase_markers),
            "message": (
                "Solicitation scout resumed in the worker slot. Poll "
                if resolved_run_dir is not None
                else "Solicitation scout started in the worker slot. Poll "
            ) + (
                "worker_status for progress through its three phases; final "
                "artifact paths and bounded report contents are delivered "
                "automatically through normal worker completion."
            ),
        })

    def _get_fixed_tool_status(self) -> dict[str, Any] | None:
        """Read fixed-tool status from the bound AgentNode, if available."""
        getter = getattr(
            getattr(self, "_worker_agent", None),
            "get_fixed_tool_status",
            None,
        )
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception:
            logger.exception("Failed to read fixed-tool status")
            return None

    def _worker_backend_instructions(self) -> str:
        """Render task-type guidance without exposing backend bindings."""
        task_types = self._configured_worker_task_types()
        if not task_types:
            return (
                "This agent has no configured worker task types. Omit task_type "
                "and reason to use its configured default backend. The backend "
                "field is a HARD OVERRIDE used only when the user explicitly "
                "names a backend in the current instruction; copy that name "
                "verbatim and never populate backend from your own judgment."
            )
        guidelines = []
        for task_type, definition in task_types.items():
            description = definition.get("description", "").strip()
            guidelines.append(
                f"- {task_type}: {description}" if description else f"- {task_type}"
            )
        return (
            "task_type is REQUIRED on every dispatch, together with a concise "
            "one-line reason. A dispatch that omits task_type is REFUSED and no "
            "worker runs — there is no unnamed default shape. Pick the closest "
            "type below rather than leaving it out.\n"
            "For ordinary dispatches, select a task TYPE only. mesh.yaml owns the "
            "type-to-backend binding; do not reason about or name the bound backend. "
            "Task-type choice is per launch only; there is no session override.\n"
            "Standing task-type guidelines:\n"
            + "\n".join(guidelines)
            + "\n"
            "ABSOLUTE RULE: an explicit user instruction naming a worker backend "
            "overrides every guideline above verbatim. Only in that case, copy the "
            "user's exact name into backend — never populate it from your own "
            "judgment. A backend override still requires a task_type, and is "
            "refused on a type that has a configured Plan-Execute-Verify "
            "workflow, because a bare backend cannot express per-phase backends "
            "and would silently drop the workflow.\n"
            "If a selected capped backend fails with a quota error or exit code 1, "
            "retry the task on the configured default instead of failing it."
        )

    # ── Open/closed model access gate ────────────────────────────────────
    #
    # The model-level twin of the isolation network gate.  An agent configured
    # with ``worker_closed_models: false`` must be structurally unable to run a
    # closed model, on EVERY dispatch branch and for EVERY phase backend.
    #
    # Two properties make that hold rather than merely usually hold:
    #   * the gate runs on the resolved WorkerSelection, at the single choke
    #     point, so a new branch inside the resolver is covered by default;
    #   * classification fails closed — an unknown backend, or one with no
    #     ``access`` in backends.yaml, counts as closed.
    #
    # A violation is a REFUSAL, never a fallback.  Substituting the configured
    # default would run a closed model silently, which is the precise failure
    # this gate exists to make impossible.

    def _resolve_worker_backend_name(self, backend_name: str | None) -> str:
        """Resolve a selection's backend to the name that will actually run.

        An empty backend on a WorkerSelection is not "no backend" — it means
        the agent's configured default runs.  The gate has to classify what
        will really execute, so resolve that here.
        """
        name = str(backend_name or "").strip()
        if name:
            return name
        return str(getattr(self, "_default_worker_backend", "") or "").strip()

    def _classify_worker_backend(self, backend_name: str | None) -> str:
        """Classify one backend as ``open`` or ``closed``. Fail-closed."""
        name = self._resolve_worker_backend_name(backend_name)
        if not name:
            # Nothing resolvable to classify: treat as closed.
            return BACKEND_ACCESS_CLOSED
        access_map = getattr(self, "_worker_backend_access", {}) or {}
        raw = access_map.get(name)
        if raw is None and name == _DEFAULT_BACKEND_SENTINEL:
            # `default` is the compatibility alias for whatever this agent
            # falls back to. It is normally a concrete backends.yaml entry with
            # its own access, but follow it to the agent's configured default
            # when it is not, so the alias is never classified by its name.
            target = str(
                getattr(self, "_default_worker_backend", "") or ""
            ).strip()
            if target and target != name:
                raw = access_map.get(target)
        return classify_backend_access(raw)

    def _closed_worker_backends(
        self, selection: WorkerSelection
    ) -> list[tuple[str, str]]:
        """Return ``(role, backend)`` for every closed backend a run would use."""
        offenders: list[tuple[str, str]] = []

        def _check(role: str, backend_name: str | None) -> None:
            if self._classify_worker_backend(backend_name) != BACKEND_ACCESS_OPEN:
                offenders.append((
                    role,
                    self._resolve_worker_backend_name(backend_name)
                    or "<unresolved default>",
                ))

        _check("backend", selection.backend)
        pev = selection.pev
        if pev is not None:
            for phase in ("plan", "execute", "verify", "compose_backend"):
                phase_backend = getattr(pev, phase, None)
                if phase_backend:
                    _check(f"pev.{phase}", phase_backend)
        prompts = selection.prompts
        if prompts is not None and prompts.sync_backend:
            _check("prompts.sync_backend", prompts.sync_backend)
        return offenders

    def _gate_worker_selection(self, selection: WorkerSelection) -> WorkerSelection:
        """Refuse a selection that would run a closed model on an open-only agent."""
        if getattr(self, "_worker_closed_models", True):
            return selection
        if selection.refusal:
            # Already refused; no worker starts either way.
            return selection

        offenders = self._closed_worker_backends(selection)
        if not offenders:
            return selection

        if selection.user_override:
            branch = "A verbatim user backend override"
        elif selection.task_type == "custom":
            branch = "A custom worker selection"
        elif selection.task_type:
            branch = f"Worker task type {selection.task_type!r}"
        else:
            branch = "The configured default worker backend"
        rendered = ", ".join(
            f"{role}={backend_name}" for role, backend_name in offenders
        )
        refusal = (
            f"{branch} resolves to closed model backends ({rendered}), but "
            "this agent is configured with worker_closed_models: false and may "
            "run open-weight models only. A backend with no 'access' "
            "classification in backends.yaml counts as closed. No worker was "
            "started, and the configured default was NOT substituted — that "
            "would run a closed model without saying so. Re-dispatch on a task "
            "type bound to an open backend, or change the agent's "
            "configuration."
        )
        logger.warning("[WORKER] DISPATCH REFUSED: %s", refusal)
        return WorkerSelection(refusal=refusal)

    def _resolve_worker_selection(
        self,
        task_type: str | None,
        backend: str | None,
        reason: str | None,
        *,
        staged_metadata: dict[str, Any] | None = None,
        allow_custom: bool = False,
    ) -> WorkerSelection:
        """Resolve a dispatch selection, then apply the open/closed gate.

        The resolution itself lives in ``_select_worker_selection``; keeping the
        gate in the wrapper means every branch — present and future — is gated
        without each return path having to remember.
        """
        return self._gate_worker_selection(
            self._select_worker_selection(
                task_type,
                backend,
                reason,
                staged_metadata=staged_metadata,
                allow_custom=allow_custom,
            )
        )

    def _select_worker_selection(
        self,
        task_type: str | None,
        backend: str | None,
        reason: str | None,
        *,
        staged_metadata: dict[str, Any] | None = None,
        allow_custom: bool = False,
    ) -> WorkerSelection:
        """Resolve a task type or verbatim user override, falling back safely."""
        self._worker_backend_override = None
        requested_type = str(task_type or "").strip()
        requested_backend = str(backend or "").strip()
        normalized_reason = self._normalize_backend_reason(reason)
        default = getattr(self, "_default_worker_backend", "") or ""

        # Custom policies are an internal launch surface.  The public
        # worker_launch schema deliberately cannot supply its PEV or prompt
        # payload, so it must not be able to activate this escape hatch by
        # naming task_type="custom".  A trusted caller stages the complete
        # metadata through _stage_trusted_custom_worker_selection_metadata().
        if requested_type == "custom":
            if not allow_custom or staged_metadata is None:
                warning = (
                    "Custom worker selections require trusted staged metadata; "
                    f"using configured default {default!r}."
                )
                logger.warning("[WORKER] %s", warning)
                return WorkerSelection(warning=warning)
            if not requested_backend:
                warning = (
                    "Custom worker selection omitted its backend; using "
                    f"configured default {default!r}."
                )
                logger.warning("[WORKER] %s", warning)
                return WorkerSelection(warning=warning)
            if requested_backend not in set(self._known_worker_backends()):
                warning = (
                    f"Custom worker backend {requested_backend!r} is not present "
                    f"in llm_backends; using configured default {default!r}."
                )
                logger.warning("[WORKER] %s", warning)
                return WorkerSelection(warning=warning)
            if not normalized_reason:
                warning = (
                    "Custom worker selection requires a one-line reason; using "
                    f"configured default {default!r}."
                )
                logger.warning("[WORKER] %s", warning)
                return WorkerSelection(warning=warning)

            raw_pev = staged_metadata.get("worker_pev")
            raw_prompts = staged_metadata.get("worker_prompt_config")
            try:
                pev_config = normalize_pev_task_config(raw_pev)
                prompt_config = (
                    TaskPromptConfig.from_dict(raw_prompts)
                    if isinstance(raw_prompts, dict) else None
                )
            except (TypeError, ValueError) as exc:
                warning = (
                    "Custom worker selection has invalid staged configuration "
                    f"({exc}); using configured default {default!r}."
                )
                logger.warning("[WORKER] %s", warning)
                return WorkerSelection(warning=warning)

            if raw_prompts is not None and not isinstance(raw_prompts, dict):
                warning = (
                    "Custom worker prompt configuration must be a mapping; using "
                    f"configured default {default!r}."
                )
                logger.warning("[WORKER] %s", warning)
                return WorkerSelection(warning=warning)

            configured_backends = {requested_backend}
            if pev_config is not None:
                configured_backends.update(
                    backend_name for backend_name in (
                        pev_config.plan, pev_config.execute, pev_config.verify,
                        pev_config.compose_backend,
                    ) if backend_name
                )
            if prompt_config is not None and prompt_config.sync_backend:
                configured_backends.add(prompt_config.sync_backend)
            missing = sorted(
                backend_name for backend_name in configured_backends
                if backend_name not in set(self._known_worker_backends())
            )
            if missing:
                warning = (
                    "Custom worker selection references backends not present in "
                    f"llm_backends ({', '.join(missing)}); using configured default "
                    f"{default!r}."
                )
                logger.warning("[WORKER] %s", warning)
                return WorkerSelection(warning=warning)
            return WorkerSelection(
                backend=requested_backend,
                task_type=requested_type,
                reason=normalized_reason,
                pev=pev_config,
                prompts=prompt_config,
            )

        # A dispatch must say what SHAPE of work it is.  Falling back to the
        # configured default on a missing task type is the same silent
        # downgrade as the backend-override case below: the caller gets some
        # workflow rather than the one they meant, and is never told.  Refuse
        # instead — every ordinary-worker shape has a named type too, so
        # naming one costs nothing and the choice becomes auditable.
        if not requested_type:
            configured_types = self._configured_worker_task_types()
            if configured_types:
                refusal = (
                    "No worker task type was supplied, and this agent's dispatch "
                    "shape is defined by its configured task types "
                    f"({', '.join(sorted(configured_types))}). Running the "
                    "configured default instead would silently pick a workflow "
                    "you did not choose, so no worker was started. Re-dispatch "
                    "with a task_type and a one-line reason."
                )
                logger.warning("[WORKER] DISPATCH REFUSED: %s", refusal)
                return WorkerSelection(refusal=refusal)
            # No configured task types: there is nothing to name, so the
            # configured default is the only shape available.
            if not requested_backend:
                return WorkerSelection()

        if requested_backend:
            known = set(self._known_worker_backends())
            if requested_backend not in known:
                warning = (
                    f"Requested user-override worker backend {requested_backend!r} "
                    f"is not present in llm_backends; using configured default "
                    f"{default!r}."
                )
                logger.warning("[WORKER] %s", warning)
                return WorkerSelection(warning=warning)
            task_prompts: TaskPromptConfig | None = None
            if requested_type:
                # A verbatim backend override carries no per-phase policy, so
                # letting it win over a PEV-configured task type does not
                # "run that task type on another backend" — it drops the
                # workflow entirely and dispatches an ordinary single-shot
                # worker with no plan and no verify, silently.  Refuse rather
                # than downgrade: the caller has to say which one it meant.
                configured = self._configured_worker_task_types().get(requested_type)
                if configured is not None and configured.get("pev") is not None:
                    refusal = (
                        f"Worker task type {requested_type!r} runs a "
                        f"Plan-Execute-Verify workflow whose per-phase backends "
                        f"come from configuration, and a verbatim backend "
                        f"override ({requested_backend!r}) cannot express them. "
                        f"Dispatching both would drop the PEV workflow without "
                        f"saying so, and no worker was started. Dispatch "
                        f"{requested_type!r} with no backend to keep the "
                        f"configured workflow, or name the backend with no task "
                        f"type to run an ordinary worker on it."
                    )
                    logger.warning("[WORKER] DISPATCH REFUSED: %s", refusal)
                    return WorkerSelection(refusal=refusal)
                if configured is not None:
                    task_prompts = configured.get("prompts")
                logger.warning(
                    "[WORKER] Both task_type=%r and backend=%r were supplied; "
                    "the verbatim user backend override takes precedence.",
                    requested_type,
                    requested_backend,
                )
            return WorkerSelection(
                backend=requested_backend,
                task_type=requested_type,
                reason=normalized_reason,
                user_override=True,
                # Backend overrides cannot describe a PEV phase map, but task
                # prompts are an orthogonal text bundle that must survive.
                prompts=task_prompts,
            )

        task_types = self._configured_worker_task_types()
        if requested_type not in task_types:
            warning = (
                f"Requested worker task type {requested_type!r} is not configured "
                f"for this agent; using configured default {default!r}."
            )
            logger.warning("[WORKER] %s", warning)
            return WorkerSelection(warning=warning)
        if not normalized_reason:
            warning = (
                f"Requested worker task type {requested_type!r} without the "
                "required one-line reason; using the configured default."
            )
            logger.warning("[WORKER] %s", warning)
            return WorkerSelection(
                backend=task_types[requested_type]["backend"],
                task_type=requested_type,
                reason="",
                pev=task_types[requested_type].get("pev"),
                prompts=task_types[requested_type].get("prompts"),
                warning=warning,
            )
        return WorkerSelection(
            backend=task_types[requested_type]["backend"],
            task_type=requested_type,
            reason=normalized_reason,
            pev=task_types[requested_type].get("pev"),
            prompts=task_types[requested_type].get("prompts"),
        )

    @staticmethod
    def _normalize_backend_reason(reason: str | None) -> str:
        """Return a bounded, single-line backend-selection reason."""
        compact = " ".join(str(reason or "").split())
        return compact[:240]

    @staticmethod
    def _clear_worker_selection_metadata(metadata: dict[str, Any]) -> None:
        for key in (
            "worker_backend",
            "worker_backend_reason",
            "worker_backend_user_override",
            "worker_task_type",
            "worker_backend_warning",
            "worker_backend_refusal",
            "worker_pev",
            "worker_prompt_config",
        ):
            metadata.pop(key, None)

    def _write_worker_selection_metadata(
        self,
        trigger: Message,
        selection: WorkerSelection,
    ) -> None:
        # Mutate the trigger's metadata IN PLACE.  Re-binding it to a fresh
        # dict orphans every reference a caller is already holding, and
        # _start_worker holds one across its call to
        # _validate_staged_worker_selection: the dispatch brief and the
        # idempotency keys it stamps afterwards were landing on the discarded
        # copy, so PEV workers ran on trigger.content and the duplicate-
        # admission guard never saw a dispatch key (2026-07-29).
        metadata = getattr(trigger, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            trigger.metadata = metadata
        self._clear_worker_selection_metadata(metadata)
        if selection.backend:
            metadata["worker_backend"] = selection.backend
        if selection.task_type:
            metadata["worker_task_type"] = selection.task_type
            metadata["worker_backend_reason"] = selection.reason
        if selection.pev is not None:
            metadata["worker_pev"] = selection.pev.as_dict()
        if selection.prompts is not None:
            metadata["worker_prompt_config"] = selection.prompts.as_dict()
        if selection.user_override:
            metadata["worker_backend_user_override"] = True
        if selection.warning:
            metadata["worker_backend_warning"] = selection.warning
        if selection.refusal:
            metadata["worker_backend_refusal"] = selection.refusal

    def _stage_worker_selection_metadata(
        self,
        trigger: Message,
        parsed: dict[str, Any],
    ) -> WorkerSelection:
        """Resolve and stage a direct-tool or harness dispatch selection."""
        selection = self._resolve_worker_selection(
            parsed.get("task_type"),
            parsed.get("backend"),
            parsed.get("backend_reason") or parsed.get("reason"),
        )
        self._write_worker_selection_metadata(trigger, selection)
        return selection

    @staticmethod
    def _render_dispatch_request(
        spec: dict[str, Any],
        *,
        source: str,
        task: str,
    ) -> str:
        """Render the request-shaped history exemplar for one dispatch door."""
        task_type = str(spec.get("task_type") or "").strip()
        backend = str(spec.get("backend") or "").strip()
        reason = str(
            spec.get("backend_reason") or spec.get("reason") or ""
        ).strip()
        if source == "tool":
            fields = [f"<task>{html.escape(task)}</task>"]
            if task_type:
                fields.append(
                    f"<task_type>{html.escape(task_type)}</task_type>"
                )
            if backend:
                fields.append(f"<backend>{html.escape(backend)}</backend>")
            if reason:
                fields.append(f"<reason>{html.escape(reason)}</reason>")
            return (
                '<mesh_call name="worker_launch">'
                + "".join(fields)
                + "</mesh_call>"
            )

        lines = ["<dispatch_worker>", f"task: {task}"]
        if task_type:
            lines.append(f"task_type: {task_type}")
        if backend:
            lines.append(f"backend: {backend}")
        if reason:
            lines.append(f"reason: {reason}")
        lines.append("</dispatch_worker>")
        return "\n".join(lines)

    def _render_dispatch_acknowledgment(
        self,
        receipt: DispatchReceipt,
        metadata: dict[str, Any],
    ) -> str:
        """Render the sole human-facing worker-launch acknowledgment."""
        lines = [
            f"Worker dispatched: {receipt.worker_id or 'unknown worker'}.",
            f"Task: {receipt.task_description}",
        ]
        if receipt.backend:
            lines.append(f"Backend: {receipt.backend}")
        stamp = self._selection_stamp(metadata, receipt.backend)
        if stamp:
            lines.append(stamp)
        return "\n".join(lines)

    def _record_dispatch_outcome(
        self,
        receipt: DispatchReceipt,
    ) -> DispatchReceipt:
        """Publish one immutable result to every status/adapter surface."""
        self._last_dispatch_receipt = receipt
        if receipt.dispatch_key:
            if not hasattr(self, "_dispatch_receipts"):
                self._dispatch_receipts = {}
            self._dispatch_receipts[receipt.dispatch_key] = receipt
        return receipt

    # ── Autonomous admission guard (plan §8.4, §8.7, §8.9; §17 items 1–2) ──
    #
    # The controller prompt is the primary enforcement; this seam is
    # defense-in-depth against a controller that ignores it.  Every failure
    # mode of the guard itself fails *open*: a missing tag, an unreadable
    # ledger, or an unexpected exception must never convert a legitimate
    # dispatch into a refusal.  Only a positively-read, positively-exhausted
    # budget refuses.

    def _autonomous_mode_enabled(self) -> bool:
        return bool(
            getattr(self._config, "autonomous_agent_mode_enabled", False)
        )

    def _extract_autonomous_project_key(self, task: str) -> str | None:
        """Return the ``[PROJECT: project:<slug>]`` key tagged in a brief.

        Returns ``None`` (fail open) when the tag is absent or its key does
        not match the dossier layer's own key grammar.  Shape authority lives
        in ``project_dossier.PROJECT_KEY_RE`` — this never re-derives it.
        """
        from .project_dossier import PROJECT_KEY_RE

        match = _AUTONOMOUS_PROJECT_TAG_RE.search(task or "")
        if match is None:
            return None
        key = match.group(1).strip()
        if not PROJECT_KEY_RE.match(key):
            logger.warning(
                "[AUTONOMOUS] Ignoring malformed project tag %r in dispatch "
                "brief; admission guard failing open.",
                key,
            )
            return None
        return key

    def _stamp_autonomous_project_tag(
        self,
        trigger: Message,
        brief: ResolvedDispatchBrief,
    ) -> ResolvedDispatchBrief:
        """Ensure an autonomous-session brief carries its ``[PROJECT: …]`` tag.

        The mandate tells the router to write the tag itself, but prompt
        enforcement alone does not hold: a brief that omits it makes the hard
        admission guard fail open, so the dispatch spends no budget and the
        completion carries no session scope.  The scope is already known here
        — the wake runtime stamped it on the trigger and a model cannot forge
        it — so the tag is applied mechanically instead of being wished for.
        """
        if not self._autonomous_mode_enabled():
            return brief
        metadata = getattr(trigger, "metadata", None)
        if not isinstance(metadata, dict) or not metadata.get(
            "autonomous_session"
        ):
            return brief
        key = str(metadata.get("autonomous_project_key") or "").strip()
        if not key:
            return brief
        from .project_dossier import PROJECT_KEY_RE

        if not PROJECT_KEY_RE.match(key):
            return brief
        if self._extract_autonomous_project_key(brief.text) is not None:
            return brief
        tagged = f"[PROJECT: {key}]\n\n{brief.text}".strip()
        metadata["worker_task_description"] = tagged
        logger.info(
            "[AUTONOMOUS] Dispatch brief omitted its project tag; stamped "
            "[PROJECT: %s] from the trusted session scope.",
            key,
        )
        return ResolvedDispatchBrief(tagged, brief.tier)

    def _autonomous_session_id(self, trigger: Message) -> str:
        """Resolve the session this dispatch belongs to.

        Prefers an id already stamped on the trigger (a controller wake
        carries its own), then the router turn id, then a fresh uuid.  The id
        is only ever router-minted; a model cannot choose it.
        """
        metadata = getattr(trigger, "metadata", None)
        if isinstance(metadata, dict):
            for field_name in ("autonomous_session_id", "router_turn_id"):
                existing = str(metadata.get(field_name) or "").strip()
                if existing:
                    return existing
        return f"as-{uuid.uuid4().hex[:12]}"

    def _resolve_autonomous_admission(
        self,
        trigger: Message,
        *,
        source: str,
        spec: dict[str, Any],
        brief: ResolvedDispatchBrief,
    ) -> tuple[dict[str, Any] | None, DispatchReceipt | None]:
        """Check the project budget before admitting a worker.

        Returns ``(scope, refusal)``.  ``scope`` is non-None only for an
        autonomous dispatch that was admitted; ``refusal`` is non-None only
        when the budget was read successfully and is exhausted.
        """
        if not self._autonomous_mode_enabled():
            return None, None

        key = self._extract_autonomous_project_key(brief.text)
        if key is None:
            logger.warning(
                "[AUTONOMOUS] No [PROJECT: project:<slug>] tag in the dispatch "
                "brief; the hard admission guard is failing open for this "
                "launch (prompt enforcement remains in force)."
            )
            return None, None

        try:
            from .project_dossier import check_budget

            state = check_budget(key, self._state_paths)
            remaining = int(state.get("remaining", 0))
            limit = int(state.get("limit", 0))
            used = int(state.get("used", 0))
        except Exception as exc:
            # Budget *infrastructure* failure is not a policy decision.  A
            # corrupt ledger or an unreadable dossier must not be able to
            # deny every worker the agent would otherwise run.
            logger.warning(
                "[AUTONOMOUS] Could not read the worker budget for %s (%s); "
                "admission guard failing open.",
                key,
                exc,
            )
            return None, None

        session_id = self._autonomous_session_id(trigger)
        if remaining <= 0:
            payload = {
                "status": "autonomous_budget_exhausted",
                "project_entity_key": key,
                "session_id": session_id,
                "limit": "daily",
                "used": used,
                "allowed": limit,
                "next_available_at": state.get("resets_at", ""),
            }
            logger.warning(
                "[AUTONOMOUS] Admission refused for %s: %d of %d worker "
                "admissions already used today; resets at %s.",
                key,
                used,
                limit,
                state.get("resets_at", "unknown"),
            )
            refusal = self._dispatch_refusal(
                trigger,
                source=source,
                spec=spec,
                brief=brief,
                status="autonomous_budget_exhausted",
                message=(
                    "Autonomous worker budget exhausted for "
                    f"{key}: {used} of {limit} admissions used today. "
                    f"Resets at {state.get('resets_at', 'unknown')}. Do not "
                    "retry with a different task, dispatch syntax, or "
                    "backend — close the session and record the refusal.\n"
                    + json.dumps(payload, sort_keys=True)
                ),
                autonomous_session=True,
                project_key=key,
                session_id=session_id,
            )
            return None, refusal

        return (
            {
                "project_key": key,
                "session_id": session_id,
                "remaining_before": remaining,
                "limit": limit,
            },
            None,
        )

    def _charge_autonomous_admission(self, scope: dict[str, Any]) -> None:
        """Charge one admission after the worker actually started."""
        key = scope.get("project_key") or ""
        try:
            from .project_dossier import spend_budget

            state = spend_budget(key, 1, self._state_paths)
        except Exception as exc:
            # The worker is already running; a ledger write failure cannot be
            # undone by refusing it retroactively.  Surface it loudly instead.
            logger.warning(
                "[AUTONOMOUS] Worker admitted for %s but the budget ledger "
                "was not charged (%s); the count is now understated.",
                key,
                exc,
            )
            return
        logger.info(
            "[AUTONOMOUS] Charged 1 worker admission to %s (session %s): "
            "%d of %d used today.",
            key,
            scope.get("session_id", ""),
            int(state.get("used", 0)),
            int(state.get("limit", 0)),
        )

    @staticmethod
    def autonomous_completion_metadata(
        trigger: Message | None,
    ) -> dict[str, Any]:
        """Return the trusted autonomous-session keys stamped on a trigger.

        Empty for every ordinary interactive dispatch, which is what lets a
        completion consumer distinguish "worker completed for autonomous
        session X" from "worker completed for interactive use".
        """
        metadata = getattr(trigger, "metadata", None)
        if not isinstance(metadata, dict):
            return {}
        if not metadata.get("autonomous_session"):
            return {}
        completion_metadata = {
            "autonomous_session": True,
            "autonomous_project_key": str(
                metadata.get("autonomous_project_key") or ""
            ),
            "autonomous_session_id": str(
                metadata.get("autonomous_session_id") or ""
            ),
            "autonomous_report_to": str(
                metadata.get("autonomous_report_to") or ""
            ),
        }
        plan = RouterV2._extract_autonomous_session_plan(
            str(metadata.get("autonomous_session_plan") or "")
        )
        if plan:
            completion_metadata["autonomous_session_plan"] = plan
        return completion_metadata

    def _dispatch_refusal(
        self,
        trigger: Message,
        *,
        source: str,
        spec: dict[str, Any],
        brief: ResolvedDispatchBrief,
        status: str,
        message: str,
        worker_id: str | None = None,
        autonomous_session: bool = False,
        project_key: str = "",
        session_id: str = "",
    ) -> DispatchReceipt:
        request_record = self._render_dispatch_request(
            spec,
            source=source,
            task=brief.text,
        )
        receipt = DispatchReceipt(
            dispatch_key="",
            status=status,
            worker_id=worker_id,
            slot_index=None,
            origin_message_id=str(getattr(trigger, "id", "") or ""),
            router_turn_id=str(
                (getattr(trigger, "metadata", {}) or {}).get(
                    "router_turn_id", ""
                )
            ),
            task_description=brief.text,
            backend=None,
            message=message,
            source=source,
            brief_tier=brief.tier.value,
            task_type=str(spec.get("task_type") or "").strip(),
            reason=self._normalize_backend_reason(
                spec.get("backend_reason") or spec.get("reason")
            ),
            request_record=request_record,
            autonomous_session=autonomous_session,
            project_key=project_key,
            session_id=session_id,
        )
        logger.warning("[WORKER] DISPATCH REFUSED: %s", message)
        return self._record_dispatch_outcome(receipt)

    def _validate_dispatch_brief(
        self,
        trigger: Message,
        *,
        source: str,
        spec: dict[str, Any],
        router_task_description: str = "",
    ) -> tuple[ResolvedDispatchBrief, DispatchReceipt | None]:
        """Resolve and fail closed on untrusted or degenerate worker briefs."""
        brief = resolve_dispatch_brief(trigger, router_task_description)
        if brief.tier is DispatchBriefTier.TRIGGER_CONTENT:
            refusal = (
                "The worker brief was missing and resolved only from the "
                "trigger message content. That terminal fallback is not a "
                "trusted dispatch brief, so no worker was started."
            )
            return brief, self._dispatch_refusal(
                trigger,
                source=source,
                spec=spec,
                brief=brief,
                status="refused",
                message=refusal,
            )
        try:
            minimum = max(
                1,
                int(
                    getattr(
                        getattr(self, "_config", None),
                        "min_worker_brief_chars",
                        120,
                    )
                    or 120
                ),
            )
        except (TypeError, ValueError):
            minimum = 120
        if (
            brief.tier is DispatchBriefTier.METADATA
            and len(brief.text.strip()) < minimum
        ):
            refusal = (
                f"The worker brief is {len(brief.text.strip())} characters; "
                f"at least {minimum} characters are required. No worker was "
                "started."
            )
            return brief, self._dispatch_refusal(
                trigger,
                source=source,
                spec=spec,
                brief=brief,
                status="refused",
                message=refusal,
            )
        return brief, None

    async def _dispatch_worker(
        self,
        trigger: Message,
        spec: dict[str, Any],
        *,
        source: str,
    ) -> DispatchReceipt:
        """Authoritative dispatch seam shared by XML and native-tool doors."""
        if source not in {"xml", "tool"}:
            raise ValueError(f"unsupported worker dispatch source: {source!r}")
        if not hasattr(self, "_router_call_worker_launches"):
            self._router_call_worker_launches = []
        if not hasattr(self, "_router_call_worker_task_keys"):
            self._router_call_worker_task_keys = set()
        metadata = getattr(trigger, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            trigger.metadata = metadata

        task = str(spec.get("task") or "").strip()
        if task:
            # This is an explicit carrier, even when a tool's synthetic
            # trigger deliberately uses the same text as its content.
            metadata["worker_task_description"] = task
            metadata["worker_brief_provenance"] = f"{source}_dispatch_spec"

        brief, refusal = self._validate_dispatch_brief(
            trigger,
            source=source,
            spec=spec,
            # A public dispatch must not inherit stale router-global state.
            router_task_description="",
        )
        if refusal is not None:
            return refusal

        # §8.7 step 3: the project tag is the admission guard's scope carrier.
        # Stamp it from trusted session metadata before any brief-derived
        # decision (task key, admission, worker payload) is taken.
        brief = self._stamp_autonomous_project_tag(trigger, brief)

        task_key = self._worker_task_key(brief.text)
        if task_key in self._router_call_worker_task_keys:
            return self._dispatch_refusal(
                trigger,
                source=source,
                spec=spec,
                brief=brief,
                status="duplicate_in_turn",
                message=(
                    "A worker launch for this task already ran in this router "
                    "turn; no duplicate worker was started."
                ),
            )
        if self._router_call_worker_launches:
            return self._dispatch_refusal(
                trigger,
                source=source,
                spec=spec,
                brief=brief,
                status="duplicate_in_turn",
                message=(
                    "Only one worker launch is allowed per router turn. Start "
                    "additional workers from separate user turns."
                ),
            )
        for slot in self._active_worker_slots():
            if self._worker_task_key(slot.task_description) == task_key:
                return self._dispatch_refusal(
                    trigger,
                    source=source,
                    spec=spec,
                    brief=brief,
                    status="duplicate_running_task",
                    worker_id=slot.worker_id,
                    message=(
                        "A worker is already running this same task; no "
                        "duplicate worker was started."
                    ),
                )

        selection = self._stage_worker_selection_metadata(trigger, spec)
        if selection.refusal:
            return self._dispatch_refusal(
                trigger,
                source=source,
                spec=spec,
                brief=brief,
                status="refused",
                message=selection.refusal,
            )

        # §8.7 step 3–4: resolve autonomous scope and check the budget only
        # after duplicate/capacity refusals, so a refusal that never reaches
        # _start_worker() cannot consume an admission.
        autonomous_scope, budget_refusal = self._resolve_autonomous_admission(
            trigger,
            source=source,
            spec=spec,
            brief=brief,
        )
        if budget_refusal is not None:
            return budget_refusal
        if autonomous_scope is not None:
            # Preserve the plan before a worker can outlive the rolling-history
            # window.  Only a trigger already stamped by the wake runtime may
            # acquire this carrier; tag-only legacy admissions remain unchanged.
            if metadata.get("autonomous_session"):
                plan = self._extract_autonomous_session_plan(
                    str(metadata.get("autonomous_session_plan") or "")
                )
                if not plan:
                    plan = self._session_plan_from_history(
                        autonomous_scope["session_id"]
                    )
                if plan:
                    metadata["autonomous_session_plan"] = plan
                else:
                    logger.warning(
                        "[AUTONOMOUS] Dispatch for session %s has no valid "
                        "SESSION PLAN carrier; its continuation will receive "
                        "an explicit recovery marker.",
                        autonomous_scope["session_id"],
                    )
            # Item 2: stamp trusted session scope on the launch trigger. This
            # is the same object _handle_worker_complete() receives, so the
            # completion trigger can carry it without a side channel.
            metadata["autonomous_session"] = True
            metadata["autonomous_project_key"] = autonomous_scope["project_key"]
            metadata["autonomous_session_id"] = autonomous_scope["session_id"]

        previous_task = getattr(self, "_current_task_description", "")
        self._current_task_description = brief.text
        metadata["worker_brief_validated"] = True
        if autonomous_scope is not None:
            # §8.7: charge at reservation, not after a successful start. A
            # backend that fails to start still consumes the admission, so a
            # controller whose every launch fails burns its budget and locks
            # out instead of retry-storming against broken infrastructure.
            self._charge_autonomous_admission(autonomous_scope)
        logger.info(
            "[WORKER] LAUNCH (via %s): %s task=%r",
            source,
            self._nickname,
            self._bounded_slot_text(brief.text, 120),
        )
        launched = await self._start_worker(trigger)
        base = getattr(self, "_last_dispatch_receipt", None)
        if not launched:
            self._current_task_description = previous_task
            if autonomous_scope is not None:
                # Deliberately not refunded (§8.7): the admission is spent.
                logger.warning(
                    "[WORKER] autonomous admission consumed by a failed "
                    "start: project=%s session=%s",
                    autonomous_scope["project_key"],
                    autonomous_scope["session_id"],
                )
            if base is None:
                return self._dispatch_refusal(
                    trigger,
                    source=source,
                    spec=spec,
                    brief=brief,
                    status="start_failed",
                    message="Worker startup failed without a dispatch receipt.",
                )
            outcome = replace(
                base,
                source=source,
                brief_tier=brief.tier.value,
                task_type=str(
                    trigger.metadata.get("worker_task_type") or ""
                ),
                reason=self._normalize_backend_reason(
                    trigger.metadata.get("worker_backend_reason")
                ),
                request_record=self._render_dispatch_request(
                    spec,
                    source=source,
                    task=brief.text,
                ),
            )
            if not outcome.message:
                outcome = replace(
                    outcome,
                    message=(
                        "The worker was not started; dispatch status is "
                        f"{outcome.status}."
                    ),
                )
            logger.warning(
                "[WORKER] DISPATCH FAILED status=%s: %s",
                outcome.status,
                outcome.message,
            )
            return self._record_dispatch_outcome(outcome)

        if base is None:
            base = DispatchReceipt(
                dispatch_key="",
                status="running",
                worker_id=getattr(self, "_current_worker_id", None),
                slot_index=None,
                origin_message_id=str(getattr(trigger, "id", "") or ""),
                router_turn_id="",
                task_description=brief.text,
                backend=selection.backend or None,
            )
        selected_backend = (
            base.backend
            or getattr(self, "_current_worker_backend", None)
            or selection.backend
            or getattr(self, "_default_worker_backend", "")
            or None
        )
        outcome = replace(
            base,
            status="running",
            backend=selected_backend,
            source=source,
            brief_tier=brief.tier.value,
            task_type=str(trigger.metadata.get("worker_task_type") or ""),
            reason=self._normalize_backend_reason(
                trigger.metadata.get("worker_backend_reason")
            ),
            request_record=self._render_dispatch_request(
                spec,
                source=source,
                task=brief.text,
            ),
            autonomous_session=autonomous_scope is not None,
            project_key=(
                autonomous_scope["project_key"] if autonomous_scope else ""
            ),
            session_id=(
                autonomous_scope["session_id"] if autonomous_scope else ""
            ),
        )
        outcome = replace(
            outcome,
            acknowledgment=self._render_dispatch_acknowledgment(
                outcome,
                trigger.metadata,
            ),
        )
        outcome = self._record_dispatch_outcome(outcome)
        self._router_call_worker_task_keys.add(task_key)
        if outcome.worker_id:
            self._router_call_worker_launches.append(outcome.worker_id)
        self._last_worker_launch = {
            "worker_id": outcome.worker_id,
            "worker_backend": outcome.backend,
            "worker_task_type": outcome.task_type or None,
            "worker_backend_reason": outcome.reason or None,
            "worker_backend_user_override": bool(
                trigger.metadata.get("worker_backend_user_override")
            ),
            "worker_pev": trigger.metadata.get("worker_pev"),
            "worker_prompt_config": trigger.metadata.get(
                "worker_prompt_config"
            ),
            "worker_backend_warning": trigger.metadata.get(
                "worker_backend_warning"
            ),
            "task": brief.text,
            "status": "dispatched",
            "dispatch_status": outcome.status,
            "dispatch_key": outcome.dispatch_key or None,
            "slot_index": outcome.slot_index,
            "acknowledgment": outcome.acknowledgment,
            "request_record": outcome.request_record,
        }
        return outcome

    def _stage_trusted_custom_worker_selection_metadata(
        self,
        trigger: Message,
        *,
        backend: str,
        reason: str,
        pev: PevTaskConfig | dict[str, Any] | None = None,
        prompts: TaskPromptConfig | dict[str, Any] | None = None,
    ) -> WorkerSelection:
        """Stage an internal custom worker policy for one immediate launch.

        This is intentionally not reachable through the LLM-facing
        ``worker_launch`` tool.  The private trigger-id receipt prevents
        arbitrary inbound message metadata from supplying an alternate PEV or
        prompt policy at the common dispatch boundary.
        """
        staged = getattr(trigger, "metadata", None)
        if not isinstance(staged, dict):
            staged = {}
        staged.update({
            "worker_backend": backend,
            "worker_task_type": "custom",
            "worker_backend_reason": reason,
        })
        if pev is not None:
            staged["worker_pev"] = (
                pev.as_dict() if isinstance(pev, PevTaskConfig) else pev
            )
        if prompts is not None:
            staged["worker_prompt_config"] = (
                prompts.as_dict() if isinstance(prompts, TaskPromptConfig) else prompts
            )
        trigger.metadata = staged
        trusted = getattr(self, "_trusted_custom_worker_selection_ids", None)
        if trusted is None:
            trusted = set()
            self._trusted_custom_worker_selection_ids = trusted
        trusted.add(trigger.id)
        selection = self._resolve_worker_selection(
            "custom",
            backend,
            reason,
            staged_metadata=staged,
            allow_custom=True,
        )
        if selection.warning:
            trusted.discard(trigger.id)
        self._write_worker_selection_metadata(trigger, selection)
        return selection

    def _validate_staged_worker_selection(
        self,
        trigger: Message,
    ) -> WorkerSelection:
        """Re-resolve staged metadata at the common direct/XML boundary."""
        metadata = dict(getattr(trigger, "metadata", {}) or {})
        backend = str(metadata.get("worker_backend") or "").strip()
        task_type = str(metadata.get("worker_task_type") or "").strip()
        reason = metadata.get("worker_backend_reason")
        if metadata.get("worker_backend_user_override"):
            selection = self._resolve_worker_selection(task_type, backend, None)
            # A bare backend override cannot safely express per-phase PEV
            # backends. Keep its task-type prompt bundle, but always run it as
            # the ordinary single-backend worker the user selected.
            if selection.pev is not None:
                selection = replace(selection, pev=None)
        elif task_type == "custom":
            trusted = getattr(self, "_trusted_custom_worker_selection_ids", set())
            selection = self._resolve_worker_selection(
                task_type,
                backend,
                reason,
                staged_metadata=metadata,
                allow_custom=trigger.id in trusted,
            )
            trusted.discard(trigger.id)
        elif task_type:
            selection = self._resolve_worker_selection(task_type, None, reason)
            if selection.backend and selection.backend != backend:
                warning = (
                    f"Worker task type {task_type!r} no longer resolves to staged "
                    f"backend {backend!r}; using the configured default."
                )
                logger.warning("[WORKER] %s", warning)
                selection = WorkerSelection(warning=warning)
            elif selection.pev is not None:
                staged_pev = metadata.get("worker_pev")
                if staged_pev != selection.pev.as_dict():
                    warning = (
                        f"Worker task type {task_type!r} PEV configuration no longer "
                        "matches the staged dispatch; using the configured default."
                    )
                    logger.warning("[WORKER] %s", warning)
                    selection = WorkerSelection(warning=warning)
            if selection.backend and selection.prompts is not None:
                staged_prompts = metadata.get("worker_prompt_config")
                if staged_prompts != selection.prompts.as_dict():
                    warning = (
                        f"Worker task type {task_type!r} prompt configuration no "
                        "longer matches the staged dispatch; using the configured "
                        "default."
                    )
                    logger.warning("[WORKER] %s", warning)
                    selection = WorkerSelection(warning=warning)
            elif selection.backend and metadata.get("worker_prompt_config") is not None:
                warning = (
                    f"Worker task type {task_type!r} has no prompt configuration "
                    "but staged prompt metadata was supplied; using the configured "
                    "default."
                )
                logger.warning("[WORKER] %s", warning)
                selection = WorkerSelection(warning=warning)
        elif backend:
            warning = (
                f"Unattributed worker backend metadata {backend!r} was rejected; "
                "backend is valid only as a staged task type or user override."
            )
            logger.warning("[WORKER] %s", warning)
            selection = WorkerSelection(warning=warning)
        else:
            return self._gate_worker_selection(WorkerSelection(
                warning=metadata.get("worker_backend_warning"),
                refusal=metadata.get("worker_backend_refusal"),
            ))
        # Several branches above replace the resolved selection with a bare
        # fall-back-to-default one, which _resolve_worker_selection never saw.
        # Re-gate here so a staleness fallback cannot land on a closed default.
        selection = self._gate_worker_selection(selection)
        self._write_worker_selection_metadata(trigger, selection)
        return selection

    def _format_worker_launch_ack(
        self, worker_id: str | None = None, task: str = "", backend: str | None = None,
    ) -> str:
        receipt = getattr(self, "_last_dispatch_receipt", None)
        if (
            receipt is not None
            and receipt.status == "running"
            and receipt.acknowledgment
            and (worker_id is None or receipt.worker_id == worker_id)
        ):
            return receipt.acknowledgment
        return self._worker_backend_stamp(backend)

    def _ensure_worker_launch_ack(
        self, response_text: str, worker_id: str | None = None,
        task: str = "", backend: str | None = None,
        receipt: DispatchReceipt | None = None,
    ) -> str:
        outcome = receipt or getattr(self, "_last_dispatch_receipt", None)
        stamp = (
            outcome.acknowledgment
            if outcome is not None and outcome.acknowledgment
            else self._worker_backend_stamp(backend)
        )
        if not stamp:
            return response_text
        if outcome is not None and outcome.acknowledgment:
            # The LLM may still narrate the familiar receipt despite the prompt.
            # Remove both a verbatim system-shaped block and receipt-shaped
            # lines before adding the one authoritative system rendering.
            response_text = response_text.replace(stamp, "")
            response_text = self._strip_model_dispatch_receipt_lines(
                response_text
            )
        return f"{response_text.rstrip()}\n\n{stamp}" if response_text else stamp

    @staticmethod
    def _strip_model_dispatch_receipt_lines(response_text: str) -> str:
        """Remove the rendered selection-stamp shape from model-authored text."""
        return re.sub(
            r"(?im)^.*\btype=[A-Za-z0-9_-]+\b.*\(reason:.*$\n?",
            "",
            str(response_text or ""),
        ).strip()

    @classmethod
    def _dispatch_history_record(
        cls,
        response_text: str,
        receipt: DispatchReceipt,
    ) -> str:
        """Keep conversational framing plus the canonical dispatch request."""
        if receipt.status != "running":
            # A refused launch is evidence of failure, not a successful request
            # exemplar. Persist the refusal so the router cannot learn the
            # missing/fallback brief as something it should reproduce.
            return receipt.message or receipt.request_record
        prefix = cls._strip_model_dispatch_receipt_lines(response_text)
        if prefix:
            return f"{prefix}\n\n{receipt.request_record}"
        return receipt.request_record

    def _selection_stamp(
        self,
        metadata: dict[str, Any] | None,
        backend: str | None,
    ) -> str:
        selected = str(backend or "").strip()
        if not selected or not isinstance(metadata, dict):
            return ""
        if metadata.get("worker_backend_user_override"):
            return f"backend={selected} (user override)"
        task_type = str(metadata.get("worker_task_type") or "").strip()
        reason = self._normalize_backend_reason(metadata.get("worker_backend_reason"))
        if task_type and reason:
            return f"type={task_type} → backend={selected} (reason: {reason})"
        return ""

    def _worker_backend_stamp(self, backend: str | None) -> str:
        """Render the mandatory human-visible task-type/override stamp."""
        trigger = getattr(self, "_pending_trigger", None)
        metadata = getattr(trigger, "metadata", {}) if trigger is not None else {}
        return self._selection_stamp(metadata, backend)

    async def _tool_set_worker_backend(self, backend: str) -> str:
        """Reject stale calls from pre-upgrade router contexts."""
        import json as _json
        self._worker_backend_override = None
        return _json.dumps({
            "status": "disabled",
            "worker_backend": self._default_worker_backend or None,
            "session_override": None,
            "message": "Worker backend is fixed by the agent configuration.",
        })

    async def _tool_reset_worker_backend(self) -> str:
        """Reject stale calls from pre-upgrade router contexts."""
        import json as _json
        self._worker_backend_override = None
        return _json.dumps({
            "status": "disabled",
            "worker_backend": getattr(self, "_default_worker_backend", "") or None,
            "session_override": None,
            "message": "Worker backend is fixed by the agent configuration.",
        })

    async def _tool_worker_launch(
        self,
        task: str,
        task_type: str | None = None,
        backend: str | None = None,
        reason: str | None = None,
    ) -> str:
        """Thin native-tool adapter over the authoritative dispatch seam."""
        import json as _json
        from datetime import datetime, timezone

        _trig_from, _trig_to = self._trigger_nodes()
        trigger = Message(
            id=f"synth-{uuid.uuid4().hex[:8]}",
            from_node=_trig_from or self._node_id,
            to_node=_trig_to or self._node_id,
            type=MessageType.MESSAGE,
            content=task,
            timestamp=datetime.now(timezone.utc).isoformat(),
            # The synthetic trigger IS the object _handle_worker_complete()
            # later receives, so the session scope has to be seeded here or it
            # is lost for the whole dispatch — no project tag on the brief, no
            # admission charge, and no autonomous closeout on completion.
            metadata=dict(self._current_autonomous_scope or {}),
        )
        spec = {
            "task": task,
            "task_type": task_type,
            "backend": backend,
            "reason": reason,
        }
        outcome = await self._dispatch_worker(
            trigger,
            spec,
            source="tool",
        )
        active = self._active_worker_slots()
        running = active[-1] if active else None
        if outcome.status == "running":
            status = "dispatched"
        elif outcome.status in {"capacity_full", "duplicate"}:
            status = "already_running"
        else:
            status = outcome.status
        payload = {
            "status": status,
            "dispatch_status": outcome.status,
            "message": outcome.message,
            "worker_id": outcome.worker_id,
            "worker_backend": outcome.backend,
            "worker_task_type": trigger.metadata.get("worker_task_type"),
            "worker_backend_reason": trigger.metadata.get("worker_backend_reason"),
            "worker_backend_user_override": bool(
                trigger.metadata.get("worker_backend_user_override")
            ),
            "worker_pev": trigger.metadata.get("worker_pev"),
            "worker_prompt_config": trigger.metadata.get("worker_prompt_config"),
            "dispatch_key": outcome.dispatch_key or None,
            "slot_index": outcome.slot_index,
            "task": outcome.task_description,
            "brief_tier": outcome.brief_tier,
            "acknowledgment_source": outcome.acknowledgment_source,
            "launched_worker_ids": list(
                getattr(self, "_router_call_worker_launches", [])
            ),
        }
        if status == "already_running" or outcome.status == "duplicate_running_task":
            running_task = (
                running.task_description
                if running is not None
                else getattr(self, "_current_task_description", "")
            )
            if len(running_task) > 160:
                running_task = running_task[:160] + "..."
            payload.update({
                "running_worker_id": (
                    running.worker_id if running is not None
                    else outcome.worker_id
                ),
                "running_worker_backend": (
                    running.backend if running is not None
                    else getattr(self, "_current_worker_backend", None)
                ),
                "running_worker_task": running_task,
                "running_worker_elapsed_seconds": (
                    round(time.monotonic() - running.start_time, 1)
                    if running is not None
                    and getattr(running, "start_time", None)
                    else None
                ),
                "active_worker_count": len(active),
                "max_concurrent_workers": self._configured_worker_capacity(),
            })
        if outcome.status == "refused":
            payload["requested_task_type"] = task_type
            payload["requested_backend"] = backend
        backend_warning = trigger.metadata.get("worker_backend_warning")
        if backend_warning:
            payload["worker_backend_warning"] = backend_warning
        return _json.dumps(payload)

    async def dispatch_controller_worker(
        self,
        task: str,
        *,
        backend: str,
        budget_seconds: int,
        allowed_tools: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Launch and await one controller-owned leaf through the normal worker path."""
        from datetime import datetime, timezone

        trigger = Message(
            id=f"controller-{uuid.uuid4().hex[:12]}",
            from_node=self._node_id,
            to_node=self._node_id,
            type=MessageType.MESSAGE,
            content=task,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "autonomous_controller_leaf": True,
                "autonomous_controller_allowed_tools": list(allowed_tools),
            },
        )
        selection = self._stage_worker_selection_metadata(
            trigger,
            {"backend": backend, "reason": "Explicit autonomous-pilot backend"},
        )
        if not selection.backend or selection.backend != backend:
            return {
                "error": f"controller backend override {backend!r} was rejected",
                "backend": selection.backend or None,
            }

        previous_description = self._current_task_description
        self._current_task_description = task
        launched = await self._start_worker(trigger)
        if not launched:
            self._current_task_description = previous_description
            return {
                "error": "worker capacity is full; controller leaf was not launched",
                "backend": backend,
            }

        receipt = getattr(self, "_last_dispatch_receipt", None)
        worker_id = receipt.worker_id if receipt is not None else None
        if not worker_id:
            return {
                "error": "worker launched without an observable worker id",
                "backend": backend,
            }
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        self._controller_worker_waiters[worker_id] = waiter
        try:
            outcome = await asyncio.wait_for(
                asyncio.shield(waiter), timeout=max(1, budget_seconds)
            )
            return dict(outcome)
        except asyncio.TimeoutError:
            await self.cancel_worker(worker_id=worker_id)
            return {
                "worker_id": worker_id,
                "backend": backend,
                "error": f"controller leaf exceeded {budget_seconds}s",
            }
        finally:
            self._controller_worker_waiters.pop(worker_id, None)

    def _resolve_controller_worker(
        self,
        worker_id: str | None,
        outcome: dict[str, Any],
    ) -> None:
        if not worker_id:
            return
        waiter = self._controller_worker_waiters.get(worker_id)
        if waiter is not None and not waiter.done():
            waiter.set_result(outcome)

    def _skill_draft_recent_context(self) -> str:
        """Render the existing worker snapshot as auditable drafting evidence."""
        rendered: list[str] = []
        for turn in self._build_worker_context():
            content = turn.content if isinstance(turn.content, str) else str(turn.content)
            if not content.strip():
                continue
            labels = [f"role={turn.role}"]
            if turn.from_node:
                labels.append(f"from={turn.from_node}")
            if turn.to_node:
                labels.append(f"to={turn.to_node}")
            if turn.meta.get("worker_report_received"):
                labels.append("worker_report=true")
            if turn.meta.get("worker_origin"):
                labels.append(f"worker_origin={turn.meta['worker_origin']}")
            rendered.append(f"[{' '.join(labels)}]\n{content.strip()}")
        return "\n\n".join(rendered)

    async def _tool_skill_draft(
        self,
        task_summary: str,
        source_files: list[str] | None = None,
        trace_path: str = "",
    ) -> str:
        """Launch an on-demand skill-card drafting worker in the normal slot."""
        import json as _json
        from datetime import datetime, timezone

        summary = " ".join(str(task_summary or "").split())
        if not summary:
            return _json.dumps({
                "status": "error",
                "message": "skill_draft.task_summary is required.",
            })
        if source_files is None:
            source_files = []
        if not isinstance(source_files, list) or not all(
            isinstance(path, str) and path.strip() for path in source_files
        ):
            return _json.dumps({
                "status": "error",
                "message": "skill_draft.source_files must be a list of absolute paths.",
            })
        if not isinstance(trace_path, str):
            return _json.dumps({
                "status": "error",
                "message": "skill_draft.trace_path must be an absolute path string.",
            })
        trace_path = trace_path.strip()
        if trace_path and not Path(trace_path).is_absolute():
            return _json.dumps({
                "status": "error",
                "message": "skill_draft.trace_path must be an absolute path.",
            })

        if not hasattr(self, "_router_call_worker_launches"):
            self._router_call_worker_launches = []
        if getattr(self, "_router_call_worker_launches", []):
            return _json.dumps({
                "status": "duplicate_in_turn",
                "message": (
                    "Only one worker-owning tool may launch per router turn; "
                    "the skill draft was not started."
                ),
                "launched_worker_ids": list(self._router_call_worker_launches),
            })

        try:
            package = build_skill_draft_package(
                self._nickname,
                summary,
                recent_context=self._skill_draft_recent_context(),
                source_files=source_files,
                trace_path=trace_path or None,
                trace_root=(
                    self._state_paths.worker_traces_dir
                    if self._state_paths is not None
                    else None
                ),
                staging_root=(
                    self._state_paths.skill_drafts_dir
                    if self._state_paths is not None
                    else None
                ),
                isolation_policy=getattr(self, "_isolation_policy", None),
            )
        except SkillCardError as exc:
            return _json.dumps({
                "status": "error",
                "message": str(exc),
            })

        trigger_from, trigger_to = self._trigger_nodes()
        trigger = Message(
            id=f"skill-draft-{uuid.uuid4().hex[:8]}",
            from_node=trigger_from or self._node_id,
            to_node=trigger_to or self._node_id,
            type=MessageType.MESSAGE,
            content=package.task,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "skill_draft": True,
                "skill_draft_run_id": package.run_id,
                "skill_draft_staging_path": str(package.staging_path),
                "skill_draft_owner": self._nickname,
                "skill_draft_source_files": [
                    str(path) for path in package.source_files
                ],
                "skill_draft_trace_path": (
                    str(package.trace_path) if package.trace_path else None
                ),
                "skill_draft_trace_fingerprint": package.trace_fingerprint,
                "skill_draft_trace_truncated": package.trace_truncated,
            },
        )

        previous_task = self._current_task_description
        self._current_task_description = package.task
        logger.info(
            "[SKILLS] LAUNCH DRAFT: %s drafting %r (run_id=%s)",
            self._nickname,
            summary[:120],
            package.run_id,
        )
        launched = await self._start_worker(trigger)
        if not launched:
            self._current_task_description = previous_task
            try:
                package.staging_path.parent.rmdir()
            except OSError:
                pass
            return _json.dumps({
                "status": "already_running",
                "running_worker_id": self._current_worker_id,
                "message": (
                    "The worker slot is occupied; no drafting worker was started."
                ),
            })

        receipt = getattr(self, "_last_dispatch_receipt", None)
        worker_id = receipt.worker_id if receipt is not None else None
        worker_slot = self._slot_for_worker(worker_id)
        if worker_id:
            self._router_call_worker_launches.append(worker_id)
        self._last_worker_launch = {
            "worker_id": worker_id,
            "worker_backend": (
                worker_slot.backend if worker_slot is not None else None
            )
            or self._default_worker_backend
            or None,
            "worker_backend_reason": None,
            "task": f"Draft skill card: {summary}",
            "status": "dispatched",
            "skill_draft_run_id": package.run_id,
        }
        return _json.dumps({
            "status": "dispatched",
            "tool": "skill_draft",
            "worker_id": worker_id,
            "owner_agent": self._nickname,
            "run_id": package.run_id,
            "staging_path": str(package.staging_path),
            "proposal_directory": str(self._skill_store.agent_dir / ".proposals"),
            "trace_path": str(package.trace_path) if package.trace_path else None,
            "trace_fingerprint": package.trace_fingerprint,
            "trace_truncated": package.trace_truncated,
            "human_approval_required": True,
            "message": (
                "Drafting worker launched. It can create only a validated proposal; "
                "activation remains a separate human action."
            ),
        })

    async def _tool_worker_list(self) -> str:
        """Return a compact view of every stable slot."""
        return json.dumps({
            "revision": int(getattr(self, "_slot_revision", 0)),
            "capacity": len(self._ensure_slot_table()),
            "active": self._active_worker_count(),
            "slots": [
                self._slot_summary(slot) for slot in self._ensure_slot_table()
            ],
        })

    def _render_user_worker_status(self) -> str:
        """Render the worker list as a concise human-facing status reply.

        ``_tool_worker_list`` deliberately remains JSON for the router tool
        contract. This renderer is the separate delivery surface for the
        user-only diagnostic fast path.
        """
        active = self._active_worker_slots()
        if not active:
            return "Idle - no workers running."

        now = time.monotonic()
        descriptions = []
        for slot in active:
            elapsed = max(0, int(now - slot.start_time)) if slot.start_time else 0
            worker_id = slot.worker_id or f"worker slot {slot.index}"
            task = self._bounded_slot_text(slot.task_description, 160)
            if not task:
                task = "the current request"
            descriptions.append(f"{worker_id}, {elapsed}s elapsed - {task}")

        if len(descriptions) == 1:
            return f"Working: {descriptions[0]}."
        return "Workers running:\\n" + "\\n".join(
            f"- {description}" for description in descriptions
        )

    async def _tool_worker_status(
        self,
        worker_id: str | None = None,
        max_lines: int = 100,
    ) -> str:
        """Return compact status or a targeted worker transcript."""
        active = self._active_worker_slots()
        if not worker_id:
            if len(active) != 1:
                return await self._tool_worker_list()
            worker_id = active[0].worker_id

        slot = self._slot_for_worker(worker_id)
        if slot is None:
            return json.dumps({
                "status": "not_found",
                "worker_id": worker_id,
                "slots": [
                    self._slot_summary(item)
                    for item in self._ensure_slot_table()
                ],
            })

        progress = (
            slot.snapshot[slot.snapshot_start:]
            if slot.snapshot_start < len(slot.snapshot)
            else []
        )
        max_lines = max(1, min(int(max_lines), 500))
        activity_lines = [
            f"[{turn.from_node or turn.role}] {turn.content}"
            for turn in progress
        ]
        bounded = activity_lines[-max_lines:]
        tool_calls = sum(
            1
            for turn in progress
            if (getattr(turn, "meta", None) or {}).get("tool_calls")
        )
        payload = self._slot_summary(slot)
        payload.update({
            "status": "ok",
            "tool_calls_so_far": tool_calls,
            "activity_lines": (
                "\n".join(bounded) if bounded else "[No activity yet]"
            ),
            "full_transcript_available": (
                len(activity_lines) > max_lines
            ),
            "dispatch_key": slot.dispatch_key,
            "router_turn_id": slot.router_turn_id,
        })
        return json.dumps(payload)

    async def _tool_worker_cancel(
        self,
        worker_id: str | None = None,
        cancel_all: bool = False,
        reason: str = "",
    ) -> str:
        """Cancel an explicit worker or all workers; ambiguity fails closed."""
        active = self._active_worker_slots()
        if cancel_all:
            cancelled: list[str] = []
            for slot in list(active):
                if await self.cancel_worker(
                    worker_id=slot.worker_id,
                    reason=reason,
                ):
                    cancelled.append(slot.worker_id or "")
            return json.dumps({
                "status": "cancelled",
                "cancelled_worker_ids": cancelled,
            })
        if worker_id is None:
            if len(active) == 1:
                worker_id = active[0].worker_id
            elif len(active) > 1:
                return json.dumps({
                    "status": "ambiguous",
                    "message": (
                        "Multiple workers are active; supply worker_id or "
                        "set cancel_all=true."
                    ),
                    "slots": [self._slot_summary(slot) for slot in active],
                })
            else:
                return json.dumps({
                    "status": "not_running",
                    "slots": [
                        self._slot_summary(slot)
                        for slot in self._ensure_slot_table()
                    ],
                })
        cancelled = await self.cancel_worker(
            worker_id=worker_id,
            reason=reason,
        )
        return json.dumps({
            "status": "cancelled" if cancelled else "not_found",
            "worker_id": worker_id,
        })


    def _trigger_nodes(self) -> tuple[str | None, str | None]:
        """Resolve the current trigger's (from_node, to_node).

        Prefers the per-async-task contextvar set by _call_router_full over the
        shared instance attributes, so a CC-monitor delivery and a concurrent
        BUSY handler don't clobber each other's reply destination (Bug 9)."""
        ctx = _CC_TRIGGER_CTX.get()
        if ctx is not None:
            return ctx
        return self._current_trigger_from_node, self._current_trigger_to_node


    # =========================================================================
    # Response sending with history storage
    # =========================================================================

    @staticmethod
    def _sanitize_outbound(text: str) -> str:
        """Strip internal LLM artifacts from outbound messages.

        Removes <thinking> blocks, raw XML tool calls, and <invoke> tags
        that DeepSeek sometimes emits in forced-synthesis or freeform output.
        Content inside backtick spans (inline `code` and fenced ```blocks```)
        is preserved — stripping only applies to bare XML outside code.
        """
        # Protect backtick-wrapped content from XML stripping.
        # Inside backticks, strip XML tags but keep their text content
        # (so `<file_read>/tmp</file_read>` becomes `/tmp`, not ``).
        _placeholders: list[str] = []
        _xml_tag_re = re.compile(
            r"</?(?:thinking|bash_exec|file_read|file_edit|file_create|"
            r"file_write|invoke|tool_call)\b[^>]*>"
        )

        # NUL-delimited placeholders cannot be produced by normal model text,
        # but choose a collision-free prefix anyway so even adversarial input
        # containing a literal ``\x00BT0\x00`` round-trips safely.
        _placeholder_prefix = "\x00BT"
        while _placeholder_prefix in text:
            _placeholder_prefix += "_"

        def _hold(m: re.Match) -> str:
            # Code examples may intentionally show internal tool XML. Remove
            # the recognized tags while retaining the example's text.
            cleaned = _xml_tag_re.sub("", m.group(0))
            _placeholders.append(cleaned)
            return f"{_placeholder_prefix}{len(_placeholders) - 1}\x00"

        # Match a balanced run of one or more backticks. Using the same run
        # length as the closing delimiter supports fenced blocks and Markdown
        # spans such as ``code containing `backticks` ``. Unbalanced runs are
        # deliberately left unprotected and therefore cannot swallow the rest
        # of an outbound message.
        _backtick_span_re = re.compile(
            r"(?<!`)(?P<ticks>`+)(?!`)(?P<body>.*?)(?<!`)(?P=ticks)(?!`)",
            re.DOTALL,
        )
        text = _backtick_span_re.sub(_hold, text)

        # Strip <thinking>...</thinking> blocks (DeepSeek reasoning)
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
        # Strip common XML tool call blocks
        for tag in (
            "bash_exec", "file_read", "file_edit", "file_create",
            "file_write", "invoke", "tool_call",
        ):
            text = re.sub(rf"<{tag}\b[^>]*>.*?</{tag}>", "", text, flags=re.DOTALL)
        # Strip self-closing variants like <no_response/>
        text = re.sub(r"<(?:bash_exec|file_read|file_edit|file_create|file_write|invoke|tool_call)\b[^/]*/\s*>", "", text)
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Restore protected backtick content
        def _restore(m: re.Match) -> str:
            return _placeholders[int(m.group(1))]

        text = re.sub(
            re.escape(_placeholder_prefix) + r"(\d+)\x00", _restore, text
        )

        return text.strip()

    async def _send_and_store(
        self,
        content: str,
        in_reply_to: Message | None,
        meta: dict | None = None,
        *,
        history_content: str | None = None,
        include_tool_visibility: bool = True,
        mark_router_turn_sent: bool = True,
    ) -> None:
        """
        Send a response and store it in the router's ConversationHistory.

        All router responses (acks, busy responses, completion responses)
        go through this method to ensure they appear in history.
        """
        from datetime import datetime, timezone

        content = self._sanitize_outbound(content)
        if not content:
            logger.debug("_send_and_store: content empty after sanitization, skipping")
            return

        # Harness-backed routers return natural text after running their own
        # internal tool loop, bypassing AgentNode._execute_send_message().
        # Consume the same per-turn ledger here so both delivery paths expose
        # an identical Contract §5 block. RouterV2 is never the worker's
        # delivery surface, preserving the worker-context exclusion.
        if include_tool_visibility and self._tool_visibility_pending:
            content = append_tools_called_block(content, self._last_router_call_tools)
            self._tool_visibility_pending = False

        # Send the message
        await self._send_fn(content, in_reply_to)
        if mark_router_turn_sent:
            self._last_router_call_sent_message = True

        # Store as a Turn in ConversationHistory (durable, summarizable).
        # Dispatch acknowledgments are user-facing receipts; the model-facing
        # history instead keeps the request shape it must reproduce.
        persisted_content = (
            self._sanitize_outbound(history_content)
            if history_content is not None
            else content
        )
        if not persisted_content:
            persisted_content = content
        turn_meta = {"router_response": True}
        if meta:
            turn_meta.update(meta)
        self._append_turn(Turn(
            role="outgoing",
            content=persisted_content,
            timestamp=datetime.now(timezone.utc),
            from_node=self._node_id or self._nickname,
            to_node=in_reply_to.from_node if in_reply_to else "",
            meta=turn_meta,
        ))

    # =========================================================================
    # Worker lifecycle
    # =========================================================================

    async def _select_memory_context(self, query: str, max_entries: int = 2) -> str:
        """Select top memories and render as XML for worker injection.

        Called by _start_worker when memory_retrieval_redesign_enabled.
        The returned block is authoritative for that worker preparation and is
        carried to the worker on its own trigger metadata.  The instance fields
        remain compatibility views for router prompt dedup and diagnostics
        only; callers must not reread them after this coroutine yields, and
        worker prompt assembly must never read them at all — under concurrency
        the last dispatch to select would otherwise overwrite what an earlier,
        still-running worker sees.
        """
        selected_ids: set[str] = set()
        self._injected_memory_ids = selected_ids
        self._injected_memory_context = ""

        if not self._memory or not isinstance(self._memory, MemorySystemV2):
            return ""
        if not query:
            return ""

        map_context = self._get_last_n_turns_text(5)
        toc = await self._memory.build_toc(
            query_text=query,
            k=getattr(self._config, "memory_toc_size", 30),
            context_text=map_context,
        )
        if not toc:
            return ""

        selected = [e for e in toc[:max_entries] if e.score > 0.3]
        if not selected:
            return ""

        parts: list[str] = []
        for entry in selected:
            full = self._memory._store.get(entry.id) if self._memory._store else None
            if full is None:
                continue
            selected_ids.add(entry.id)
            text = full.reflection if full.reflection else full.summary
            date_str = full.created_at.strftime("%Y-%m-%d")
            parts.append(
                f'<memory id="{entry.id}" date="{date_str}" '
                f'topic="{full.topic_label}" score="{entry.score:.2f}">\n'
                f'{text}\n</memory>'
            )

        if not parts:
            return ""

        # Inject relevant project maps for worker context
        map_context = self._get_last_n_turns_text(5)
        map_block = await self._memory.render_relevant_maps_block(map_context)
        if map_block:
            parts.append(map_block)

        block = (
            "<router_injected_context>\n"
            + "\n".join(parts)
            + "\n</router_injected_context>"
        )
        self._injected_memory_context = block
        logger.info(
            "Router injected %d memories + maps for query='%s'",
            len(parts), query[:80],
        )
        return block

    def _get_last_n_turns_text(self, n: int = 5) -> str:
        """Extract text from the last n conversation turns for context embedding."""
        turns = list(self._history.window)[-n:]
        parts: list[str] = []
        for t in turns:
            text = t.content if isinstance(t.content, str) else str(t.content)
            parts.append(text)
        return "\n".join(parts)

    def _build_worker_context(self, trigger: Message | None = None) -> list[Turn]:
        """Build a token-trimmed worker context snapshot.

        The worker sees the same history the router does — including cancelled
        worker artifacts. Trims to worker_context_window_tokens.
        """
        window = list(self._history.window)
        budget = self._config.worker_context_window_tokens

        # Trim oldest turns to fit within budget (keep most recent)
        total = sum(estimate_tokens(t.content) for t in window)
        start = 0
        while total > budget and start < len(window) - 1:
            total -= estimate_tokens(window[start].content)
            start += 1

        if start > 0:
            logger.info(
                f"RouterV2 worker context: trimmed {start} oldest turns "
                f"to fit within {budget} token budget (~{total} tokens remaining)"
            )
            window = window[start:]

        todo_context = self._render_todo_context(
            self._conversation_id_from_message(
                trigger if trigger is not None else self._pending_trigger
            )
        )
        if todo_context:
            from datetime import datetime, timezone
            window.append(Turn(
                role="system",
                content=todo_context,
                timestamp=datetime.now(timezone.utc),
                from_node="system",
                to_node=self._node_id,
                meta={"context_block": "conversation_todos"},
            ))

        return window

    async def _start_worker(self, trigger: Message) -> bool:
        """Atomically reserve, prepare, and start one worker slot."""
        if not hasattr(self, "_dispatch_lock"):
            self._dispatch_lock = asyncio.Lock()
        if not hasattr(self, "_dispatch_receipts"):
            self._dispatch_receipts = {}
        if not isinstance(getattr(trigger, "metadata", None), dict):
            trigger.metadata = {}
        metadata = trigger.metadata

        self._worker_backend_override = None
        selection = self._validate_staged_worker_selection(trigger)
        # Re-read: selection staging owns the metadata dict, and every stamp
        # below must land on whatever the trigger actually carries into the
        # worker.  Writing to a stale local reference is silent.
        metadata = trigger.metadata
        if selection.refusal:
            self._last_dispatch_receipt = DispatchReceipt(
                dispatch_key="",
                status="refused",
                worker_id=None,
                slot_index=None,
                origin_message_id=str(trigger.id or ""),
                router_turn_id=str(metadata.get("router_turn_id") or ""),
                task_description=str(trigger.content),
                message=selection.refusal,
            )
            logger.warning("[WORKER] DISPATCH REFUSED: %s", selection.refusal)
            return False

        fixed_tool_name = str(metadata.get("fixed_tool") or "").strip()
        task_description = (
            str(metadata.get("worker_task_description") or "").strip()
            or getattr(self, "_current_task_description", "")
            or str(trigger.content)
        )
        metadata["worker_task_description"] = task_description
        task_fingerprint = self._worker_task_key(task_description)

        origin_message_id = str(
            metadata.get("origin_message_id")
            or getattr(self, "_current_origin_message_id", "")
            or trigger.id
            or f"anonymous-{id(trigger):x}"
        )
        router_turn_id = str(
            metadata.get("router_turn_id")
            or getattr(self, "_current_router_turn_id", "")
            or f"direct-{origin_message_id}"
        )
        explicit_ordinal = metadata.get("launch_ordinal")
        if explicit_ordinal is None:
            if getattr(self, "_current_router_turn_id", ""):
                self._current_launch_ordinal = (
                    int(getattr(self, "_current_launch_ordinal", 0)) + 1
                )
                launch_ordinal = self._current_launch_ordinal
            else:
                # Direct compatibility calls are distinct unless the caller
                # explicitly supplies an ordinal for replay.
                launch_ordinal = int(getattr(self, "_worker_id_counter", 0)) + 1
        else:
            launch_ordinal = int(explicit_ordinal)
        metadata["origin_message_id"] = origin_message_id
        metadata["router_turn_id"] = router_turn_id
        metadata["launch_ordinal"] = launch_ordinal
        dispatch_key = uuid.uuid5(
            uuid.NAMESPACE_URL,
            (
                f"{getattr(self, '_node_id', self._nickname)}"
                f"\0{origin_message_id}\0{launch_ordinal}"
            ),
        ).hex
        metadata["dispatch_key"] = dispatch_key

        slot: WorkerSlot | None = None
        worker_id: str | None = None
        async with self._dispatch_lock:
            existing = getattr(self, "_dispatch_receipts", {}).get(dispatch_key)
            if existing is not None:
                self._last_dispatch_receipt = DispatchReceipt(
                    **{
                        **existing.__dict__,
                        "status": "duplicate",
                        "message": "Dispatch replay matched an existing receipt.",
                    }
                )
                return False

            interactive_active = bool(
                getattr(getattr(self, "_cc_mgr", None), "_cc_tmux_session", None)
                or getattr(
                    getattr(self, "_harness_session_mgr", None), "_state", "idle"
                ) in {"starting", "active", "warm_idle", "stopping"}
            )
            active_slots = self._active_worker_slots()
            refusal = ""
            status = ""
            if interactive_active:
                status = "refused"
                refusal = "An interactive session is active and exclusively owns execution."
            elif any(
                slot.task_fingerprint == task_fingerprint
                for slot in active_slots
                if task_fingerprint
            ):
                status = "duplicate"
                refusal = "An active worker already owns the same task fingerprint."
            elif len(active_slots) >= self._configured_worker_capacity():
                status = "capacity_full"
                refusal = "All configured worker slots are occupied."
            else:
                slot = next(
                    (
                        candidate
                        for candidate in self._ensure_slot_table()
                        if not candidate.active
                    ),
                    None,
                )
                if slot is None:
                    status = "capacity_full"
                    refusal = "All configured worker slots are occupied."

            if slot is None:
                receipt = DispatchReceipt(
                    dispatch_key=dispatch_key,
                    status=status,
                    worker_id=None,
                    slot_index=None,
                    origin_message_id=origin_message_id,
                    router_turn_id=router_turn_id,
                    task_description=task_description,
                    backend=selection.backend or None,
                    message=refusal,
                )
                self._dispatch_receipts[dispatch_key] = receipt
                self._last_dispatch_receipt = receipt
                return False

            # Reserve STARTING before any preparation await.
            if slot.lifecycle != WorkerLifecycle.EMPTY:
                self._reset_slot(slot)
            self._worker_id_counter = (
                int(getattr(self, "_worker_id_counter", 0)) + 1
            )
            worker_id = f"{self._nickname}-worker{self._worker_id_counter}"
            started_event = asyncio.Event()
            slot.lifecycle = WorkerLifecycle.STARTING
            slot.worker_id = worker_id
            slot.origin_message_id = origin_message_id
            slot.router_turn_id = router_turn_id
            slot.dispatch_key = dispatch_key
            slot.task_fingerprint = task_fingerprint
            slot.task_description = task_description
            slot.selection_metadata = copy.deepcopy(metadata)
            slot.trigger = trigger
            slot.started_event = started_event
            slot.start_time = time.monotonic()
            slot.backend = selection.backend or None
            slot.pev = selection.pev
            slot.kind = "fixed_tool" if fixed_tool_name else "worker"
            slot.fixed_tool_name = fixed_tool_name or None
            receipt = DispatchReceipt(
                dispatch_key=dispatch_key,
                status="starting",
                worker_id=worker_id,
                slot_index=slot.index,
                origin_message_id=origin_message_id,
                router_turn_id=router_turn_id,
                task_description=task_description,
                backend=slot.backend,
            )
            self._dispatch_receipts[dispatch_key] = receipt
            # Bound process-local tombstones.
            while len(self._dispatch_receipts) > 256:
                self._dispatch_receipts.pop(next(iter(self._dispatch_receipts)))
            self._last_dispatch_receipt = receipt
            self._bump_slot_revision()
            self._sync_worker_compat_views()
            self._set_current_worker_slot(slot)
            self._state = RouterState.BUSY

        assert slot is not None and worker_id is not None
        controller_leaf = bool(metadata.get("autonomous_controller_leaf"))
        try:
            use_toc = getattr(
                self._config, "memory_retrieval_redesign_enabled", False
            )
            if use_toc and not fixed_tool_name and not controller_leaf:
                query = (
                    trigger.content
                    if isinstance(trigger.content, str)
                    else str(trigger.content)
                )
                memory_context = await self._select_memory_context(query)
                metadata["worker_injected_memory_context"] = memory_context

            selected_skill_ids: tuple[str, ...] = ()
            skill_selected_at: str | None = None
            skill_store = getattr(self, "_skill_store", None)
            if not fixed_tool_name and not controller_leaf and skill_store is not None:
                try:
                    selections = skill_store.select_with_scores(task_description)
                    skill_context = skill_store.render_selected_block(selections)
                    if selections and skill_context:
                        from datetime import datetime, timezone

                        selected_skill_ids = tuple(
                            item.card["id"] for item in selections
                        )
                        skill_selected_at = datetime.now(timezone.utc).isoformat()
                        metadata["governed_skill_ids"] = list(selected_skill_ids)
                        metadata["governed_skill_context"] = skill_context
                        metadata["governed_skill_selected_at"] = skill_selected_at
                except (OSError, SkillCardError) as exc:
                    logger.warning("[SKILLS] Worker retrieval failed closed: %s", exc)

            if controller_leaf:
                snapshot = []
            else:
                context_builder = self._build_worker_context
                if inspect.signature(context_builder).parameters:
                    snapshot = context_builder(trigger)
                else:
                    # Compatibility for focused tests and extensions that
                    # still expose the historical zero-argument hook.
                    snapshot = context_builder()
            slot.snapshot = snapshot
            slot.snapshot_start = len(snapshot)
            slot.skill_card_ids = selected_skill_ids
            slot.skill_selected_at = skill_selected_at
            slot.selection_metadata = copy.deepcopy(metadata)

            context_factory = getattr(
                getattr(self, "_worker_agent", None),
                "_create_worker_execution_context",
                None,
            )
            execution_context = None
            if callable(context_factory):
                execution_context = context_factory(
                    worker_id=worker_id,
                    trigger=trigger,
                    task_description=task_description,
                    snapshot=snapshot,
                    started_event=slot.started_event,
                )
            slot.execution_context = execution_context

            try:
                worker_coro = self._run_worker(
                    worker_id,
                    trigger,
                    snapshot,
                    execution_context=execution_context,
                )
            except TypeError:
                # Compatibility for focused tests that replace _run_worker
                # with the historical three-argument coroutine.
                try:
                    worker_coro = self._run_worker(
                        worker_id, trigger, snapshot
                    )
                except TypeError:
                    worker_coro = self._run_worker(trigger)
            slot.task = asyncio.create_task(worker_coro)
            self._sync_worker_compat_views()
            self._set_current_worker_slot(slot)

            if execution_context is not None and slot.started_event is not None:
                await asyncio.wait_for(slot.started_event.wait(), timeout=10.0)
                startup_error = getattr(execution_context, "startup_error", None)
                if startup_error:
                    raise RuntimeError(startup_error)
            elif slot.started_event is not None:
                # Bare worker functions used by focused tests have no
                # AgentNode context installation step.
                slot.started_event.set()

            if slot.worker_id != worker_id:
                # A very short worker may complete and release its slot after
                # signalling startup but before this coroutine resumes. It was
                # admitted and did run; never resurrect the cleared slot as a
                # phantom RUNNING worker.
                running_receipt = DispatchReceipt(
                    **{
                        **receipt.__dict__,
                        "status": "running",
                    }
                )
                self._dispatch_receipts[dispatch_key] = running_receipt
                self._last_dispatch_receipt = running_receipt
                return True

            if (
                slot.task is not None
                and slot.task.done()
                and slot.task.exception() is not None
            ):
                raise slot.task.exception()
            slot.lifecycle = WorkerLifecycle.RUNNING
            running_receipt = DispatchReceipt(
                **{
                    **receipt.__dict__,
                    "status": "running",
                }
            )
            self._dispatch_receipts[dispatch_key] = running_receipt
            self._last_dispatch_receipt = running_receipt
            self._bump_slot_revision()
            self._sync_worker_compat_views()
            self._set_current_worker_slot(slot)

            selection_stamp = self._selection_stamp(metadata, slot.backend)
            if not fixed_tool_name:
                try:
                    self._start_flush_monitor(trigger, worker_id=worker_id)
                except TypeError:
                    # Historical focused tests replace this method with the
                    # original one-argument callback.
                    self._start_flush_monitor(trigger)
            try:
                self._start_watchdog(trigger, worker_id=worker_id)
            except TypeError:
                self._start_watchdog(trigger)
            if hasattr(self, "_last_router_call_tools"):
                self._last_router_call_tools.append(
                    ("worker_launch", selection_stamp or "")
                )
            logger.info(
                "[WORKER] START: %s worker %s slot=%d task=%r",
                self._nickname,
                worker_id,
                slot.index,
                self._bounded_slot_text(task_description, 100),
            )
            return True
        except Exception as exc:
            logger.exception("[WORKER] startup failed for %s", worker_id)
            if slot.task is not None and not slot.task.done():
                slot.task.cancel()
            slot.lifecycle = WorkerLifecycle.FAILED
            slot.failure = str(exc)
            slot.completed_at = time.monotonic()
            failed = DispatchReceipt(
                **{
                    **receipt.__dict__,
                    "status": "start_failed",
                    "message": str(exc),
                }
            )
            self._dispatch_receipts[dispatch_key] = failed
            self._last_dispatch_receipt = failed
            self._bump_slot_revision()
            self._sync_worker_compat_views()
            remaining = self._refresh_primary_worker_slot()
            self._state = (
                RouterState.BUSY if remaining is not None else RouterState.IDLE
            )
            if (
                not hasattr(self, "_worker_agent")
                and not hasattr(self, "_worker_fn")
                and isinstance(exc, RuntimeError)
            ):
                # Compatibility with the historical guard test, which uses a
                # deliberately incomplete RouterV2 and a sentinel preparation
                # exception to prove that admission proceeded.
                raise
            return False

    async def _run_worker(
        self,
        worker_id: str,
        trigger: Message,
        snapshot: list[Turn],
        execution_context: Any | None = None,
    ) -> None:
        """Run worker and handle completion/error."""
        try:
            if execution_context is not None:
                result = await self._worker_fn(
                    snapshot,
                    trigger,
                    execution_context,
                )
            else:
                result = await self._worker_fn(snapshot, trigger)
            if result.error is not None:
                await self._handle_worker_error(
                    result.error,
                    trigger,
                    worker_id=worker_id,
                )
            else:
                await self._handle_worker_complete(
                    result,
                    trigger,
                    worker_id=worker_id,
                )
        except asyncio.CancelledError:
            slot = self._slot_for_worker(worker_id)
            elapsed_s = round(time.monotonic() - slot.start_time, 1) if slot else 0
            logger.info(f"[WORKER] CANCELLED: {self._nickname} worker {worker_id} cancelled after {elapsed_s}s")
            raise
        except Exception as e:
            await self._handle_worker_error(e, trigger, worker_id=worker_id)

    def _record_skill_outcomes(
        self,
        trigger: Message,
        worker_id: str | None,
        *,
        result: str,
        note: str,
    ) -> None:
        """Append fail-closed runtime receipts for cards selected at dispatch."""
        metadata = trigger.metadata if isinstance(trigger.metadata, dict) else {}
        card_ids = metadata.get("governed_skill_ids") or []
        skill_store = getattr(self, "_skill_store", None)
        if not isinstance(card_ids, list) or not card_ids or skill_store is None:
            return
        slot = getattr(self, "_worker_slots", {}).get(worker_id or "")
        task_summary = (
            slot.task_description if slot is not None
            else (trigger.content if isinstance(trigger.content, str) else str(trigger.content))
        )
        selected_at = metadata.get("governed_skill_selected_at")
        memory_id = metadata.get("canonical_memory_id") or metadata.get("memory_id")
        if not isinstance(memory_id, str):
            memory_id = None
        for card_id in card_ids:
            if not isinstance(card_id, str):
                continue
            try:
                skill_store.append_outcome(
                    card_id,
                    task_summary=task_summary,
                    task_ref=trigger.id or worker_id or "worker",
                    result=result,
                    disposition="unknown",
                    selected_at=selected_at if isinstance(selected_at, str) else None,
                    memory_id=memory_id,
                    note=note,
                )
            except (OSError, SkillCardError) as exc:
                logger.warning(
                    "[SKILLS] Failed to append outcome for %s: %s",
                    card_id,
                    exc,
                )

    # =========================================================================
    # Worker Synthesis
    # =========================================================================

    def _render_worker_trace(
        self,
        worker_id: str,
        result: WorkerResult,
        *,
        max_result_lines: int | None,
    ) -> str:
        """Render the full worker evidence, optionally truncating each result.

        Combines two sources:
        1. In-flight LLM history — all LLM responses, tool calls, tool results
        2. Cumulative CC events — tool call/result data with per-result line caps
        """
        def _truncate(text: str) -> str:
            if max_result_lines is None:
                return text
            lines = text.split('\n')
            if len(lines) <= max_result_lines:
                return text
            remaining = len(lines) - max_result_lines
            return (
                '\n'.join(lines[:max_result_lines])
                + f"\n  ... ({remaining} more lines truncated)"
            )

        parts = [f"<worker_trace worker='{worker_id}'>"]
        has_content = False
        history_has_tool_content = False

        # Source 1: In-flight LLM history (HistoryMessage objects)
        history = result.worker_in_flight_history
        if history:
            for i, msg in enumerate(history):
                role = getattr(msg, 'role', getattr(msg, 'from_node', 'unknown'))
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                parts.append(f"\n[Turn {i+1}, role={role}]")
                parts.append(_truncate(content))
                if content.strip():
                    has_content = True
                    if (
                        "Tool execution results:" in content
                        or "[CC Tool Activity]" in content
                    ):
                        history_has_tool_content = True

        # Source 2: Full CC tool events (cumulative across all iterations)
        # The bounded synthesis prompt already receives tool calls/results in
        # worker_in_flight_history for harness/Codex-style workers.  Repeating
        # the raw CC event stream there doubled Ada's 2026-07-20 report prompt.
        # Keep raw events in the unbounded archive, or as a fallback when the
        # history stream is empty.
        cc_events = result.worker_cc_events or []
        if cc_events and (max_result_lines is None or not history_has_tool_content):
            parts.append("\n[CC Tool Events]")
            for event in cc_events:
                if event.event_type == "tool_call":
                    args_str = str(event.data) if isinstance(event.data, dict) else str(event.data)
                    parts.append(f"[{event.tool_name}] {args_str}")
                    has_content = True
                elif event.event_type == "tool_result":
                    result_str = str(event.data)
                    parts.append(f"  → {_truncate(result_str)}")

        # Source 3: Buffered messages (fallback for backends like codex-5.5
        # that run their own internal tool loop and return a single response,
        # leaving in-flight history and CC events empty).
        if result.buffered_messages and (
            not has_content or max_result_lines is None
        ):
            parts.append("\n[Worker Output Messages]")
            for to_node, msg_content in result.buffered_messages:
                dest = to_node or "user"
                parts.append(f"\n[Message to {dest}]")
                parts.append(_truncate(msg_content))
                has_content = True

        # The terminal assistant response is part of the durable evidence even
        # when a backend reported through a separate buffered send_report.
        # Keep synthesis behavior unchanged; only the unbounded archive adds it.
        if max_result_lines is None and str(result.response or "").strip():
            parts.append("\n[Final Worker Response]")
            parts.append(str(result.response).strip())

        parts.append("\n</worker_trace>")
        return "\n".join(parts)

    def _build_worker_trace(self, worker_id: str, result: WorkerResult) -> str:
        """Build the bounded ephemeral trace used by synthesis."""
        return self._render_worker_trace(
            worker_id,
            result,
            max_result_lines=self._config.synthesis_trace_max_lines,
        )

    def _build_full_worker_trace(self, worker_id: str, result: WorkerResult) -> str:
        """Build the untruncated trace persisted for later skill distillation."""
        return self._render_worker_trace(
            worker_id,
            result,
            max_result_lines=None,
        )

    def _synthesis_timeout_seconds(self) -> float:
        """Return the backend-configured worker synthesis timeout."""
        config = getattr(getattr(self, "_llm_client", None), "config", None)
        raw_timeout = getattr(config, "synthesis_timeout", 180)
        try:
            timeout = float(raw_timeout)
        except (TypeError, ValueError):
            timeout = 180.0
        return max(timeout, 1.0)

    @staticmethod
    def _extract_report_content(result: WorkerResult) -> str:
        """Return the worker report body when the worker used send_report."""
        if not getattr(result, "report_sent", False):
            return ""
        messages = [
            str(content).strip()
            for _, content in (result.buffered_messages or [])
            if str(content).strip()
        ]
        return "\n\n".join(messages)

    def _append_worker_report_turn(
        self,
        worker_id: str | None,
        *,
        submitted: bool,
    ) -> None:
        """Record that a worker completion is entering the report path."""
        from datetime import datetime, timezone as _tz

        note = (
            "completed and submitted report"
            if submitted
            else "completed without send_report; report synthesized from its "
            "output to close the autonomous session"
        )
        self._append_turn(Turn(
            role="system",
            content=f"[Worker {worker_id} {note}]",
            timestamp=datetime.now(_tz.utc),
            from_node="system",
            meta={
                "worker_report_received": True,
                "worker_id": worker_id,
                "worker_report_synthesized": not submitted,
            },
        ))

    def _build_worker_report_trigger(
        self,
        worker_id: str | None,
        report_content: str,
        trigger: Message,
        *,
        synthesized: bool = False,
    ) -> Message:
        """Build the internal trigger that re-enters the router with tools.

        Shared by the send_report path and the autonomous closeout reroute so
        both re-enter the ReAct loop through exactly one construction.
        """
        from datetime import datetime, timezone as _tz

        autonomous_metadata = self.autonomous_completion_metadata(trigger)
        autonomous_report_to = str(
            autonomous_metadata.get("autonomous_report_to") or ""
        ).strip()

        return Message(
            type=MessageType.MESSAGE,
            from_node=f"worker:{worker_id}",
            to_node=self._node_id,
            content=report_content,
            timestamp=datetime.now(_tz.utc).isoformat(),
            metadata={
                "worker_report": True,
                "worker_id": worker_id,
                "report_fallback_content": report_content,
                "worker_report_synthesized": synthesized,
                # Report-as-trigger processing happens after the worker
                # slot is cleaned up. Carry the original task's reply
                # target explicitly so AgentNode never infers the dead
                # worker as the completion destination.
                "response_destination": (
                    autonomous_report_to
                    or (
                        trigger.to_node
                        if trigger.to_node
                        and trigger.to_node.startswith("channel:")
                        else trigger.from_node
                    )
                ),
                # Trusted autonomous-session scope (§17 item 2). Set by
                # _dispatch_worker() at admission; router-minted, never
                # model-supplied. Absent for interactive dispatches.
                **autonomous_metadata,
            },
        )

    @staticmethod
    def _format_unsynthesized_worker_report(
        worker_id: str,
        report_content: str,
        *,
        reason: str,
    ) -> str:
        """Mark and return a worker report delivered without router synthesis."""
        report_content = report_content.strip()
        return (
            f"[Unsynthesized worker report from {worker_id}: "
            f"router synthesis {reason}; delivering the worker report verbatim.]\n\n"
            f"{report_content}"
        )

    def _truncate_tool_result(
        self,
        content: str,
        max_lines: int | None = None,
        max_chars: int | None = None,
    ) -> tuple[str, bool, int, int]:
        """Truncate a tool result for history-append (trace-as-history C2).

        Returns (content, was_truncated, original_lines, original_chars).
        Strategy:
          1. If content has more than max_lines newlines, drop the tail.
          2. Otherwise (or if still too long after line trim), cap by max_chars.
        Uses config defaults when args are None.

        Plan: docs/plans/trace-as-history-impl-2026-04-27.md (C2)
        Spec: docs/plans/trace-as-history-2026-04-27.md §2.4
        """
        # C3: read directly from RouterV2Config (now a real field). The
        # getattr fallback is kept as defense-in-depth for tests or older
        # configs that might not have the field.
        if max_lines is None:
            max_lines = getattr(self._config, "tool_result_max_lines", 80)
        if max_chars is None:
            max_chars = getattr(self._config, "tool_result_max_chars", 6400)

        original_chars = len(content)
        lines = content.split("\n")
        original_lines = len(lines)
        truncated = False

        if original_lines > max_lines:
            remaining = original_lines - max_lines
            content = "\n".join(lines[:max_lines])
            content += (
                f"\n[truncated: {remaining} more lines, "
                f"{original_chars} chars total]"
            )
            truncated = True

        # Final char-cap pass (handles single-line dumps + already-truncated)
        if len(content) > max_chars:
            cut = content[:max_chars]
            content = (
                f"{cut}\n[truncated: {original_chars - max_chars} more chars, "
                f"{original_chars} chars total]"
            )
            truncated = True

        return content, truncated, original_lines, original_chars

    def _extract_trace_turns(
        self,
        result: WorkerResult,
        worker_id: str,
    ) -> list[Turn]:
        """Convert a WorkerResult's in-flight history / CC events into a list
        of Turn objects suitable for appending to the conversation history.

        Output ordering: matches the time order of the worker's execution.
        Tool-call Turns get role="assistant"; tool-result Turns get role="tool".
        The final-text Turn is NOT included here — the caller is responsible
        for appending it (it has different meta and send-side semantics).

        Sources:
          - result.worker_in_flight_history (list[HistoryMessage]) for
            mesh-harness, openai, anthropic backends.
          - result.worker_cc_events (list[CCToolEvent]) for CC backends.

        Scope (C2.1 fix): On the HistoryMessage path, only entries with
        source == "in_flight" are converted — these are the inner-loop
        appends from mesh/harness/loop.py. Pre-dispatch history (the mesh
        conversation blob from _build_history_for_llm()) carries the default
        source == "persisted" and is excluded. Using `source` as the
        discriminator is robust to _manage_in_flight_context returning a new
        list with a pruning marker inserted.

        Embedded tool_results (C2.1 fix): the harness emits tool_results as
        a single system-role HistoryMessage whose content is
        "Tool execution results:\\n<tool_result name=\"X\">...</tool_result>"
        joined by blank lines. One Turn is emitted per embedded
        <tool_result> block so retrieval-plan TOC dedup can match by
        tool_name.

        Each tool result is truncated via _truncate_tool_result; the
        truncation flags are stored in meta. Real call_ids are recovered
        from the assistant <tool_call id="..."> attribute when present;
        otherwise synthetic ids of the form "{worker_id}-call-N" are used.

        Plan: docs/plans/trace-as-history-impl-2026-04-27.md (C2, C2.1)
        Spec: docs/plans/trace-as-history-2026-04-27.md §2.5, §2.6.2
        """
        from datetime import datetime, timezone as _tz
        now = datetime.now(_tz.utc)
        turns: list[Turn] = []

        # Patterns for embedded XML inside HistoryMessage content.
        # Harness format (mesh/harness/loop.py):
        #   <tool_call name="X" id="Y">...</tool_call>     (in assistant content)
        #   <tool_result name="X">...</tool_result>        (in system content)
        tool_call_re = re.compile(
            r'<tool_call\s+([^>]*)>\n?(.*?)\n?</tool_call>',
            re.DOTALL,
        )
        tool_result_re = re.compile(
            r'<tool_result\s+([^>]*)>\n?(.*?)\n?</tool_result>',
            re.DOTALL,
        )
        attr_re = re.compile(r'(\w+)\s*=\s*["\']([^"\']*)["\']')

        def _parse_attrs(s: str) -> dict[str, str]:
            return {m.group(1): m.group(2) for m in attr_re.finditer(s)}

        # --- Source 1: HistoryMessage list (non-CC backends) ---
        # Scope filter: inner-loop entries only. Pre-dispatch entries from
        # _build_history_for_llm() carry source="persisted" (the default) and
        # must NOT be persisted as trace.
        history_all = result.worker_in_flight_history or []
        history = [
            m for m in history_all
            if getattr(m, "source", "persisted") == "in_flight"
        ]
        pending_call_id: str | None = None  # links assistant turn to next tool_result
        for msg in history:
            raw_content = getattr(msg, "content", "")
            content = raw_content if isinstance(raw_content, str) else str(raw_content)
            from_node = getattr(msg, "from_node", "")

            # Tool-result message: harness convention is system-role with
            # "Tool execution results:" prefix and one or more embedded
            # <tool_result name="X">...</tool_result> blocks.
            stripped = content.lstrip()
            is_tool_result = (
                content.startswith("Tool execution results:")
                or stripped.startswith("<tool_result")
            )

            if is_tool_result:
                tr_matches = list(tool_result_re.finditer(content))
                if not tr_matches:
                    # Pruning marker or other system message (e.g. "[N previous
                    # tool result(s) omitted...]"). Persist as a single
                    # tool_result Turn so the trace remains traceable.
                    trunc, was_trunc, orig_lines, orig_chars = self._truncate_tool_result(content)
                    turns.append(Turn(
                        role="tool",
                        content=trunc,
                        timestamp=now,
                        from_node="system",
                        meta={
                            "trace_block": "tool_result",
                            "tool_call_id": pending_call_id or "",
                            "truncated": was_trunc,
                            "original_lines": orig_lines,
                            "original_chars": orig_chars,
                            "tool_success": True,
                            "worker_id": worker_id,
                        },
                    ))
                else:
                    for m in tr_matches:
                        attrs = _parse_attrs(m.group(1))
                        body = m.group(2)
                        tool_name = attrs.get("name", "")
                        # Recover real call_id from for_call attribute if present;
                        # else fall back to the synthetic id from the preceding
                        # assistant Turn.
                        real_call_id = attrs.get("for_call") or (pending_call_id or "")
                        success_attr = attrs.get("success", "").lower()
                        success_val = (success_attr != "false")  # default True
                        trunc, was_trunc, orig_lines, orig_chars = self._truncate_tool_result(body)
                        rendered = (
                            f'<tool_result for_call="{real_call_id}" tool_name="{tool_name}">\n'
                            f"{trunc}\n"
                            f"</tool_result>"
                        )
                        turns.append(Turn(
                            role="tool",
                            content=rendered,
                            timestamp=now,
                            from_node="system",
                            meta={
                                "trace_block": "tool_result",
                                "tool_name": tool_name,
                                "tool_call_id": real_call_id,
                                "tool_success": success_val,
                                "truncated": was_trunc,
                                "original_lines": orig_lines,
                                "original_chars": orig_chars,
                                "worker_id": worker_id,
                            },
                        ))
                pending_call_id = None
            else:
                # Assistant turn: may contain reasoning + zero-or-more
                # <tool_call> XML blocks. Skip pure-text turns (those are
                # the final response, appended by the caller separately).
                tc_matches = list(tool_call_re.finditer(content))
                if not tc_matches:
                    continue
                # Best-effort: real call_id from first <tool_call id="...">.
                first_attrs = _parse_attrs(tc_matches[0].group(1))
                real_id = first_attrs.get("id") or f"{worker_id}-call-{len(turns)}"
                tool_name = first_attrs.get("name", "")
                pending_call_id = real_id
                a_meta: dict[str, Any] = {
                    "trace_block": "tool_call",
                    "tool_call_id": real_id,
                    "worker_id": worker_id,
                }
                if tool_name:
                    a_meta["tool_name"] = tool_name
                turns.append(Turn(
                    role="assistant",
                    content=content,
                    timestamp=now,
                    from_node=self._node_id,
                    meta=a_meta,
                ))

        # --- Source 2: CC events (CC backend) ---
        cc_events = result.worker_cc_events or []
        for event in cc_events:
            event_type = getattr(event, "event_type", None)
            tool_name = getattr(event, "tool_name", "")
            call_id = getattr(event, "call_id", "")
            data = getattr(event, "data", None)

            if event_type == "tool_call":
                args_repr = data if isinstance(data, str) else str(data)
                content = (
                    f'<tool_call name="{tool_name}" id="{call_id}">\n'
                    f"{args_repr}\n"
                    f"</tool_call>"
                )
                turns.append(Turn(
                    role="assistant",
                    content=content,
                    timestamp=now,
                    from_node=self._node_id,
                    meta={
                        "trace_block": "tool_call",
                        "tool_name": tool_name,
                        "tool_call_id": call_id,
                        "tool_args": data if isinstance(data, dict) else None,
                        "worker_id": worker_id,
                    },
                ))
            elif event_type == "tool_result":
                raw = str(data) if data is not None else ""
                trunc, was_trunc, orig_lines, orig_chars = self._truncate_tool_result(raw)
                content = (
                    f'<tool_result for_call="{call_id}">\n'
                    f"{trunc}\n"
                    f"</tool_result>"
                )
                turns.append(Turn(
                    role="tool",
                    content=content,
                    timestamp=now,
                    from_node="system",
                    meta={
                        "trace_block": "tool_result",
                        "tool_name": tool_name,
                        "tool_call_id": call_id,
                        "tool_success": True,
                        "truncated": was_trunc,
                        "original_lines": orig_lines,
                        "original_chars": orig_chars,
                        "worker_id": worker_id,
                    },
                ))

        return turns

    def _build_worker_digest(self, worker_id: str) -> str:
        """Build a single-Turn compact trace of worker activity (mechanical, token-capped).

        Walks the worker snapshot delta and formats tool calls + results
        with ~100 line truncation per result. Token-capped at worker_digest_max_tokens.
        """
        slot = self._slot_for_worker(worker_id)
        snapshot = slot.snapshot if slot is not None else self._worker_snapshot
        snapshot_start = (
            slot.snapshot_start
            if slot is not None else self._worker_snapshot_start
        )
        if not snapshot:
            return ""
        delta = snapshot[snapshot_start:]
        if not delta:
            return ""

        max_tokens = self._config.worker_digest_max_tokens  # default 15000
        RESULT_MAX_LINES = 100

        # Count total tool operations for header
        tool_count = sum(1 for t in delta if (t.meta or {}).get("tool_calls")
                         or (t.meta or {}).get("cc_tool_events"))

        header = f"<worker_digest worker='{worker_id}' tools_used={tool_count}>"
        footer = "</worker_digest>"
        overhead = estimate_tokens(header) + estimate_tokens(footer) + 10
        budget = max_tokens - overhead

        lines: list[str] = []
        step = 0
        running_tokens = 0

        def truncate_by_lines(text: str, max_lines: int = RESULT_MAX_LINES) -> str:
            """Truncate to ~max_lines, preserving line structure."""
            text_lines = text.split('\n')
            if len(text_lines) <= max_lines:
                return text
            truncated = '\n'.join(text_lines[:max_lines])
            remaining = len(text_lines) - max_lines
            return f"{truncated}\n  ... ({remaining} more lines, {len(text)} chars total)"

        i = 0
        while i < len(delta):
            turn = delta[i]
            meta = turn.meta or {}
            line = ""

            if meta.get("tool_calls"):
                step += 1
                call_text = turn.content  # Already formatted "[Tool: name(args)]"

                # Grab paired result (tool_calls and tool_results are always adjacent)
                result_text = ""
                if i + 1 < len(delta) and (delta[i + 1].meta or {}).get("tool_results"):
                    raw = delta[i + 1].content.replace("[Tool Results]\n", "", 1)
                    result_text = truncate_by_lines(raw)
                    i += 1

                line = f"[{step}] {call_text}\n  → {result_text}" if result_text else f"[{step}] {call_text}"

            elif meta.get("cc_tool_events"):
                cc_content = turn.content.replace("[CC Tool Activity]\n", "")
                for cc_line in cc_content.split("\n"):
                    cc_line = cc_line.strip()
                    if not cc_line:
                        continue
                    step += 1
                    entry = f"[{step}] {cc_line}"
                    lt = estimate_tokens(entry)
                    if running_tokens + lt > budget:
                        lines.append(f"[...truncated, ~{len(delta) - i} entries remaining]")
                        return f"{header}\n" + "\n".join(lines) + f"\n{footer}"
                    lines.append(entry)
                    running_tokens += lt
                i += 1
                continue

            elif not meta and turn.content and turn.role == "outgoing":
                # capturing_send message
                step += 1
                preview = truncate_by_lines(turn.content)
                to = turn.to_node or '?'
                line = f"[{step}] send_message(to='{to}')\n  → {preview}"
            else:
                i += 1
                continue

            lt = estimate_tokens(line)
            if running_tokens + lt > budget:
                lines.append(f"[...truncated, ~{len(delta) - i} entries remaining]")
                break
            lines.append(line)
            running_tokens += lt
            i += 1

        return f"{header}\n" + "\n".join(lines) + f"\n{footer}"

    async def _build_synthesis_context(self) -> str:
        """Build a full router context block for the synthesis prompt.

        Mirrors _build_router_prompt structure: identity, personality,
        memory blocks (v2: representative + map + log; v1: profile render),
        and recent conversation turns — so the synthesis LLM has the same
        knowledge the router has when generating a response.
        """
        parts = []

        # Identity
        parts.append(f"You are {self._nickname} ({self._agent_type}).")

        # Personality
        if self._memory:
            personality = self._memory.get_personality()
            if personality:
                parts.append(f"<personality>\n{personality}\n</personality>")

        # Memory blocks — same as _build_router_prompt
        if self._memory and isinstance(self._memory, MemorySystemV2):
            rep_block = await self._memory.render_representative_block()
            if rep_block:
                parts.append(rep_block)
            map_context = self._get_last_n_turns_text(5)
            map_block = await self._memory.render_relevant_maps_block(map_context)
            if map_block:
                parts.append(map_block)
            log_block = await self._memory.render_recent_log_block()
            if log_block:
                parts.append(log_block)
            if self._relevant_context:
                parts.append(
                    f"<relevant_memories>\n{self._relevant_context}\n</relevant_memories>"
                )
        elif self._memory:
            profile = self._memory.light_profile
            memory_block = await self._memory.render(
                profile, query=self._latest_user_message,
            )
            if memory_block:
                parts.append(memory_block)

        # v2 conversation summary
        if self._memory and isinstance(self._memory, MemorySystemV2):
            summary_block = await self._memory.render_summary_block()
            if summary_block:
                parts.append(summary_block)

        # Recent conversation (last N turns — no truncation)
        max_turns = self._config.synthesis_context_turns
        durable = self._history.build_context_for_llm()
        # Strip tool-call visibility blocks (Contract §5)
        for _msg in durable:
            if isinstance(getattr(_msg, 'content', None), str):
                _msg.content = strip_tools_called_block(_msg.content)
        if durable:
            recent = durable[-max_turns:]
            turn_lines = []
            for msg in recent:
                from_node = getattr(msg, 'from_node', 'unknown')
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                turn_lines.append(f"[{from_node}] {content}")
            parts.append("\nRecent conversation:\n" + "\n".join(turn_lines))

        return "\n".join(parts)

    async def _synthesize_worker_output(
        self,
        trace_text: str,
        trigger: Message,
    ) -> str:
        """Run synthesis LLM call with router context and worker trace.

        Injects identity, personality, memory blocks (v2: representative +
        map + log + retrieved; v1: profile render), and recent conversation
        turns so the synthesis LLM can write in the agent's voice and frame
        the response in conversational context.

        If the combined prompt exceeds synthesis_max_tokens, the worker trace
        is truncated from the beginning (oldest content discarded first).
        """
        max_tokens = self._config.synthesis_max_tokens  # default 150k

        # Build router context (identity + personality + recent turns)
        context_block = await self._build_synthesis_context()
        context_tokens = estimate_tokens(context_block)

        # Budget: total - instructions overhead - context
        instructions_overhead = 500  # ~500 tokens for the template text
        trace_budget = max(max_tokens - instructions_overhead - context_tokens, 10_000)
        trace_tokens = estimate_tokens(trace_text)

        if trace_tokens > trace_budget:
            # Truncate from the beginning — keep the most recent content
            # Use character-level approximation: ~4 chars per token
            keep_chars = trace_budget * 4
            discarded = len(trace_text) - keep_chars
            trace_text = (
                f"[... {discarded:,} chars / ~{trace_tokens - trace_budget:,} tokens truncated ...]\n"
                + trace_text[-keep_chars:]
            )
            logger.info(
                f"RouterV2 synthesis: truncated worker trace from ~{trace_tokens} "
                f"to ~{trace_budget} tokens"
            )

        # Build prompt: instructions template contains both trace and context slots
        user_request = trigger.content if trigger else "(unknown request)"
        prompt = (
            f"Original user request: {user_request}\n\n"
            f"{SYNTHESIZE_INSTRUCTIONS.format(worker_trace=trace_text, context_block=context_block)}"
        )

        logger.info(
            f"RouterV2 synthesis prompt: ~{estimate_tokens(prompt)} tokens "
            f"(context: ~{context_tokens}, trace: ~{estimate_tokens(trace_text)})"
        )

        # Synthesis uses the router LLM client (consolidated).
        synthesis_client = self._llm_client
        timeout_s = self._synthesis_timeout_seconds()
        logger.info(
            "RouterV2 synthesis using router LLM client "
            "(timeout=%ss)",
            timeout_s,
        )

        try:
            response = await asyncio.wait_for(
                synthesis_client.complete(prompt),
                timeout=timeout_s,
            )
            return response.strip()
        except asyncio.TimeoutError:
            logger.error(
                "RouterV2 synthesis LLM call timed out after %ss", timeout_s
            )
            return ""
        except Exception as e:
            logger.error(f"RouterV2 synthesis LLM call failed: {e}")
            return ""

    async def _complete_via_trace(
        self,
        result: WorkerResult,
        trigger: Message,
        worker_id: str,
    ) -> None:
        """Append worker trace Turns + final-text Turn to history; deliver final text directly.

        Implements docs/plans/trace-as-history-2026-04-27.md §2.5.1.
        Replaces the synthesis branch when trace_as_history_enabled is on.

        Skips synthesis entirely. Tool calls + tool results are appended to
        history as Turns (with the meta["trace_block"] convention) so the
        next worker dispatch sees them naturally.
        """
        from datetime import datetime, timezone as _tz

        now = datetime.now(_tz.utc)

        # Step 1: extract trace Turns (assistant tool_call + tool tool_result pairs)
        trace_turns = self._extract_trace_turns(result, worker_id)

        # Step 2: append trace Turns under state lock
        async with self._state_lock:
            for t in trace_turns:
                self._append_turn(t)

            # Mirror synthesis-path bookkeeping: clear snapshot start so the
            # flush monitor doesn't re-emit anything we just folded into history.
            slot = self._slot_for_worker(worker_id)
            snapshot = slot.snapshot if slot is not None else self._worker_snapshot
            if snapshot:
                if slot is not None:
                    slot.snapshot_start = len(snapshot)
                else:
                    self._worker_snapshot_start = len(snapshot)

        # Step 3: deliver final text (handle empty + buffered fallback)
        final_text = (result.response or "").strip()
        is_channel = (
            trigger and trigger.to_node and trigger.to_node.startswith("channel:")
        )

        if not final_text and result.buffered_messages:
            # Buffered-fallback: same shape as today's synthesis-fallback,
            # but no synthesis was attempted in the first place.
            logger.warning(
                f"RouterV2 trace-mode fallback: sending all "
                f"{len(result.buffered_messages)} buffered messages"
            )
            for _to_node, content in result.buffered_messages:
                t_is_channel = bool(_to_node) and _to_node.startswith("channel:")
                if t_is_channel:
                    self._append_turn(Turn(
                        role="outgoing",
                        content=content,
                        timestamp=now,
                        from_node=self._node_id,
                        to_node=_to_node,
                        meta={
                            "router_response": True,
                            "trace_fallback": True,
                            "worker_id": worker_id,
                        },
                    ))
                else:
                    await self._send_and_store(
                        content, trigger,
                        meta={"trace_fallback": True, "worker_id": worker_id},
                    )
            total_chars = sum(len(c) for _, c in result.buffered_messages)
            logger.info(
                f"RouterV2 worker {worker_id} complete (trace-mode fallback), "
                f"sent {len(result.buffered_messages)} messages, {total_chars} chars"
            )
            return

        if not final_text:
            final_text = "Done."

        if is_channel:
            # Worker already delivered to channel directly via capturing_send;
            # store-only (mirrors today's behavior in synthesis branch).
            self._append_turn(Turn(
                role="outgoing",
                content=final_text,
                timestamp=now,
                from_node=self._node_id,
                to_node=trigger.to_node,
                meta={
                    "router_response": True,
                    "trace_mode": True,
                    "worker_id": worker_id,
                },
            ))
            logger.info(
                f"RouterV2 worker {worker_id} complete (trace-mode, channel — stored only), "
                f"{len(trace_turns)} trace turns + final {len(final_text)} chars"
            )
        else:
            await self._send_and_store(
                final_text, trigger,
                meta={"trace_mode": True, "worker_id": worker_id},
            )
            logger.info(
                f"RouterV2 worker {worker_id} complete (trace-mode), "
                f"{len(trace_turns)} trace turns + final {len(final_text)} chars"
            )

    async def _handle_worker_complete(
        self,
        result: WorkerResult,
        trigger: Message,
        worker_id: str | None = None,
    ) -> None:
        """
        Handle successful worker completion.

        When trace-as-history is enabled:
        1. Append worker trace Turns + final-text Turn to history
        2. Deliver final text directly (no synthesis)

        When synthesis is enabled (default):
        1. Build worker trace (full fidelity) and digest (mechanical, token-capped)
        2. Append digest Turn to history
        3. Run synthesis LLM call (outside lock)
        4. Send synthesized response to user
        5. Skip _merge_worker_context() — digest replaces raw snapshot merge

        When synthesis is disabled (passthrough fallback):
        1. Send buffered/passthrough messages directly
        2. Merge worker context (old behavior)
        3. Transition to IDLE
        """
        worker_id = worker_id or getattr(self, "_current_worker_id", None)
        completion_reached = False
        _report_content = ""
        _controller_cleaned = False
        _report_trigger = None
        _legacy_direct_report = False

        try:
            # Phase 1: State bookkeeping under lock (fast, synchronous)
            async with self._state_lock:
                slot = self._slot_for_worker(worker_id)
                worker_id = worker_id or (
                    slot.worker_id if slot is not None else self._current_worker_id
                )

                # Log cumulative token usage from worker
                start_time = (
                    slot.start_time
                    if slot is not None else self._worker_start_time
                )
                elapsed_s = (
                    round(time.monotonic() - start_time, 1)
                    if start_time else 0
                )
                if result.usage:
                    u = result.usage
                    logger.info(
                        f"[WORKER] COMPLETE: {self._nickname} worker {worker_id} finished "
                        f"(duration={elapsed_s}s, "
                        f"in={u['input_tokens']} out={u['output_tokens']} "
                        f"total={u['total_tokens']} llm_calls={u.get('llm_calls', '?')})"
                    )
                else:
                    logger.info(
                        f"[WORKER] COMPLETE: {self._nickname} worker {worker_id} finished "
                        f"(duration={elapsed_s}s)"
                    )

                # Stop flush monitor before synthesis
                try:
                    self._stop_flush_monitor(worker_id=worker_id)
                except TypeError:
                    self._stop_flush_monitor()

            # Persist the full execution evidence before any delivery branch or
            # cleanup can discard WorkerResult internals.  Prompt logs capture
            # launch context only; this private archive is the source consumed
            # by a later skill_draft.  Drafting-worker traces remain auditable
            # but are excluded from automatic latest-procedure selection.
            try:
                archived_trace = None
                if self._config.worker_trace_persist:
                    archived_trace = persist_completed_worker_trace(
                        self._nickname,
                        worker_id or "worker",
                        self._build_full_worker_trace(worker_id or "worker", result),
                        task_summary=(
                            slot.task_description
                            if slot is not None
                            else (
                                trigger.content
                                if isinstance(trigger.content, str)
                                else str(trigger.content)
                            )
                        ),
                        kind=(
                            "skill_draft"
                            if isinstance(trigger.metadata, dict)
                            and trigger.metadata.get("skill_draft")
                            else "worker"
                        ),
                    )
                if archived_trace is not None:
                    logger.info(
                        "[SKILLS] Archived completed worker trace: %s",
                        archived_trace,
                    )
            except (OSError, SkillCardError) as exc:
                # Trace persistence must never convert a completed task into a
                # delivery failure. skill_draft will degrade to router context
                # and surface the missing evidence in mandatory caveats.
                logger.warning(
                    "[SKILLS] Could not archive completed worker trace for %s: %s",
                    worker_id,
                    exc,
                )

            if isinstance(trigger.metadata, dict) and trigger.metadata.get(
                "autonomous_controller_leaf"
            ):
                content = self._extract_report_content(result) or result.response or ""
                controller_tool_calls = [
                    {
                        "name": getattr(event, "tool_name", ""),
                        "data": getattr(event, "data", None),
                    }
                    for event in (result.worker_cc_events or [])
                    if getattr(event, "event_type", "") == "tool_call"
                ]
                async with self._state_lock:
                    self._cleanup_worker_state(worker_id=worker_id)
                _controller_cleaned = True
                completion_reached = True
                self._resolve_controller_worker(
                    worker_id,
                    {
                        "worker_id": worker_id,
                        "backend": (slot.backend if slot is not None else None)
                        or trigger.metadata.get("worker_backend"),
                        "content": content,
                        "usage": result.usage,
                        "tool_calls": controller_tool_calls,
                        "error": str(result.error) if result.error else "",
                    },
                )
                return

            # Phase 2: Delivery path selection (outside lock)
            _report_trigger = None
            worker_snapshot = (
                slot.snapshot if slot is not None else self._worker_snapshot
            )

            if result.report_sent and result.buffered_messages:
                # --- REPORT-AS-TRIGGER PATH (Agent-Worker Contract §4) ---
                # Worker used send_report — deliver as internal trigger after cleanup.
                report_content = "\n\n".join(
                    content for _, content in result.buffered_messages
                )
                _report_content = report_content
                logger.info(
                    f"[WORKER] Report-as-trigger: {self._nickname} worker {worker_id} "
                    f"({len(report_content)} chars) → agent ReAct loop"
                )
                self._append_worker_report_turn(worker_id, submitted=True)
                _report_trigger = self._build_worker_report_trigger(
                    worker_id,
                    report_content,
                    trigger,
                )
            elif (trace_mode := getattr(self._config, "trace_as_history_enabled", False)):
                # --- TRACE-AS-HISTORY PATH ---
                # If cancel-flush already appended trace Turns, skip to avoid duplicates.
                if not self._trace_already_appended_on_cancel(slot, worker_id):
                    await self._complete_via_trace(result, trigger, worker_id or "worker")
                else:
                    logger.info(
                        f"RouterV2 worker {worker_id} complete (trace-mode, "
                        f"cancel-flush already appended trace; skipping)"
                    )
                    self._clear_trace_appended_on_cancel(slot, worker_id)
            elif self._config.synthesize_enabled and self._llm_client:
                # --- SYNTHESIS PATH ---
                from datetime import datetime, timezone as tz

                # 1. Build worker trace (full fidelity, uncapped, ephemeral)
                trace_text = self._build_worker_trace(worker_id or "worker", result)

                # 2. Build worker digest (mechanical, token-capped)
                digest_text = self._build_worker_digest(worker_id or "worker")

                # 3. Append digest Turn to history (before building synthesis prompt
                #    so the synthesis LLM sees it in the rolling window)
                if digest_text:
                    self._append_turn(Turn(
                        role="system",
                        content=digest_text,
                        timestamp=datetime.now(tz.utc),
                        from_node="system",
                        meta={"worker_digest": True, "worker_id": worker_id},
                    ))

                # 4. Append worker's outgoing channel messages to durable history.
                #    capturing_send delivered these to the channel immediately but
                #    only wrote them to the ephemeral snapshot — not the router's
                #    ConversationHistory. Without this, the LLM won't see the
                #    worker's actual output on subsequent turns.
                if worker_snapshot:
                    for _snap_turn in worker_snapshot:
                        if (_snap_turn.role == "outgoing"
                                and _snap_turn.to_node
                                and _snap_turn.to_node.startswith("channel:")):
                            self._append_turn(Turn(
                                role="outgoing",
                                content=_snap_turn.content,
                                timestamp=_snap_turn.timestamp,
                                from_node=_snap_turn.from_node,
                                to_node=_snap_turn.to_node,
                                meta={"worker_channel_message": True, "worker_id": worker_id},
                            ))

                # 5. Clear snapshot start to prevent flush monitor from double-counting
                if worker_snapshot:
                    if slot is not None:
                        slot.snapshot_start = len(worker_snapshot)
                    else:
                        self._worker_snapshot_start = len(worker_snapshot)

                # 6. Verbatim buffered delivery (deliver_buffered_verbatim).
                #    The worker already composed message(s) for the dispatch
                #    origin — deliver that content verbatim instead of asking
                #    the synthesis LLM to relay it (which sometimes describes
                #    the message as "already sent" and discards the content).
                #    Capture/buffering is unchanged: the origin still receives
                #    exactly ONE message at completion — multiple buffered
                #    messages are concatenated in order into a single delivery.
                #    Synthesis still runs when the buffer holds nothing
                #    addressed to the origin.
                _origin_msgs = [
                    _content
                    for _to_node, _content in (result.buffered_messages or [])
                    if _to_node == trigger.from_node
                ]
                if self.autonomous_completion_metadata(trigger):
                    # --- AUTONOMOUS CLOSEOUT REROUTE ---
                    # A worker that skipped send_report would otherwise land in
                    # the fallback delivery below: a text-only message with no
                    # tool turn. The closeout half of the mandate — session
                    # report, dossier task completion, next wake — only exists
                    # inside a ReAct turn, so a session would end silently
                    # mid-cycle. Reuse the synthesis this path already performs,
                    # then route it through the report-as-trigger handler so the
                    # router gets its tool-capable turn with the session scope
                    # inherited from the dispatch trigger.
                    synthesized = await self._synthesize_worker_output(
                        trace_text, trigger
                    )
                    report_content = (
                        (synthesized or "").strip()
                        or "\n\n".join(
                            str(_content).strip()
                            for _to_node, _content in (
                                result.buffered_messages or []
                            )
                            if str(_content).strip()
                        )
                        or str(result.response or "").strip()
                        or "The worker completed without producing output."
                    )
                    _report_content = report_content
                    self._append_worker_report_turn(worker_id, submitted=False)
                    _report_trigger = self._build_worker_report_trigger(
                        worker_id,
                        report_content,
                        trigger,
                        synthesized=True,
                    )
                    logger.info(
                        "[AUTONOMOUS] Worker %s completed without send_report; "
                        "rerouting %d synthesized chars through the "
                        "report-as-trigger path to close session %s.",
                        worker_id,
                        len(report_content),
                        (trigger.metadata or {}).get(
                            "autonomous_session_id", ""
                        ),
                    )
                elif (getattr(self._config, "deliver_buffered_verbatim", False)
                        and _origin_msgs):
                    if len(_origin_msgs) == 1:
                        _combined = _origin_msgs[0]
                    else:
                        _combined = "\n\n---\n\n".join(
                            f"— message {_i} of {len(_origin_msgs)} —\n\n{_c}"
                            for _i, _c in enumerate(_origin_msgs, 1)
                        )
                    await self._send_and_store(
                        _combined, trigger,
                        meta={"verbatim_buffered_delivery": True,
                              "worker_id": worker_id},
                    )
                    logger.info(
                        f"RouterV2 worker {worker_id} complete "
                        f"(verbatim buffered delivery), delivered "
                        f"{len(_origin_msgs)} buffered message(s) as one, "
                        f"{len(_combined)} chars"
                    )
                else:
                    # 6b. Synthesize
                    synthesized = await self._synthesize_worker_output(trace_text, trigger)
                    report_content = self._extract_report_content(result)

                    # 7. Fallback: if synthesis fails/empty, send ALL buffered messages
                    #    (not just the last one — earlier messages may contain the
                    #    actual substantive content the worker produced).
                    if not synthesized and report_content:
                        await self._send_and_store(
                            self._format_unsynthesized_worker_report(
                                worker_id or "worker",
                                report_content,
                                reason="failed",
                            ),
                            trigger,
                            meta={
                                "unsynthesized_worker_report": True,
                                "worker_id": worker_id,
                            },
                        )
                        logger.info(
                            "RouterV2 worker %s complete "
                            "(unsynthesized send_report fallback), delivered "
                            "%d report chars",
                            worker_id,
                            len(report_content),
                        )
                    elif not synthesized and result.buffered_messages:
                        logger.warning(
                            f"RouterV2 synthesis fallback: sending all "
                            f"{len(result.buffered_messages)} buffered messages"
                        )
                        for _to_node, _content in result.buffered_messages:
                            is_channel = (
                                _to_node and _to_node.startswith("channel:")
                            )
                            if is_channel:
                                from datetime import datetime, timezone as _tz
                                self._append_turn(Turn(
                                    role="outgoing",
                                    content=_content,
                                    timestamp=datetime.now(_tz.utc),
                                    from_node=self._node_id,
                                    to_node=_to_node,
                                    meta={"router_response": True, "synthesis_fallback": True},
                                ))
                            else:
                                await self._send_and_store(_content, trigger)
                        total_chars = sum(len(c) for _, c in result.buffered_messages)
                        logger.info(
                            f"RouterV2 worker {worker_id} complete (synthesis fallback), "
                            f"sent {len(result.buffered_messages)} messages, {total_chars} chars"
                        )
                    elif not synthesized:
                        synthesized = result.response or "Done."
                        await self._send_and_store(synthesized, trigger)
                        logger.info(
                            f"RouterV2 worker {worker_id} complete (no synthesis, no buffer), "
                            f"sent {len(synthesized)} chars"
                        )
                    else:
                        # 8. Fallback completion delivery. Reaching this branch means
                        #    the worker did NOT use send_report (that path returned
                        #    above as a report-as-trigger). Always deliver the
                        #    synthesized result to the dispatch origin, including a
                        #    channel. Intermediate worker messages must not suppress
                        #    the only durable completion report.
                        is_channel = (
                            trigger.to_node and trigger.to_node.startswith("channel:")
                        )
                        await self._send_and_store(
                            synthesized,
                            trigger,
                            meta={
                                "fallback_completion_delivery": True,
                                "worker_id": worker_id,
                            },
                        )
                        logger.info(
                            f"[WORKER] Fallback completion delivery: {self._nickname} "
                            f"worker {worker_id} did not call send_report; delivered "
                            f"{len(synthesized)} synthesized chars to "
                            f"{'channel ' + trigger.to_node if is_channel else 'requester'}"
                        )
            else:
                # --- PASSTHROUGH PATH (synthesis disabled) ---
                # Send last buffered message or result.response
                if result.buffered_messages:
                    for _to_node, content in result.buffered_messages:
                        await self._send_and_store(content, trigger)
                elif result.response:
                    # Messages were already sent in real-time via capturing_send
                    pass
                else:
                    await self._send_and_store("Done.", trigger)

                # Append worker's outgoing channel messages (same fix as synthesis path)
                if worker_snapshot:
                    from datetime import datetime, timezone as _tz2
                    for _snap_turn in worker_snapshot:
                        if (_snap_turn.role == "outgoing"
                                and _snap_turn.to_node
                                and _snap_turn.to_node.startswith("channel:")):
                            self._append_turn(Turn(
                                role="outgoing",
                                content=_snap_turn.content,
                                timestamp=_snap_turn.timestamp,
                                from_node=_snap_turn.from_node,
                                to_node=_snap_turn.to_node,
                                meta={"worker_channel_message": True, "worker_id": worker_id},
                            ))

                # Merge worker context (old behavior)
                async with self._state_lock:
                    self._merge_worker_context(result.context, worker_id)

                logger.info(
                    f"RouterV2 worker {worker_id} complete (passthrough), "
                    f"sent worker response directly"
                )

            # Phase 3: Post-delivery bookkeeping (under lock)
            async with self._state_lock:
                # Session-level memory reflection
                if self._memory:
                    stats = self._compute_episode_stats(result, trigger)
                    # Subtract tools already flushed mid-worker to avoid double-counting
                    already_flushed = (
                        slot.flush_tools_already_flushed
                        if slot is not None
                        else self._flush_tools_already_flushed
                    )
                    if already_flushed > 0:
                        stats.tool_calls = max(
                            0, stats.tool_calls - already_flushed
                        )
                    trigger_text = trigger.content if isinstance(trigger.content, str) else str(trigger.content)
                    self._accumulate_session_stats(stats, trigger_text, result, worker_id)
            completion_reached = True

        finally:
            if completion_reached:
                self._record_skill_outcomes(
                    trigger,
                    worker_id,
                    result="unknown",
                    note=(
                        "Worker completed; follow-through and verifier results "
                        "were not mechanically observable."
                    ),
                )
            # Phase 4: Cleanup always runs (under lock), even if send/LLM failed
            if not _controller_cleaned:
                async with self._state_lock:
                    if _report_trigger is not None:
                        report_slot = self._slot_for_worker(worker_id)
                        if (
                            report_slot is not None
                            and not report_slot.report_accepted
                        ):
                            report_slot.lifecycle = WorkerLifecycle.REPORTING
                            report_slot.report_accepted = True
                            self._bump_slot_revision()
                            if hasattr(self, "_report_wake_queue"):
                                self._enqueue_report_wake(_report_trigger)
                            else:
                                # Narrow legacy tests build RouterV2 via
                                # __new__. Preserve their synchronous seam;
                                # constructed routers always own the queue.
                                _legacy_direct_report = True
                        elif report_slot is None and not hasattr(
                            self, "_report_wake_queue"
                        ):
                            _legacy_direct_report = True
                    try:
                        self._cleanup_worker_state(worker_id=worker_id)
                    except TypeError:
                        self._cleanup_worker_state()

            # Persist history to disk after worker completion
            try:
                self.save_history()
            except Exception as e:
                logger.warning(f"Failed to save history after worker complete: {e}")

        # Report triggers are normally drained asynchronously after their slot
        # has been accepted into the ordered wake queue and released.
        if _legacy_direct_report and _report_trigger is not None:
            self._last_router_call_sent_message = False
            try:
                await self.on_message(_report_trigger)
            except Exception as exc:
                logger.exception(
                    "RouterV2 report-as-trigger processing failed for %s: %s",
                    worker_id,
                    exc,
                )
                await self._send_and_store(
                    self._format_unsynthesized_worker_report(
                        worker_id or "worker",
                        _report_content,
                        reason="failed",
                    ),
                    _report_trigger,
                    meta={
                        "unsynthesized_worker_report": True,
                        "worker_id": worker_id,
                    },
                )
            else:
                if not getattr(self, "_last_router_call_sent_message", False):
                    await self._send_and_store(
                        self._format_unsynthesized_worker_report(
                            worker_id or "worker",
                            _report_content,
                            reason="timed out or produced no visible response",
                        ),
                        _report_trigger,
                        meta={
                            "unsynthesized_worker_report": True,
                            "worker_id": worker_id,
                        },
                    )

    async def _handle_worker_error(
        self,
        error: Exception,
        trigger: Message,
        worker_id: str | None = None,
    ) -> None:
        """Handle worker failure — MUST always notify user, never fail silently."""
        worker_id = worker_id or getattr(self, "_current_worker_id", None)
        controller_capture = bool(
            isinstance(trigger.metadata, dict)
            and trigger.metadata.get("autonomous_controller_leaf")
        )
        try:
            async with self._state_lock:
                slot = self._slot_for_worker(worker_id)
                worker_id = worker_id or (
                    slot.worker_id if slot is not None else self._current_worker_id
                )
            start_time = (
                slot.start_time if slot is not None else self._worker_start_time
            )
            elapsed_s = (
                round(time.monotonic() - start_time, 1)
                if start_time else 0
            )
            if slot is not None:
                slot.lifecycle = WorkerLifecycle.FAILED
                slot.failure = str(error)
                self._bump_slot_revision()
            logger.error(f"[WORKER] FAILED: {self._nickname} worker {worker_id} error after {elapsed_s}s: {error}")

            if not controller_capture:
                error_msg = f"[Worker failed: {error}]"
                try:
                    await self._send_and_store(error_msg, trigger)
                except Exception as send_err:
                    logger.error(
                        f"[WORKER] FAILED TO NOTIFY USER: {self._nickname} worker {worker_id} "
                        f"— original error: {error}, send error: {send_err}"
                    )
                    try:
                        await self._send_fn(error_msg, trigger)
                    except Exception:
                        logger.critical(
                            f"[WORKER] SILENT FAILURE: {self._nickname} worker {worker_id} "
                            f"— could not deliver error to user. Error: {error}"
                        )
        finally:
            self._record_skill_outcomes(
                trigger,
                worker_id,
                result="failed",
                note=f"Worker failed before a verified terminal result: {error}",
            )
            # Cleanup always runs, even if _send_and_store fails
            async with self._state_lock:
                try:
                    self._cleanup_worker_state(worker_id=worker_id)
                except TypeError:
                    self._cleanup_worker_state()

            # Persist history to disk after worker error
            try:
                self.save_history()
            except Exception as e:
                logger.warning(f"Failed to save history after worker error: {e}")
            if controller_capture:
                self._resolve_controller_worker(
                    worker_id,
                    {
                        "worker_id": worker_id,
                        "backend": trigger.metadata.get("worker_backend"),
                        "content": "",
                        "usage": None,
                        "error": str(error),
                    },
                )

    def _compute_episode_stats(self, result: WorkerResult, trigger: Message) -> EpisodeStats:
        """
        Compute EpisodeStats from a worker result for reflection gating.

        Counts only user-visible messages (user↔agent), excluding tool
        calls/results and system/internal messages.
        """
        tool_calls = 0
        num_user_visible_turns = 0
        total_user_visible_chars = 0
        agent_response_chars = 0
        has_errors = result.error is not None

        # Count the trigger itself as a user-visible turn
        trigger_content = trigger.content if isinstance(trigger.content, str) else str(trigger.content)
        num_user_visible_turns += 1
        total_user_visible_chars += len(trigger_content)

        for entry in getattr(result, "context", []):
            msg = entry.message if hasattr(entry, "message") else entry
            # Turn uses .meta, Message uses .metadata — check both
            metadata = getattr(msg, "meta", None) or getattr(msg, "metadata", None) or {}
            msg_type = getattr(msg, "type", None)
            from_node = getattr(msg, "from_node", "")
            content = getattr(msg, "content", "")
            content_str = content if isinstance(content, str) else str(content)

            # Count mesh tool calls
            if metadata.get("tool_calls"):
                calls = metadata["tool_calls"]
                if isinstance(calls, list):
                    tool_calls += len(calls)
                else:
                    tool_calls += 1

            # Count CC (Claude Code) internal tool calls
            if metadata.get("cc_tool_events"):
                cc_count = metadata.get("cc_tool_calls", 0)
                tool_calls += cc_count if cc_count else 1

            # Skip tool requests/results for user-visible counting
            if msg_type in (MessageType.TOOL_REQUEST, MessageType.TOOL_RESULT):
                continue
            if metadata.get("tool_calls") or metadata.get("tool_results"):
                continue
            if metadata.get("cc_tool_events"):
                continue

            # User-visible messages: from user or to user/channel only
            to_node = getattr(msg, "to_node", "") or ""
            if from_node.startswith("user:"):
                num_user_visible_turns += 1
                total_user_visible_chars += len(content_str)
            elif from_node.startswith("agent:") and (
                to_node.startswith("user:") or to_node.startswith("channel:")
            ):
                num_user_visible_turns += 1
                total_user_visible_chars += len(content_str)
                agent_response_chars += len(content_str)

        return EpisodeStats(
            tool_calls=tool_calls,
            num_user_visible_turns=num_user_visible_turns,
            total_user_visible_chars=total_user_visible_chars,
            agent_response_chars=agent_response_chars,
            has_errors=has_errors,
        )

    def _accumulate_session_stats(
        self,
        stats: EpisodeStats,
        trigger_text: str,
        result: WorkerResult,
        worker_id: str,
    ) -> None:
        """Accumulate completion stats into the current session.

        If there's been a gap since the last completion, evaluate the
        previous session's stats for reflection first, then start a new session.
        """
        now = time.monotonic()

        # Check for session gap — evaluate previous session if gap detected
        if self._session_stats is not None:
            gap = now - self._session_last_completion_time
            if gap >= self._session_gap_secs:
                # Gap detected — flush the previous session
                self._flush_session_reflection()
                # Start fresh session
                self._session_stats = None

        # Start new session or accumulate
        if self._session_stats is None:
            self._session_stats = EpisodeStats()
            self._session_trigger_text = trigger_text

        self._session_stats.merge(stats)
        self._session_last_completion_time = now
        self._session_last_result = result
        self._session_last_worker_id = worker_id

        logger.debug(
            f"Session stats accumulated: tools={self._session_stats.tool_calls}, "
            f"turns={self._session_stats.num_user_visible_turns}, "
            f"chars={self._session_stats.total_user_visible_chars}, "
            f"agent_chars={self._session_stats.agent_response_chars}"
        )

    def _flush_session_reflection(self) -> None:
        """Evaluate accumulated session stats and fire reflection if warranted."""
        if self._session_stats is None or not self._memory:
            return
        # Memory Formation v3: short-circuit legacy session reflection. The v3
        # path forms memories via `form_un_formed` triggered by time/token/
        # shutdown/startup — no per-session reflection call.
        if getattr(self._memory, "_formation_v3_enabled", False):
            return

        stats = self._session_stats
        result = self._session_last_result
        trigger_text = self._session_trigger_text
        worker_id = self._session_last_worker_id or "unknown"

        if self._memory.should_reflect(result, stats):
            asyncio.create_task(
                self._memory.reflect_on_completion(trigger_text, result, worker_id)
            )
            logger.info(
                f"RouterV2 fired session reflection for {worker_id} "
                f"(tools={stats.tool_calls}, turns={stats.num_user_visible_turns}, "
                f"chars={stats.total_user_visible_chars}, "
                f"agent_chars={stats.agent_response_chars})"
            )
        else:
            logger.debug(
                f"Session did not meet reflection threshold "
                f"(tools={stats.tool_calls}, turns={stats.num_user_visible_turns}, "
                f"chars={stats.total_user_visible_chars}, "
                f"agent_chars={stats.agent_response_chars})"
            )

    # =========================================================================
    # Intra-worker periodic flush monitor
    # =========================================================================

    def _start_flush_monitor(
        self,
        trigger: Message,
        worker_id: str | None = None,
    ) -> None:
        """Start the background monitor that fires mid-worker reflections."""
        if not self._flush_interval_tools or not self._memory:
            return
        slot = self._slot_for_worker(worker_id or self._current_worker_id)
        if slot is None:
            self._flush_snapshot_cursor = self._worker_snapshot_start
            self._flush_tools_since_last = 0
            self._flush_tools_already_flushed = 0
            self._flush_monitor_task = asyncio.create_task(
                self._monitor_worker_tools(trigger)
            )
            return
        slot.flush_snapshot_cursor = slot.snapshot_start
        slot.flush_tools_since_last = 0
        slot.flush_tools_already_flushed = 0
        slot.flush_task = asyncio.create_task(
            self._monitor_worker_tools(trigger, worker_id=slot.worker_id)
        )

    def _stop_flush_monitor(self, worker_id: str | None = None) -> None:
        """Cancel the flush monitor task."""
        slot = self._slot_for_worker(worker_id)
        if slot is not None:
            if slot.flush_task and not slot.flush_task.done():
                slot.flush_task.cancel()
            slot.flush_task = None
            return
        if self._flush_monitor_task and not self._flush_monitor_task.done():
            self._flush_monitor_task.cancel()
        self._flush_monitor_task = None

    async def _monitor_worker_tools(
        self,
        trigger: Message,
        worker_id: str | None = None,
    ) -> None:
        """Poll the worker snapshot and fire reflections at tool-count intervals.

        Runs as a background task alongside the worker.  Every 10 seconds,
        scans new snapshot entries for tool calls.  When the accumulated
        count since the last flush crosses ``_flush_interval_tools``, fires
        a mid-worker reflection and resets the counter.
        """
        trigger_text = trigger.content if isinstance(trigger.content, str) else str(trigger.content)
        try:
            while True:
                await asyncio.sleep(10)
                slot = self._slot_for_worker(worker_id)
                snapshot = slot.snapshot if slot is not None else self._worker_snapshot
                if not snapshot:
                    continue

                # Count new tool calls since last check
                new_tools = 0
                end = len(snapshot)
                cursor = (
                    slot.flush_snapshot_cursor
                    if slot is not None else self._flush_snapshot_cursor
                )
                for entry in snapshot[cursor:end]:
                    meta = getattr(entry, "meta", None) or getattr(entry, "metadata", None) or {}
                    if meta.get("tool_calls"):
                        calls = meta["tool_calls"]
                        new_tools += len(calls) if isinstance(calls, list) else 1
                    if meta.get("cc_tool_events"):
                        cc_count = meta.get("cc_tool_calls", 0)
                        new_tools += cc_count if cc_count else 1

                if slot is not None:
                    slot.flush_snapshot_cursor = end
                    slot.flush_tools_since_last += new_tools
                    tools_since_last = slot.flush_tools_since_last
                else:
                    self._flush_snapshot_cursor = end
                    self._flush_tools_since_last += new_tools
                    tools_since_last = self._flush_tools_since_last

                if tools_since_last >= self._flush_interval_tools:
                    # Build a synthetic EpisodeStats for this chunk
                    chunk_stats = EpisodeStats(
                        tool_calls=tools_since_last,
                    )
                    active_worker_id = (
                        slot.worker_id
                        if slot is not None else self._current_worker_id
                    ) or "unknown"

                    # Flush: fire reflection for the accumulated chunk
                    if self._session_stats is None:
                        self._session_stats = EpisodeStats()
                        self._session_trigger_text = trigger_text
                    self._session_stats.merge(chunk_stats)
                    self._session_last_completion_time = time.monotonic()
                    self._session_last_worker_id = active_worker_id

                    # Build a lightweight WorkerResult from the snapshot so far
                    # (the reflection needs context to summarize)
                    snapshot_copy = list(snapshot[:end])
                    mid_result = WorkerResult(
                        response="(mid-worker checkpoint)",
                        context=snapshot_copy,
                    )
                    self._session_last_result = mid_result

                    logger.info(
                        f"RouterV2 mid-worker flush: {tools_since_last} tools "
                        f"since last flush (worker {active_worker_id})"
                    )
                    self._flush_session_reflection()
                    # Reset for next interval
                    if slot is not None:
                        slot.flush_tools_already_flushed += tools_since_last
                        slot.flush_tools_since_last = 0
                    else:
                        self._flush_tools_already_flushed += tools_since_last
                        self._flush_tools_since_last = 0
                    self._session_stats = None

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Flush monitor error: {e}")

    # =========================================================================
    # Worker Watchdog — periodic check-in on worker progress
    # =========================================================================

    def _start_watchdog(
        self,
        trigger: Message,
        worker_id: str | None = None,
    ) -> None:
        """Start the periodic watchdog timer."""
        if not self._config.watchdog_interval_minutes:
            logger.debug("Watchdog disabled (watchdog_interval_minutes=0)")
            return
        logger.info(
            f"Watchdog started: interval={self._config.watchdog_interval_minutes}min, "
            f"worker={self._current_worker_id or 'worker'}"
        )
        slot = self._slot_for_worker(worker_id or self._current_worker_id)
        task = asyncio.create_task(
            self._watchdog_loop(trigger, worker_id=worker_id)
        )
        if slot is not None:
            slot.watchdog_task = task
            if slot.worker_id == self._current_worker_id:
                # Legacy diagnostic view for the selected primary slot.
                self._watchdog_task = task
        else:
            self._watchdog_task = task

    def _stop_watchdog(self, worker_id: str | None = None) -> None:
        """Cancel the watchdog timer."""
        slot = self._slot_for_worker(worker_id)
        if slot is not None:
            task = slot.watchdog_task
            if task and not task.done():
                task.cancel()
            slot.watchdog_task = None
            if self._watchdog_task is task:
                self._watchdog_task = None
            return
        was_running = self._watchdog_task and not self._watchdog_task.done()
        if was_running:
            self._watchdog_task.cancel()
            logger.info(f"Watchdog stopped for worker={self._current_worker_id or 'worker'}")
        self._watchdog_task = None

    async def _watchdog_loop(
        self,
        trigger: Message,
        worker_id: str | None = None,
    ) -> None:
        """Periodic check-in loop. Fires _watchdog_tick every interval."""
        interval = self._config.watchdog_interval_minutes * 60
        tick_count = 0
        try:
            while True:
                await asyncio.sleep(interval)
                slot = self._slot_for_worker(worker_id)
                if slot is not None and not slot.active:
                    logger.info(
                        f"Watchdog loop exiting: state={slot.lifecycle.value} "
                        f"(no longer BUSY after {tick_count} tick(s))"
                    )
                    break
                if slot is None and self._state != RouterState.BUSY:
                    break
                tick_count += 1
                logger.info(
                    f"Watchdog tick #{tick_count} firing for "
                    f"worker={worker_id or self._current_worker_id or 'worker'}"
                )
                if getattr(self, "_router_turn_lock", None) is not None and (
                    self._router_turn_lock.locked()
                ):
                    logger.info("Watchdog deferred: user router turn is active")
                    continue
                async with self._router_turn_lock:
                    await self._watchdog_tick(trigger, worker_id=worker_id)
        except asyncio.CancelledError:
            logger.debug(
                f"Watchdog loop cancelled after {tick_count} tick(s)"
            )

    async def _watchdog_tick(
        self,
        trigger: Message,
        worker_id: str | None = None,
    ) -> None:
        """Single watchdog evaluation. Builds prompt, calls LLM, parses result."""
        slot = self._slot_for_worker(worker_id)
        worker_id = (
            slot.worker_id if slot is not None else worker_id
        ) or self._current_worker_id or "worker"
        elapsed = 0.0
        worker_start_time = (
            slot.start_time if slot is not None else self._worker_start_time
        )
        if worker_start_time:
            elapsed = time.monotonic() - worker_start_time

        # Log what the LLM will see
        activity_lines = self._build_worker_activity_lines(worker_id)
        worker_progress = self._get_worker_progress(worker_id)
        cc_events = (
            list(getattr(slot.execution_context, "current_cc_events", []) or [])
            if slot is not None and slot.execution_context is not None
            else (
                self._cc_events_fn()
                if self._cc_events_fn and self._worker_snapshot else []
            )
        )
        harness_events = self._harness_events_fn(n=10) if self._harness_events_fn else []
        pending_trigger = slot.trigger if slot is not None else self._pending_trigger
        task_summary = self._summarize_trigger(pending_trigger)
        logger.info(
            f"Watchdog tick for {worker_id}: elapsed={elapsed:.0f}s, "
            f"progress_turns={len(worker_progress)}, "
            f"cc_events={len(cc_events)}, "
            f"harness_events={len(harness_events)}, "
            f"activity_lines={len(activity_lines)}, "
            f"task={task_summary!r:.120}"
        )
        if activity_lines:
            # Log last few activity lines at DEBUG for forensic review
            recent = activity_lines[-5:]
            for line in recent:
                logger.debug(f"  watchdog context: {line[:300]}")

        try:
            raw_response = await self._call_router_full(
                msg=trigger,
                busy=True,
                watchdog=True,
                worker_id=worker_id,
                pending_trigger=pending_trigger,
                worker_start_time=worker_start_time,
            )
        except Exception as e:
            logger.warning(
                f"Watchdog LLM call failed for {worker_id} "
                f"(elapsed={elapsed:.0f}s): {e}"
            )
            return

        # Worker may have completed during the LLM call
        if slot is not None and not slot.active:
            logger.info(
                f"Watchdog: worker {worker_id} completed during check-in "
                f"(elapsed={elapsed:.0f}s), discarding response"
            )
            return
        if slot is None and self._state != RouterState.BUSY:
            # Compatibility path for legacy callers with no authoritative
            # slot: the mutable state transition is still the completion
            # signal and must suppress a stale watchdog notification.
            logger.info(
                f"Watchdog: worker {worker_id} completed during check-in "
                f"(elapsed={elapsed:.0f}s), discarding response"
            )
            return

        # Log the full raw LLM response for forensic review
        logger.debug(f"Watchdog raw response for {worker_id}: {raw_response}")

        if self._is_nominal_watchdog_response(raw_response):
            logger.info(
                f"Watchdog check-in for {worker_id}: NOMINAL "
                f"(elapsed={elapsed:.0f}s, "
                f"activity_lines={len(activity_lines)})"
            )
        else:
            parsed = self._parse_router_response(raw_response)
            if parsed["no_response"]:
                logger.info(
                    f"Watchdog for {worker_id}: <no_response>, suppressing "
                    f"(elapsed={elapsed:.0f}s)"
                )
                return
            notify_text = parsed["response"] or raw_response
            logger.info(
                f"Watchdog NOTIFY for {worker_id} "
                f"(elapsed={elapsed:.0f}s): {notify_text[:500]}"
            )
            await self._send_and_store(notify_text, trigger)

    def _is_nominal_watchdog_response(self, response: str) -> bool:
        """Check if watchdog response indicates nothing unusual.

        Empty/whitespace response = nothing to report (nominal).
        Also matches 'nothing to report' anywhere in the message, with optional 'to'.
        Handles: 'Nothing to report.', 'nothing report', 'Overall, nothing to report here.'
        """
        if not response.strip():
            return True
        return bool(re.search(r'nothing\s+(to\s+)?report', response.strip(), re.IGNORECASE))

    def _cleanup_worker_state(self, worker_id: str | None = None) -> None:
        """Release only the targeted stable slot."""
        slot = self._slot_for_worker(worker_id)
        if slot is not None:
            try:
                self._stop_flush_monitor(worker_id=worker_id)
            except TypeError:
                self._stop_flush_monitor()
            try:
                self._stop_watchdog(worker_id=worker_id)
            except TypeError:
                self._stop_watchdog()
            if slot.execution_context is not None:
                cancel_event = getattr(
                    slot.execution_context, "cancel_event", None
                )
                if cancel_event is not None:
                    cancel_event.set()
            slot.cleanup_complete = True
            slot.completed_at = time.monotonic()
            if getattr(self, "_current_worker_id", None) == worker_id:
                # Do not let the legacy singleton fallback resurrect the
                # currently executing completion task after its fixed slot
                # has been released.
                self._set_current_worker_slot(None)
            self._reset_slot(slot)
        elif worker_id is None:
            # Legacy single-worker tests construct no slot table.
            self._stop_flush_monitor()
            self._stop_watchdog()
            for candidate in list(self._ensure_slot_table()):
                if candidate.active:
                    self._reset_slot(candidate)

        active_slots = self._active_worker_slots()
        if active_slots:
            self._refresh_primary_worker_slot()
        else:
            self._state = RouterState.IDLE
            self._set_current_worker_slot(None)
        # Note: _ephemeral_peeks NOT cleared here — planning peeks may be active (RouterV3)

    # =========================================================================
    # Non-LLM handlers (fallback)
    # =========================================================================

    async def _handle_busy(
        self,
        msg: Message,
        worker_id: str | None,
        pending_trigger: Message | None,
        worker_start_time: float | None,
    ) -> None:
        """Handle a message that arrives while worker is busy (non-LLM fallback)."""
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        if self._is_cancel_request(content):
            await self._handle_cancel_request(msg, worker_id)
            return

        if self._is_user_status_query(msg, content):
            elapsed = 0.0
            if worker_start_time:
                elapsed = time.monotonic() - worker_start_time
            status_msg = f"Still working on your request ({elapsed:.0f}s elapsed)..."
            await self._send_and_store(status_msg, msg)
            return

        trigger_summary = self._summarize_trigger(pending_trigger)
        busy_msg = f"Got it. Let me finish {trigger_summary} first, then I'll get back to you."
        await self._send_and_store(busy_msg, msg)

    def _is_status_query(self, content: str) -> bool:
        """Return whether *all* of a short message is a status query.

        This predicate is intentionally conservative. False negatives fall
        through to the normal router turn, while a false positive discards a
        substantive message by answering a diagnostic instead.
        """
        normalized = " ".join(str(content or "").lower().split())
        if not normalized or len(normalized) > 80:
            return False
        query = normalized.strip(" \\t\\r\\n.,!?")
        patterns = [
            " ".join(pattern.lower().split())
            for pattern in self._config.status_patterns
        ]
        if normalized in patterns:
            return True
        if any(
            pattern == pattern.rstrip(".,!?") and query == pattern
            for pattern in patterns
        ):
            return True
        # "any update?" and analogous short variants are natural status
        # queries, but retain the full-query rule rather than substring match.
        return any(
            normalized == f"any {pattern}"
            or (
                pattern == pattern.rstrip(".,!?")
                and query == f"any {pattern}"
            )
            for pattern in patterns
        )

    def _is_user_status_query(self, msg: Message, content: str) -> bool:
        """Gate the diagnostic status fast path to actual user messages."""
        return bool((msg.from_node or "").startswith("user:")) and self._is_status_query(content)

    def _is_cancel_request(self, content: str) -> bool:
        """Check if content is an exact-phrase request to cancel the current worker."""
        content_lower = content.lower().strip()
        if content_lower in {"stop all", "cancel all", "stop all workers", "cancel all workers"}:
            return True
        return any(
            content_lower == pattern
            or content_lower.startswith(pattern + " ")
            for pattern in self._config.cancel_patterns
        )

    def _cancel_target_from_content(
        self,
        content: str,
    ) -> tuple[str | None, bool]:
        lowered = content.lower().strip()
        cancel_all = lowered in {
            "stop all",
            "cancel all",
            "stop all workers",
            "cancel all workers",
        }
        for slot in self._active_worker_slots():
            if slot.worker_id and slot.worker_id.lower() in lowered:
                return slot.worker_id, cancel_all
        return None, cancel_all

    async def _handle_cancel_request(self, msg: Message, worker_id: str | None) -> None:
        """Cancel the current worker and notify the user.

        Acquires _state_lock internally — callers must NOT hold the lock.
        """
        logger.info(f"RouterV2 cancel requested by {msg.from_node}, cancelling worker {worker_id}")
        target, cancel_all = self._cancel_target_from_content(
            msg.content if isinstance(msg.content, str) else str(msg.content)
        )
        active = self._active_worker_slots()
        if len(active) > 1 and target is None and not cancel_all:
            await self._send_and_store(
                (
                    "Multiple workers are active. Specify a worker ID or say "
                    "\"stop all\":\n"
                    + "\n".join(
                        f"- {slot.worker_id}: {slot.task_description[:120]}"
                        for slot in active
                    )
                ),
                msg,
            )
            return
        cancelled = await self.cancel_worker(
            msg,
            worker_id=target,
            cancel_all=cancel_all,
        )
        if cancelled:
            cancel_msg = f"Cancelled the current task. What would you like me to do instead?"
            await self._send_and_store(cancel_msg, msg, meta={"worker_cancelled": True})
        else:
            # Worker already finished between the check and cancel
            await self._send_and_store("The task just finished — nothing to cancel.", msg)

    # Regex for "set context to <path>" variants — captures the path after "to"
    _SET_CONTEXT_RE = re.compile(
        r"^set\s+(?:your\s+)?(?:project\s+)?context\s+to\s+(.+)",
        re.IGNORECASE,
    )

    def _extract_set_context_path(self, content: str) -> str | None:
        """Extract project path from a 'set context to <path>' command.

        Returns the path string if matched, None otherwise.
        Only fires when memory is MemorySystemV2.
        """
        if not isinstance(self._memory, MemorySystemV2):
            return None
        m = self._SET_CONTEXT_RE.match(content.strip())
        if m:
            return m.group(1).strip().strip("'\"")
        return None

    async def _handle_set_context_request(self, msg: Message, path: str) -> None:
        """Pre-router intercept: run set_project_context and confirm.

        Called from on_message() which already holds _state_lock.
        Sends confirmation with project name and map details, then returns
        (skipping classification entirely).
        """
        logger.info(
            "Pre-router set_project_context intercept: path=%s, from=%s",
            path, msg.from_node,
        )

        # Curate the outgoing project's map before switching
        outgoing = self._memory._active_project
        if outgoing:
            try:
                raw_text = self._memory._format_turns_as_text(
                    list(self._history._window)
                )
                if raw_text.strip():
                    logger.info(
                        "Context switch: curating outgoing map '%s' "
                        "(%d window turns)",
                        outgoing, len(self._history._window),
                    )
                    await self._memory.curate_active_map(
                        raw_text, len(self._history._window),
                    )
            except Exception:
                logger.warning(
                    "Context switch: outgoing map curation failed for '%s'",
                    outgoing, exc_info=True,
                )

        try:
            result = await self._memory.set_project_context(path)
        except Exception as e:
            logger.error("set_project_context failed: %s", e)
            await self._send_and_store(
                f"Failed to set project context: {e}", msg
            )
            return

        # Build a detailed confirmation — read from file, not DB
        project_name = self._memory._active_project or "unknown"
        content_text = await self._memory.get_map(project_name)
        if content_text:
            char_count = len(content_text)
            word_count = len(content_text.split())
            confirm = (
                f"Project context set: **{project_name}**\n"
                f"Map: {char_count:,} chars, ~{word_count:,} words"
            )
        else:
            confirm = result  # fallback to the raw status message

        logger.info("set_project_context result: %s", confirm)
        await self._send_and_store(confirm, msg)

    # Regex for "review map" variants — broad enough to catch natural phrasing
    _REVIEW_MAP_RE = re.compile(
        r"^(?:please\s+)?(?:review|refresh|update|check)\s+"
        r"(?:the\s+|my\s+|our\s+|your\s+)?"
        r"(?:project\s+)?map"
        r"(?:[,.]?\s*(?:please|thanks|thx))?\s*$",
        re.IGNORECASE,
    )

    def _is_review_map_request(self, content: str) -> bool:
        """Return True if the message is a 'review map' intercept trigger.

        Only fires when memory is MemorySystemV2 with an active project.
        """
        if not isinstance(self._memory, MemorySystemV2):
            return False
        return bool(self._REVIEW_MAP_RE.match(content.strip()))

    async def _handle_review_map_request(self, msg: Message) -> None:
        """Pre-router intercept: run map review and report results.

        Called from on_message() which already holds _state_lock.
        Sends a confirmation with map changes, then returns
        (skipping classification entirely). The review LLM resolves
        ambiguities itself using its tools — no questions sent to user.
        """
        logger.info(
            "Pre-router map_review intercept: from=%s", msg.from_node,
        )
        await self._send_and_store("Reviewing the project map against current state…", msg)
        try:
            result = await self._memory.review_active_map()
        except Exception as e:
            logger.error("map_review failed: %s", e)
            await self._send_and_store(f"Map review failed: {e}", msg)
            return

        summary = result.get("summary", "Review complete.")
        logger.info("map_review result: updated=%s", result.get("updated"))
        await self._send_and_store(summary, msg)

    def _summarize_trigger(self, trigger: Message | None) -> str:
        """Create a brief summary of what we're working on."""
        if not trigger:
            return "the current request"

        content = trigger.content if isinstance(trigger.content, str) else str(trigger.content)

        if len(content) > 50:
            return f'"{content[:47]}..."'

        return f'"{content}"'

    # =========================================================================
    # Context merge
    # =========================================================================

    def _merge_worker_context(
        self,
        worker_context: list[Any],
        worker_id: str | None = None,
    ) -> None:
        """
        Merge worker's new entries back into router ConversationHistory with attribution.

        Under snapshot-and-merge architecture, the worker appends Turn objects
        to the mutable snapshot list. The delta (entries after _worker_snapshot_start)
        represents the worker's tool calls and responses.
        """
        wid = worker_id or self._current_worker_id or "worker"
        slot = self._slot_for_worker(worker_id)
        snapshot = slot.snapshot if slot is not None else self._worker_snapshot
        snapshot_start = (
            slot.snapshot_start
            if slot is not None else self._worker_snapshot_start
        )

        # Extract delta from snapshot (entries the worker added during execution)
        if snapshot and snapshot_start < len(snapshot):
            delta = snapshot[snapshot_start:]
            # Include all entries including outgoing — worker messages are
            # sent in real-time via capturing_send() and need to appear in
            # the router's conversation history.
            logger.debug(f"RouterV2 merging {len(delta)} snapshot entries from {wid}")

            for turn in delta:
                # Tag with worker_origin attribution
                meta = dict(turn.meta) if turn.meta else {}
                meta["worker_origin"] = wid
                self._append_turn(Turn(
                    role=turn.role,
                    content=turn.content,
                    timestamp=turn.timestamp,
                    from_node=turn.from_node,
                    to_node=turn.to_node,
                    meta=meta,
                ))
        else:
            # Fallback: legacy merge from list of HistoryEntry/Message objects
            from datetime import datetime, timezone

            new_messages = worker_context
            if not new_messages:
                return

            logger.debug(f"RouterV2 merging {len(new_messages)} legacy entries from {wid}")

            for entry in new_messages:
                if isinstance(entry, Turn):
                    meta = dict(entry.meta) if entry.meta else {}
                    meta["worker_origin"] = wid
                    self._append_turn(Turn(
                        role=entry.role,
                        content=entry.content,
                        timestamp=entry.timestamp,
                        from_node=entry.from_node,
                        to_node=entry.to_node,
                        meta=meta,
                    ))
                elif hasattr(entry, 'message'):
                    msg = entry.message
                    direction = entry.direction
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    ts = msg.timestamp
                    if isinstance(ts, str) and ts:
                        try:
                            ts = datetime.fromisoformat(ts)
                        except ValueError:
                            ts = datetime.now(timezone.utc)
                    elif not ts:
                        ts = datetime.now(timezone.utc)
                    self._append_turn(Turn(
                        role=direction,
                        content=content,
                        timestamp=ts,
                        from_node=msg.from_node or "",
                        to_node=msg.to_node,
                        meta={"worker_origin": wid},
                    ))

        # Check if summarization is needed after merge
        self._check_and_trigger_summarization()

    def _truncate_context(self) -> None:
        """Legacy truncation — no longer used, summarization handles context limits."""
        pass  # ConversationHistory handles summarization-based limits

    def _check_and_trigger_summarization(self) -> None:
        """Check if summarization is needed and trigger it in the background.

        For memory v2: partitions window, checkpoints dropped turns,
        runs the window drop pipeline (topic segmentation → significance
        gate → reflection → log entries → map curation → conversation
        summary update), then drops old half.

        Also checks for stale maps: if the active project map hasn't been
        curated in MAP_CURATION_STALE_HOURS and there's enough window
        content, triggers passive curation (no window drop).

        For memory v1: existing flow (LLM summarization of old half).
        """
        if not self._llm_client:
            return

        if (not self._history._summarization_enabled
                and not isinstance(self._memory, MemorySystemV2)):
            return

        # Ensure periodic curation timer is running (lazy start on first message)
        self._ensure_curation_timer()

        W = self._history.window_budget
        window_tokens = self._history.estimate_window_tokens()

        # v2 staleness-based curation: if map is stale and window has
        # content, curate without dropping the window.
        if (isinstance(self._memory, MemorySystemV2)
                and not self._v2_drop_in_progress
                and not self._v2_curate_in_progress
                and window_tokens >= W // 2):
            age = self._memory.active_map_age_hours()
            if age is not None and age > MAP_CURATION_STALE_HOURS:
                logger.info(
                    "Map stale (%.1fh > %dh threshold), triggering passive "
                    "curation with %d window tokens",
                    age, MAP_CURATION_STALE_HOURS, window_tokens,
                )
                self._v2_curate_in_progress = True
                self._v2_turns_at_last_curation = len(self._history._window)
                self._v2_curate_task = asyncio.create_task(
                    self._v2_passive_curate()
                )

        if not self._history.needs_summarization():
            return

        # Check if memory system is v2
        if isinstance(self._memory, MemorySystemV2):
            if self._v2_drop_in_progress:
                logger.debug("v2 window drop already in progress, skipping")
                return
            logger.info(
                f"RouterV2 rolling window trigger (v2): "
                f"window={window_tokens} tokens >= 2×W={2 * W}, "
                f"partitioning and running window drop pipeline"
            )
            self._v2_drop_in_progress = True
            self._v2_turns_at_last_curation = len(self._history._window)
            self._v2_drop_task = asyncio.create_task(self._v2_window_drop())
            return

        # v1 path: LLM summarization
        logger.info(
            f"RouterV2 rolling window trigger: "
            f"window={window_tokens} tokens >= 2×W={2 * W}, "
            f"triggering background summarization"
        )
        asyncio.create_task(
            self._history.summarize(
                self._llm_client,
            )
        )

    async def _v2_passive_curate(self) -> None:
        """Run map curation on current window without dropping it.

        Triggered when the active project map is stale (hasn't been updated
        in MAP_CURATION_STALE_HOURS) and there's content in the window.
        This ensures maps stay current even if the 2W window drop threshold
        is never reached (e.g., due to frequent agent restarts).
        """
        try:
            raw_text = self._memory._format_turns_as_text(
                list(self._history._window)
            )
            if not raw_text.strip():
                return
            logger.info(
                "Passive map curation starting: %d window turns, %d chars",
                len(self._history._window), len(raw_text),
            )
            await self._memory.curate_active_map(
                raw_text, len(self._history._window),
            )
            logger.info("Passive map curation completed")
        except Exception:
            logger.error("Passive map curation failed", exc_info=True)
        finally:
            self._v2_curate_in_progress = False

    # =========================================================================
    # Periodic map curation timer
    # =========================================================================

    def _ensure_curation_timer(self) -> None:
        """Start the periodic curation timer if not already running."""
        if self._v2_curation_timer_task and not self._v2_curation_timer_task.done():
            return  # Already running
        if not self._config.map_curation_interval_minutes:
            return  # Disabled
        if not isinstance(self._memory, MemorySystemV2):
            return
        interval = self._config.map_curation_interval_minutes
        logger.info(
            "Starting periodic map curation timer: every %dm, min %d turns",
            interval, self._config.map_curation_min_turns,
        )
        self._v2_turns_at_last_curation = len(self._history._window)
        self._v2_curation_timer_task = asyncio.create_task(
            self._periodic_curation_loop()
        )

    async def _periodic_curation_loop(self) -> None:
        """Sleep → check activity gate → curate. Repeats until cancelled."""
        interval = self._config.map_curation_interval_minutes * 60
        min_turns = self._config.map_curation_min_turns
        try:
            while True:
                await asyncio.sleep(interval)

                # Skip if another curation or window drop is already running
                if self._v2_curate_in_progress or self._v2_drop_in_progress:
                    logger.debug("Periodic curation skipped: another curation in progress")
                    continue

                # Activity gate: enough new turns since last curation?
                current_turns = len(self._history._window)
                new_turns = current_turns - self._v2_turns_at_last_curation
                if new_turns < min_turns:
                    logger.debug(
                        "Periodic curation skipped: %d new turns < %d minimum",
                        new_turns, min_turns,
                    )
                    continue

                # Fire curation
                logger.info(
                    "Periodic map curation triggered: %d new turns since last curation",
                    new_turns,
                )
                self._v2_curate_in_progress = True
                self._v2_turns_at_last_curation = current_turns
                try:
                    await self._v2_passive_curate()
                except Exception:
                    logger.error("Periodic map curation failed", exc_info=True)
                # _v2_curate_in_progress is reset inside _v2_passive_curate's finally block

        except asyncio.CancelledError:
            logger.debug("Periodic curation timer cancelled")

    async def _v2_window_drop(self) -> None:
        """Memory v2 window drop: partition, checkpoint, process, drop."""
        try:
            # Partition window and get old half turns
            old_half = self._history.partition_and_drop_old()
            if not old_half:
                logger.warning("v2 window drop: partition returned empty old_half")
                return

            # Checkpoint before processing
            self._memory.checkpoint_dropped_turns(old_half)

            # Run the pipeline
            await self._memory.on_window_drop(old_half)

            # Clear checkpoint on success
            self._memory.clear_checkpoint()

        except Exception:
            logger.error("v2 window drop pipeline failed", exc_info=True)
        finally:
            self._v2_drop_in_progress = False

    async def _flush_worker_buffer_on_cancel(
        self,
        trigger: Message | None,
        worker_id: str | None = None,
    ) -> int:
        """Flush any buffered worker output before cancellation tears down state.

        Reads _worker_buffered_messages and _worker_response_text from
        self._worker_agent (set at __init__ from worker_fn.__self__). Sends
        each buffered DM via _send_and_store, prefixing the FIRST message
        with "[CANCELLED] ". Returns the count of messages flushed.

        No-op when:
        - _worker_agent is None (worker_fn was a bare function — tests only)
        - buffer is empty AND _worker_response_text is empty (passthrough mode
          or worker produced no output before cancel)
        - trigger is None and only DM-targeted output is buffered (no route)
        """
        from datetime import datetime, timezone as _tz

        agent = self._worker_agent
        if agent is None:
            return 0

        slot = self._slot_for_worker(worker_id)
        execution_context = (
            slot.execution_context if slot is not None else None
        )
        buffered = list(
            getattr(execution_context, "buffered_messages", None)
            or getattr(agent, "_worker_buffered_messages", [])
            or []
        )
        response_text = (
            getattr(execution_context, "response_text", "")
            or getattr(agent, "_worker_response_text", "")
            or ""
        )

        # Trace-as-history: capture partial trace from in-flight worker state
        # before cleanup zeroes it. Orphaned tool calls (call without matching
        # result) are handled correctly by _extract_trace_turns.
        if getattr(self._config, "trace_as_history_enabled", False):
            partial_history = list(
                getattr(execution_context, "in_flight_history", None)
                or getattr(agent, "_worker_in_flight_history", None)
                or []
            )
            partial_cc_events = list(
                getattr(execution_context, "current_cc_events", None)
                or getattr(agent, "_current_cc_events", None)
                or []
            )
            if partial_history or partial_cc_events:
                synth_result = WorkerResult(
                    response="",
                    context=[],
                    worker_in_flight_history=partial_history,
                    worker_cc_events=partial_cc_events,
                )
                trace_turns = self._extract_trace_turns(
                    synth_result, worker_id or self._current_worker_id or "cancelled"
                )
                for t in trace_turns:
                    self._append_turn(t)
                if trace_turns:
                    self._mark_trace_appended_on_cancel(
                        slot, worker_id or self._current_worker_id
                    )
                    logger.info(
                        f"RouterV2 cancel: appended {len(trace_turns)} partial trace turns"
                    )

        sent = 0
        if buffered:
            for i, (to_node, content) in enumerate(buffered):
                prefix = "[CANCELLED] " if i == 0 else ""
                payload = prefix + content
                is_channel = bool(to_node) and to_node.startswith("channel:")
                if is_channel:
                    # Worker already delivered channel messages directly
                    # (capturing_send bypasses the buffer for channels). This
                    # branch is defensive; if a future change buffers a
                    # channel message, record it in history without re-send.
                    self._append_turn(Turn(
                        role="outgoing",
                        content=payload,
                        timestamp=datetime.now(_tz.utc),
                        from_node=self._node_id,
                        to_node=to_node,
                        meta={"router_response": True, "cancel_flush": True},
                    ))
                    sent += 1
                elif trigger is not None:
                    await self._send_and_store(
                        payload, trigger, meta={"cancel_flush": True}
                    )
                    sent += 1
                # else: trigger=None and DM target — no route, drop silently
        elif response_text and trigger is not None:
            # No explicit send_message calls reached the buffer, but the
            # worker produced final-text output via capturing_send. Surface it.
            await self._send_and_store(
                f"[CANCELLED] {response_text}",
                trigger,
                meta={"cancel_flush": True},
            )
            sent = 1
        # else: nothing to flush

        return sent

    async def _cancel_worker_unlocked(
        self,
        trigger: Message | None = None,
        *,
        worker_id: str | None = None,
        cancel_all: bool = False,
        reason: str = "",
        preserve_external: bool = False,
    ) -> bool:
        """Cancel in-flight worker without acquiring the state lock.

        Caller MUST already hold self._state_lock.
        Returns True if a worker was cancelled, False if none was running.

        If `trigger` is provided, flushes any buffered worker output (with a
        [CANCELLED] prefix on the first message) before tearing down state.
        Flush is bounded by a 2-second timeout to keep the state lock free
        for subsequent operations even if the transport hangs.
        """
        active = self._active_worker_slots()
        if cancel_all:
            targets = list(active)
        elif worker_id is not None:
            target = self._slot_for_worker(worker_id)
            targets = [target] if target is not None and target.active else []
        elif len(active) == 1:
            targets = [active[0]]
        else:
            # Zero workers or ambiguous multi-worker cancellation.
            targets = []

        cancelled_any = False
        for slot in targets:
            if slot is None:
                continue
            slot.lifecycle = WorkerLifecycle.CANCELLING
            slot.latest_activity = reason or "Cancellation requested"
            self._bump_slot_revision()
            execution_context = slot.execution_context
            if execution_context is not None:
                cancel_event = getattr(execution_context, "cancel_event", None)
                abort_event = getattr(execution_context, "abort_event", None)
                if cancel_event is not None:
                    cancel_event.set()
                if abort_event is not None:
                    abort_event.set()
            if slot.kind == "fixed_tool" and not preserve_external:
                cancel_fixed = getattr(self._worker_agent, "cancel_fixed_tool", None)
                if callable(cancel_fixed):
                    try:
                        await cancel_fixed(reason="router worker cancellation")
                    except Exception:
                        logger.exception("Failed to terminate fixed-tool subprocess")
            if slot.task is not None and not slot.task.done():
                slot.task.cancel()
                try:
                    await slot.task
                except (asyncio.CancelledError, Exception):
                    pass

            # Flush buffered output before cleanup zeroes everything.
            # Bounded by 2s — workers buffer 1-3 small DMs in normal cases;
            # if _send_and_store can't drain those in 2s, transport is sick
            # and we'd rather bail than hold the state lock indefinitely.
            try:
                flushed = await asyncio.wait_for(
                    self._flush_worker_buffer_on_cancel(
                        # The caller's trigger is the authority to deliver a
                        # cancellation flush. Internal reset/relaunch paths
                        # intentionally pass None and must not acquire a route
                        # merely because the fixed slot retains its original
                        # dispatch trigger.
                        trigger,
                        worker_id=slot.worker_id,
                    ),
                    timeout=2.0,
                )
                if flushed:
                    logger.info(
                        "RouterV2 cancel: flushed %d buffered worker message(s)",
                        flushed,
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    "RouterV2 cancel: flush timed out after 2s; proceeding with cleanup"
                )
            except Exception as e:
                logger.warning(
                    "RouterV2 cancel: flush failed: %s", e, exc_info=True
                )

            cancelled_worker_id = slot.worker_id
            self._cleanup_worker_state(worker_id=cancelled_worker_id)
            logger.info("RouterV2 worker %s cancelled", cancelled_worker_id)
            cancelled_any = True

        return cancelled_any

    async def cancel_worker(
        self,
        trigger: Message | None = None,
        *,
        worker_id: str | None = None,
        cancel_all: bool = False,
        reason: str = "",
        preserve_external: bool = False,
    ) -> bool:
        """
        Cancel in-flight worker (if any).

        Returns True if a worker was cancelled, False if none was running.
        If `trigger` is provided, buffered worker output is flushed to the
        trigger sender with a [CANCELLED] prefix before cleanup.
        """
        async with self._state_lock:
            return await self._cancel_worker_unlocked(
                trigger,
                worker_id=worker_id,
                cancel_all=cancel_all,
                reason=reason,
                preserve_external=preserve_external,
            )

    async def reset(self) -> None:
        """
        Reset router state (used by reset_context).

        Clears context and cancels any in-flight work.
        """
        async with self._state_lock:
            await self._cancel_worker_unlocked(cancel_all=True)
            self._context = []
            logger.info("RouterV2 reset complete")
