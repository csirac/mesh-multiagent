# SPDX-License-Identifier: Apache-2.0
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
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable, Any, TYPE_CHECKING

from pathlib import Path

from .protocol import Message, MessageType
from .conversation_history import ConversationHistory, Turn, ROUTER_SUMMARY_PROMPT
from .llm import estimate_tokens
from .memory.system import EpisodeStats
from .memory.system_v2 import MemorySystemV2
from .storage import MessageStore

if TYPE_CHECKING:
    from .llm import LLMClient

logger = logging.getLogger(__name__)


# =============================================================================
# Router Tool Set — restricted to read-only + single-action tools
# =============================================================================

# Tools the full router is allowed to use. Worker-only tools (bash_exec, file_edit,
# file_write, file_create, file_diff, agent_shutdown, browser_*, plaid mutations)
# are excluded. The router can inspect and query but not mutate the filesystem.
ROUTER_TOOL_NAMES: set[str] = {
    # Information retrieval (read-only)
    "channel_list", "channel_members", "tool_help", "mesh_list", "worker_launch", "worker_status",
    "mesh_status", "agent_status", "current_time", "get_working_directory",
    "file_read",
    "exa_search", "exa_fetch_full", "extract_url",
    "literature_search", "literature_fulltext",
    "arxiv_search", "arxiv_get", "arxiv_fulltext",
    "pubmed_search", "pubmed_get", "pubmed_fulltext", "pubmed_related",
    "plaid_link_status", "plaid_accounts", "plaid_transactions",
    "synthetic_quota", "claude_code_usage", "tamu_cookie_status", "tamu_auth_refresh",

    # Single-action (quick mutations)
    "account_get_current", "account_list", "account_set_current",
    "schedule_wake", "schedule_list", "schedule_cancel",
    "set_working_directory",
    "gmail_list_from_date", "gmail_get_email", "gmail_search_emails",
    "gmail_send_message", "gmail_reply_to",
    "gmail_create_draft", "gmail_draft_reply",
    "calendar_list_on_date", "calendar_create_event", "calendar_delete_event",
    "notes_search", "notes_get", "notes_list", "notes_read",
    "notes_add", "notes_delete", "notes_edit",
    "remember", "memory_list", "memory_get", "memory_search",
    "memory_add", "memory_delete",
    "history_search",
    "personality_get", "personality_set",
    "todo_list", "todo_add", "todo_update", "todo_toggle", "todo_remove",
    "todo_reorder", "todo_set_section_order",

    # Map tools (v2 memory — router can edit maps inline)
    "map_list", "map_get", "map_edit", "map_create", "set_project_context",

    # Messaging
    "send_message",
}

# Backend types that have their own internal tool loops (ReAct / TAOR).
# Router-native worker tools are not exposed to these backends; they dispatch
# through the backend-neutral <dispatch_worker> response block instead.
HARNESS_BACKENDS: frozenset[str] = frozenset({"codex", "claude-code", "mesh-harness"})

# ReAct loop cap for direct (non-harness) backends.
# Raised to 30 for CC-session turns (start→trust→send→monitor cycle needs many iterations).
REACT_MAX_ITERS: int = 30

# Tools that are only available to direct (non-harness) router backends.
# Harness backends have their own internal tool loops and cannot use these.
# Workers never see these tools (they use their own tool_names).
WORKER_ROUTER_TOOLS: frozenset[str] = frozenset({"worker_launch", "worker_status"})

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


# =============================================================================
# Map curation staleness threshold (hours)
# =============================================================================
MAP_CURATION_STALE_HOURS = 12  # Trigger passive curation if map not updated in this many hours


# =============================================================================
# XML Tool Call Fallback Parser
# =============================================================================

_XML_TOOL_CALL_RE = re.compile(
    r'<tool_call\s+name="([^"]+)">(.*?)</tool_call>',
    re.DOTALL,
)
_XML_TOOL_CALLS_BLOCK_RE = re.compile(
    r'<tool_calls>\s*(.*?)\s*</tool_calls>',
    re.DOTALL,
)
_XML_PARAM_RE = re.compile(
    r'<(\w+)>(.*?)</\1>',
    re.DOTALL,
)
# Anthropic-style: <invoke name="tool"><parameter name="key">value</parameter></invoke>
_XML_INVOKE_RE = re.compile(
    r'<(?:antml:)?invoke\s+name="([^"]+)"[^>]*>(.*?)</(?:antml:)?invoke>',
    re.DOTALL,
)
_XML_INVOKE_PARAM_RE = re.compile(
    r'<(?:antml:)?parameter\s+name="([^"]+)"[^>]*>(.*?)</(?:antml:)?parameter>',
    re.DOTALL,
)
_XML_INVOKE_BLOCK_RE = re.compile(
    r'<(?:antml:)?function_calls>\s*(.*?)\s*</(?:antml:)?function_calls>',
    re.DOTALL,
)


def extract_xml_tool_calls(text: str) -> list:
    """Parse tool calls from XML embedded in LLM response text.

    Handles two XML formats models may emit:

    Format 1 (DeepSeek style):
        <tool_call name="cc_get_screen">
          <lines>50</lines>
        </tool_call>

    Format 2 (Anthropic style, with or without antml: prefix):
        <invoke name="cc_get_screen">
          <parameter name="session">cc-session-123</parameter>
        </invoke>

    Returns a list of ToolCall objects (from mesh.tools).
    """
    from .tools import ToolCall

    calls: list[ToolCall] = []

    # --- Format 1: <tool_call> ---
    block_match = _XML_TOOL_CALLS_BLOCK_RE.search(text)
    search_text = block_match.group(1) if block_match else text

    for m in _XML_TOOL_CALL_RE.finditer(search_text):
        name = m.group(1)
        body = m.group(2).strip()
        arguments: dict[str, str] = {}
        for pm in _XML_PARAM_RE.finditer(body):
            arguments[pm.group(1)] = pm.group(2).strip()
        calls.append(ToolCall(
            name=name,
            arguments=arguments,
            raw_xml=m.group(0),
            call_id=f"xml-fallback-{uuid.uuid4().hex[:8]}",
        ))

    # --- Format 2: <invoke> / <invoke> ---
    if not calls:
        invoke_block = _XML_INVOKE_BLOCK_RE.search(text)
        invoke_search = invoke_block.group(1) if invoke_block else text

        for m in _XML_INVOKE_RE.finditer(invoke_search):
            name = m.group(1)
            body = m.group(2).strip()
            arguments: dict[str, str] = {}
            for pm in _XML_INVOKE_PARAM_RE.finditer(body):
                arguments[pm.group(1)] = pm.group(2).strip()
            calls.append(ToolCall(
                name=name,
                arguments=arguments,
                raw_xml=m.group(0),
                call_id=f"xml-fallback-{uuid.uuid4().hex[:8]}",
            ))

    return calls


def strip_xml_tool_calls(text: str) -> str:
    """Remove XML tool call blocks from response text."""
    result = _XML_TOOL_CALLS_BLOCK_RE.sub('', text)
    result = _XML_TOOL_CALL_RE.sub('', result)
    result = _XML_INVOKE_BLOCK_RE.sub('', result)
    result = _XML_INVOKE_RE.sub('', result)
    return result.strip()


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
{{"needs_response": true, "needs_worker": false, "response": "Hey Alan, doing well! What's on your mind?"}}
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
Dispatch by calling the Mesh worker_launch tool. For XML-backed router
backends, emit exactly:
<mesh_call name="worker_launch"><task>...</task></mesh_call>
Do not use Codex/Claude's own internal task delegation, team-worker launcher,
shell commands, or project-editing surface to do router dispatch work.

You MAY send conversational text before the tool call. For example, say
"Okay, let me check on that now," then call the tool with the detailed task.
"""

_WORKER_DISPATCH_BLOCK_INSTRUCTIONS = """\
Dispatch by including this block at the END of your response, after any
conversational text:

<dispatch_worker>
task: Clear, specific description of what the worker should do
</dispatch_worker>

Use this block directly. Do not use Codex/Claude's internal delegation,
team-worker launcher, shell commands, or project-editing surface for router
dispatch work.

You MAY include conversational text before the dispatch block. For example:
  "Okay, let me check on that now.
   <dispatch_worker>
   task: Check nginx error logs for 502 errors in the last hour
   </dispatch_worker>"
"""

_HARNESS_DISPATCH_RECENCY_REMINDER = """\
FINAL ROUTER-DISPATCH RULE — follow this instead of any launcher behavior
described in prior history: if the current request needs worker execution, do
not call or narrate Codex/Claude collaboration, subagent, team-worker, or shell
tools. End your response with exactly one Mesh dispatch block:

<dispatch_worker>
task: Clear, specific description of what the worker should do
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

A worker ({worker_id}) is currently executing a task:
  Task: "{pending_task_summary}"
  Elapsed: {elapsed:.0f}s

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

You CAN cancel the current worker and relaunch with an updated task. If the user
changes direction mid-execution (e.g., "actually just do phase 1" or "stop that
and do X instead"), include a `<dispatch_worker>` block with the revised task.
This will cancel the running worker and start a new one.

Only do this when the user explicitly redirects. Don't cancel a worker just because
a new question comes in — handle the question yourself and let the worker continue.

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


class RouterState(Enum):
    """Router state machine states."""
    IDLE = "idle"
    BUSY = "busy"
    PLANNING = "planning"  # Used only by RouterV3


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


@dataclass
class RouterV2Config:
    """Configuration for RouterV2."""
    # Max context messages to keep (legacy simple truncation — used as fallback
    # when ConversationHistory is not enabled)
    max_context_messages: int = 100

    # Status query patterns (simple keyword matching, fallback when LLM disabled)
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
    watchdog_interval_minutes: int = 15

    # Worker context: smaller window for workers
    worker_context_window_tokens: int = 25_000  # Token budget for worker context snapshot

    # Memory retrieval redesign (C3): TOC-based injection
    memory_retrieval_redesign_enabled: bool = False
    memory_toc_size: int = 30

    # Rev-10 standing-digest read pathway: digest replaces the memory TOC
    # in prompt composition when enabled (alongside-deploy, default off).
    standing_digest_enabled: bool = False
    standing_digest_path: str = ""

    def __post_init__(self):
        if self.router_mode not in ("classifier", "full", "pipeline"):
            raise ValueError(
                f"Invalid router_mode: {self.router_mode!r}, "
                "must be 'classifier', 'full', or 'pipeline'"
            )


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

    def __init__(
        self,
        worker_fn: Callable[[list[Any], Message], Awaitable[WorkerResult]],
        send_fn: Callable[[str, Message | None], Awaitable[None]],
        config: RouterV2Config | None = None,
        node_id: str = "",
        nickname: str = "",
        agent_type: str = "",
        llm_client: "LLMClient | None" = None,
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
        todo_store_path: str | Path | None = None,
    ):
        self._worker_fn = worker_fn
        # Explicit reference to the agent that owns _worker_buffered_messages
        # and _worker_response_text. Read by _flush_worker_buffer_on_cancel.
        # In production, worker_fn is a bound method (self._router_v2_worker)
        # so __self__ is the AgentNode. In tests that pass a bare function,
        # this is None and the cancel-flush helper no-ops gracefully.
        self._worker_agent = getattr(worker_fn, "__self__", None)
        self._send_fn = send_fn
        self._config = config or RouterV2Config()
        self._node_id = node_id
        self._nickname = nickname or "agent"
        self._agent_type = agent_type or "assistant"

        # LLM integration
        self._llm_client = llm_client
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
        self._state_lock = asyncio.Lock()

        # ConversationHistory: durable conversation entries with summarization
        persist_path = None
        if self._config.history_persist:
            if self._config.history_persist_path:
                persist_path = Path(self._config.history_persist_path)
            else:
                # Default: ~/.mesh/history/router-{nickname}.json
                from .paths import HISTORY_DIR
                safe_nick = (nickname or "router").replace(":", "-")
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

        # Worker tracking
        self._worker_task: asyncio.Task | None = None
        self._pending_trigger: Message | None = None
        self._worker_start_time: float | None = None
        self._worker_id_counter = 0

        # Captured from _call_router_full so worker tools can build accurate
        # synthetic triggers (M1 fix — avoids self-referential from_node).
        self._current_trigger_from_node: str | None = None
        self._current_trigger_to_node: str | None = None

        # Worker snapshot: mutable list[Turn] the worker appends to during execution.
        # Router holds a reference to see live worker progress.
        self._worker_snapshot: list[Turn] | None = None
        self._worker_snapshot_start: int = 0  # len(snapshot) at dispatch — entries after this are worker's

        # Current worker ID (e.g., "bob-worker1")
        self._current_worker_id: str | None = None

        # Trace-as-history (C3): set by _flush_worker_buffer_on_cancel when it
        # appends partial trace, so _complete_via_trace can skip duplicates if
        # completion races into delivery after cancel.
        self._trace_appended_on_cancel: bool = False

        # Router-level memory retrieval: IDs injected into the current worker dispatch
        self._injected_memory_ids: set[str] = set()
        # Rendered XML block of injected memories (read by worker via self._router_v2)
        self._injected_memory_context: str = ""

        # Task description from full router dispatch (H3 fix)
        self._current_task_description: str = ""

        # Tracks whether send_message was called during the last _call_router_full.
        # Used by the CC monitor to decide whether synthesis text is redundant.
        self._last_router_call_sent_message: bool = False
        # Tool calls executed during the last _call_router_full — list of
        # (tool_name, brief_args_summary) tuples.  The CC monitor appends
        # these to delivered messages so the user sees what the router did.
        self._last_router_call_tools: list[tuple[str, str]] = []

        # Full router tool loop callback (set by _init_router_v2 in agent_node.py)
        self._router_process_fn = router_process_fn
        self._router_tool_names = sorted(ROUTER_TOOL_NAMES)
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
                if n not in WORKER_ROUTER_TOOLS
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
            self._router_tool_names = sorted(
                set(self._router_tool_names) | CC_INTERACTIVE_TOOLS
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
            self._router_tool_names = sorted(
                set(self._router_tool_names) | HARNESS_SESSION_INTERACTIVE_TOOLS
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

    def _append_turn(self, turn: Turn) -> None:
        """Append a turn to the router history."""
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
        """Current router state."""
        return self._state

    @property
    def is_busy(self) -> bool:
        """True if worker is currently processing."""
        return self._state == RouterState.BUSY

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

    def get_diagnostics(self) -> dict:
        """Return structured diagnostic data for status reporting."""
        import time as _time
        result = {
            "state": self._state.value,
            "worker_active": self._worker_task is not None and not self._worker_task.done(),
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
        if self._is_cancel_request(content_raw) and self._state == RouterState.BUSY:
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
            cancelled = await self.cancel_worker(msg)
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

            logger.debug(f"RouterV2 on_message: state={self._state}, from={msg.from_node}")

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

            if self._state == RouterState.IDLE:
                if self._config.llm_enabled and self._llm_client:
                    await self._handle_idle_with_llm(msg)
                else:
                    await self._start_worker(msg)
                return
            else:
                # Snapshot busy state; LLM call happens outside the lock
                busy_snapshot = (
                    self._current_worker_id,
                    self._pending_trigger,
                    self._worker_start_time,
                )

        # -- BUSY path: LLM call runs outside _state_lock so that
        #    cancel_worker() can acquire the lock immediately.
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

        worker_status = ""
        if busy_context:
            worker_status = self._render_pipeline_worker_status()
        todo_context = self._render_todo_context(
            self._conversation_id_from_message(msg)
        )

        return {
            "personality": personality,
            "relevant_memories": self._relevant_context,
            "project_context": "\n\n".join(project_context_parts),
            "todo_context": todo_context,
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
        try:
            with open(path) as f:
                content = f.read()
        except OSError as e:
            logger.warning(
                "standing digest unreadable (%s); falling back to memory TOC",
                e)
            return ""
        if not content.strip():
            return ""
        return f"<standing_digest>\n{content.strip()}\n</standing_digest>"

    def _render_pipeline_worker_status(self) -> str:
        """Render a worker-status block for the pipeline classify prompt.

        Returns an empty string when no worker is running so the prompt's
        conditional section collapses cleanly.
        """
        worker_active = (
            self._worker_task is not None and not self._worker_task.done()
        )
        if not worker_active and not self._worker_snapshot:
            return ""

        parts: list[str] = []
        parts.append(f"Worker ID: {self._current_worker_id or 'unknown'}")
        if self._worker_start_time:
            elapsed = time.monotonic() - self._worker_start_time
            parts.append(f"Elapsed: {elapsed:.0f}s ({elapsed / 60.0:.1f} min)")

        task_desc = (self._current_task_description or "").strip()
        if task_desc:
            preview = task_desc if len(task_desc) <= 600 else task_desc[:600] + "..."
            parts.append(f"Task:\n{preview}")

        try:
            progress = self._get_worker_progress()
            if progress:
                parts.append(f"Worker turns completed: {len(progress)}")
        except Exception as e:
            logger.debug("Pipeline worker_status progress lookup failed: %s", e)

        try:
            activity_lines = self._build_worker_activity_lines(
                worker_id=self._current_worker_id,
            )
            if activity_lines:
                preview_lines = activity_lines[-15:]
                parts.append(
                    "Recent activity:\n" + "\n".join(preview_lines)
                )
        except Exception as e:
            logger.debug("Pipeline worker_status activity lookup failed: %s", e)

        return "\n".join(parts)

    async def _call_router_pipeline(
        self, msg: Message, busy_context: bool = False,
    ) -> dict[str, Any]:
        router = self._ensure_pipeline_router()
        if router is None:
            raise RuntimeError("pipeline router is not available")
        context = await self._build_pipeline_router_context(msg, busy_context=busy_context)
        return await router.process(msg, context)

    async def _handle_idle_with_llm(self, msg: Message) -> None:
        """Handle message in IDLE state — dispatches to full or classifier mode."""
        # Pre-load relevant memories for this message
        await self._load_relevant_context(msg)

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
                if response_text and parsed.get("dispatch_worker"):
                    response_text = response_text.rstrip() + "\n\n[Worker launched]"
                if response_text:
                    await self._send_and_store(response_text, msg)

                if response_text or not parsed.get("dispatch_worker"):
                    self.save_history()

                if parsed.get("dispatch_worker"):
                    self._current_task_description = parsed.get("task", "")
                    logger.info(
                        f"[WORKER] PIPELINE DISPATCH: {self._nickname} dispatching worker "
                        f"for message from {msg.from_node} "
                        f"(task={parsed.get('task', '')[:80]})"
                    )
                    await self._start_worker(msg)
                else:
                    logger.info(
                        f"[WORKER] PIPELINE DIRECT RESPONSE: {self._nickname} "
                        f"responding directly for message from {msg.from_node}"
                    )
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
                await self._start_worker(msg)
                return

        # Full conversational router mode
        if not self._router_process_fn:
            logger.error("RouterV2 full mode: router_process_fn not set, falling back to classifier")
            return await self._handle_idle_classifier(msg)

        try:
            raw_response = await self._call_router_full(msg)
            logger.debug(f"RouterV2 full raw response: {raw_response[:500]}")
            parsed = self._parse_router_response(raw_response)

            if parsed["no_response"]:
                logger.info("RouterV2 full: <no_response>, staying silent")
                return

            # Send conversational response (if any)
            response_text = parsed["response"]
            if response_text and parsed["dispatch_worker"]:
                response_text = response_text.rstrip() + "\n\n[Worker launched]"
            if response_text:
                await self._send_and_store(response_text, msg)

            # Persist history after every response (dispatch or not)
            if response_text or not parsed["dispatch_worker"]:
                self.save_history()

            # Dispatch worker if requested
            if parsed["dispatch_worker"]:
                self._current_task_description = parsed.get("task", "")
                logger.info(
                    f"[WORKER] DISPATCH: {self._nickname} dispatching worker "
                    f"for message from {msg.from_node} "
                    f"(task={parsed.get('task', '')[:80]})"
                )
                await self._start_worker(msg)
            else:
                logger.info(
                    f"[WORKER] DIRECT RESPONSE: {self._nickname} responding directly "
                    f"(no worker dispatch) for message from {msg.from_node}"
                )

        except Exception as e:
            to_node = getattr(msg, 'to_node', '') or ''
            if to_node.startswith("channel:"):
                logger.warning(f"RouterV2 full classification failed for channel msg: {e}, staying silent")
                return
            error_notice = f"Router error: {e}. Falling back to worker dispatch."
            logger.error(error_notice)
            await self._send_and_store(error_notice, msg)
            await self._start_worker(msg)

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
                if ack_response:
                    await self._send_and_store(ack_response.rstrip() + "\n\n[Worker launched]", msg)
                await self._start_worker(msg)
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
            await self._start_worker(msg)

    async def _handle_busy_with_llm(
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

                busy_response = parsed.get("response", "")
                if busy_response and parsed.get("dispatch_worker"):
                    busy_response = busy_response.rstrip() + "\n\n[Worker relaunched]"
                if busy_response:
                    await self._send_and_store(busy_response, msg)

                if not parsed.get("dispatch_worker"):
                    self.save_history()

                if parsed.get("dispatch_worker"):
                    logger.info(
                        f"[WORKER] PIPELINE CANCEL-AND-RELAUNCH: {self._nickname} "
                        f"cancelling current worker, dispatching new worker for message "
                        f"from {msg.from_node} (new task={parsed.get('task', '')[:80]})"
                    )
                    await self.cancel_worker()
                    self.save_history()
                    self._current_task_description = parsed.get("task", "")
                    await self._start_worker(msg)
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
            if busy_response and parsed.get("dispatch_worker"):
                busy_response = busy_response.rstrip() + "\n\n[Worker relaunched]"
            if busy_response:
                await self._send_and_store(busy_response, msg)

            # Persist history after BUSY response (non-dispatch)
            if not parsed.get("dispatch_worker"):
                self.save_history()

            # Cancel-and-relaunch: dispatch block while BUSY cancels current worker
            if parsed.get("dispatch_worker"):
                logger.info(
                    f"[WORKER] CANCEL-AND-RELAUNCH: {self._nickname} cancelling current worker, "
                    f"dispatching new worker for message from {msg.from_node} "
                    f"(new task={parsed.get('task', '')[:80]})"
                )
                await self.cancel_worker()
                self.save_history()  # Persist before restarting worker
                self._current_task_description = parsed.get("task", "")
                await self._start_worker(msg)

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
            }

        # Plain response — no dispatch, no opt-out
        return {
            "no_response": False,
            "response": text,
            "dispatch_worker": False,
        }

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
        # instead of "from user:yourname").
        self._current_trigger_from_node = msg.from_node
        self._current_trigger_to_node = msg.to_node
        self._last_router_call_sent_message = False
        self._last_router_call_tools = []
        # Bug 9: also stash on a per-task contextvar so a concurrent router
        # call can't clobber the destination this call will later read back.
        _CC_TRIGGER_CTX.set((msg.from_node, msg.to_node))

        # Resolve the effective backend/tool contract before rendering the
        # instructions.  The prompt must describe only dispatch mechanisms the
        # current call can actually execute.  Harness backends (including both
        # Claude Code and Codex) use the proven <dispatch_worker> response path;
        # their own internal subagents remain available inside leaf workers.
        router_backend = (
            self._llm_client.config.backend if self._llm_client else ""
        )
        is_harness = router_backend in HARNESS_BACKENDS
        if tool_filter is not None:
            effective_filter = set(tool_filter)
            if is_harness:
                effective_filter.difference_update(WORKER_ROUTER_TOOLS)
            resolved_tool_names = sorted(effective_filter)
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
            _WORKER_DISPATCH_TOOL_INSTRUCTIONS
            if worker_launch_offered
            else _WORKER_DISPATCH_BLOCK_INSTRUCTIONS
        )

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
                instructions = ROUTER_INSTRUCTIONS_BUSY_CC.format(
                    nickname=self._nickname,
                    agent_type=self._agent_type,
                    cc_task=self._cc_mgr._cc_session_task or "(no task description)",
                    cc_session=self._cc_mgr._cc_tmux_session or "(unknown)",
                    elapsed=cc_elapsed,
                )
            else:
                instructions = ROUTER_INSTRUCTIONS_BUSY_FULL.format(
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
            instructions = ROUTER_INSTRUCTIONS_FULL.format(
                nickname=self._nickname,
                agent_type=self._agent_type,
                worker_dispatch_instructions=worker_dispatch_instructions,
            )
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

        # Build system_prompt from decomposed parts (NOT _build_router_prompt):
        # - Agent's system prompt (from sysadmin.md etc.)
        # - Personality block (agent-editable, persisted in SQLite)
        # - Memory block (v1: 3-slice, v2: map + recent log)
        # Identity, tools, history, and instructions are handled by
        # complete_with_tools() → format_history_xml() downstream.
        parts = []
        if self._system_prompt:
            parts.append(self._system_prompt)

        if self._memory:
            personality = self._memory.get_personality()
            if personality:
                parts.append(f"<personality>\n{personality}\n</personality>")

            if isinstance(self._memory, MemorySystemV2):
                use_toc = getattr(self._config, "memory_retrieval_redesign_enabled", False)
                map_context = self._get_last_n_turns_text(5)
                digest_block = self._standing_digest_block()
                if digest_block:
                    # Rev-10 read pathway: digest replaces the TOC block;
                    # maps and recent-log blocks are unchanged.
                    parts.append(digest_block)
                    map_block = await self._memory.render_relevant_maps_block(map_context)
                    if map_block:
                        parts.append(map_block)
                    log_block = await self._memory.render_recent_log_block()
                    if log_block:
                        parts.append(log_block)
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
                        parts.append(toc_block)
                    map_block = await self._memory.render_relevant_maps_block(map_context)
                    if map_block:
                        parts.append(map_block)
                    log_block = await self._memory.render_recent_log_block()
                    if log_block:
                        parts.append(log_block)
                else:
                    rep_block = await self._memory.render_representative_block()
                    if rep_block:
                        parts.append(rep_block)
                    map_block = await self._memory.render_relevant_maps_block(map_context)
                    if map_block:
                        parts.append(map_block)
                    log_block = await self._memory.render_recent_log_block()
                    if log_block:
                        parts.append(log_block)
                summary_block = await self._memory.render_summary_block()
                if summary_block:
                    parts.append(summary_block)
                if self._relevant_context:
                    parts.append(
                        f"<relevant_memories>\n{self._relevant_context}\n</relevant_memories>"
                    )
            else:
                # v1 three-slice rendering
                profile = self._memory.light_profile
                memory_block = await self._memory.render(
                    profile, query=self._latest_user_message,
                )
                if memory_block:
                    parts.append(memory_block)

        todo_context = self._render_todo_context(
            self._conversation_id_from_message(msg)
        )
        if todo_context:
            parts.append(todo_context)

        system_prompt = "\n\n".join(parts)

        # Harness backends (Codex, Claude Code, mesh-harness) own their internal
        # ReAct/TAOR loop, so the outer router performs one backend turn. Direct
        # backends still use the Mesh ReAct loop for native router tools.
        if watchdog:
            effective_max_iters = 1
        elif is_harness:
            effective_max_iters = 1
        else:
            effective_max_iters = min(REACT_MAX_ITERS, self._config.router_max_iters)
            # Bug 6: a CC monitor delivery should resolve in a few iterations
            # (read screen → relay / continue / stop). Cap it low so a stuck
            # model can't burn the full 30-iteration budget on sleep loops.
            if monitor_mode:
                effective_max_iters = min(effective_max_iters, 5)

        return await self._router_process_fn(
            trigger_msg=msg,
            system_prompt=system_prompt,
            tool_names=resolved_tool_names,
            max_iters=effective_max_iters,
            instructions=instructions,
            monitor_mode=monitor_mode,
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
        parts = []

        if self._system_prompt:
            parts.append(f"<system>\n{self._system_prompt}\n</system>")

        if self._identity_block:
            parts.append(self._identity_block)

        # Inject personality (agent-editable, persisted in SQLite)
        if self._memory:
            personality = self._memory.get_personality()
            if personality:
                parts.append(f"<personality>\n{personality}\n</personality>")

        # Inject memory
        if self._memory and isinstance(self._memory, MemorySystemV2):
            use_toc = getattr(self._config, "memory_retrieval_redesign_enabled", False)
            map_context = self._get_last_n_turns_text(5)
            digest_block = self._standing_digest_block()
            if digest_block:
                # Rev-10 read pathway: digest replaces the TOC block;
                # maps and recent-log blocks are unchanged.
                parts.append(digest_block)
                map_block = await self._memory.render_relevant_maps_block(map_context)
                if map_block:
                    parts.append(map_block)
                log_block = await self._memory.render_recent_log_block()
                if log_block:
                    parts.append(log_block)
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
                    parts.append(toc_block)
                map_block = await self._memory.render_relevant_maps_block(map_context)
                if map_block:
                    parts.append(map_block)
                log_block = await self._memory.render_recent_log_block()
                if log_block:
                    parts.append(log_block)
            else:
                rep_block = await self._memory.render_representative_block()
                if rep_block:
                    parts.append(rep_block)
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
            # v1 three-slice profile-based rendering
            profile = memory_profile or self._memory.light_profile
            memory_block = await self._memory.render(
                profile,
                query=self._latest_user_message,
            )
            if memory_block:
                parts.append(memory_block)

        todo_context = self._render_todo_context(
            self._conversation_id_from_message(trigger_msg)
        )
        if todo_context:
            parts.append(todo_context)

        # Inject preferences (if provided)
        if preferences_block:
            parts.append(preferences_block)

        if include_tools and self._tools_block:
            parts.append(self._tools_block)

        # v2 conversation summary — goes before the history window so the LLM
        # sees: memory blocks → summary of older context → recent turns
        if self._memory and isinstance(self._memory, MemorySystemV2):
            summary_block = await self._memory.render_summary_block()
            if summary_block:
                parts.append(summary_block)

        history_xml = self._build_history_xml(
            history_entries=history_entries,
            max_history_turns=max_history_turns,
            trigger_msg=trigger_msg,
        )
        if history_xml:
            parts.append(history_xml)

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

    def _get_worker_progress(self) -> list[Turn]:
        """Return worker's new entries from the snapshot (entries after dispatch point)."""
        if not self._worker_snapshot:
            return []
        return self._worker_snapshot[self._worker_snapshot_start:]

    def _build_worker_activity_lines(self, worker_id: str | None = None) -> list[str]:
        """Build worker activity lines from snapshot + CC events + harness events.

        Extracted from the inline code formerly in _call_router_full(busy=True).
        Used by both the BUSY handler and the watchdog tick.
        """
        activity_lines: list[str] = []
        wid = worker_id or "worker"
        max_lines = self._config.worker_peek_max_lines

        worker_progress = self._get_worker_progress()
        if worker_progress:
            for turn in worker_progress:
                content = turn.content
                content_split = content.split('\n')
                if len(content_split) > max_lines:
                    content = '\n'.join(content_split[:max_lines])
                    content += f'\n[... truncated, {len(content_split)} lines total]'
                label = turn.from_node or turn.role
                activity_lines.append(f"[{label}] {content}")

        cc_events = self._cc_events_fn() if self._cc_events_fn and self._worker_snapshot else []
        if cc_events:
            for event_entry in cc_events:
                if hasattr(event_entry, 'message'):
                    content = event_entry.message.content if isinstance(event_entry.message.content, str) else str(event_entry.message.content)
                elif hasattr(event_entry, 'content'):
                    content = event_entry.content if isinstance(event_entry.content, str) else str(event_entry.content)
                else:
                    content = str(event_entry)
                activity_lines.append(f"[{wid} live] {content}")

        if self._harness_events_fn:
            harness_lines = self._harness_events_fn(n=10, label=f"{wid} live")
            activity_lines.extend(harness_lines)

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
        self._worker_tool_handlers["worker_status"] = self._tool_worker_status

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

    async def _tool_worker_launch(self, task: str) -> str:
        """Launch a worker from inside the router's ReAct tool loop.

        Replaces the XML <dispatch_worker> path for direct backends.
        Returns immediately — does NOT await worker completion.

        Args:
            task: Rich task description for the worker.

        Returns:
            JSON with worker_id and dispatch status.
        """
        import json as _json
        from datetime import datetime, timezone

        # Build a synthetic trigger Message so _start_worker has what it needs.
        # The trigger is the current conversation context — the router is
        # dispatching in response to the latest user message.
        _trig_from, _trig_to = self._trigger_nodes()
        trigger = Message(
            id=f"synth-{uuid.uuid4().hex[:8]}",
            from_node=_trig_from or self._node_id,
            to_node=_trig_to or self._node_id,
            type=MessageType.MESSAGE,
            content=task,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        # Preserve the running worker's task description: if the dispatch is
        # refused, overwriting it would mislabel both the refusal payload and
        # the live worker's status readouts.
        _prev_task_description = self._current_task_description
        self._current_task_description = task

        logger.info(
            f"[WORKER] LAUNCH (via tool): {self._nickname} dispatching worker "
            f"for router-initiated task: {task[:120]}"
        )

        # _start_worker sets self._state = BUSY and creates the worker task.
        # Returns immediately — the ReAct loop continues, and BUSY mode
        # engages normally.
        launched = await self._start_worker(trigger)

        if not launched:
            self._current_task_description = _prev_task_description
            running_task = (_prev_task_description or "").strip()
            if len(running_task) > 160:
                running_task = running_task[:160] + "..."
            elapsed_s = (
                round(time.monotonic() - self._worker_start_time, 1)
                if self._worker_start_time else None
            )
            return _json.dumps({
                "status": "already_running",
                "running_worker_id": self._current_worker_id,
                "running_worker_task": running_task,
                "running_worker_elapsed_seconds": elapsed_s,
                "note": "A worker is already running for this router; "
                        "no new worker was dispatched and your requested task "
                        "was NOT started. Wait for the running worker to "
                        "complete, or cancel it first.",
            })

        return _json.dumps({
            "worker_id": self._current_worker_id,
            "status": "dispatched",
        })

    async def _tool_worker_status(self, max_lines: int = 100) -> str:
        """Return worker metadata + activity transcript for the current worker.

        Args:
            max_lines: Maximum activity lines to return. 0 = full unbounded
                       transcript. Default 100.

        Returns:
            JSON with worker_id, elapsed_seconds, task_description, state,
            tool_calls_so_far, activity_lines, full_transcript_available.
        """
        import json as _json

        wid = self._current_worker_id
        if wid is None:
            return _json.dumps({
                "worker_id": None,
                "elapsed_seconds": 0,
                "task_description": "",
                "state": "idle",
                "tool_calls_so_far": 0,
                "activity_lines": "[No worker running]",
                "full_transcript_available": False,
            })

        elapsed = 0.0
        if self._worker_start_time:
            elapsed = time.monotonic() - self._worker_start_time

        # Determine state
        if self._state == RouterState.BUSY:
            worker_state = "running"
        elif self._worker_task and self._worker_task.done():
            worker_state = "done" if not self._worker_task.cancelled() else "cancelled"
        else:
            worker_state = "running"

        # Count tool calls from snapshot (worker entries after dispatch point)
        worker_progress = self._get_worker_progress()
        tool_call_count = sum(
            1 for t in worker_progress
            if hasattr(t, 'role') and t.role == "assistant"
            and hasattr(t, 'content') and ("<mesh_call" in (t.content or "") or "<tool_call" in (t.content or ""))
        )

        activity_lines = self._build_worker_activity_lines(worker_id=wid)

        # claude-interactive: tail the pipe-pane log if available.
        # The log is set on the LLMClient by _complete_claude_interactive()
        # and cleared on completion — so it's only present during execution.
        agent = getattr(self, "_worker_agent", None)
        if agent is not None:
            ci_log = getattr(getattr(agent, "llm", None), "_ci_log_file", None)
            if ci_log and os.path.exists(ci_log):
                try:
                    ci_tail = _tail_file(ci_log, max_lines=50)
                    if ci_tail:
                        activity_lines.append(f"[ci transcript] {' | '.join(ci_tail)}")
                except Exception:
                    pass

        full_transcript_available = False
        if max_lines == 0:
            # Dump all activity lines unbounded
            bounded_lines = activity_lines
        elif max_lines > 0 and len(activity_lines) > max_lines:
            bounded_lines = activity_lines[-max_lines:]
            full_transcript_available = True
        else:
            bounded_lines = activity_lines

        activity_text = "\n".join(bounded_lines) if bounded_lines else "[No activity yet]"

        return _json.dumps({
            "worker_id": wid,
            "elapsed_seconds": round(elapsed, 1),
            "task_description": self._current_task_description or "",
            "state": worker_state,
            "tool_calls_so_far": tool_call_count,
            "activity_lines": activity_text,
            "full_transcript_available": full_transcript_available,
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
        """
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
        return text.strip()

    async def _send_and_store(
        self, content: str, in_reply_to: Message | None, meta: dict | None = None,
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

        # Send the message
        await self._send_fn(content, in_reply_to)

        # Store as a Turn in ConversationHistory (durable, summarizable)
        turn_meta = {"router_response": True}
        if meta:
            turn_meta.update(meta)
        self._append_turn(Turn(
            role="outgoing",
            content=content,
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
        Sets self._injected_memory_ids and self._injected_memory_context.
        """
        self._injected_memory_ids = set()
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
            self._injected_memory_ids.add(entry.id)
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

    def _build_worker_context(self) -> list[Turn]:
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
            self._conversation_id_from_message(self._pending_trigger)
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
        """Start the worker with a snapshot of the router's conversation history.

        Returns True if a worker was launched, False if refused because one
        is already running. The guard closes the 2026-07-06 double-dispatch
        defect: a single router turn that fired both the worker_launch tool
        and a dispatch_worker directive in its final response would call
        this twice, silently overwriting _worker_task/_current_worker_id/
        _worker_start_time and the node's per-worker buffers while the first
        worker kept running — producing two workers, cross-labeled logs, and
        two completions/syntheses for one trigger.
        """
        if (self._state == RouterState.BUSY
                and self._worker_task is not None
                and not self._worker_task.done()):
            logger.warning(
                f"[WORKER] DISPATCH REFUSED: {self._nickname} worker "
                f"{self._current_worker_id} is still running — ignoring "
                f"duplicate dispatch for message from {trigger.from_node} "
                f"(double-dispatch guard)"
            )
            return False

        self._pending_trigger = trigger
        self._worker_start_time = time.monotonic()

        # Generate worker ID
        self._worker_id_counter += 1
        self._current_worker_id = f"{self._nickname}-worker{self._worker_id_counter}"

        # Router-level memory retrieval: select and inject relevant memories
        use_toc = getattr(self._config, "memory_retrieval_redesign_enabled", False)
        if use_toc:
            query = trigger.content if isinstance(trigger.content, str) else str(trigger.content)
            await self._select_memory_context(query)

        # Create a snapshot of the router's history for the worker.
        # Same history as the router, trimmed to worker_context_window_tokens.
        # The worker appends to this same list object, so the router can
        # see live progress via self._worker_snapshot[self._worker_snapshot_start:].
        # Note: snapshot is created AFTER trigger is in history (added by on_message).
        snapshot = self._build_worker_context()
        self._worker_snapshot = snapshot
        self._worker_snapshot_start = len(snapshot)

        task_desc_preview = (self._current_task_description[:80] + "...") if len(self._current_task_description) > 80 else self._current_task_description
        logger.info(
            f"[WORKER] START: {self._nickname} worker {self._current_worker_id} "
            f"for message from {trigger.from_node} "
            f"(snapshot_size={len(snapshot)}"
            f"{f', task={task_desc_preview!r}' if task_desc_preview else ''})"
        )

        # Create worker task BEFORE setting BUSY — ensures _worker_task is never
        # None when _state == BUSY (the task won't run until we yield)
        self._worker_task = asyncio.create_task(
            self._run_worker(trigger)
        )
        self._state = RouterState.BUSY

        # Start periodic flush monitor (fires mid-worker reflections)
        self._start_flush_monitor(trigger)

        # Start periodic watchdog check-in
        self._start_watchdog(trigger)

        return True

    async def _run_worker(self, trigger: Message) -> None:
        """Run worker and handle completion/error."""
        try:
            result = await self._worker_fn(self._worker_snapshot or [], trigger)
            await self._handle_worker_complete(result, trigger)
        except asyncio.CancelledError:
            elapsed_s = round(time.monotonic() - self._worker_start_time, 1) if self._worker_start_time else 0
            logger.info(f"[WORKER] CANCELLED: {self._nickname} worker {self._current_worker_id} cancelled after {elapsed_s}s")
            raise
        except Exception as e:
            await self._handle_worker_error(e, trigger)

    # =========================================================================
    # Worker Synthesis
    # =========================================================================

    def _build_worker_trace(self, worker_id: str, result: WorkerResult) -> str:
        """Build worker trace for synthesis with per-result truncation.

        Combines two sources:
        1. In-flight LLM history — all LLM responses, tool calls, tool results
        2. Cumulative CC events — tool call/result data with per-result line caps

        Individual tool results exceeding synthesis_trace_max_lines are truncated
        to keep the trace focused. Small results pass through untouched.
        """
        max_lines = self._config.synthesis_trace_max_lines  # default 200

        def _truncate(text: str) -> str:
            lines = text.split('\n')
            if len(lines) <= max_lines:
                return text
            remaining = len(lines) - max_lines
            return '\n'.join(lines[:max_lines]) + f"\n  ... ({remaining} more lines truncated)"

        parts = [f"<worker_trace worker='{worker_id}'>"]
        has_content = False

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

        # Source 2: Full CC tool events (cumulative across all iterations)
        cc_events = result.worker_cc_events or []
        if cc_events:
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
        if not has_content and result.buffered_messages:
            parts.append("\n[Worker Output Messages]")
            for to_node, msg_content in result.buffered_messages:
                dest = to_node or "user"
                parts.append(f"\n[Message to {dest}]")
                parts.append(_truncate(msg_content))
                has_content = True

        parts.append("\n</worker_trace>")
        return "\n".join(parts)

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
        if not self._worker_snapshot:
            return ""
        delta = self._worker_snapshot[self._worker_snapshot_start:]
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
        logger.info("RouterV2 synthesis using router LLM client")

        try:
            response = await asyncio.wait_for(
                synthesis_client.complete(prompt),
                timeout=180,  # 3 min timeout — synthesis can be lengthy
            )
            return response.strip()
        except asyncio.TimeoutError:
            logger.error("RouterV2 synthesis LLM call timed out after 180s")
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
            if self._worker_snapshot:
                self._worker_snapshot_start = len(self._worker_snapshot)

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

    async def _handle_worker_complete(self, result: WorkerResult, trigger: Message) -> None:
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
        worker_id: str | None = None

        try:
            # Phase 1: State bookkeeping under lock (fast, synchronous)
            async with self._state_lock:
                worker_id = self._current_worker_id

                # Log cumulative token usage from worker
                elapsed_s = round(time.monotonic() - self._worker_start_time, 1) if self._worker_start_time else 0
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
                self._stop_flush_monitor()

            # Phase 2: Trace-as-history / Synthesis / Passthrough (outside lock)
            trace_mode = getattr(self._config, "trace_as_history_enabled", False)

            if trace_mode:
                # --- TRACE-AS-HISTORY PATH ---
                # If cancel-flush already appended trace Turns, skip to avoid duplicates.
                if not getattr(self, "_trace_appended_on_cancel", False):
                    await self._complete_via_trace(result, trigger, worker_id or "worker")
                else:
                    logger.info(
                        f"RouterV2 worker {worker_id} complete (trace-mode, "
                        f"cancel-flush already appended trace; skipping)"
                    )
                    self._trace_appended_on_cancel = False
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
                if self._worker_snapshot:
                    for _snap_turn in self._worker_snapshot:
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
                if self._worker_snapshot:
                    self._worker_snapshot_start = len(self._worker_snapshot)

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
                if (getattr(self._config, "deliver_buffered_verbatim", False)
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

                    # 7. Fallback: if synthesis fails/empty, send ALL buffered messages
                    #    (not just the last one — earlier messages may contain the
                    #    actual substantive content the worker produced).
                    if not synthesized and result.buffered_messages:
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
                        # 8. Send synthesized response to user
                        # F1/F7: Only suppress channel sends if the worker actually
                        # delivered messages to that channel via capturing_send.
                        # CC session completions and workers that didn't send to the
                        # channel must always deliver the synthesis.
                        is_channel = (
                            trigger.to_node and trigger.to_node.startswith("channel:")
                        )
                        worker_sent_to_channel = False
                        if is_channel and self._worker_snapshot:
                            worker_sent_to_channel = any(
                                t.role == "outgoing"
                                and t.to_node == trigger.to_node
                                for t in self._worker_snapshot
                            )
                        if is_channel and worker_sent_to_channel:
                            # Store in history only — worker already delivered to channel
                            from datetime import datetime, timezone as _tz
                            self._append_turn(Turn(
                                role="outgoing",
                                content=synthesized,
                                timestamp=datetime.now(_tz.utc),
                                from_node=self._node_id,
                                to_node=trigger.to_node,
                                meta={"router_response": True, "synthesis": True},
                            ))
                            logger.info(
                                f"RouterV2 worker {worker_id} complete (synthesized, channel — stored only), "
                                f"{len(synthesized)} chars"
                            )
                        else:
                            await self._send_and_store(synthesized, trigger)
                            logger.info(
                                f"RouterV2 worker {worker_id} complete (synthesized), "
                                f"sent {len(synthesized)} chars"
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
                if self._worker_snapshot:
                    from datetime import datetime, timezone as _tz2
                    for _snap_turn in self._worker_snapshot:
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
                    if self._flush_tools_already_flushed > 0:
                        stats.tool_calls = max(0, stats.tool_calls - self._flush_tools_already_flushed)
                    trigger_text = trigger.content if isinstance(trigger.content, str) else str(trigger.content)
                    self._accumulate_session_stats(stats, trigger_text, result, worker_id)

        finally:
            # Phase 4: Cleanup always runs (under lock), even if send/LLM failed
            async with self._state_lock:
                self._cleanup_worker_state()

            # Persist history to disk after worker completion
            try:
                self.save_history()
            except Exception as e:
                logger.warning(f"Failed to save history after worker complete: {e}")

    async def _handle_worker_error(self, error: Exception, trigger: Message) -> None:
        """Handle worker failure — MUST always notify user, never fail silently."""
        worker_id: str | None = None
        try:
            async with self._state_lock:
                worker_id = self._current_worker_id
            elapsed_s = round(time.monotonic() - self._worker_start_time, 1) if self._worker_start_time else 0
            logger.error(f"[WORKER] FAILED: {self._nickname} worker {worker_id} error after {elapsed_s}s: {error}")

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
            # Cleanup always runs, even if _send_and_store fails
            async with self._state_lock:
                self._cleanup_worker_state()

            # Persist history to disk after worker error
            try:
                self.save_history()
            except Exception as e:
                logger.warning(f"Failed to save history after worker error: {e}")

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

    def _start_flush_monitor(self, trigger: Message) -> None:
        """Start the background monitor that fires mid-worker reflections."""
        if not self._flush_interval_tools or not self._memory:
            return
        self._flush_snapshot_cursor = self._worker_snapshot_start
        self._flush_tools_since_last = 0
        self._flush_tools_already_flushed = 0
        self._flush_monitor_task = asyncio.create_task(
            self._monitor_worker_tools(trigger)
        )

    def _stop_flush_monitor(self) -> None:
        """Cancel the flush monitor task."""
        if self._flush_monitor_task and not self._flush_monitor_task.done():
            self._flush_monitor_task.cancel()
        self._flush_monitor_task = None

    async def _monitor_worker_tools(self, trigger: Message) -> None:
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
                snapshot = self._worker_snapshot
                if not snapshot:
                    continue

                # Count new tool calls since last check
                new_tools = 0
                end = len(snapshot)
                for entry in snapshot[self._flush_snapshot_cursor:end]:
                    meta = getattr(entry, "meta", None) or getattr(entry, "metadata", None) or {}
                    if meta.get("tool_calls"):
                        calls = meta["tool_calls"]
                        new_tools += len(calls) if isinstance(calls, list) else 1
                    if meta.get("cc_tool_events"):
                        cc_count = meta.get("cc_tool_calls", 0)
                        new_tools += cc_count if cc_count else 1

                self._flush_snapshot_cursor = end
                self._flush_tools_since_last += new_tools

                if self._flush_tools_since_last >= self._flush_interval_tools:
                    # Build a synthetic EpisodeStats for this chunk
                    chunk_stats = EpisodeStats(
                        tool_calls=self._flush_tools_since_last,
                    )
                    worker_id = self._current_worker_id or "unknown"

                    # Flush: fire reflection for the accumulated chunk
                    if self._session_stats is None:
                        self._session_stats = EpisodeStats()
                        self._session_trigger_text = trigger_text
                    self._session_stats.merge(chunk_stats)
                    self._session_last_completion_time = time.monotonic()
                    self._session_last_worker_id = worker_id

                    # Build a lightweight WorkerResult from the snapshot so far
                    # (the reflection needs context to summarize)
                    snapshot_copy = list(snapshot[:end])
                    mid_result = WorkerResult(
                        response="(mid-worker checkpoint)",
                        context=snapshot_copy,
                    )
                    self._session_last_result = mid_result

                    logger.info(
                        f"RouterV2 mid-worker flush: {self._flush_tools_since_last} tools "
                        f"since last flush (worker {worker_id})"
                    )
                    self._flush_session_reflection()
                    # Reset for next interval
                    self._flush_tools_already_flushed += self._flush_tools_since_last
                    self._session_stats = None
                    self._flush_tools_since_last = 0

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Flush monitor error: {e}")

    # =========================================================================
    # Worker Watchdog — periodic check-in on worker progress
    # =========================================================================

    def _start_watchdog(self, trigger: Message) -> None:
        """Start the periodic watchdog timer."""
        if not self._config.watchdog_interval_minutes:
            logger.debug("Watchdog disabled (watchdog_interval_minutes=0)")
            return
        logger.info(
            f"Watchdog started: interval={self._config.watchdog_interval_minutes}min, "
            f"worker={self._current_worker_id or 'worker'}"
        )
        self._watchdog_task = asyncio.create_task(
            self._watchdog_loop(trigger)
        )

    def _stop_watchdog(self) -> None:
        """Cancel the watchdog timer."""
        was_running = self._watchdog_task and not self._watchdog_task.done()
        if was_running:
            self._watchdog_task.cancel()
            logger.info(f"Watchdog stopped for worker={self._current_worker_id or 'worker'}")
        self._watchdog_task = None

    async def _watchdog_loop(self, trigger: Message) -> None:
        """Periodic check-in loop. Fires _watchdog_tick every interval."""
        interval = self._config.watchdog_interval_minutes * 60
        tick_count = 0
        try:
            while True:
                await asyncio.sleep(interval)
                if self._state != RouterState.BUSY:
                    logger.info(
                        f"Watchdog loop exiting: state={self._state.name} "
                        f"(no longer BUSY after {tick_count} tick(s))"
                    )
                    break
                tick_count += 1
                logger.info(
                    f"Watchdog tick #{tick_count} firing for "
                    f"worker={self._current_worker_id or 'worker'}"
                )
                await self._watchdog_tick(trigger)
        except asyncio.CancelledError:
            logger.debug(
                f"Watchdog loop cancelled after {tick_count} tick(s)"
            )

    async def _watchdog_tick(self, trigger: Message) -> None:
        """Single watchdog evaluation. Builds prompt, calls LLM, parses result."""
        worker_id = self._current_worker_id or "worker"
        elapsed = 0.0
        if self._worker_start_time:
            elapsed = time.monotonic() - self._worker_start_time

        # Log what the LLM will see
        activity_lines = self._build_worker_activity_lines(worker_id)
        worker_progress = self._get_worker_progress()
        cc_events = self._cc_events_fn() if self._cc_events_fn and self._worker_snapshot else []
        harness_events = self._harness_events_fn(n=10) if self._harness_events_fn else []
        task_summary = self._summarize_trigger(self._pending_trigger)
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
                pending_trigger=self._pending_trigger,
                worker_start_time=self._worker_start_time,
            )
        except Exception as e:
            logger.warning(
                f"Watchdog LLM call failed for {worker_id} "
                f"(elapsed={elapsed:.0f}s): {e}"
            )
            return

        # Worker may have completed during the LLM call
        if self._state != RouterState.BUSY:
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

        Matches 'nothing to report' anywhere in the message, with optional 'to'.
        Handles: 'Nothing to report.', 'nothing report', 'Overall, nothing to report here.'
        """
        return bool(re.search(r'nothing\s+(to\s+)?report', response.strip(), re.IGNORECASE))

    def _cleanup_worker_state(self) -> None:
        """Reset worker-related state."""
        self._stop_flush_monitor()
        self._stop_watchdog()
        self._state = RouterState.IDLE
        self._worker_task = None
        self._pending_trigger = None
        self._worker_start_time = None
        self._current_worker_id = None
        self._current_task_description = ""
        self._injected_memory_ids = set()
        self._injected_memory_context = ""
        self._worker_snapshot = None
        self._worker_snapshot_start = 0
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

        if self._is_status_query(content):
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
        """Check if content is a status query."""
        content_lower = content.lower()
        return any(pattern in content_lower for pattern in self._config.status_patterns)

    def _is_cancel_request(self, content: str) -> bool:
        """Check if content is an exact-phrase request to cancel the current worker."""
        content_lower = content.lower().strip()
        return content_lower in self._config.cancel_patterns

    async def _handle_cancel_request(self, msg: Message, worker_id: str | None) -> None:
        """Cancel the current worker and notify the user.

        Acquires _state_lock internally — callers must NOT hold the lock.
        """
        logger.info(f"RouterV2 cancel requested by {msg.from_node}, cancelling worker {worker_id}")
        cancelled = await self.cancel_worker(msg)
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

        # Extract delta from snapshot (entries the worker added during execution)
        if self._worker_snapshot and self._worker_snapshot_start < len(self._worker_snapshot):
            delta = self._worker_snapshot[self._worker_snapshot_start:]
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
        self, trigger: Message | None
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

        buffered = list(getattr(agent, "_worker_buffered_messages", []) or [])
        response_text = getattr(agent, "_worker_response_text", "") or ""

        # Trace-as-history: capture partial trace from in-flight worker state
        # before cleanup zeroes it. Orphaned tool calls (call without matching
        # result) are handled correctly by _extract_trace_turns.
        if getattr(self._config, "trace_as_history_enabled", False):
            partial_history = list(getattr(agent, "_worker_in_flight_history", None) or [])
            partial_cc_events = list(getattr(agent, "_current_cc_events", None) or [])
            if partial_history or partial_cc_events:
                synth_result = WorkerResult(
                    response="",
                    context=[],
                    worker_in_flight_history=partial_history,
                    worker_cc_events=partial_cc_events,
                )
                trace_turns = self._extract_trace_turns(
                    synth_result, self._current_worker_id or "cancelled"
                )
                for t in trace_turns:
                    self._append_turn(t)
                if trace_turns:
                    self._trace_appended_on_cancel = True
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
        self, trigger: Message | None = None
    ) -> bool:
        """Cancel in-flight worker without acquiring the state lock.

        Caller MUST already hold self._state_lock.
        Returns True if a worker was cancelled, False if none was running.

        If `trigger` is provided, flushes any buffered worker output (with a
        [CANCELLED] prefix on the first message) before tearing down state.
        Flush is bounded by a 2-second timeout to keep the state lock free
        for subsequent operations even if the transport hangs.
        """
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except (asyncio.CancelledError, Exception):
                pass

            # Flush buffered output before cleanup zeroes everything.
            # Bounded by 2s — workers buffer 1-3 small DMs in normal cases;
            # if _send_and_store can't drain those in 2s, transport is sick
            # and we'd rather bail than hold the state lock indefinitely.
            try:
                flushed = await asyncio.wait_for(
                    self._flush_worker_buffer_on_cancel(trigger),
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

            self._cleanup_worker_state()
            logger.info("RouterV2 worker cancelled")
            return True

        return False

    async def cancel_worker(self, trigger: Message | None = None) -> bool:
        """
        Cancel in-flight worker (if any).

        Returns True if a worker was cancelled, False if none was running.
        If `trigger` is provided, buffered worker output is flushed to the
        trigger sender with a [CANCELLED] prefix before cleanup.
        """
        async with self._state_lock:
            return await self._cancel_worker_unlocked(trigger)

    async def reset(self) -> None:
        """
        Reset router state (used by reset_context).

        Clears context and cancels any in-flight work.
        """
        async with self._state_lock:
            await self._cancel_worker_unlocked()
            self._context = []
            logger.info("RouterV2 reset complete")
