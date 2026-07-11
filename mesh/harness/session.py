"""
Persistent harness *session* loop.

The standard harness (`run_loop` / `run_native_loop` in loop.py) runs a task to
completion and exits. A *session* keeps the same TAOR core alive between turns:
the subprocess stays running, reads JSONL commands from stdin, and surfaces
structured `session.*` events on stdout. A router-side manager
(mesh/harness_session_manager.py) drives it — sending tasks, steering it at
iteration boundaries, and reacting to lifecycle events — exactly analogous to
the CC interactive session path, but without scraping a terminal.

This module deliberately does NOT modify loop.py. It reuses loop.py's proven
building blocks (`_execute_tool_calls`, `estimate_native_tokens`, retry/backoff,
the forced-synthesis threshold) and runs the native OpenAI-messages path, which
is what the local-model backends (qwen36) already use in worker mode.

stdin command protocol (one JSON object per line):
    {"type": "task",     "content": "..."}   new work item → resumes the loop
    {"type": "steer",    "content": "..."}   injected at the next iteration
                                             boundary as a [DRIVER STEERING] msg
    {"type": "continue", "content": "..."}   resume after a checkpoint yield
                                             (optional content = steering nudge)
    {"type": "reset",    "content": "..."}   clear history, seed with content
                                             (used after context_exhausted)
    {"type": "status"}                        emit a session.status snapshot
    {"type": "abort"}                         drain, finish, exit

stdout events: see protocol.py (`session.*`, plus the standard tool_call /
tool_result / usage / error / thread.finished events from the shared loop).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from typing import Any

from ..llm import LLMClient
from ..tools import ToolRegistry, ToolCall
from . import protocol
from .loop import (
    FORCED_SYNTHESIS_FRACTION,
    KEEP_RECENT_RESULTS,
    LLM_RETRIES,
    _backoff_for_attempt,
    _execute_tool_calls,
    _is_transient,
    _truncate_extreme_result,
    estimate_native_tokens,
    PhaseCompleteSignal,
)

logger = logging.getLogger(__name__)


# ── Anthropic format helpers ──────────────────────────────────────────────
# The session maintains messages in OpenAI format internally (role-based with
# tool_call_id). These helpers convert to/from Anthropic Messages API format
# at the LLM call boundary so thinking blocks, tool_use, and tool_result are
# preserved correctly.

def _openai_to_anthropic_messages(
    messages: list[dict],
    thinking_blocks_by_idx: dict[int, list[dict]],
) -> list[dict]:
    """Convert OpenAI-format message list to Anthropic Messages format.

    - system messages are stripped (handled separately as top-level param)
    - assistant messages with tool_calls become content block lists (tool_use)
    - tool messages become user messages with tool_result content blocks
    - thinking/redacted_thinking blocks are injected from the parallel map
    - consecutive same-role messages are merged (Anthropic requires alternation)
    """
    result: list[dict] = []

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role == "system":
            continue

        if role == "assistant":
            content_blocks: list[dict] = []
            # Inject preserved thinking blocks from this turn
            for tb in thinking_blocks_by_idx.get(idx, []):
                content_blocks.append(tb)
            # Text content
            text = msg.get("content") or ""
            if text:
                content_blocks.append({"type": "text", "text": text})
            # Tool use blocks
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "input": args,
                })
            if not content_blocks:
                content_blocks.append({"type": "text", "text": ""})
            result.append({"role": "assistant", "content": content_blocks})

        elif role == "tool":
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            # Anthropic requires tool_result inside a user message. Merge
            # with the preceding user message if it already has tool_results,
            # otherwise create a new user message.
            if result and result[-1].get("role") == "user":
                prev_content = result[-1].get("content")
                if isinstance(prev_content, list):
                    prev_content.append(tool_result_block)
                else:
                    result[-1]["content"] = [tool_result_block]
            else:
                result.append({"role": "user", "content": [tool_result_block]})

        elif role == "user":
            text = msg.get("content", "")
            # Merge consecutive user messages (Anthropic requires alternation)
            if result and result[-1].get("role") == "user":
                prev_content = result[-1].get("content")
                if isinstance(prev_content, str):
                    result[-1]["content"] = prev_content + "\n\n" + text
                elif isinstance(prev_content, list):
                    prev_content.append({"type": "text", "text": text})
                else:
                    result[-1]["content"] = text
            else:
                result.append({"role": "user", "content": text})

    return result


def _openai_tools_to_anthropic(openai_tools: list[dict]) -> list[dict]:
    """Convert OpenAI function-calling tool defs to Anthropic tool format."""
    result = []
    for tool in openai_tools:
        func = tool.get("function", {})
        result.append({
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
        })
    return result


# Loop-state labels surfaced in session.status and checkpoint digests.
STATE_STARTING = "starting"
STATE_ACTIVE = "active"
STATE_WARM_IDLE = "warm_idle"
STATE_STOPPING = "stopping"

# In-flight pruning: once history exceeds this fraction of the soft limit (but
# before forced synthesis at 0.97), drop the oldest tool-result bodies, keeping
# the most recent KEEP_RECENT_RESULTS. Keeps a long-running session alive
# without the opaque compaction loops the CC path suffered.
PRUNE_THRESHOLD_FRACTION = 0.85


async def _make_stdin_reader() -> asyncio.StreamReader:
    """Wrap sys.stdin in an asyncio StreamReader for non-blocking line reads."""
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol_factory = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol_factory, sys.stdin)
    return reader


class _CommandChannel:
    """Background stdin reader. Parses one JSON command per line onto a queue.

    A blank line or EOF is ignored/closes; malformed JSON is logged and skipped
    (a dropped command must never crash the session)."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[dict] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._closed = False

    async def start(self) -> None:
        reader = await _make_stdin_reader()
        self._task = asyncio.create_task(self._read_loop(reader))

    async def _read_loop(self, reader: asyncio.StreamReader) -> None:
        while True:
            try:
                line = await reader.readline()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("session stdin read error: %s", e)
                break
            if not line:
                self._closed = True
                break
            text = line.decode(errors="replace").strip()
            if not text:
                continue
            try:
                cmd = json.loads(text)
            except json.JSONDecodeError:
                logger.warning("session: dropping malformed command line: %s", text[:200])
                continue
            if isinstance(cmd, dict) and cmd.get("type"):
                await self.queue.put(cmd)
            else:
                logger.warning("session: command missing 'type', dropped: %s", text[:200])

    def drain_nowait(self) -> list[dict]:
        """Return all queued commands without blocking."""
        cmds: list[dict] = []
        while True:
            try:
                cmds.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return cmds

    async def wait(self) -> dict:
        """Block for the next command."""
        return await self.queue.get()


def _prune_native_history(messages: list[dict], soft_limit: int) -> int:
    """Drop the oldest tool-result bodies once over the prune threshold.

    Keeps the system message, all user/assistant messages, and the most recent
    KEEP_RECENT_RESULTS tool messages intact. Older tool bodies are replaced
    with a short omission marker so call_id pairing and conversation structure
    stay valid. Returns the number of tool results pruned.
    """
    prune_threshold = int(soft_limit * PRUNE_THRESHOLD_FRACTION)
    if estimate_native_tokens(messages) < prune_threshold:
        return 0

    tool_idxs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    # Leave the most recent KEEP_RECENT_RESULTS untouched.
    prunable = tool_idxs[:-KEEP_RECENT_RESULTS] if len(tool_idxs) > KEEP_RECENT_RESULTS else []
    pruned = 0
    for i in prunable:
        body = messages[i].get("content") or ""
        if body.startswith("[pruned"):
            continue
        messages[i]["content"] = f"[pruned: {len(body)} chars of older tool output dropped to free context]"
        pruned += 1
    if pruned:
        before = estimate_native_tokens(messages)
        logger.info("session: pruned %d old tool results (~%dK tokens now)", pruned, before // 1000)
        protocol.emit_context_pruned(pruned, before, estimate_native_tokens(messages))
    return pruned


async def _summarize_progress(
    llm_client: LLMClient,
    messages: list[dict],
    max_chars: int = 4000,
) -> str:
    """Best-effort self-summary of work so far, for context_exhausted/reset.

    A toolless single call. Falls back to a mechanical digest of the recent
    transcript if the LLM call fails — the summary must never crash the loop.
    """
    digest_parts: list[str] = []
    for m in messages[-12:]:
        role = m.get("role")
        if role == "tool":
            digest_parts.append(f"[tool result]: {str(m.get('content', ''))[:300]}")
        elif role == "assistant":
            c = m.get("content") or ""
            tcs = m.get("tool_calls") or []
            if tcs:
                names = ", ".join(tc.get("function", {}).get("name", "?") for tc in tcs)
                digest_parts.append(f"[assistant → {names}]: {c[:300]}")
            elif c.strip():
                digest_parts.append(f"[assistant]: {c[:300]}")
        elif role == "user":
            digest_parts.append(f"[user]: {str(m.get('content', ''))[:300]}")
    mechanical = "\n".join(digest_parts)[:max_chars]

    summary_messages = [
        {"role": "system", "content": "You are summarizing a coding session that ran out of context budget."},
        {"role": "user", "content": (
            "Summarize the work done so far so it can seed a fresh session: the "
            "task, what was accomplished, files changed, and what remains. Be "
            "concise (a few hundred words max).\n\n--- TRANSCRIPT TAIL ---\n" + mechanical
        )},
    ]
    try:
        content, _tc, _usage = await llm_client.complete_multi_turn(messages=summary_messages, tools=None)
        text = (content or "").strip()
        return text[:max_chars] if text else mechanical
    except Exception as e:
        logger.warning("session: progress summary failed (%s) — using mechanical digest", e)
        return mechanical or "[no transcript available]"


async def run_session_loop(
    llm_client: LLMClient,
    system_prompt: str,
    tool_registry: ToolRegistry,
    tool_names: list[str] | None,
    initial_task: str = "",
    *,
    node_id: str = "harness-session",
    max_iterations: int = 100,
    soft_limit: int = 200_000,
    agent_socket_path: str | None = None,
    checkpoint_interval: int = 0,
) -> None:
    """Run a persistent session loop over native OpenAI messages.

    Blocks on stdin between tasks. Returns only when an `abort` command arrives
    or stdin closes. All communication is via protocol.* events on stdout.
    """
    session_id = uuid.uuid4().hex[:12]
    backend = llm_client.config.backend
    model = llm_client.config.model

    channel = _CommandChannel()
    await channel.start()

    protocol.emit_session_started(session_id, backend, model, checkpoint_interval)
    protocol.emit_thread_started(session_id, backend, model)

    is_anthropic = backend == "anthropic"
    openai_tools = tool_registry.get_openai_tools(tool_names) if tool_names else None
    anthropic_tools = (
        _openai_tools_to_anthropic(openai_tools) if is_anthropic and openai_tools else None
    )
    synthesis_threshold = int(soft_limit * FORCED_SYNTHESIS_FRACTION)

    sys_content = system_prompt
    tool_prompt = ""
    try:
        tool_prompt = tool_registry.format_tools_prompt(tool_names, backend="openai") if tool_names else ""
    except Exception:
        tool_prompt = ""
    if tool_prompt:
        sys_content = f"{sys_content}\n\n{tool_prompt}" if sys_content else tool_prompt

    messages: list[dict] = [{"role": "system", "content": sys_content}]
    if initial_task:
        messages.append({"role": "user", "content": initial_task})

    # Anthropic thinking blocks indexed by the messages-list position of the
    # assistant message they belong to. Preserved across turns so the API
    # receives them back (required for thinking continuity).
    _thinking_by_msg_idx: dict[int, list[dict]] = {}

    cumulative_usage: dict[str, Any] = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_creation_tokens": 0, "cache_read_tokens": 0,
        "reasoning_tokens": 0, "total_tokens": 0,
        "llm_calls": 0, "backend": backend, "model": model,
    }

    recent_tools: list[str] = []   # ring buffer of recent tool names
    files_touched: list[str] = []
    iteration = 0
    loop_state = STATE_STARTING
    aborted = False
    response_content = ""

    def _status_digest() -> dict[str, Any]:
        return {
            "session_id": session_id,
            "iteration": iteration,
            "loop_state": loop_state,
            "recent_tools": recent_tools[-8:],
            "files_touched": files_touched[-12:],
            "tokens": {
                "input": cumulative_usage["input_tokens"],
                "output": cumulative_usage["output_tokens"],
                "estimated_context": estimate_native_tokens(messages),
                "soft_limit": soft_limit,
            },
            "llm_calls": cumulative_usage["llm_calls"],
        }

    def _note_files(tc: ToolCall) -> None:
        for key in ("file_path", "path", "filename"):
            v = tc.arguments.get(key) if isinstance(tc.arguments, dict) else None
            if isinstance(v, str) and v and v not in files_touched:
                files_touched.append(v)

    async def _handle_idle_commands() -> str:
        """Block until a command resumes work or aborts. Returns one of:
        'resume' (new task/steer/reset appended), 'abort'."""
        nonlocal aborted, messages
        while True:
            cmd = await channel.wait()
            ctype = cmd.get("type")
            content = cmd.get("content", "") or ""
            if ctype == "abort":
                aborted = True
                return "abort"
            if ctype == "status":
                protocol.emit_session_status(
                    iteration, loop_state, recent_tools[-8:], files_touched[-12:], cumulative_usage,
                )
                continue
            if ctype == "task":
                messages.append({"role": "user", "content": content})
                return "resume"
            if ctype in ("steer", "continue"):
                if content:
                    messages.append({"role": "user", "content": f"[DRIVER STEERING] {content}"})
                    return "resume"
                # bare continue/steer with no content while idle: ignore, keep waiting
                continue
            if ctype == "reset":
                seed = content or "[session reset]"
                messages = [messages[0], {"role": "user", "content": seed}]
                _thinking_by_msg_idx.clear()
                protocol.emit_session_reset_ack(iteration, estimate_native_tokens(messages))
                return "resume"
            logger.warning("session: unknown idle command type %r", ctype)

    def _apply_inflight_commands() -> str:
        """Drain queued commands at an iteration boundary. Returns 'abort',
        'reset', or '' (continue)."""
        nonlocal aborted, messages
        for cmd in channel.drain_nowait():
            ctype = cmd.get("type")
            content = cmd.get("content", "") or ""
            if ctype == "abort":
                aborted = True
                return "abort"
            elif ctype == "steer":
                if content:
                    messages.append({"role": "user", "content": f"[DRIVER STEERING] {content}"})
                logger.info("session: steering injected at iteration %d", iteration)
            elif ctype == "task":
                # New work mid-run: append as additional user input.
                messages.append({"role": "user", "content": content})
            elif ctype == "status":
                protocol.emit_session_status(
                    iteration, loop_state, recent_tools[-8:], files_touched[-12:], cumulative_usage,
                )
            elif ctype == "reset":
                seed = content or "[session reset]"
                messages = [messages[0], {"role": "user", "content": seed}]
                _thinking_by_msg_idx.clear()
                protocol.emit_session_reset_ack(iteration, estimate_native_tokens(messages))
                return "reset"
            elif ctype == "continue":
                if content:
                    messages.append({"role": "user", "content": f"[DRIVER STEERING] {content}"})
            else:
                logger.warning("session: unknown in-flight command type %r", ctype)
        return ""

    async def _await_checkpoint_decision() -> str:
        """Block for the driver's verdict after a checkpoint yield. Returns
        'continue' or 'abort'."""
        nonlocal aborted, messages
        while True:
            cmd = await channel.wait()
            ctype = cmd.get("type")
            content = cmd.get("content", "") or ""
            if ctype == "abort":
                aborted = True
                return "abort"
            if ctype == "status":
                protocol.emit_session_status(
                    iteration, loop_state, recent_tools[-8:], files_touched[-12:], cumulative_usage,
                )
                continue
            if ctype in ("continue", "steer", "task"):
                if content:
                    tag = "[DRIVER STEERING] " if ctype != "task" else ""
                    messages.append({"role": "user", "content": f"{tag}{content}"})
                return "continue"
            if ctype == "reset":
                seed = content or "[session reset]"
                messages = [messages[0], {"role": "user", "content": seed}]
                _thinking_by_msg_idx.clear()
                protocol.emit_session_reset_ack(iteration, estimate_native_tokens(messages))
                return "continue"
            logger.warning("session: unknown checkpoint command type %r", ctype)

    # ---- Main session lifecycle ----
    try:
        # No task on argv: wait for the driver's first task/steer before working.
        if not initial_task:
            loop_state = STATE_WARM_IDLE
            protocol.emit_session_awaiting_input("", iteration, cumulative_usage)
            if await _handle_idle_commands() == "abort":
                aborted = True

        while not aborted:
            loop_state = STATE_ACTIVE
            # Inner TAOR run on the current message stack until it yields.
            while iteration < max_iterations and not aborted:
                iteration += 1

                decision = _apply_inflight_commands()
                if decision == "abort":
                    break
                # 'reset' just continues with the reseeded stack.

                # --- Budget management ---
                _prune_native_history(messages, soft_limit)
                estimated_tokens = estimate_native_tokens(messages)
                if estimated_tokens >= synthesis_threshold:
                    loop_state = STATE_WARM_IDLE
                    summary = await _summarize_progress(llm_client, messages)
                    protocol.emit_session_context_exhausted(
                        iteration, summary, estimated_tokens, soft_limit,
                    )
                    verdict = await _handle_idle_commands()
                    if verdict == "abort":
                        break
                    loop_state = STATE_ACTIVE
                    continue

                protocol.emit_turn_started(iteration)

                # --- LLM call with retry ---
                response_content = ""
                tool_calls: list[ToolCall] = []
                llm_ok = False
                for attempt in range(1 + LLM_RETRIES):
                    try:
                        if is_anthropic:
                            anth_msgs = _openai_to_anthropic_messages(
                                messages, _thinking_by_msg_idx,
                            )
                            content, tc_list, _usage = await llm_client.complete_multi_turn_anthropic(
                                messages=anth_msgs, tools=anthropic_tools,
                            )
                        else:
                            content, tc_list, _usage = await llm_client.complete_multi_turn(
                                messages=messages, tools=openai_tools,
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
                            logger.exception("session LLM call failed at iteration %d", iteration)
                            break
                if not llm_ok:
                    # Treat a hard LLM failure as an idle yield so the driver can
                    # steer or abort rather than the session dying silently.
                    loop_state = STATE_WARM_IDLE
                    protocol.emit_session_awaiting_input(
                        "[LLM call failed — awaiting driver input]", iteration, cumulative_usage,
                    )
                    verdict = await _handle_idle_commands()
                    if verdict == "abort":
                        break
                    loop_state = STATE_ACTIVE
                    continue

                # Accumulate usage
                if llm_client._last_usage:
                    u = llm_client._last_usage
                    for key in ("input_tokens", "output_tokens", "cache_creation_tokens",
                                "cache_read_tokens", "reasoning_tokens", "total_tokens"):
                        cumulative_usage[key] += u.get(key, 0)
                    cumulative_usage["llm_calls"] += 1

                protocol.emit_assistant_message(response_content[:500] if response_content else "", iteration)

                # --- Yield to WARM_IDLE on a final (toolless) answer ---
                if not tool_calls:
                    final_text = (response_content or "").strip()
                    if final_text:
                        messages.append({"role": "assistant", "content": final_text})
                        if is_anthropic and getattr(llm_client, "_last_thinking_blocks", None):
                            _thinking_by_msg_idx[len(messages) - 1] = list(llm_client._last_thinking_blocks)
                    loop_state = STATE_WARM_IDLE
                    protocol.emit_usage(cumulative_usage)
                    protocol.emit_session_awaiting_input(final_text, iteration, cumulative_usage)
                    verdict = await _handle_idle_commands()
                    if verdict == "abort":
                        break
                    loop_state = STATE_ACTIVE
                    continue

                # --- Execute tool calls ---
                assistant_msg: dict[str, Any] = {"role": "assistant"}
                if response_content:
                    assistant_msg["content"] = response_content
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.call_id or uuid.uuid4().hex[:12],
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in tool_calls
                ]
                messages.append(assistant_msg)
                if is_anthropic and getattr(llm_client, "_last_thinking_blocks", None):
                    _thinking_by_msg_idx[len(messages) - 1] = list(llm_client._last_thinking_blocks)

                for i, tc in enumerate(tool_calls):
                    recent_tools.append(tc.name)
                    _note_files(tc)
                    call_id = tc.call_id or assistant_msg["tool_calls"][i]["id"]
                    protocol.emit_tool_call(
                        tc.name,
                        tc.arguments if isinstance(tc.arguments, dict) else {},
                        call_id,
                    )

                max_result_chars = int(soft_limit * 0.4)
                try:
                    results = await _execute_tool_calls(
                        tool_calls, tool_registry, agent_socket_path, max_result_chars,
                        tool_names,
                    )
                except PhaseCompleteSignal as sig:
                    loop_state = STATE_WARM_IDLE
                    protocol.emit_usage(cumulative_usage)
                    protocol.emit_session_awaiting_input(sig.summary, iteration, cumulative_usage)
                    verdict = await _handle_idle_commands()
                    if verdict == "abort":
                        break
                    loop_state = STATE_ACTIVE
                    continue

                for i, (name, result_str, _ok) in enumerate(results):
                    tc = tool_calls[i] if i < len(tool_calls) else None
                    call_id = (tc.call_id if tc else None) or assistant_msg["tool_calls"][i]["id"]
                    protocol.emit_tool_result(name, result_str[:500], call_id, _ok)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _truncate_extreme_result(result_str, soft_limit),
                    })

                # --- Checkpoint yield (driver-paced cadence) ---
                if checkpoint_interval and iteration % checkpoint_interval == 0:
                    loop_state = STATE_WARM_IDLE
                    protocol.emit_session_checkpoint(iteration, _status_digest())
                    verdict = await _await_checkpoint_decision()
                    if verdict == "abort":
                        break
                    loop_state = STATE_ACTIVE

            else:
                # Inner loop hit max_iterations without an explicit yield/abort.
                if iteration >= max_iterations and not aborted:
                    loop_state = STATE_WARM_IDLE
                    protocol.emit_error(
                        f"Hit max iterations ({max_iterations}) — yielding to driver",
                        iteration,
                    )
                    protocol.emit_usage(cumulative_usage)
                    protocol.emit_session_awaiting_input(
                        (response_content or "").strip()
                        or "[Hit max iterations without a final answer]",
                        iteration, cumulative_usage,
                    )
                    verdict = await _handle_idle_commands()
                    if verdict == "abort":
                        break
                    # Driver resumed: grant a fresh iteration budget.
                    max_iterations += max_iterations
                    loop_state = STATE_ACTIVE
                    continue

            if aborted:
                break

        loop_state = STATE_STOPPING
    finally:
        protocol.emit_usage(cumulative_usage)
        status = "aborted" if aborted else "finished"
        protocol.emit_thread_finished(
            session_id, iteration,
            (response_content or "").strip(),
            {**cumulative_usage, "status": status},
        )
        if channel._task and not channel._task.done():
            channel._task.cancel()
