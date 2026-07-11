"""
Observer/controller router for CC-session agents.

Replaces RouterV2 for agents with context_mode: cc-session.

IDLE mode:  Forward messages to CCSession.send_turn(), observe the stream
            for rolling-window context and AgentState updates. Zero LLM calls.
BUSY mode:  Answer status queries and handle cancellation using its own LLM
            (grounded in live CC turns from the stream). Respond or cancel —
            never queue.
Catch-up:   Track BUSY-period conversation. When transitioning BUSY→IDLE,
            prepend verbatim transcript to the next message via CC invocation.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
from pathlib import Path
import time
from collections.abc import AsyncIterator
from xml.sax.saxutils import escape as xml_escape
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Literal, Protocol, TYPE_CHECKING

from .cc_session import CCSession, CCStreamEvent
from .llm import _build_subprocess_env
from .conversation_history import ConversationHistory, Turn
from .protocol import Message, make_message, make_tool_activity

if TYPE_CHECKING:
    from .llm import LLMClient
    from .memory.system_v2 import MemorySystemV2
    from .router_v2 import RouterV2Config

logger = logging.getLogger(__name__)


class CCRouterState(Enum):
    IDLE = "idle"
    BUSY = "busy"


@dataclass
class AgentState:
    """Observable state of the CC-session agent."""
    state: CCRouterState = CCRouterState.IDLE
    current_task: str = ""
    turn_started_at: float = 0.0
    recent_tools: list[str] = field(default_factory=list)
    last_text: str = ""
    text_parts: list[str] = field(default_factory=list)


class BusyPolicy(Protocol):
    """Pluggable policy for BUSY-mode message handling."""
    def classify(self, state: AgentState, msg: Message) -> Literal["status", "cancel", "queue"]:
        ...


class HeuristicPolicy:
    """Default BUSY policy: regex-based intent classification."""
    _CANCEL_RE = re.compile(r"^(stop|cancel|abort|kill|halt)\b", re.IGNORECASE)
    _STATUS_RE = re.compile(
        r"(status|how.s it going|what.s happening|progress|update|how far|eta|busy)",
        re.IGNORECASE,
    )

    def classify(self, state: AgentState, msg: Message) -> Literal["status", "cancel", "queue"]:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        content = content.strip()

        if self._CANCEL_RE.search(content):
            return "cancel"
        if self._STATUS_RE.search(content):
            return "status"
        # Short messages from users default to status inquiry
        if msg.from_node and msg.from_node.startswith("user:") and len(content) < 120:
            return "status"
        return "queue"


@dataclass
class CatchupEntry:
    """A message received while BUSY, to be forwarded on next IDLE transition."""
    from_node: str
    content: str
    timestamp: str


class RouterCC:
    """
    Observer/controller router for CC-session agents.

    Replaces RouterV2 for agents with context_mode: cc-session.
    """

    def __init__(
        self,
        cc_session: CCSession,
        send_fn: Callable[[str, Message | None], Awaitable[None]],
        config: 'RouterV2Config',
        node_id: str,
        nickname: str,
        agent_type: str,
        llm_client: 'LLMClient | None' = None,
        system_prompt: str = "",
        identity_block: str = "",
        memory_system: 'MemorySystemV2 | None' = None,
        busy_policy: BusyPolicy | None = None,
        raw_send_fn: Callable[[Message], Awaitable[None]] | None = None,
    ):
        self._cc = cc_session
        self._send_fn = send_fn
        self._config = config
        self._node_id = node_id
        self._nickname = nickname
        self._agent_type = agent_type
        self._llm_client = llm_client
        self._system_prompt = system_prompt
        self._identity_block = identity_block
        self._memory = memory_system
        self._policy = busy_policy or HeuristicPolicy()
        self._raw_send_fn = raw_send_fn

        # State
        self._state = CCRouterState.IDLE
        self._state_lock = asyncio.Lock()
        self._agent_state = AgentState()

        # Conversation history (for persistence / context)
        persist_path = None
        if self._config.history_persist:
            if self._config.history_persist_path:
                persist_path = Path(self._config.history_persist_path)
            else:
                from .paths import HISTORY_DIR
                persist_path = HISTORY_DIR / f"{nickname}.json"
        self._history = ConversationHistory(
            soft_token_limit=self._config.history_soft_limit_tokens,
            hard_token_limit=self._config.history_hard_limit_tokens,
            window_budget=self._config.history_window_tokens or None,
            persist_path=persist_path,
        )

        # Passive channel messages (received via add_to_history_only)
        self._passive_messages: list[CatchupEntry] = []

        # Busy-mode tracking
        self._catchup: list[CatchupEntry] = []
        self._current_trigger: Message | None = None
        self._cc_turn_task: asyncio.Task | None = None

        # Live activity stream (for status peeking)
        self._live_events: list[dict[str, Any]] = []
        self._max_live_events = 50

        # Worker start time (monotonic, for status elapsed calculation)
        self._worker_start_mono: float | None = None

        self._cc_sent_message = False  # Set True when CC calls send_message

        # Cancel coordination flag — tells _run_cc_turn's finally block
        # to skip cleanup (the cancel handler does it instead)
        self._cancel_in_progress = False

        # Status response guard
        self._status_response_pending = False

    # ── Public properties (compatibility with agent_node status) ──

    @property
    def state(self) -> CCRouterState:
        """Current router state."""
        return self._state

    @property
    def history(self) -> ConversationHistory:
        """Router conversation history."""
        return self._history

    @property
    def _worker_start_time(self) -> float | None:
        """Worker start time (monotonic) for elapsed calculation (compat with RouterV2)."""
        return self._worker_start_mono

    @property
    def _worker_task(self) -> asyncio.Task | None:
        """Worker task for .done() check (compat with RouterV2)."""
        return self._cc_turn_task

    # Token tracking — CC manages its own prompt, so these are 0
    _last_prompt_tokens: int = 0
    _static_prompt_tokens: int = 0

    def get_diagnostics(self) -> dict:
        """Return structured diagnostic data for status reporting."""
        result: dict[str, Any] = {
            "state": self._state.value,
            "worker_active": self._cc_turn_task is not None and not self._cc_turn_task.done(),
            "worker_elapsed_seconds": (
                round(time.time() - self._agent_state.turn_started_at, 1)
                if self._agent_state.turn_started_at and self._state == CCRouterState.BUSY
                else None
            ),
            "pending_trigger_from": (
                self._current_trigger.from_node if self._current_trigger else None
            ),
            "pending_trigger_preview": (
                str(self._current_trigger.content)[:100]
                if self._current_trigger else None
            ),
            "recent_tools": self._agent_state.recent_tools[-5:],
            "live_events_count": len(self._live_events),
            "passive_messages_queued": len(self._passive_messages),
            "catchup_queued": len(self._catchup),
            "cc_sent_message": self._cc_sent_message,
        }
        return result

    def set_context(self, context: list[Any]) -> None:
        """Seed the router from legacy history (list of HistoryEntry).

        Converts HistoryEntry objects into Turn objects and appends to
        the ConversationHistory. Used when loading from persisted worker history.
        """
        for entry in context:
            if hasattr(entry, 'message'):
                msg = entry.message
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                ts = msg.timestamp or datetime.now(timezone.utc).isoformat()
                self._history.append(Turn(
                    role=entry.direction,
                    content=content,
                    timestamp=ts,
                    from_node=msg.from_node or "",
                    to_node=msg.to_node,
                ))
            elif isinstance(entry, Message):
                content = entry.content if isinstance(entry.content, str) else str(entry.content)
                ts = entry.timestamp or datetime.now(timezone.utc).isoformat()
                self._history.append(Turn(
                    role="incoming",
                    content=content,
                    timestamp=ts,
                    from_node=entry.from_node or "",
                    to_node=entry.to_node,
                ))

    # ── Public interface (mirrors RouterV2) ──────────────────────

    async def on_message(self, msg: Message) -> None:
        """Handle an incoming message."""
        content_raw = msg.content if isinstance(msg.content, str) else str(msg.content)

        async with self._state_lock:
            # Always record in history
            self._append_to_history(msg)

            if self._state == CCRouterState.IDLE:
                await self._handle_idle(msg)
            else:
                # BUSY mode
                intent = self._policy.classify(self._agent_state, msg)
                if intent == "cancel":
                    await self._handle_cancel(msg)
                elif intent == "status":
                    # Spawn as background task to avoid holding the lock
                    if not self._status_response_pending:
                        self._status_response_pending = True
                        asyncio.create_task(self._respond_status(msg))
                else:
                    # Queue for catchup
                    self._catchup.append(CatchupEntry(
                        from_node=msg.from_node,
                        content=content_raw,
                        timestamp=msg.timestamp,
                    ))

    async def add_to_history_only(self, msg: Message) -> None:
        """Add a message to history without triggering dispatch.

        Also accumulates the message for catchup injection on the next
        CC turn, so CC-session agents see passive channel context.
        """
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        async with self._state_lock:
            self._append_to_history(msg)
            self._passive_messages.append(CatchupEntry(
                from_node=msg.from_node,
                content=content,
                timestamp=msg.timestamp or datetime.now(timezone.utc).isoformat(),
            ))
        try:
            self._history.save()
        except Exception as e:
            logger.warning(f"Failed to save history after passive add: {e}")

    def load_history(self) -> int:
        """Load persisted router history from disk. Returns count of entries loaded."""
        return self._history.load()

    def save_history(self) -> None:
        """Persist router history to disk."""
        self._history.save()

    def _flush_session_reflection(self) -> None:
        """No-op for RouterCC — session reflection is handled by CC itself."""
        pass

    def get_current_activity(self) -> list[dict[str, Any]]:
        """Return recent live events for status peeking."""
        return list(self._live_events)

    async def cancel_worker(self) -> bool:
        """Cancel the current CC turn.

        Safe to call from outside the state lock.  Sets _cancel_in_progress
        so the task's finally block yields cleanup to us.
        """
        if self._cc_turn_task and not self._cc_turn_task.done():
            self._cancel_in_progress = True
            # Kill CC subprocess first — unblocks pipe reads in _parse_stream
            await self._cc.stop()
            # Cancel and await the task (finally block sees _cancel_in_progress
            # and skips its own state transition / fallback send)
            self._cc_turn_task.cancel()
            try:
                await self._cc_turn_task
            except (asyncio.CancelledError, Exception):
                pass
            async with self._state_lock:
                self._transition_to_idle()
            self._cancel_in_progress = False
            return True
        return False

    async def reset(self) -> None:
        """Reset router state."""
        await self.cancel_worker()
        async with self._state_lock:
            self._history.window.clear()
            self._history.summary = None
            self._catchup.clear()
            self._passive_messages.clear()
            self._live_events.clear()

    # ── IDLE mode ────────────────────────────────────────────────

    async def _handle_idle(self, msg: Message) -> None:
        """Handle a message in IDLE state. Transitions to BUSY, starts CC turn."""
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        self._state = CCRouterState.BUSY
        self._agent_state = AgentState(
            state=CCRouterState.BUSY,
            current_task=content[:200],
            turn_started_at=time.time(),
        )
        self._current_trigger = msg
        self._cc_sent_message = False
        self._worker_start_mono = time.monotonic()
        self._live_events.clear()

        # Build catchup XML from BUSY-period messages + passive channel messages
        catchup_xml = self._build_catchup_xml()
        self._catchup.clear()
        self._passive_messages.clear()

        # Launch CC turn as background task
        self._cc_turn_task = asyncio.create_task(
            self._run_cc_turn(msg, content, catchup_xml)
        )

    async def _run_cc_turn(
        self,
        trigger: Message,
        message_text: str,
        catchup_xml: str,
    ) -> None:
        """Run a CC turn and observe the stream. Called as a background task."""
        # Ack watchdog disabled — the CC process handles its own
        # acknowledgments via send_message (see COMMUNICATION_CADENCE_BLOCK).

        try:
            async for event in self._cc.send_turn(message_text, catchup_xml):
                await self._observe_event(event, trigger)

        except asyncio.CancelledError:
            logger.info(f"[{self._nickname}] CC turn cancelled")
            raise
        except Exception as e:
            logger.error(f"[{self._nickname}] CC turn error: {e}", exc_info=True)
            try:
                await self._send_fn(
                    f"Sorry, I hit an error: {e}", trigger
                )
            except Exception:
                pass
        finally:
            # If the cancel handler is managing cleanup, bail out —
            # _handle_cancel / cancel_worker already transitions to IDLE
            # and holds the state lock, so we must not re-acquire it.
            if self._cancel_in_progress:
                return

            # Fallback: if CC completed without calling send_message,
            # forward ALL accumulated text output to the channel.
            if not self._cc_sent_message and self._agent_state.text_parts:
                fallback_text = "\n\n".join(
                    part.strip() for part in self._agent_state.text_parts
                    if part.strip()
                )
                if fallback_text:
                    logger.warning(
                        f"[{self._nickname}] CC completed without send_message — "
                        f"forwarding {len(self._agent_state.text_parts)} text chunks "
                        f"({len(fallback_text)} chars)"
                    )
                    try:
                        await self._send_fn(fallback_text, trigger)
                    except Exception as e:
                        logger.error(f"[{self._nickname}] Fallback send failed: {e}")

            # Record agent's full output in history
            full_text = "\n\n".join(
                part.strip() for part in self._agent_state.text_parts
                if part.strip()
            ) if self._agent_state.text_parts else ""
            if full_text:
                self._history.append(Turn(
                    role="assistant",
                    content=full_text[:4000],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    from_node=self._node_id,
                ))

            # Transition back to IDLE
            async with self._state_lock:
                self._transition_to_idle()

            # Persist history
            try:
                self.save_history()
            except Exception as e:
                logger.warning(f"Failed to save history after turn: {e}")

    def _transition_to_idle(self) -> None:
        """Transition from BUSY to IDLE. Must be called under _state_lock."""
        self._state = CCRouterState.IDLE
        self._agent_state.state = CCRouterState.IDLE
        self._current_trigger = None
        self._cc_turn_task = None
        self._worker_start_mono = None
        self._status_response_pending = False

    # ── Context for lightweight subprocesses ─────────────────────

    def _build_context_preamble(self, max_turns: int = 10) -> str:
        """Build a compact context block for lightweight sonnet subprocesses.

        Includes identity, personality, and recent conversation turns so
        ack and status responses sound like the agent, not a stranger.
        """
        parts: list[str] = []

        # Identity
        if self._identity_block:
            parts.append(self._identity_block.strip())

        # Personality
        if self._memory:
            try:
                personality = self._memory.get_personality()
                if personality:
                    parts.append(f"<personality>\n{personality}\n</personality>")
            except Exception:
                pass

        # Recent conversation (last N turns from history window)
        window = self._history.window[-max_turns:] if self._history.window else []
        if window:
            conv_lines = ["<recent_conversation>"]
            for turn in window:
                sender = turn.from_node or turn.role
                text = turn.content[:300] if turn.content else ""
                conv_lines.append(f"[{sender}]: {text}")
            conv_lines.append("</recent_conversation>")
            parts.append("\n".join(conv_lines))

        return "\n\n".join(parts)

    # ── BUSY mode handlers ───────────────────────────────────────

    async def _handle_cancel(self, msg: Message) -> None:
        """Handle a cancel request. Must be called under _state_lock.

        Sets _cancel_in_progress so _run_cc_turn's finally block skips its
        own cleanup (state transition, fallback text send) — we handle that
        here instead.  This avoids the double-transition and the deadlock
        that occurs when the finally block tries to re-acquire _state_lock.
        """
        logger.info(f"[{self._nickname}] Cancel requested by {msg.from_node}")

        # Signal to _run_cc_turn's finally block to yield cleanup to us
        self._cancel_in_progress = True

        # Kill CC subprocess first — unblocks pipe reads in _parse_stream
        await self._cc.stop()

        # Cancel and await the turn task.  The finally block sees
        # _cancel_in_progress and skips state transition + fallback text,
        # so it won't try to re-acquire the lock we already hold.
        if self._cc_turn_task and not self._cc_turn_task.done():
            self._cc_turn_task.cancel()
            try:
                await self._cc_turn_task
            except (asyncio.CancelledError, Exception):
                pass

        self._transition_to_idle()
        self._cancel_in_progress = False
        await self._send_fn("Cancelled.", msg)

    async def _respond_status(self, msg: Message) -> None:
        """Generate a contextual status response. Runs as a background task."""
        try:
            task_text = self._agent_state.current_task
            elapsed = time.time() - self._agent_state.turn_started_at
            recent_tools = self._agent_state.recent_tools[-5:]

            prompt = self._build_busy_prompt(
                task_text, elapsed, recent_tools, msg,
                live_events=list(self._live_events),
                text_parts=list(self._agent_state.text_parts),
            )

            # Use lightweight CC subprocess for the response
            claude_bin = shutil.which("claude") or "claude"
            cmd = [
                claude_bin, "-p",
                "--model", "sonnet",
                "--max-turns", "2",
                "--output-format", "text",
            ]

            env = _build_subprocess_env()
            env.pop("CLAUDE_CODE_ENTRYPOINT", None)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=30,
            )
            response = stdout.decode("utf-8").strip()

            # Detect CC error messages
            if response and response.startswith("Error:"):
                logger.warning(f"[{self._nickname}] Status subprocess returned error: {response}")
                response = ""

            if not response:
                response = (
                    f"I'm currently working on: {task_text[:100]}. "
                    f"Been at it for {elapsed:.0f}s."
                )

            await self._send_fn(response, msg)
        except Exception as e:
            logger.error(f"[{self._nickname}] Status response error: {e}")
            try:
                await self._send_fn(
                    f"I'm working on something right now — {e}",
                    msg,
                )
            except Exception:
                pass
        finally:
            self._status_response_pending = False

    def _build_busy_prompt(
        self,
        task_text: str,
        elapsed: float,
        recent_tools: list[str],
        query_msg: Message,
        live_events: list[dict[str, Any]] | None = None,
        text_parts: list[str] | None = None,
    ) -> str:
        """Build the prompt for a BUSY-mode status response.

        Includes full tool call/result log and accumulated CC text output
        so the status responder can give an informed, specific answer.
        """
        query = query_msg.content if isinstance(query_msg.content, str) else str(query_msg.content)
        tools_str = ", ".join(recent_tools) if recent_tools else "none yet"
        context = self._build_context_preamble(max_turns=8)

        # Build activity log from live events
        activity_log = ""
        if live_events:
            # Show last 20 events (tool_call + tool_result pairs)
            recent = live_events[-20:]
            lines = []
            start_ts = self._agent_state.turn_started_at or (
                recent[0]["timestamp"] if recent else time.time()
            )
            for evt in recent:
                rel_secs = evt["timestamp"] - start_ts
                etype = evt.get("event_type", "")
                tool = evt.get("tool_name", "?")
                data = evt.get("data", "")
                if etype == "tool_call":
                    # Show tool name + args (truncated)
                    args_preview = data[:300] if data else ""
                    lines.append(f"  [{rel_secs:6.1f}s] CALL  {tool}: {args_preview}")
                elif etype == "tool_result":
                    result_preview = data[:300] if data else ""
                    lines.append(f"  [{rel_secs:6.1f}s] RESULT {tool}: {result_preview}")
            activity_log = (
                f"## CC Activity Log ({len(live_events)} total events, "
                f"showing last {len(recent)})\n"
                + "\n".join(lines)
            )

        # Build accumulated text output
        text_output = ""
        if text_parts:
            # Show last few text chunks (most recent context)
            recent_text = text_parts[-5:]
            combined = "\n".join(t.strip() for t in recent_text if t.strip())
            if combined:
                text_output = f"## CC Text Output (latest)\n{combined[:1000]}"

        sections = [
            f"{context}\n\n"
            f"You are currently busy working on a task.\n\n"
            f"## Current Task\n{task_text}\n\n"
            f"## Status\n"
            f"- Elapsed: {elapsed:.0f} seconds\n"
            f"- Total tool calls: {len([e for e in (live_events or []) if e.get('event_type') == 'tool_call'])}\n"
            f"- Recent tool names: {tools_str}\n",
        ]
        if activity_log:
            sections.append(activity_log)
        if text_output:
            sections.append(text_output)
        sections.append(
            f"## Incoming Message\n"
            f"From: {query_msg.from_node}\n"
            f"Content: {query}\n\n"
            f"Respond to this message in first person. Give a specific, informed "
            f"status update based on the activity log above. Mention what you've "
            f"done so far, what you're currently doing, and estimate progress. "
            f"Be concise but specific (2-4 sentences). Stay in character."
        )

        return "\n\n".join(sections)

    # ── Stream observer ──────────────────────────────────────────

    async def _observe_event(self, event: CCStreamEvent, trigger: Message) -> None:
        """Process a stream event from CC. Updates state and pushes TOOL_ACTIVITY."""
        # Track text output — accumulate all chunks, deduplicate consecutive identical
        if event.type == "text" and event.content:
            stripped = event.content.strip()
            if stripped and (
                not self._agent_state.text_parts
                or self._agent_state.text_parts[-1].strip() != stripped
            ):
                self._agent_state.text_parts.append(event.content)
            self._agent_state.last_text = event.content

        # Track tool calls
        if event.type == "tool_use":
            self._agent_state.recent_tools.append(event.tool_name)
            # Keep bounded
            if len(self._agent_state.recent_tools) > 20:
                self._agent_state.recent_tools = self._agent_state.recent_tools[-10:]

            # Push TOOL_ACTIVITY to the trigger's sender
            if self._raw_send_fn and trigger.from_node:
                try:
                    activity_msg = make_tool_activity(
                        from_node=self._node_id,
                        to_node=trigger.from_node,
                        event_type="tool_call",
                        tool_name=event.tool_name,
                        tool_source="cc",
                        data={"args": event.content[:500], "preview": ""},
                        in_reply_to=trigger.id,
                    )
                    await self._raw_send_fn(activity_msg)
                except Exception as e:
                    logger.debug(f"Failed to push tool_call activity: {e}")

            # Record in live events for status peeking
            self._live_events.append({
                "event_type": "tool_call",
                "tool_name": event.tool_name,
                "data": event.content[:500],
                "timestamp": time.time(),
            })

            # Intercept send_message: deliver via the agent's persistent
            # transport instead of relying on the MCP server's ephemeral
            # WebSocket (which is broken — never registers with router).
            if event.tool_name and "send_message" in event.tool_name:
                delivered = False
                if self._raw_send_fn:
                    try:
                        args = json.loads(event.content) if event.content else {}
                        to_node = args.get("to", "")
                        msg_content = args.get("content", "")
                        if to_node and msg_content:
                            delivery_msg = make_message(
                                from_node=self._node_id,
                                to_node=to_node,
                                content=msg_content,
                                in_reply_to=trigger.id if trigger else None,
                            )
                            await self._raw_send_fn(delivery_msg)
                            delivered = True
                            self._cc_sent_message = True
                            logger.info(
                                f"[{self._nickname}] Intercepted send_message: "
                                f"{self._node_id} → {to_node} "
                                f"({len(msg_content)} chars)"
                            )
                    except Exception as e:
                        logger.error(
                            f"[{self._nickname}] send_message interception "
                            f"failed: {e}"
                        )

                if not delivered:
                    logger.warning(
                        f"[{self._nickname}] CC called {event.tool_name} "
                        f"but interception failed — text fallback will fire"
                    )

        # Track tool results
        if event.type == "tool_result":
            if self._raw_send_fn and trigger.from_node:
                try:
                    activity_msg = make_tool_activity(
                        from_node=self._node_id,
                        to_node=trigger.from_node,
                        event_type="tool_result",
                        tool_name=event.tool_name,
                        tool_source="cc",
                        data={"result": event.content[:500], "success": True},
                        in_reply_to=trigger.id,
                    )
                    await self._raw_send_fn(activity_msg)
                except Exception as e:
                    logger.debug(f"Failed to push tool_result activity: {e}")

            self._live_events.append({
                "event_type": "tool_result",
                "tool_name": event.tool_name,
                "data": event.content[:500],
                "timestamp": time.time(),
            })

        # Trim live events
        if len(self._live_events) > self._max_live_events:
            self._live_events = self._live_events[-self._max_live_events:]

    # ── Catchup ──────────────────────────────────────────────────

    def _build_catchup_xml(self) -> str:
        """Build XML transcript of messages received while BUSY + passive channel messages.

        Passive channel messages (from add_to_history_only) are included so
        CC-session agents see channel context they weren't directly @mentioned in.
        """
        all_entries = list(self._passive_messages) + list(self._catchup)
        if not all_entries:
            return ""

        # Sort by timestamp for chronological ordering
        all_entries.sort(key=lambda e: e.timestamp)

        lines = ["<catchup_messages>"]
        for entry in all_entries:
            safe_content = xml_escape(entry.content)
            lines.append(
                f'  <message from="{xml_escape(entry.from_node)}" '
                f'time="{xml_escape(entry.timestamp)}">{safe_content}</message>'
            )
        lines.append("</catchup_messages>")
        return "\n".join(lines)

    # ── History helpers ──────────────────────────────────────────

    def _append_to_history(self, msg: Message) -> None:
        """Append a message to the conversation history."""
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        ts = msg.timestamp
        if isinstance(ts, str) and ts:
            pass  # Use as-is
        else:
            ts = datetime.now(timezone.utc).isoformat()

        role = "user" if msg.from_node and msg.from_node.startswith("user:") else "assistant"
        self._history.append(Turn(
            role=role,
            content=content,
            timestamp=ts,
            from_node=msg.from_node,
            to_node=msg.to_node,
        ))
