"""HarnessSessionManager — native interactive harness session subsystem.

This is the native replacement for the CC interactive session path
(mesh/cc_session_manager.py). Instead of launching Claude Code in a tmux pane
and scraping the terminal, it spawns the standalone harness in *session mode*
(`python -m mesh.harness session`) as a child process and communicates over
structured pipes:

    stdout  →  JSONL events (protocol.py): tool_call, tool_result, usage,
               session.awaiting_input, session.checkpoint,
               session.context_exhausted, thread.finished, error
    stdin   →  JSONL commands: task, steer, continue, abort, status, reset

The router LLM is the *driver*. It starts a session, gets a live activity
stream (tool_call/tool_result relayed as TOOL_ACTIVITY, exactly like the
traditional worker path), and is woken on lifecycle events (awaiting_input,
checkpoint, context_exhausted, finished, fatal error) to steer, abort, or relay
results.

Architecturally this mirrors CCSessionManager: the manager owns all session
STATE and METHODS, holds a back-reference to the owning RouterV2 as ``self.r``,
and reaches the router only for a small fixed surface:

    self.r._state                       router state machine (IDLE/BUSY)
    self.r._node_id, self.r._nickname   identity
    self.r._append_turn(...)            write a Turn into router history
    self.r._call_router_full(...)       run the router LLM (event delivery)
    self.r._send_and_store(...)         fallback outbound delivery
    self.r._trigger_nodes()             resolve current trigger (from, to)
    self.r._worker_agent                owning AgentNode (for ._conn and
                                        _push_mesh_tool_activity)

Tool REGISTRATION stays in RouterV2 (_init_harness_session_handlers there binds
the four harness_* tools to this manager's methods). The driver instructions
and session tool set are exposed as class attributes.

What it deliberately does NOT carry over from the CC path: tmux launch,
capture-pane scraping, the input-injection handshake, bracketed-paste / paste-
collapse workarounds, ❯ detection, and the CC-pathology monitor rules (nohup
children, suggested-prompt execution, fabrication-from-blank-screens). Native
events are born structured, and stdin writes are lossless, so none of that
machinery is needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time

from .protocol import Message, MessageType, make_tool_activity
from .conversation_history import Turn
from .harness import protocol as hp
# RouterState lives in router_v2; importing at module top is safe because
# router_v2 imports THIS module lazily (inside RouterV2.__init__).
from .router_v2 import RouterState

logger = logging.getLogger(__name__)


_HARNESS_SESSION_INSTRUCTIONS = """

─── HARNESS INTERACTIVE SESSION ───

You can drive a native interactive worker session to do multi-step work.

## This is your way to execute code

A harness session (harness_start_session) runs a persistent worker that edits
files, runs commands, and works across many turns on any backend (including
local models at zero marginal cost). You drive it: give it a task, watch the
activity stream, steer it if it drifts, and relay results when it finishes.

## When to use a session

**USE a session when the task involves:**
• Writing or editing files (code changes, config edits, creating scripts)
• Multi-step shell operations (investigate → fix → verify)
• Open-ended exploration or debugging
• Any task needing 3+ tool calls to complete

**Do NOT use a session when:**
• The task can be answered from memory or conversation context alone
• A single tool call suffices
• It is a lookup or status check

## Workflow

Single-call pattern (preferred):

  harness_start_session(task="<clear, scoped task description>")

This spawns the worker AND sends the task in one call. A background event pump
watches it automatically. You're done — respond with a brief status message and
wait for the [Harness Session …] event.

What happens next:
• tool_call / tool_result events stream to the user as live activity.
• When the worker yields (produces a final answer with no more tool calls),
  you receive a [Harness Session — Awaiting Input] event with its final text.
  Read it and decide: was the task accomplished?
   - YES → relay results to the user via send_message. Do NOT stop the session;
     it stays warm for follow-up work.
   - NO  → send guidance with harness_send_input(kind="steer", content="…").
• If you set a checkpoint interval, you receive [Harness Session — Checkpoint]
  events every N iterations; reply by sending harness_send_input(kind="continue")
  (optionally with a steering nudge) or stop the session.
• If the worker exhausts its context budget you receive
  [Harness Session — Context Exhausted] with a self-summary; send
  harness_send_input(kind="reset", content="<seed summary>") to continue fresh,
  or stop.

## Steering

harness_send_input(content="…", kind="steer"|"task"|"continue"|"reset"|"abort")
  • steer    — a correction injected at the next iteration boundary
  • task     — a new/additional work item
  • continue — resume after a checkpoint (optional content = nudge)
  • reset    — clear history and seed with content (after context exhaustion)
  • abort    — stop the worker

Steering is lossless and applied at the next iteration boundary (worst case,
after the current tool call finishes).

## Rules

1. One session at a time. Starting a new TASK on a fresh session cold-starts a
   clean worker; the system manages lifecycle.
2. Do NOT poll. The event pump wakes you on every lifecycle event. Use
   harness_get_status only when explicitly troubleshooting.
3. Do NOT do the work yourself with direct tools — the session IS your tool.
   Send investigation requests to it via harness_send_input.
4. Only stop a session early (harness_stop_session) if it has clearly drifted
   off-task or entered a degenerate loop; provide a `rationale`.
5. NEVER stop a session based on a status snapshot showing 0 iterations or
   0 tokens unless an explicit error event was received. A fresh session may
   still be initializing — wait for the next lifecycle event before acting.
6. Report only what the worker actually produced — never invent results.
"""

# Tools available to the driver while a session is active — only session
# management + messaging. Everything else is excluded so the router LLM cannot
# bypass the session and do the work itself.
_HARNESS_SESSION_TOOLS: frozenset[str] = frozenset({
    "harness_get_status", "harness_send_input", "harness_stop_session",
    "send_message", "sleep",
})

# How long to wait for graceful shutdown before escalating signals.
_STOP_SIGTERM_GRACE_S = 15.0
_STOP_SIGKILL_GRACE_S = 15.0  # additional wait after SIGTERM (total ~30s)


class HarnessSessionManager:
    """Owns the native interactive harness session lifecycle for one RouterV2."""

    INTERACTIVE_INSTRUCTIONS = _HARNESS_SESSION_INSTRUCTIONS
    SESSION_TOOLS = _HARNESS_SESSION_TOOLS

    def __init__(self, router, session_llm_config=None) -> None:
        self.r = router
        # Resolved LLMConfig for the named session backend (e.g.
        # mesh-harness-qwen36). Reads harness_* fields off it to build argv.
        self._session_cfg = session_llm_config
        self._proc: asyncio.subprocess.Process | None = None
        self._pump_task: asyncio.Task | None = None
        self._session_id: str | None = None
        self._session_trigger: Message | None = None
        self._session_task: str = ""
        self._state: str = "idle"  # idle | starting | active | warm_idle | stopping
        self._started_at: float | None = None
        self._last_event_at: float | None = None
        # Accumulated, typed state for harness_get_status — no scraping needed.
        self._iteration: int = 0
        self._recent_tools: list[str] = []        # ring buffer of tool names
        self._files_touched: list[str] = []
        self._event_tail: list[dict] = []          # ring buffer of recent events
        self._usage: dict = {}
        self._last_final_text: str = ""
        self._initial_task_delivered: bool = False

    # =========================================================================
    # Subprocess command construction
    # =========================================================================

    def _build_session_cmd(
        self,
        working_directory: str,
        max_iters: int,
        soft_limit: int,
        checkpoint_interval: int,
    ) -> list[str]:
        """Build the `python -m mesh.harness session ...` argv from config."""
        cfg = self._session_cfg
        python = sys.executable or "python3"
        cmd = [python, "-m", "mesh.harness", "session"]

        backend = getattr(cfg, "harness_backend", "") or "openai"
        model = getattr(cfg, "model", "") or "local-27b"
        cmd += ["--backend", backend, "--model", model]

        base_url = getattr(cfg, "harness_base_url", "")
        if base_url:
            cmd += ["--base-url", base_url]
        api_key = getattr(cfg, "harness_api_key", "")
        if api_key:
            cmd += ["--api-key", api_key]

        effort = getattr(cfg, "cc_effort", "")
        if effort:
            cmd += ["--effort", effort]
        thinking_budget = getattr(cfg, "thinking_budget", None)
        if thinking_budget:
            cmd += ["--thinking-budget", str(thinking_budget)]

        toolset = getattr(cfg, "harness_toolset", "") or "harness"
        cmd += ["--toolset", toolset]

        sp_file = getattr(cfg, "harness_system_prompt_file", "")
        if not sp_file:
            cand = os.path.join(os.path.dirname(__file__), "harness", "system_prompt.md")
            if os.path.isfile(cand):
                sp_file = cand
        if sp_file:
            cmd += ["--system-prompt-file", sp_file]

        cmd += ["--soft-limit", str(soft_limit)]
        cmd += ["--max-iters", str(max_iters)]
        cmd += ["--checkpoint-interval", str(checkpoint_interval)]
        cmd += ["--node-id", self.r._node_id or self.r._nickname or "harness-session"]

        socket_path = getattr(getattr(self.r, "_worker_agent", None), "_tool_socket_path", None)
        if socket_path:
            cmd += ["--agent-socket", socket_path]

        if working_directory:
            cmd += ["--cwd", working_directory]
        return cmd

    @staticmethod
    def _mask_cmd(cmd: list[str]) -> list[str]:
        masked = list(cmd)
        for i, arg in enumerate(masked):
            if i > 0 and any(k in masked[i - 1].lower() for k in ("key", "token")):
                masked[i] = f"***{arg[-4:]}" if len(arg) > 4 else "***"
        return masked

    def _subprocess_env(self) -> dict:
        try:
            from .llm import _build_subprocess_env
            env = _build_subprocess_env()
        except Exception:
            env = dict(os.environ)
        if self.r._node_id:
            env["MESH_NODE_ID"] = self.r._node_id
        return env

    def _reset_state(self, *, set_idle: bool = True) -> None:
        self._proc = None
        self._session_id = None
        self._session_trigger = None
        self._session_task = ""
        self._state = "idle"
        self._started_at = None
        self._iteration = 0
        self._recent_tools = []
        self._files_touched = []
        self._event_tail = []
        self._usage = {}
        self._initial_task_delivered = False
        if set_idle and self.r._state == RouterState.BUSY:
            self.r._state = RouterState.IDLE

    # =========================================================================
    # Tool: harness_start_session
    # =========================================================================

    async def _tool_harness_start_session(
        self,
        task: str = "",
        working_directory: str = "",
        max_iters: int = 0,
        budget: int = 0,
        checkpoint_interval: int = 0,
        **kwargs,
    ) -> str:
        """Spawn a native harness session and send it the initial task.

        A NEW task always cold-starts a fresh worker: if a session is already
        running, it is torn down first (the driver owns the kill decision; per
        the design, warm reuse is for steering pauses within a task, not across
        tasks)."""
        if self._session_cfg is None:
            return json.dumps({
                "status": "error",
                "error": "harness session backend is not configured for this agent.",
            })
        if not task:
            return json.dumps({
                "status": "error",
                "error": "task is required.",
            })

        # New task → cold start. Tear down any existing session.
        if self._proc is not None and self._proc.returncode is None:
            logger.info("[HARNESS-SESSION] New task — tearing down existing session %s", self._session_id)
            await self._terminate_proc(reason="new task cold-start")
            self._reset_state(set_idle=False)

        soft_limit = budget or getattr(self._session_cfg, "harness_soft_limit", 0) or 200_000
        max_iterations = max_iters or 100
        workdir = working_directory or self._real_cwd()

        cmd = self._build_session_cmd(workdir, max_iterations, soft_limit, checkpoint_interval)
        logger.info("[HARNESS-SESSION] launch: %s", " ".join(self._mask_cmd(cmd)))

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._subprocess_env(),
                start_new_session=True,  # own process group → teardown kills children
            )
        except Exception as e:
            logger.exception("[HARNESS-SESSION] spawn failed")
            return json.dumps({"status": "error", "error": f"spawn failed: {e}"})

        self._proc = proc
        self._state = "starting"
        self._session_task = task
        self._started_at = time.time()
        self._last_event_at = time.time()

        _trig_from, _trig_to = self.r._trigger_nodes()
        self._session_trigger = Message(
            from_node=_trig_from or self.r._node_id or "",
            to_node=_trig_to or self.r._node_id or "",
            type=MessageType.MESSAGE,
            content="[harness session trigger]",
        )

        # Start the event pump and stderr forwarder.
        self._pump_task = asyncio.create_task(self._event_pump())
        asyncio.create_task(self._forward_stderr())

        # Send the initial task as the first JSONL command (avoids argv limits).
        sent = await self._write_command({"type": "task", "content": task})
        if not sent:
            await self._terminate_proc(reason="failed to deliver initial task")
            self._reset_state()
            return json.dumps({"status": "error", "error": "failed to write initial task to session stdin"})
        self._initial_task_delivered = True

        self.r._state = RouterState.BUSY
        self._state = "active"
        return json.dumps({
            "status": "started",
            "session": self._session_id or f"pid:{proc.pid}",
            "pid": proc.pid,
            "backend": getattr(self._session_cfg, "harness_backend", "") or "openai",
            "model": getattr(self._session_cfg, "model", ""),
            "task": task[:200] + ("..." if len(task) > 200 else ""),
            "checkpoint_interval": checkpoint_interval,
            "hint": "Session started and task sent. The event pump will notify you "
                    "when it yields or finishes. Respond with a brief status message and wait.",
        })

    @staticmethod
    def _real_cwd() -> str:
        try:
            import pwd
            return pwd.getpwuid(os.getuid()).pw_dir
        except Exception:
            return os.getcwd()

    # =========================================================================
    # Tool: harness_send_input
    # =========================================================================

    async def _tool_harness_send_input(
        self, content: str = "", kind: str = "steer", **kwargs,
    ) -> str:
        """Write one JSONL command to the session's stdin. Lossless."""
        if self._proc is None or self._proc.returncode is not None:
            return json.dumps({"status": "error", "error": "No active harness session."})

        kind = (kind or "steer").lower()
        if kind not in ("steer", "task", "continue", "reset", "abort"):
            return json.dumps({"status": "error", "error": f"invalid kind '{kind}'."})

        if kind == "abort":
            return await self._tool_harness_stop_session(rationale="driver sent abort")

        ok = await self._write_command({"type": kind, "content": content})
        if not ok:
            return json.dumps({"status": "error", "error": "failed to write to session stdin (pipe closed?)"})

        # A task/steer/continue re-activates the worker.
        self.r._state = RouterState.BUSY
        self._state = "active"
        return json.dumps({
            "status": "sent",
            "kind": kind,
            "chars_sent": len(content),
        })

    # =========================================================================
    # Tool: harness_get_status
    # =========================================================================

    async def _tool_harness_get_status(self, **kwargs) -> str:
        """Return a structured digest from accumulated typed event state."""
        if self._proc is None:
            return json.dumps({"status": "no_session"})
        alive = self._proc.returncode is None
        return json.dumps({
            "status": "active" if alive else "exited",
            "session": self._session_id,
            "pid": self._proc.pid,
            "loop_state": self._state,
            "iteration": self._iteration,
            "recent_tools": self._recent_tools[-8:],
            "files_touched": self._files_touched[-12:],
            "task": self._session_task[:200],
            "tokens": {
                "input": self._usage.get("input_tokens", 0),
                "output": self._usage.get("output_tokens", 0),
                "llm_calls": self._usage.get("llm_calls", 0),
            },
            "elapsed_s": round(time.time() - self._started_at, 1) if self._started_at else None,
            "returncode": self._proc.returncode,
        })

    # =========================================================================
    # Tool: harness_stop_session
    # =========================================================================

    async def _tool_harness_stop_session(
        self, rationale: str = "", force: bool = False, **kwargs,
    ) -> str:
        """Stop the active session: abort command → SIGTERM → SIGKILL."""
        if self._proc is None:
            return json.dumps({"status": "error", "error": "No active harness session to stop."})

        if rationale:
            logger.info("[HARNESS-SESSION] stop with rationale: %s", rationale)

        sid = self._session_id or f"pid:{self._proc.pid}"
        await self._terminate_proc(reason=rationale or "stop requested", force=force)
        self._reset_state()
        return json.dumps({
            "status": "stopped",
            "session": sid,
            **({"rationale": rationale} if rationale else {}),
        })

    async def _terminate_proc(self, reason: str = "", force: bool = False) -> None:
        """Graceful abort → SIGTERM(group) → SIGKILL(group). Kills children via
        the process group (start_new_session=True at spawn)."""
        proc = self._proc
        if proc is None:
            return
        if self._pump_task and not self._pump_task.done():
            self._pump_task.cancel()

        if proc.returncode is None and not force:
            # Polite: ask the loop to drain and finish.
            await self._write_command({"type": "abort"})
            try:
                await asyncio.wait_for(proc.wait(), timeout=_STOP_SIGTERM_GRACE_S)
                logger.info("[HARNESS-SESSION] session ended gracefully (%s)", reason)
                return
            except asyncio.TimeoutError:
                pass

        # Escalate to signals on the process group.
        for sig, grace in ((signal.SIGTERM, _STOP_SIGTERM_GRACE_S), (signal.SIGKILL, _STOP_SIGKILL_GRACE_S)):
            if proc.returncode is not None:
                break
            try:
                os.killpg(os.getpgid(proc.pid), sig)
            except (ProcessLookupError, PermissionError):
                break
            try:
                await asyncio.wait_for(proc.wait(), timeout=grace)
                break
            except asyncio.TimeoutError:
                logger.warning("[HARNESS-SESSION] %s did not stop session, escalating", sig)

    async def _write_command(self, cmd: dict) -> bool:
        """Write one JSONL command line to the session subprocess stdin."""
        proc = self._proc
        if proc is None or proc.stdin is None or proc.returncode is not None:
            return False
        try:
            proc.stdin.write((json.dumps(cmd) + "\n").encode("utf-8"))
            await proc.stdin.drain()
            return True
        except (BrokenPipeError, ConnectionResetError, RuntimeError) as e:
            logger.warning("[HARNESS-SESSION] stdin write failed: %s", e)
            return False

    # =========================================================================
    # Event pump
    # =========================================================================

    async def _forward_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                text = line.decode(errors="replace").rstrip()
                if text:
                    logger.info("harness-session[%s]: %s", proc.pid, text)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("[HARNESS-SESSION] stderr forward ended: %s", e)

    async def _event_pump(self) -> None:
        """Consume JSONL events from the session subprocess stdout.

        tool_call / tool_result → live activity relay (TOOL_ACTIVITY).
        Lifecycle events (awaiting_input, checkpoint, context_exhausted, fatal
        error, thread.finished) → delivered to the router as a turn so the LLM
        can steer, reset, relay, or stop."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode(errors="replace").strip()
                if not text:
                    continue
                event = hp.parse_event(text)
                if event is None:
                    continue
                self._last_event_at = time.time()
                self._event_tail.append({"type": event.type, "data": event.data})
                if len(self._event_tail) > 50:
                    self._event_tail = self._event_tail[-50:]
                try:
                    await self._handle_event(event)
                except Exception as e:
                    logger.warning("[HARNESS-SESSION] event handler error (%s): %s", event.type, e)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("[HARNESS-SESSION] event pump ended: %s", e)
        finally:
            # If the pump ends because the process exited unexpectedly (not via
            # an explicit stop), surface it to the driver.
            rc = proc.returncode
            if rc is not None and self._state not in ("stopping", "idle"):
                await self._deliver_event(
                    f"[Harness Session — Exited]\nThe session process exited (code {rc}) "
                    f"without an explicit stop. Last task: {self._session_task[:200]}"
                )
                self._reset_state()

    async def _handle_event(self, event) -> None:
        etype = event.type
        data = event.data or {}

        if etype == "session.started":
            self._session_id = data.get("session_id")
            self._state = "active"
            return

        if etype == "turn.started":
            self._iteration = data.get("iteration", self._iteration)
            return

        if etype == "usage":
            for k in ("input_tokens", "output_tokens", "cache_creation_tokens",
                      "cache_read_tokens", "reasoning_tokens", "total_tokens", "llm_calls"):
                if k in data:
                    self._usage[k] = data[k]
            return

        if etype == "tool_call":
            name = data.get("name", "?")
            self._recent_tools.append(name)
            args = data.get("arguments", {})
            if isinstance(args, dict):
                for key in ("file_path", "path", "filename"):
                    v = args.get(key)
                    if isinstance(v, str) and v and v not in self._files_touched:
                        self._files_touched.append(v)
            await self._relay_tool_activity("tool_call", name, {
                "args": args if isinstance(args, dict) else {"input": str(args)},
                "preview": "",
            })
            return

        if etype == "tool_result":
            name = data.get("name", "?")
            preview = str(data.get("result", ""))[:200]
            await self._relay_tool_activity("tool_result", name, {
                "result": preview,
                "success": bool(data.get("success", True)),
            })
            return

        if etype == "context.pruned":
            return  # informational; status reflects it via tokens

        if etype == "session.awaiting_input":
            # Suppress the pre-task idle yield: the session emits
            # session.awaiting_input before the initial task arrives via stdin.
            # The flag is set *after* the first task write succeeds, so any
            # idle yield received before that is pre-task noise.
            if not self._initial_task_delivered:
                logger.debug("[HARNESS-SESSION] suppressed pre-task idle yield")
                return
            self._state = "warm_idle"
            self._last_final_text = data.get("final_text", "")
            final = self._last_final_text or "(no text)"
            await self._deliver_event(
                f"[Harness Session — Awaiting Input]\n"
                f"The worker finished a turn and is idle (warm). Its final output:\n\n"
                f"{final}\n\n"
                f"Decide: was the task accomplished? If YES, relay results to the user "
                f"via send_message (do NOT stop — the session stays warm). If NO, send "
                f"harness_send_input(kind=\"steer\", content=\"…\") to continue."
            )
            return

        if etype == "session.checkpoint":
            self._state = "warm_idle"
            digest = data.get("digest", {})
            await self._deliver_event(
                f"[Harness Session — Checkpoint @ iteration {data.get('iteration')}]\n"
                f"Status digest: {json.dumps(digest, indent=2)[:1500]}\n\n"
                f"Review progress. To continue, send harness_send_input(kind=\"continue\") "
                f"(optionally with a steering nudge). To stop, call harness_stop_session "
                f"with a rationale."
            )
            return

        if etype == "session.context_exhausted":
            self._state = "warm_idle"
            summary = data.get("summary", "")
            await self._deliver_event(
                f"[Harness Session — Context Exhausted]\n"
                f"The worker hit its context budget. Self-summary of work so far:\n\n"
                f"{summary[:3000]}\n\n"
                f"To continue with a fresh context seeded by this summary, send "
                f"harness_send_input(kind=\"reset\", content=\"<seed>\"). Otherwise stop "
                f"the session and relay what was accomplished."
            )
            return

        if etype == "error" and data.get("fatal"):
            await self._deliver_event(
                f"[Harness Session — Error]\n{data.get('message', 'unknown error')}"
            )
            return

        if etype == "thread.finished":
            status = (data.get("usage") or {}).get("status", "finished")
            if status == "aborted":
                # Stop path already handled; just ensure state is clean.
                return
            self._state = "warm_idle"
            return

    # =========================================================================
    # Public accessors for watchdog visibility
    # =========================================================================

    def get_recent_events(self, n: int | None = None) -> list[dict]:
        """Return the most recent *n* events from ``_event_tail`` (all if *n* is None)."""
        if n is None:
            return list(self._event_tail)
        return list(self._event_tail[-n:])

    def get_recent_event_strings(self, n: int | None = None, label: str = "worker live") -> list[str]:
        """Format recent ``_event_tail`` entries as display strings.

        Output mirrors the format consumed by ``_build_worker_activity_lines``
        so the watchdog can render harness sessions identically to CC sessions.
        """
        events = self.get_recent_events(n)
        lines: list[str] = []
        for ev in events:
            etype = ev.get("type", "?")
            data = ev.get("data") or {}
            if etype == "tool_call":
                name = data.get("name", "?")
                args = data.get("arguments", {})
                if isinstance(args, dict):
                    items = list(args.items())[:2]
                    args_str = ", ".join(f'{k}="{str(v)[:60]}"' for k, v in items)
                else:
                    args_str = str(args)[:80]
                lines.append(f"[{label}] tool_call: {name}({args_str})")
            elif etype == "tool_result":
                name = data.get("name", "?")
                result_str = str(data.get("result", ""))
                success = data.get("success", True)
                lines.append(f"[{label}] tool_result: {name} ({'ok' if success else 'err'}, {len(result_str)} chars)")
            else:
                lines.append(f"[{label}] {etype}: {str(data)[:100]}")
        return lines

    # =========================================================================
    # Activity relay + event delivery (router-facing)
    # =========================================================================

    async def _relay_tool_activity(self, event_type: str, tool_name: str, data: dict) -> None:
        """Push a session tool event to the trigger sender as TOOL_ACTIVITY
        (tool_source="harness"), parity with the worker path's
        _push_mesh_tool_activity. Best-effort; never disturbs the pump."""
        trigger = self._session_trigger
        dest = trigger.from_node if trigger else None
        if not dest or dest == self.r._node_id:
            return
        agent = getattr(self.r, "_worker_agent", None)
        pusher = getattr(agent, "_push_mesh_tool_activity", None)
        try:
            if pusher is not None:
                await pusher(
                    to_node=dest,
                    event_type=event_type,
                    tool_name=tool_name,
                    data=data,
                    in_reply_to=trigger.id if trigger else None,
                    tool_source="harness",
                )
                return
            conn = getattr(agent, "_conn", None)
            if conn is None:
                return
            await conn.send(make_tool_activity(
                from_node=self.r._node_id or self.r._nickname,
                to_node=dest,
                event_type=event_type,
                tool_name=tool_name,
                tool_source="harness",
                data=data,
                in_reply_to=trigger.id if trigger else None,
            ))
        except Exception as e:
            logger.debug("[HARNESS-SESSION] tool_activity relay failed: %s", e)

    async def _deliver_event(self, content: str) -> None:
        """Deliver a session lifecycle event to the router via _call_router_full,
        so the LLM can steer/reset/relay/stop. Mirrors CCSessionManager._cc_deliver_event."""
        trigger = self._session_trigger
        if not trigger:
            trigger = Message(
                from_node=self.r._node_id or f"agent:unknown:{self.r._nickname}",
                to_node=self.r._node_id or f"agent:unknown:{self.r._nickname}",
                type=MessageType.MESSAGE,
                content=content,
            )

        from datetime import datetime, timezone
        self.r._append_turn(Turn(
            role="system",
            content=content,
            timestamp=datetime.now(timezone.utc),
            from_node="system:harness-session",
            to_node=self.r._node_id or self.r._nickname,
            meta={"harness_session_event": True},
        ))

        task_ctx = self._session_task or "(no task description)"
        if trigger.to_node and trigger.to_node.startswith("channel:"):
            reply_dest = trigger.to_node
        else:
            reply_dest = trigger.from_node
        instructions = (
            _HARNESS_SESSION_INSTRUCTIONS
            + f"\n## Current Task\n\n{task_ctx}\n"
            + f"\n## Reply Destination\n\nWhen relaying results via send_message, send to: `{reply_dest}`\n"
        )

        try:
            await self.r._call_router_full(
                msg=trigger,
                busy=True,
                watchdog=False,
                tool_filter=_HARNESS_SESSION_TOOLS,
                instructions_override=instructions,
                monitor_mode=True,
            )
        except Exception as e:
            logger.warning("[HARNESS-SESSION] router delivery failed: %s", e)
            try:
                fallback = content if len(content) <= 4000 else content[:4000] + "\n... (truncated)"
                await self.r._send_and_store(
                    f"[Harness Session Event — delivery error]\n{fallback}", trigger,
                )
            except Exception as e2:
                logger.error("[HARNESS-SESSION] fallback delivery also failed: %s", e2)
