"""
JSONL event protocol for the harness subprocess.

Every line of stdout is a JSON object with a "type" field.
The parent process (agent_node or benchmark runner) reads these
line-by-line and reacts accordingly.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

EventType = Literal[
    "thread.started",
    "turn.started",
    "assistant.message",
    "tool_call",
    "tool_result",
    "context.pruned",
    "turn.finished",
    "thread.finished",
    "usage",
    "error",
    "checkpoint.query",
    "checkpoint.response",
    "assessor.triage",
    "assessor.phase_start",
    "assessor.phase_complete",
    "assessor.assessment",
    # Session-mode events (persistent harness session driven by a router).
    # See mesh/harness/session.py and mesh/harness_session_manager.py.
    "session.started",
    "session.awaiting_input",
    "session.checkpoint",
    "session.context_exhausted",
    "session.reset_ack",
    "session.status",
]


@dataclass
class HarnessEvent:
    type: EventType
    ts: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def emit(event: HarnessEvent, file=None) -> None:
    """Write a single JSONL event to stdout (or provided file)."""
    out = file or sys.stdout
    out.write(event.to_json() + "\n")
    out.flush()


def emit_thread_started(thread_id: str, backend: str, model: str) -> None:
    emit(HarnessEvent(type="thread.started", data={
        "thread_id": thread_id,
        "backend": backend,
        "model": model,
    }))


def emit_turn_started(iteration: int) -> None:
    emit(HarnessEvent(type="turn.started", data={"iteration": iteration}))


def emit_assistant_message(text: str, iteration: int) -> None:
    emit(HarnessEvent(type="assistant.message", data={
        "text": text,
        "iteration": iteration,
    }))


def emit_tool_call(name: str, arguments: dict[str, Any], call_id: str) -> None:
    emit(HarnessEvent(type="tool_call", data={
        "name": name,
        "arguments": arguments,
        "call_id": call_id,
    }))


def emit_tool_result(name: str, result: str, call_id: str, success: bool = True) -> None:
    emit(HarnessEvent(type="tool_result", data={
        "name": name,
        "result": result,
        "call_id": call_id,
        "success": success,
    }))


def emit_context_pruned(pruned_count: int, before_tokens: int, after_tokens: int) -> None:
    emit(HarnessEvent(type="context.pruned", data={
        "pruned_count": pruned_count,
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
    }))


def emit_usage(usage: dict[str, Any]) -> None:
    emit(HarnessEvent(type="usage", data=usage))


def emit_thread_finished(
    thread_id: str,
    iterations: int,
    final_text: str,
    usage: dict[str, Any],
) -> None:
    emit(HarnessEvent(type="thread.finished", data={
        "thread_id": thread_id,
        "iterations": iterations,
        "final_text": final_text,
        "usage": usage,
    }))


def emit_error(message: str, iteration: int | None = None, fatal: bool = True) -> None:
    emit(HarnessEvent(type="error", data={
        "message": message,
        "iteration": iteration,
        "fatal": fatal,
    }))


def emit_checkpoint_query(iteration: int, trigger_reason: str) -> None:
    emit(HarnessEvent(type="checkpoint.query", data={
        "iteration": iteration,
        "trigger_reason": trigger_reason,
    }))


def emit_checkpoint_response(
    iteration: int, response_text: str, decision: str,
) -> None:
    emit(HarnessEvent(type="checkpoint.response", data={
        "iteration": iteration,
        "response_text": response_text[:500],
        "decision": decision,
    }))


def emit_assessor_triage(decision: str, needs_plan: bool, summary: str) -> None:
    emit(HarnessEvent(type="assessor.triage", data={
        "decision": decision,
        "needs_plan": needs_plan,
        "summary": summary[:500],
    }))


def emit_assessor_phase_start(phase_id: int, phase_type: str, phase_prompt: str) -> None:
    emit(HarnessEvent(type="assessor.phase_start", data={
        "phase_id": phase_id,
        "phase_type": phase_type,
        "phase_prompt": phase_prompt[:300],
    }))


def emit_assessor_phase_complete(phase_id: int, phase_type: str) -> None:
    emit(HarnessEvent(type="assessor.phase_complete", data={
        "phase_id": phase_id,
        "phase_type": phase_type,
    }))


def emit_assessor_assessment(
    phase_id: int, decision: str, reasoning: str,
    has_revised_plan: bool = False,
) -> None:
    emit(HarnessEvent(type="assessor.assessment", data={
        "phase_id": phase_id,
        "decision": decision,
        "reasoning": reasoning[:500],
        "has_revised_plan": has_revised_plan,
    }))


# ---------------------------------------------------------------------------
# Session-mode emit helpers
#
# A "session" is a persistent harness subprocess that stays alive between
# turns, driven by a router via JSONL commands on stdin. These events let the
# router-side manager (mesh/harness_session_manager.py) track lifecycle without
# scraping a terminal — every state transition is an explicit, typed event.
# ---------------------------------------------------------------------------


def emit_session_started(session_id: str, backend: str, model: str,
                         checkpoint_interval: int = 0) -> None:
    emit(HarnessEvent(type="session.started", data={
        "session_id": session_id,
        "backend": backend,
        "model": model,
        "checkpoint_interval": checkpoint_interval,
    }))


def emit_session_awaiting_input(final_text: str, iteration: int,
                                usage: dict[str, Any] | None = None) -> None:
    """The loop produced a final answer with no tool calls and is now WARM_IDLE,
    blocking on stdin for the next `task`/`steer`/`abort` command."""
    emit(HarnessEvent(type="session.awaiting_input", data={
        "final_text": final_text,
        "iteration": iteration,
        "usage": usage or {},
    }))


def emit_session_checkpoint(iteration: int, digest: dict[str, Any]) -> None:
    """Yielded every checkpoint_interval iterations. The driver reviews the
    digest and replies with `continue` (optionally with a steering nudge) or
    `abort`."""
    emit(HarnessEvent(type="session.checkpoint", data={
        "iteration": iteration,
        "digest": digest,
    }))


def emit_session_context_exhausted(iteration: int, summary: str,
                                   estimated_tokens: int, soft_limit: int) -> None:
    """Context budget hit the forced-synthesis threshold. The driver decides
    whether to `reset` (clear history, seed with the summary) or stop."""
    emit(HarnessEvent(type="session.context_exhausted", data={
        "iteration": iteration,
        "summary": summary,
        "estimated_tokens": estimated_tokens,
        "soft_limit": soft_limit,
    }))


def emit_session_reset_ack(iteration: int, kept_tokens: int) -> None:
    emit(HarnessEvent(type="session.reset_ack", data={
        "iteration": iteration,
        "kept_tokens": kept_tokens,
    }))


def emit_session_status(iteration: int, loop_state: str,
                        recent_tools: list[str], files_touched: list[str],
                        usage: dict[str, Any]) -> None:
    emit(HarnessEvent(type="session.status", data={
        "iteration": iteration,
        "loop_state": loop_state,
        "recent_tools": recent_tools,
        "files_touched": files_touched,
        "usage": usage,
    }))


def parse_event(line: str) -> HarnessEvent | None:
    """Parse a single JSONL line into a HarnessEvent. Returns None on failure."""
    line = line.strip()
    if not line:
        return None
    try:
        d = json.loads(line)
        return HarnessEvent(
            type=d["type"],
            ts=d.get("ts", 0.0),
            data=d.get("data", {}),
        )
    except (json.JSONDecodeError, KeyError):
        return None
