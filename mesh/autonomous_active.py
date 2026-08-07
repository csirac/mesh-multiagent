"""Deterministic next-wake scheduling for autonomous active mode.

Active mode is the answer to a simple operator request: when a project is armed
and its daily budget still has room, the next session should schedule itself.
The operator's two constraints on that are equally simple — it must be *paced* (a
minimum gap between sessions, never continuous), and an agent that controls more
than one project must never end up with two sessions overlapping.

Those are scheduling rules, not judgment calls, so they live here as a pure
function.  ``plan_active_wake`` takes a snapshot of the world — the active flag,
remaining budget, the agent's pending wakes, the pacing gap, and the current
time — and returns a decision.  It touches no files, no clock, and no agent
state, which is what makes the behaviour testable and what keeps the LLM out of
the loop: the controller may write the report, but it does not get a vote on
whether the next wake happens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import yaml

from .protocol import AUTONOMOUS_WAKE_HEADER

__all__ = [
    "PendingWake",
    "ActiveWakeDecision",
    "TERMINAL_SESSION_STATUSES",
    "plan_active_wake",
    "wake_project_key",
    "report_suppresses_next_wake",
]

#: Session statuses that mean "do not continue on your own".  ``completed`` and
#: ``partial`` describe a session that ran; the project itself keeps going.  The
#: three below are the mandate's stop conditions: nothing to do, waiting on a
#: human, or a session that fell over — and the mandate is explicit that a
#: failed closeout must never loop immediately.
TERMINAL_SESSION_STATUSES = ("blocked", "failed", "no_op")

_PROJECT_KEY_LINE = re.compile(r"^\s*project_entity_key\s*:\s*(\S+)\s*$", re.MULTILINE)
_REPORT_FRONTMATTER = re.compile(
    r"\A---[ \t]*\r?\n(?P<frontmatter>.*?)\r?\n---[ \t]*(?:\r?\n|\Z)",
    re.DOTALL,
)


@dataclass(frozen=True)
class PendingWake:
    """One scheduled wake belonging to this agent.

    ``project_key`` is empty for an ordinary reminder.  Those still count for
    the no-overlap rule — an autonomous session colliding with any other wake
    is the collision Project Owner asked to avoid — but only a matching project key
    trips the dedup gate.
    """

    wake_id: str
    wake_time: datetime
    project_key: str = ""


@dataclass
class ActiveWakeDecision:
    """The outcome of one scheduling evaluation."""

    scheduled: bool
    reason: str
    wake_time: datetime | None = None
    deferred_behind: str = ""
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        out: dict = {"scheduled": self.scheduled, "reason": self.reason}
        if self.wake_time is not None:
            out["wake_time_utc"] = self.wake_time.isoformat()
        if self.deferred_behind:
            out["deferred_behind"] = self.deferred_behind
        out.update(self.detail)
        return out


def wake_project_key(prompt: str) -> str:
    """Return the project key an autonomous wake prompt names, or ``""``.

    Only prompts carrying the canonical session header count.  A user reminder
    that happens to mention a project key is not an autonomous session.
    """
    text = prompt or ""
    if AUTONOMOUS_WAKE_HEADER not in text:
        return ""
    match = _PROJECT_KEY_LINE.search(text)
    return match.group(1).strip() if match else ""


def report_suppresses_next_wake(report_text: str) -> str:
    """Return a suppression reason from a session report, or ``""``.

    The report frontmatter is where the controller records its judgment about
    the session.  Two signals stop the automatic scheduler:

    * ``status`` is one of :data:`TERMINAL_SESSION_STATUSES`.

    ``next_wake_intent: none`` is meaningful only alongside one of those
    terminal statuses.  That keeps an older or malformed controller report
    from accidentally parking an otherwise armed project: a completed or
    partial session may not know the scheduler's eventual timestamp.

    Everything else — including a malformed or absent frontmatter — leaves the
    scheduler armed.  Active mode is an explicit opt-in with hard budget and
    pacing gates behind it, so a garbled report should not silently park a
    project the operator deliberately armed.
    """
    match = _REPORT_FRONTMATTER.match(report_text or "")
    if not match:
        return ""

    try:
        frontmatter = yaml.safe_load(match.group("frontmatter"))
    except yaml.YAMLError:
        return ""
    if not isinstance(frontmatter, dict):
        return ""

    status = str(frontmatter.get("status", "")).strip().lower()
    if status in TERMINAL_SESSION_STATUSES:
        return f"session status={status}"

    return ""


def plan_active_wake(
    project_key: str,
    active: bool,
    remaining: int,
    pending: list[PendingWake],
    gap_minutes: int,
    now: datetime,
    suppressed_reason: str = "",
) -> ActiveWakeDecision:
    """Decide whether and when the next autonomous session should fire.

    The gates run in the order an operator would reason about them: is the
    project armed, does it have budget, is a wake already queued, and only then
    what time is free.
    """
    if not active:
        return ActiveWakeDecision(False, "project is not in active mode")

    if suppressed_reason:
        return ActiveWakeDecision(False, suppressed_reason)

    try:
        remaining = int(remaining)
    except (TypeError, ValueError):
        remaining = 0
    if remaining <= 0:
        return ActiveWakeDecision(False, "daily worker budget is exhausted")

    gap = timedelta(minutes=max(1, int(gap_minutes)))

    for wake in pending:
        if wake.project_key == project_key:
            return ActiveWakeDecision(
                False,
                f"a wake for {project_key} is already pending ({wake.wake_id})",
                detail={"existing_wake_id": wake.wake_id},
            )

    # Pacing: never sooner than one gap from now.
    candidate = now + gap
    deferred_behind = ""

    # No-overlap: walk this agent's other wakes in time order and push the
    # candidate past any that sits within a gap of it.  The candidate only ever
    # moves later, so one ascending pass reaches a stable answer.
    for wake in sorted(pending, key=lambda w: w.wake_time):
        if abs((candidate - wake.wake_time).total_seconds()) < gap.total_seconds():
            candidate = wake.wake_time + gap
            deferred_behind = wake.wake_id

    return ActiveWakeDecision(
        True,
        "active with budget remaining",
        wake_time=candidate,
        deferred_behind=deferred_behind,
        detail={"remaining_budget": remaining, "gap_minutes": int(gap.total_seconds() // 60)},
    )
