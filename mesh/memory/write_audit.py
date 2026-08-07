"""Per-write-attempt audit for curated-artifact writes (G-004).

The readable layer of the memory pipeline — the standing digest and the entity
dossiers — is maintained by self-curation turns.  Over-ceiling writes to those
artifacts are *refused*, never truncated (D-003), and a refusal is not carried
forward: unless the model compresses and retries inside the same turn, the
addition is silently dropped.  Until now that drop was only *observable* — an
audit had to infer incidence from refusal prose in the ``curation_turn`` trail.

This module makes it *measurable*.  Every curated-artifact write attempt
records one row, and the true retry-success and information-drop rates fall out
of a group-by.

Design notes
------------

**No new store.**  Attempts are recorded as ``curation_write_attempt`` rows in
the existing ``entity_events`` trail.  That table already carries the agent
(``actor_node``), the timestamp (``created_at``) and a stable per-turn id
(``run_key``), and its ``event_type`` column is unconstrained, so nothing but
two covering indexes is needed.  Keeping one trail means T-005 (re-run the
ceiling audit against instrumented data) and the eventual T-001 carry-forward
fix consume the same source of truth.

**This module measures; it never decides.**  Nothing here changes refusal
semantics.  ``queued`` was defined in the vocabulary before anything emitted
it, so the reporting query would not have to change when carry-forward landed.
T-001 now fills that slot: an addition the model does not land inside the turn
is written to the durable pending-additions ledger at turn end and recorded
here as ``queued`` (see :mod:`mesh.memory.pending_additions`).

**The ``outcome`` vocabulary.**  G-004 names four outcomes and defines
``retried`` as "an attempt that follows an earlier refusal of the same artifact
in the same turn", while also requiring that a successful retry record
``landed``.  Those two readings collide on the landed-retry case, so the
reconciliation is explicit:

``landed``
    The write committed.  (Also used for a retry that succeeds — this is the
    case G-004 calls out, and ``is_retry`` below keeps it distinguishable.)
``refused``
    The write was rejected, and this artifact had not been refused earlier in
    this turn.
``retried``
    The write was rejected *again* — the artifact had already been refused in
    this turn.  A failed retry.
``queued``
    The write was refused and the addition was carried forward into the
    durable pending-additions ledger, so it is not a drop.  Emitted once per
    un-landed addition, at turn end (T-001).

Every attempt additionally carries ``is_retry``, so retry-success is countable
directly as ``outcome == "landed" and is_retry`` without re-deriving turn order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

#: ``entity_events.event_type`` for one curated-artifact write attempt.
CURATION_WRITE_ATTEMPT_EVENT = "curation_write_attempt"

OUTCOME_LANDED = "landed"
OUTCOME_REFUSED = "refused"
OUTCOME_QUEUED = "queued"
OUTCOME_RETRIED = "retried"

#: The four-value outcome vocabulary from G-004.
WRITE_OUTCOMES = frozenset(
    {OUTCOME_LANDED, OUTCOME_REFUSED, OUTCOME_QUEUED, OUTCOME_RETRIED}
)

#: Outcomes that mean "the artifact did not change on this attempt".
REFUSAL_OUTCOMES = frozenset({OUTCOME_REFUSED, OUTCOME_RETRIED})

#: Per-turn resolution states, one per artifact touched.
RESOLUTION_LANDED_CLEAN = "landed_clean"
RESOLUTION_RETRY_SUCCESS = "retry_success"
RESOLUTION_TERMINAL_DROP = "terminal_drop"

#: Cap on attempts recorded per turn.  A model looping on the same refusal
#: cannot grow the trail without bound; the overflow is still counted (see
#: :attr:`WriteAttemptLog.overflow`) so the cap never silently hides volume.
WRITE_ATTEMPT_CAP = 200

#: Truncation for the human-readable refusal reason kept on each attempt.
DETAIL_CHARS = 300

#: Length of the short content hashes.  Twelve hex characters of sha256 is
#: ample to detect "did the artifact change", which is all before/after is for.
HASH_CHARS = 12


def short_hash(text: str | None) -> str:
    """Return the first :data:`HASH_CHARS` hex characters of sha256(text).

    ``None`` and ``""`` both hash as the empty string, so a write that creates
    an artifact from nothing shows a stable, comparable ``before_hash``.
    """
    payload = (text or "").encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:HASH_CHARS]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_artifact_key(agent: str) -> str:
    """Stable ``target_artifact`` identifier for an agent's standing digest."""
    return f"digest:{agent or 'unknown'}"


def essay_artifact_key(entity_key: str) -> str:
    """Stable ``target_artifact`` identifier for one entity dossier."""
    return f"essay:{entity_key}"


def entity_artifact_key(entity_key: str) -> str:
    """Stable ``target_artifact`` identifier for one entity registry record."""
    return f"entity:{entity_key}"


@dataclass(frozen=True)
class WriteAttempt:
    """One resolved write attempt against one curated artifact."""

    target_artifact: str
    tool: str
    call_ordinal: int
    outcome: str
    before_hash: str
    after_hash: str
    turn_id: str
    agent: str
    timestamp: str
    is_retry: bool = False
    measured_tokens: int | None = None
    budget_tokens: int | None = None
    over_budget: bool | None = None
    session_id: str = ""
    session_ordinal: int | None = None
    mode: str = ""
    detail: str = ""

    @property
    def changed(self) -> bool:
        """True when the artifact's content actually moved."""
        return self.before_hash != self.after_hash

    def as_details(self) -> dict[str, Any]:
        """The ``details_json`` payload for this attempt's event row."""
        return {
            "target_artifact": self.target_artifact,
            "tool": self.tool,
            "call_ordinal": self.call_ordinal,
            "outcome": self.outcome,
            "is_retry": self.is_retry,
            "measured_tokens": self.measured_tokens,
            "budget_tokens": self.budget_tokens,
            "over_budget": self.over_budget,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "turn_id": self.turn_id,
            "agent": self.agent,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "session_ordinal": self.session_ordinal,
            "mode": self.mode,
            "detail": self.detail,
        }


class WriteAttemptLog:
    """Per-turn accumulator that assigns ordinals and classifies outcomes.

    The log is the only place that knows a turn's history, so it is what turns
    a bare "this write was refused" into "this write was refused *again*" and,
    at turn end, into "this artifact was dropped".
    """

    def __init__(self, *, session_id: str = "", session_ordinal_start: int = 0):
        self.attempts: list[WriteAttempt] = []
        self.overflow: int = 0
        self.session_id = session_id
        self._session_ordinal = int(session_ordinal_start)
        #: Artifacts refused at least once so far in this turn.
        self._refused: set[str] = set()

    # ── recording ────────────────────────────────────────────────

    def record(
        self,
        *,
        target_artifact: str,
        tool: str,
        landed: bool,
        before_hash: str,
        after_hash: str,
        turn_id: str = "",
        agent: str = "",
        measured_tokens: int | None = None,
        budget_tokens: int | None = None,
        mode: str = "",
        detail: str = "",
        queued: bool = False,
        timestamp: str | None = None,
    ) -> WriteAttempt | None:
        """Classify and append one attempt.

        ``landed`` is the caller's ground truth — whether the write committed.
        Everything else (ordinal, retry status, the four-value outcome) is
        derived here from the turn's own history, so no call site has to know
        what happened earlier in the turn.

        Returns the recorded attempt, or ``None`` once the per-turn cap is hit.
        """
        self._session_ordinal += 1
        if len(self.attempts) >= WRITE_ATTEMPT_CAP:
            self.overflow += 1
            return None

        is_retry = target_artifact in self._refused
        if queued:
            outcome = OUTCOME_QUEUED
        elif landed:
            outcome = OUTCOME_LANDED
        else:
            outcome = OUTCOME_RETRIED if is_retry else OUTCOME_REFUSED

        if outcome in REFUSAL_OUTCOMES:
            self._refused.add(target_artifact)

        over_budget = None
        if measured_tokens is not None and budget_tokens is not None:
            over_budget = int(measured_tokens) > int(budget_tokens)

        attempt = WriteAttempt(
            target_artifact=target_artifact,
            tool=tool,
            call_ordinal=len(self.attempts) + 1,
            outcome=outcome,
            before_hash=before_hash,
            after_hash=after_hash,
            turn_id=turn_id,
            agent=agent,
            timestamp=timestamp or _utcnow(),
            is_retry=is_retry,
            measured_tokens=(
                None if measured_tokens is None else int(measured_tokens)
            ),
            budget_tokens=(
                None if budget_tokens is None else int(budget_tokens)
            ),
            over_budget=over_budget,
            session_id=self.session_id,
            session_ordinal=self._session_ordinal,
            mode=mode,
            detail=(detail or "")[:DETAIL_CHARS],
        )
        self.attempts.append(attempt)
        return attempt

    # ── turn-level resolution ────────────────────────────────────

    def resolution(self) -> dict[str, str]:
        """Per-artifact terminal state for this turn.

        This is the "true drop rate" signal: an artifact refused at some point
        that never lands by the end of the turn is a terminal drop — real
        information the pipeline lost.
        """
        return resolve_turn(attempt.as_details() for attempt in self.attempts)

    def summary(self) -> dict[str, Any]:
        """Compact roll-up suitable for the turn's ``curation_turn`` event."""
        resolution = self.resolution()
        counts: dict[str, int] = {}
        for attempt in self.attempts:
            counts[attempt.outcome] = counts.get(attempt.outcome, 0) + 1
        states: dict[str, int] = {}
        for state in resolution.values():
            states[state] = states.get(state, 0) + 1
        return {
            "attempts": len(self.attempts),
            "overflow": self.overflow,
            "by_outcome": counts,
            "artifacts": len(resolution),
            "by_resolution": states,
            "retry_success": states.get(RESOLUTION_RETRY_SUCCESS, 0),
            "terminal_drops": states.get(RESOLUTION_TERMINAL_DROP, 0),
            "carried_forward": states.get(OUTCOME_QUEUED, 0),
        }


def resolve_turn(attempts: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Map ``target_artifact`` to its terminal state within one turn.

    Accepts raw attempt mappings (the ``details_json`` payloads), so it works
    equally on a live :class:`WriteAttemptLog` and on rows read back out of the
    event trail.  Attempts are assumed to arrive in turn order; when they carry
    ``call_ordinal`` they are sorted by it defensively, because a windowed
    query may interleave rows from several turns.
    """
    ordered = sorted(
        attempts,
        key=lambda item: int(item.get("call_ordinal") or 0),
    )
    refused: set[str] = set()
    state: dict[str, str] = {}
    for attempt in ordered:
        artifact = str(attempt.get("target_artifact") or "")
        if not artifact:
            continue
        outcome = str(attempt.get("outcome") or "")
        if outcome in REFUSAL_OUTCOMES:
            refused.add(artifact)
            state[artifact] = RESOLUTION_TERMINAL_DROP
        elif outcome == OUTCOME_LANDED:
            state[artifact] = (
                RESOLUTION_RETRY_SUCCESS
                if artifact in refused
                else RESOLUTION_LANDED_CLEAN
            )
        elif outcome == OUTCOME_QUEUED:
            # A carried-forward addition is not lost, so it is not a drop.
            # Queueing always FOLLOWS the refusal it rescues, so it must be
            # able to clear the terminal drop that refusal recorded — the
            # whole point of T-001 is that this artifact's turn no longer
            # ends in loss.  It must not overwrite a real resolution: an
            # artifact that also landed something this turn keeps that
            # verdict, which is the stronger statement.
            if state.get(artifact) in (None, RESOLUTION_TERMINAL_DROP):
                state[artifact] = OUTCOME_QUEUED
    return state


# ── Reading the trail ────────────────────────────────────────────


def load_write_attempts(
    connection: Any,
    *,
    since: str | None = None,
    until: str | None = None,
    agent: str | None = None,
) -> list[dict[str, Any]]:
    """Read ``curation_write_attempt`` rows out of ``entity_events``.

    Read-only: this issues a single ``SELECT`` and never writes.  ``since`` and
    ``until`` are ISO-8601 strings compared against ``created_at`` (half-open:
    ``since <= created_at < until``), which sorts lexicographically for UTC
    ISO timestamps.
    """
    where = ["event_type = ?"]
    params: list[Any] = [CURATION_WRITE_ATTEMPT_EVENT]
    if since is not None:
        where.append("created_at >= ?")
        params.append(since)
    if until is not None:
        where.append("created_at < ?")
        params.append(until)
    if agent is not None:
        where.append("actor_node = ?")
        params.append(agent)
    rows = connection.execute(
        "SELECT sequence, actor_node, run_key, details_json, created_at "
        "FROM entity_events WHERE " + " AND ".join(where) + " ORDER BY sequence",
        tuple(params),
    ).fetchall()

    attempts: list[dict[str, Any]] = []
    for sequence, actor_node, run_key, details_json, created_at in rows:
        try:
            details = json.loads(details_json or "{}")
        except (TypeError, json.JSONDecodeError):
            details = {}
        if not isinstance(details, dict):
            details = {}
        # The event row's own columns win: they are written by the store, not
        # by the payload, so they cannot drift.
        details["sequence"] = int(sequence)
        details["agent"] = actor_node or details.get("agent") or ""
        details["turn_id"] = run_key or details.get("turn_id") or ""
        details["created_at"] = created_at
        attempts.append(details)
    return attempts


def summarize_attempts(
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate attempts into the per-agent report described by G-004.

    Returns ``{"agents": {agent: {...}}, "totals": {...}}`` where each block
    carries attempts by outcome, retry-success rate and terminal-drop count.
    """
    # Group by (agent, turn) so resolution is computed per turn, then roll up.
    turns: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for attempt in attempts:
        key = (
            str(attempt.get("agent") or ""),
            str(attempt.get("turn_id") or ""),
        )
        turns.setdefault(key, []).append(attempt)

    def _blank() -> dict[str, Any]:
        return {
            "attempts": 0,
            "by_outcome": {name: 0 for name in sorted(WRITE_OUTCOMES)},
            "turns": 0,
            "artifacts": 0,
            "landed_clean": 0,
            "retry_success": 0,
            "terminal_drops": 0,
            # Artifacts whose turn ended with the addition carried forward
            # into the pending ledger (T-001).  Counted separately from both
            # rates: the addition is neither resolved nor lost, so folding it
            # into either one would misstate the window.
            "carried_forward": 0,
            "refused_artifacts": 0,
            "retry_success_rate": None,
            "drop_rate": None,
        }

    agents: dict[str, dict[str, Any]] = {}
    for (agent, _turn_id), turn_attempts in turns.items():
        block = agents.setdefault(agent, _blank())
        block["turns"] += 1
        for attempt in turn_attempts:
            block["attempts"] += 1
            outcome = str(attempt.get("outcome") or "")
            if outcome in block["by_outcome"]:
                block["by_outcome"][outcome] += 1
        for state in resolve_turn(turn_attempts).values():
            block["artifacts"] += 1
            if state == RESOLUTION_LANDED_CLEAN:
                block["landed_clean"] += 1
            elif state == RESOLUTION_RETRY_SUCCESS:
                block["retry_success"] += 1
            elif state == RESOLUTION_TERMINAL_DROP:
                block["terminal_drops"] += 1
            elif state == OUTCOME_QUEUED:
                block["carried_forward"] += 1

    totals = _blank()
    for block in agents.values():
        # ``refused_artifacts`` is the denominator for both rates: every
        # artifact that hit a refusal and RESOLVED within its turn, either by
        # a successful retry or by dropping.  A carried-forward artifact is
        # deliberately outside it — its addition is still owed, so scoring it
        # as a success would flatter the window and scoring it as a drop
        # would libel it.  ``carried_forward`` reports it in its own right.
        block["refused_artifacts"] = (
            block["retry_success"] + block["terminal_drops"]
        )
        denominator = block["refused_artifacts"]
        if denominator:
            block["retry_success_rate"] = block["retry_success"] / denominator
            block["drop_rate"] = block["terminal_drops"] / denominator
        for name in ("attempts", "turns", "artifacts", "landed_clean",
                     "retry_success", "terminal_drops", "carried_forward",
                     "refused_artifacts"):
            totals[name] += block[name]
        for outcome, count in block["by_outcome"].items():
            totals["by_outcome"][outcome] += count

    if totals["refused_artifacts"]:
        totals["retry_success_rate"] = (
            totals["retry_success"] / totals["refused_artifacts"]
        )
        totals["drop_rate"] = (
            totals["terminal_drops"] / totals["refused_artifacts"]
        )

    return {"agents": agents, "totals": totals}


def report(
    connection: Any,
    *,
    since: str | None = None,
    until: str | None = None,
    agent: str | None = None,
) -> dict[str, Any]:
    """Read-only windowed report over the write-attempt trail."""
    attempts = load_write_attempts(
        connection, since=since, until=until, agent=agent
    )
    summary = summarize_attempts(attempts)
    summary["window"] = {"since": since, "until": until, "agent": agent}
    return summary


__all__ = [
    "CURATION_WRITE_ATTEMPT_EVENT",
    "DETAIL_CHARS",
    "HASH_CHARS",
    "OUTCOME_LANDED",
    "OUTCOME_QUEUED",
    "OUTCOME_REFUSED",
    "OUTCOME_RETRIED",
    "REFUSAL_OUTCOMES",
    "RESOLUTION_LANDED_CLEAN",
    "RESOLUTION_RETRY_SUCCESS",
    "RESOLUTION_TERMINAL_DROP",
    "WRITE_ATTEMPT_CAP",
    "WRITE_OUTCOMES",
    "WriteAttempt",
    "WriteAttemptLog",
    "digest_artifact_key",
    "entity_artifact_key",
    "essay_artifact_key",
    "load_write_attempts",
    "report",
    "resolve_turn",
    "short_hash",
    "summarize_attempts",
]
