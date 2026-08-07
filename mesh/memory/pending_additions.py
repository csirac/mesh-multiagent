"""Durable pending-additions ledger for over-ceiling writes (T-001 / G-001).

Compress-then-write is the primary mechanism: an over-ceiling refusal now
carries the budget-pressure order (see :mod:`mesh.memory.ceiling_rules`) so the
router LLM can make room and retry inside the same turn.  This module is the
guarantee layer underneath it — for the turn where the model declines to
compact, or compacts and still cannot fit.

The contract is narrow and absolute: **an over-ceiling addition is never a
terminal drop.**  It either lands, or it is written here and offered back at
the top of a later curation turn until it lands.

Where it lives
--------------
One row per un-landed addition in ``curation_pending_additions``, a table in
the agent's own ``MemoryStore`` database.  That choice is deliberate:

* it is per-agent and durable by construction — the same file that already
  holds the memories, the essays and the ``entity_events`` audit trail, so a
  restart, a backup and a copy all carry the queue with everything else;
* resolution is transactional against the same connection the write path
  already holds, so a landed write and its ledger row cannot disagree;
* the T-004 audit trail (``entity_events``) and this queue are read together
  when computing drop rate, and a single store keeps that a plain join rather
  than a cross-file reconciliation.

A JSON file under ``~/.mesh`` was the alternative.  It is simpler to inspect
but gives none of the three properties above.

How a row resolves
------------------
Rows are queued at **turn end**, not at refusal time.  Refusal time is the
wrong moment: the primary mechanism is a same-turn retry, so queueing on
refusal would enqueue additions the model lands seconds later and every one
would need dequeuing.  At turn end the question is already answered — the
addition is queued only if its ``new_text`` is absent from the artifact's final
content.

A queued row is resolved by :meth:`PendingAdditionLedger.resolve_landed`, which
the write path calls after every landed write: an exact ``new_text`` match is
``landed``.  A compression rewrite can instead be ``superseded``, but only with
the original citation set plus a distinctive exact marker still present in the
committed artifact.  A row without either proof stays pending and is offered
again — the queue drains only on evidence, never on optimism.  Rendered
``_clip`` previews are refused at the queue boundary so the durable payload is
never lossy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import re
from typing import Any, Sequence

logger = logging.getLogger(__name__)

#: Row states.  ``pending`` is the only state that is ever re-offered.
STATUS_PENDING = "pending"
STATUS_LANDED = "landed"
STATUS_SUPERSEDED = "superseded"

#: How many pending additions are rendered into one curation turn's drain
#: block.  The block competes for prompt budget with the batch itself, and a
#: model handed thirty queued edits at once lands none of them.  Overflow is
#: never hidden: the block states how many rows were held back.
DRAIN_RENDER_CAP = 8

#: Per-field truncation inside the rendered block.  The full text stays in the
#: row; this only bounds what one turn's prompt carries.
RENDER_TEXT_CHARS = 600

#: Hard ceiling on pending rows per artifact.  A pathological artifact that
#: refuses everything cannot grow the queue without bound.  When the cap is
#: reached the OLDEST row is kept and the new one is rejected, because the
#: oldest has been waiting longest and dropping it would be the very silent
#: loss this ledger exists to prevent.
MAX_PENDING_PER_ARTIFACT = 50


# ``_clip`` is deliberately only a render-time prompt-budget boundary.  This
# exact suffix identifies its lossy output and is therefore forbidden in the
# durable ``new_text`` payload.
_CLIPPED_PREVIEW_RE = re.compile(
    r"… \(\+\d+ chars; full text on file\)"
)
_CITATION_MARKER_RE = re.compile(r"\[m_[A-Za-z0-9]+\]")
_STABLE_IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Za-z][A-Za-z0-9_]*-\d{1,5}|[0-9a-f]{7,40}|"
    r"\d{1,5}/\d{1,5}|\d{3,}(?:,\d{3})*)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_MAX_SUBSTANTIVE_MARKERS = 24
_PHRASE_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
        "was", "were", "with",
    }
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PendingAddition:
    """One curated-artifact addition that was refused and not landed."""

    rowid: int
    agent: str
    target_artifact: str
    tool: str
    entity_key: str
    old_text: str
    new_text: str
    replace_all: bool
    reason: str
    measured_tokens: int | None
    budget_tokens: int | None
    origin_turn_id: str
    status: str
    offers: int
    created_at: str
    updated_at: str

    def as_details(self) -> dict[str, Any]:
        """Payload recorded on the ``queued`` write-attempt event."""
        return {
            "pending_id": self.rowid,
            "target_artifact": self.target_artifact,
            "tool": self.tool,
            "entity_key": self.entity_key,
            "origin_turn_id": self.origin_turn_id,
            "measured_tokens": self.measured_tokens,
            "budget_tokens": self.budget_tokens,
            "reason": self.reason,
        }


class PendingAdditionLedger:
    """Durable queue of over-ceiling additions for one agent.

    Wraps the store's ``sqlite3`` connection directly.  Every method is
    self-contained and commits its own work, so a caller inside the write path
    cannot leave the queue half-updated.
    """

    def __init__(self, connection: Any, *, agent: str = ""):
        self._conn = connection
        self._agent = agent or ""

    # ── writing ──────────────────────────────────────────────────

    def queue(
        self,
        *,
        target_artifact: str,
        tool: str,
        new_text: str,
        old_text: str = "",
        entity_key: str = "",
        replace_all: bool = False,
        reason: str = "",
        measured_tokens: int | None = None,
        budget_tokens: int | None = None,
        origin_turn_id: str = "",
    ) -> PendingAddition | None:
        """Durably record one un-landed addition.

        Returns the stored row, or ``None`` when the addition is already
        queued (same artifact and same ``new_text``) or the per-artifact cap
        is reached.  Re-queueing an identical addition is a no-op rather than
        an error: a model that hits the same ceiling on three consecutive
        turns should leave one row behind, not three.
        """
        if not target_artifact or not new_text:
            return None
        if _is_clipped_preview(new_text):
            # Refuse rather than persist a prompt rendering as the source of
            # truth.  The caller must recover the original payload; a clipped
            # preview cannot meet the ledger's no-loss guarantee.
            logger.warning(
                "refusing clipped pending-addition preview for %s",
                target_artifact,
            )
            return None
        duplicate = self._conn.execute(
            "SELECT 1 FROM curation_pending_additions "
            "WHERE target_artifact = ? AND new_text = ? AND status = ?",
            (target_artifact, new_text, STATUS_PENDING),
        ).fetchone()
        if duplicate is not None:
            return None
        (depth,) = self._conn.execute(
            "SELECT COUNT(*) FROM curation_pending_additions "
            "WHERE target_artifact = ? AND status = ?",
            (target_artifact, STATUS_PENDING),
        ).fetchone()
        if int(depth) >= MAX_PENDING_PER_ARTIFACT:
            return None

        now = _utcnow()
        cursor = self._conn.execute(
            "INSERT INTO curation_pending_additions "
            "(agent, target_artifact, tool, entity_key, old_text, new_text, "
            " replace_all, reason, measured_tokens, budget_tokens, "
            " origin_turn_id, status, offers, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                self._agent,
                target_artifact,
                tool,
                entity_key or "",
                old_text or "",
                new_text,
                1 if replace_all else 0,
                reason or "",
                None if measured_tokens is None else int(measured_tokens),
                None if budget_tokens is None else int(budget_tokens),
                origin_turn_id or "",
                STATUS_PENDING,
                0,
                now,
                now,
            ),
        )
        self._conn.commit()
        return self.get(int(cursor.lastrowid))

    def resolve_landed(
        self, target_artifact: str, content: str, *, turn_id: str = "",
    ) -> list[int]:
        """Close every pending row for ``target_artifact`` now present in it.

        Called after each landed write.  Containment is the evidence: if the
        committed artifact contains a queued row's ``new_text``, that addition
        is in the artifact and the row is ``landed``.  A compression rewrite
        cannot satisfy that exact check, so it may instead resolve as
        ``superseded`` only when *all* cited-memory markers from the queued text
        and at least one deterministic substantive marker are still present.
        Anything else stays pending — the queue never closes a row on the
        assumption that a write "probably" covered it.

        Returns the ids of the rows that closed.
        """
        if not target_artifact or not content:
            return []
        rows = self._conn.execute(
            "SELECT id, new_text FROM curation_pending_additions "
            "WHERE target_artifact = ? AND status = ?",
            (target_artifact, STATUS_PENDING),
        ).fetchall()
        resolutions: list[tuple[int, str]] = []
        for rowid, new_text in rows:
            if not new_text:
                continue
            if new_text in content:
                resolutions.append((int(rowid), STATUS_LANDED))
            elif _is_compression_supersession(new_text, content):
                resolutions.append((int(rowid), STATUS_SUPERSEDED))
        if not resolutions:
            return []
        now = _utcnow()
        self._conn.executemany(
            "UPDATE curation_pending_additions "
            "SET status = ?, updated_at = ?, resolved_turn_id = ? "
            "WHERE id = ?",
            [
                (status, now, turn_id or "", rowid)
                for rowid, status in resolutions
            ],
        )
        self._conn.commit()
        return [rowid for rowid, _ in resolutions]

    def note_offered(self, rowids: Sequence[int]) -> None:
        """Increment the offer counter for rows rendered into a turn.

        A row's ``offers`` count is how a stuck addition becomes visible: an
        addition offered ten times and still pending is a real signal about the
        artifact, not about the queue.
        """
        if not rowids:
            return
        now = _utcnow()
        self._conn.executemany(
            "UPDATE curation_pending_additions "
            "SET offers = offers + 1, updated_at = ? WHERE id = ?",
            [(now, int(rowid)) for rowid in rowids],
        )
        self._conn.commit()

    # ── reading ──────────────────────────────────────────────────

    def get(self, rowid: int) -> PendingAddition | None:
        row = self._conn.execute(
            "SELECT id, agent, target_artifact, tool, entity_key, "
            "old_text, new_text, replace_all, reason, measured_tokens, "
            "budget_tokens, origin_turn_id, status, offers, created_at, "
            "updated_at FROM curation_pending_additions WHERE id = ?",
            (int(rowid),),
        ).fetchone()
        return None if row is None else _row_to_addition(row)

    def pending(self, *, limit: int | None = None) -> list[PendingAddition]:
        """Oldest-first pending rows.

        Oldest-first is the fairness rule: an addition that has waited longest
        is offered first, so nothing starves behind a stream of newer refusals.
        """
        sql = (
            "SELECT id, agent, target_artifact, tool, entity_key, "
            "old_text, new_text, replace_all, reason, measured_tokens, "
            "budget_tokens, origin_turn_id, status, offers, created_at, "
            "updated_at FROM curation_pending_additions "
            "WHERE status = ? ORDER BY id"
        )
        params: list[Any] = [STATUS_PENDING]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [_row_to_addition(row) for row in rows]

    def pending_count(self) -> int:
        (count,) = self._conn.execute(
            "SELECT COUNT(*) FROM curation_pending_additions WHERE status = ?",
            (STATUS_PENDING,),
        ).fetchone()
        return int(count)


def _row_to_addition(row: Sequence[Any]) -> PendingAddition:
    return PendingAddition(
        rowid=int(row[0]),
        agent=row[1] or "",
        target_artifact=row[2] or "",
        tool=row[3] or "",
        entity_key=row[4] or "",
        old_text=row[5] or "",
        new_text=row[6] or "",
        replace_all=bool(row[7]),
        reason=row[8] or "",
        measured_tokens=None if row[9] is None else int(row[9]),
        budget_tokens=None if row[10] is None else int(row[10]),
        origin_turn_id=row[11] or "",
        status=row[12] or "",
        offers=int(row[13] or 0),
        created_at=row[14] or "",
        updated_at=row[15] or "",
    )


def _clip(text: str, limit: int = RENDER_TEXT_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"… (+{len(text) - limit} chars; full text on file)"


def _is_clipped_preview(text: str) -> bool:
    """Whether ``text`` is the lossy output of :func:`_clip`."""
    return bool(_CLIPPED_PREVIEW_RE.search(text or ""))


def _is_compression_supersession(new_text: str, content: str) -> bool:
    """Require citation and substantive evidence before closing a rewrite.

    Compression is intentionally allowed to rewrite prose, so semantic matching
    would be tempting here.  It would also make a false closure hard to audit.
    This deliberately uses only deterministic evidence: every citation marker
    in the queued text must survive and one distinctive identifier or exact
    multiword phrase must survive as well.  Rows without citation markers fail
    closed and remain pending.
    """
    citations = set(_CITATION_MARKER_RE.findall(new_text or ""))
    content_citations = set(_CITATION_MARKER_RE.findall(content))
    if not citations or not citations.issubset(content_citations):
        return False
    return any(
        _marker_in_content(marker, content)
        for marker in _substantive_markers(new_text)
    )


def _substantive_markers(text: str) -> set[str]:
    """Return bounded, exact-matchable evidence that survives compaction."""
    without_citations = _CITATION_MARKER_RE.sub(" ", text or "")
    markers: list[str] = []
    seen: set[str] = set()

    def add(marker: str) -> None:
        if marker not in seen and len(markers) < _MAX_SUBSTANTIVE_MARKERS:
            markers.append(marker)
            seen.add(marker)

    for match in _STABLE_IDENTIFIER_RE.finditer(without_citations):
        add(match.group(0).lower())

    # Three adjacent informative words are specific enough to pair with the
    # complete citation-set requirement.  Longer phrases add no extra safety
    # while making normal prose compaction unnecessarily brittle.
    words = [word.lower() for word in _WORD_RE.findall(without_citations)]
    for index in range(len(words) - 2):
        phrase_words = words[index:index + 3]
        if (
            len(" ".join(phrase_words)) >= 18
            and all(
                len(word) >= 4 and word not in _PHRASE_STOP_WORDS
                for word in phrase_words
            )
        ):
            add(" ".join(phrase_words))
    return set(markers)


def _marker_in_content(marker: str, content: str) -> bool:
    """Match a stable identifier or normalized exact phrase in ``content``."""
    if " " not in marker:
        return bool(
            re.search(
                rf"(?<![A-Za-z0-9]){re.escape(marker)}(?![A-Za-z0-9])",
                content,
                re.I,
            )
        )
    normalized = " ".join(word.lower() for word in _WORD_RE.findall(content))
    return f" {marker} " in f" {normalized} "


def render_pending_block(
    additions: Sequence[PendingAddition], *, total_pending: int | None = None,
) -> str:
    """Render the drain block inserted at the top of a curation turn.

    Empty string when nothing is queued, so a turn with a clean queue carries
    no drain section at all.
    """
    if not additions:
        return ""
    total = len(additions) if total_pending is None else int(total_pending)
    held_back = max(0, total - len(additions))

    lines = [
        "These additions were refused at an artifact's token ceiling on an "
        "earlier turn and have NOT landed yet. They are queued, not lost. "
        "Land them FIRST, before the new memories below: make room per the "
        "compaction rules in the constitutions, then re-issue each edit.",
        "",
        f"{total} queued addition(s)"
        + (f"; {held_back} held back for a later turn." if held_back else "."),
        "",
    ]
    for addition in additions:
        label = addition.entity_key or addition.target_artifact
        measured = (
            f"{addition.measured_tokens}/{addition.budget_tokens} tokens"
            if addition.measured_tokens is not None
            and addition.budget_tokens is not None
            else "size unrecorded"
        )
        lines.append(
            f"[queued #{addition.rowid}] {addition.tool} → {label} "
            f"(refused at {measured}; offered {addition.offers}× so far)"
        )
        if addition.reason:
            lines.append(f"  reason: {addition.reason}")
        if addition.old_text:
            lines.append(f"  old_text: {_clip(addition.old_text)!r}")
        lines.append(f"  new_text: {_clip(addition.new_text)!r}")
        if addition.replace_all:
            lines.append("  replace_all: true")
        lines.append("")

    lines.append(
        "If an addition is genuinely obsolete — its content already appears "
        "in the artifact in other words, or a later memory superseded it — "
        "land the superseding version instead; the queued row closes only "
        "when its exact text, or its citations plus a distinctive marker, "
        "are present in the artifact."
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "DRAIN_RENDER_CAP",
    "MAX_PENDING_PER_ARTIFACT",
    "RENDER_TEXT_CHARS",
    "STATUS_LANDED",
    "STATUS_PENDING",
    "STATUS_SUPERSEDED",
    "PendingAddition",
    "PendingAdditionLedger",
    "render_pending_block",
]
