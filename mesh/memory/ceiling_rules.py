"""The compaction contract carried by an over-ceiling refusal (T-001 / G-001).

Over-ceiling writes to a curated artifact are refused, never truncated (D-003).
That is correct, but a refusal is only actionable if the model is told *how* to
make room.  Until now both refusals pointed at "the constitution" without
carrying it, and a live curation turn does not have the fold driver's
constitution in context — the fold driver injects it, the curation turn does
not.  So the pointer resolved to nothing and the addition was dropped.

This module holds the pointer's target: the budget-pressure order, distilled
from the authoritative sources and worded to be actionable inside a tool
result.

Sources
-------
``mesh/fold_driver/prompts/standing_digest/constitution.txt``
    "## Must-keep categories and budget-pressure order" (Timeline first,
    Narrative second by liveness, must-keep tightened last and never emptied)
    and "## Narrative lifecycle and graduation" (condition (iii): never delete
    the sole surviving copy of a judgment).
``mesh/fold_driver/prompts/standing_digest/fold_step_edit.txt``
    Rule 3 (react to real measurements, never estimates) and rule 7 (compact
    only when actually over the ceiling; never pad toward it).

Why inline rather than injected
-------------------------------
Injecting the whole constitution into every curation turn would spend ~2.5K
prompt tokens per turn on text that is mostly fold-mechanism-specific (file
handles, ``[[Mn]]`` citation handles, the fold-completion envelope) and wrong
for a live turn — and it still would not be attached to the refusal, which is
the moment the rules are needed.  Carrying a compact, artifact-scoped summary
in the refusal itself is self-contained, costs nothing on turns that never hit
a ceiling, and is directly testable.

This module is deliberately dependency-free so both write paths can import it:
``agent_node`` for the digest and ``memory.entities`` for the dossier.
``memory.curation`` already imports ``memory.entities``, so anything the
dossier path needs cannot live in ``curation``.
"""

from __future__ import annotations

#: The measurement discipline, shared by both artifacts.  Stated last in a
#: refusal because it governs what the model does *after* it edits.
_MEASUREMENT_RULE = (
    "Re-measure with token_count after each edit and react to the "
    "measurement — do not estimate your own output length. Compact only as "
    "far as the ceiling requires: never pad, and never invent content."
)

#: Budget-pressure order for the standing digest's seven-section skeleton.
DIGEST_COMPACTION_RULES = f"""\
COMPACTION — the budget-pressure order (standing-digest constitution). Make \
room, then retry this same write in this turn.
  1. Timeline FIRST. Merge runs of older quiet one-liners into date-range \
span entries ("**2026-03-27-31:** ..."), oldest quiet runs first. A span must \
preserve every [m_<id>] tag its source lines carried and must not flatten an \
anomaly — the outlier stays named inside the span. Take this step before \
touching Narrative.
  2. Narrative SECOND. Compress by LIVENESS, not age: oldest QUIET arcs \
first, preserving each arc's interpretive conclusion over its event detail. \
An arc still referenced by Open threads or touched by recent Timeline is \
compressed only after the quiet arcs are exhausted, and a live arc's core is \
never reduced below a full paragraph.
  3. Must-keep sections LAST (People, Projects, Open threads, Agent \
narrative, Essays). Tighten by dropping the LEAST IMPORTANT content by your \
judgment of the whole catalog — never the merely oldest — and NEVER empty a \
section.
An arc leaves Narrative only once its conclusion survives as a resolvable \
[m_<id>] memory record. Never delete the sole surviving copy of a judgment.
{_MEASUREMENT_RULE}"""

#: Budget-pressure order for an entity dossier.  The digest's seven-section
#: skeleton does not apply, but the ordering principle does: quiet detail goes
#: before live detail, and conclusions outrank events.
DOSSIER_COMPACTION_RULES = f"""\
COMPACTION — the budget-pressure order. Make room, then retry this same write \
in this turn.
  1. Quiet material FIRST. Merge runs of one-event sentences about the same \
settled thread into a single sentence that keeps the conclusion and every \
[m_<id>] tag those sentences carried.
  2. Conclusions OVER event detail. Compress by LIVENESS, not age: a live \
thread keeps its detail until the quiet ones are exhausted. Keep what a \
future reader needs in order to act; drop the blow-by-blow.
  3. Standing sections LAST (purpose/identity, open threads) — tighten them, \
never empty them.
Never drop a claim's citation to save room: an uncited claim is removed, not \
softened. Never delete the sole surviving copy of a judgment — if a \
conclusion survives only here, keep it and cut elsewhere.
{_MEASUREMENT_RULE}"""

#: Stated on every refusal.  The guarantee is two-sided and the model needs
#: both halves: the write will not be silently truncated, and a refusal it
#: does not resolve is not silently lost either.
CARRY_FORWARD_NOTE = (
    "The write is refused, never truncated. If you cannot land it this turn "
    "it is queued durably and offered back on a later curation turn, so "
    "nothing is lost — but landing it now is better."
)


def digest_ceiling_refusal(measured: int, ceiling: int) -> str:
    """The refusal returned when a ``digest_edit`` would exceed the ceiling."""
    return (
        f"Error: digest would be {int(measured)} tokens, over the "
        f"{int(ceiling)}-token ceiling by {int(measured) - int(ceiling)}. "
        f"{CARRY_FORWARD_NOTE}\n\n{DIGEST_COMPACTION_RULES}"
    )


def dossier_ceiling_refusal(measured: int, budget: int) -> str:
    """The refusal message body when a dossier publish exceeds its budget.

    Returned without an ``Error:`` prefix because the dossier path raises this
    as an :class:`~mesh.memory.entities.EntityError` and the caller prefixes
    it, whereas the digest path returns its refusal string directly.
    """
    return (
        f"dossier is {int(measured)} tokens, over the {int(budget)}-token "
        f"ceiling by {int(measured) - int(budget)}. {CARRY_FORWARD_NOTE}\n\n"
        f"{DOSSIER_COMPACTION_RULES}"
    )


__all__ = [
    "CARRY_FORWARD_NOTE",
    "DIGEST_COMPACTION_RULES",
    "DOSSIER_COMPACTION_RULES",
    "digest_ceiling_refusal",
    "dossier_ceiling_refusal",
]
