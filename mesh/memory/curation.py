"""Entity, group, and digest self-curation: shared contract and validators.

The agent maintains its own long-term state — entity registry, entity groups,
entity/group dossiers, and the standing digest — continuously, in-process,
using its existing router tool loop, triggered immediately after each
successful memory-formation batch.  See
``docs/plans/entity-self-curation.md``.

This module holds the pieces every participant needs and nothing that needs a
router, an agent node, or a memory system:

* :class:`CurationBatch` — the immutable post-commit notification payload;
* the curation tool-name sets (§3.6) and the phase-aware allowlist;
* :func:`render_update_instruction` — anchor-insertion prompt assembly (§2);
* :class:`CurationExecutionContext` — one turn's mode plus the shadow overlay
  (§3.6);
* the pre-commit validators shared by ``essay_edit`` and ``digest_edit``
  (§6.3, §6.4) and the canonical group-roster renderer (§5.2).

Keeping it separate from ``system_v2`` avoids an import cycle: the memory
system emits batches, the router consumes them, and the agent node registers
handlers for both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable

from .entities import canonical_dossier_hash
from .write_audit import WriteAttemptLog


# ── Tool scopes (§3.6) ────────────────────────────────────────────

#: The three entity mutations landed in Phase 1 plus the reused correction tool.
SELF_CURATION_ENTITY_TOOLS = frozenset({
    "entity_create",
    "entity_merge",
    "entity_edit",
    "entity_link_correct",
})

#: Phase 2 group mutations.  Offered only when group curation is enabled.
SELF_CURATION_GROUP_TOOLS = frozenset({
    "entity_group_create",
    "entity_group_member_add",
    "entity_group_member_remove",
})

#: Every name whose curation execution must go through the scoped dispatcher.
SELF_CURATION_MUTATION_TOOLS = (
    SELF_CURATION_ENTITY_TOOLS
    | SELF_CURATION_GROUP_TOOLS
    | frozenset({"essay_edit", "digest_edit"})
)

#: Read-only evidence and measurement tools.
SELF_CURATION_READ_TOOLS = frozenset({
    "memory_get",
    "memory_search",
    "essay_get",
    "essay_list",
    "digest_get",
    "token_count",
})

#: Names that only a curation scope may execute at all.  ``entity_link_correct``
#: is deliberately absent: interactive write-mode correction predates curation.
SELF_CURATION_ONLY_TOOLS = (
    (SELF_CURATION_ENTITY_TOOLS - {"entity_link_correct"})
    | SELF_CURATION_GROUP_TOOLS
)

#: Convenience arguments on ``entity_link_correct`` that reach
#: ``create_user_named_entity()``.  A curation turn has no user message, so
#: these are rejected inside a curation scope (§3.2).
CURATION_FORBIDDEN_LINK_ARGS = frozenset({
    "new_entity_type",
    "new_display_name",
    "new_identity_note",
    "aliases",
    "naming_surface",
    "memory_patch",
})


def curation_tool_names(*, groups_enabled: bool = False) -> frozenset[str]:
    """Return the phase-aware curation allowlist.

    Phase 1 omits the group tools so the update instruction never asks for a
    mutation the turn cannot execute.
    """
    names = (
        SELF_CURATION_ENTITY_TOOLS
        | SELF_CURATION_READ_TOOLS
        | frozenset({"essay_edit", "digest_edit"})
    )
    if groups_enabled:
        names = names | SELF_CURATION_GROUP_TOOLS
    return frozenset(names)


# ── The batch payload (§4.1) ──────────────────────────────────────


@dataclass(frozen=True)
class CurationBatch:
    """One successful formation commit, as seen by the curation turn.

    Deliberately small and immutable: the renderer resolves each memory ID
    against the committed store rather than trusting a mutable ``MemoryEntry``
    retained across the lock wait.
    """

    reason: str
    memory_ids: tuple[str, ...] = ()
    entity_keys: tuple[str, ...] = ()
    formed_at: str = ""

    def turn_id(self, node_id: str) -> str:
        first = self.memory_ids[0] if self.memory_ids else "empty"
        return f"curation-{node_id}-{self.formed_at}-{first}"


def curation_window_key(turn_id: str) -> str:
    """Derive the durable evidence-window key for one curation batch.

    Entity activation counts ``COUNT(DISTINCT window_key)``: the rule is that
    an entity earns activation by recurring across independent evidence
    occasions rather than in a single burst.  Windowed formation supplies that
    key via ``formation_v3.formation_window_key``; the self-curation path
    linked memories with ``window_key = NULL``, so every link it made was
    invisible to the count and entities it created could never activate.

    A curation batch is the right unit here — ``CurationBatch`` is exactly one
    successful formation commit as seen by the curation turn, so distinct
    batches are distinct evidence occasions in precisely the sense activation
    means.  ``turn_id`` already identifies the batch durably (it is what the
    entity-event log records as ``source_message_id``), so the key is derived
    from it rather than invented: the same batch always yields the same key,
    and a different batch never does.

    Namespaced and shaped like a formation window key (16 hex chars) so the
    two can share a column without a formation key ever colliding with one.
    """
    serialized = "curation-window\x1f" + (turn_id or "")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


# ── Citation and bracket-token validation (§6.3) ──────────────────

#: Any ``[[…]]`` placeholder is a defect in a committed body.  Same regex the
#: offline fold used (``essay_fold.BRACKET_TOKEN_RE``).
BRACKET_TOKEN_RE = re.compile(r"\[\[[^\]\r\n]+\]\]")

#: A well-formed citation as it appears in a dossier or the digest.  The
#: underscore delimiter is canonical: it is what ``memory_get`` accepts, what
#: the constitutions instruct, and what the batch block models.
CITATION_RE = re.compile(r"\[m_([0-9a-fA-F]{4,40})\]")

#: Legacy colon-delimited citation surface.  Some agents' digests were written
#: before the batch block modelled the canonical form, and emit ``[m:<id>]``.
#: Those citations are real memory references — they were simply invisible to
#: :data:`CITATION_RE`, which silently disarmed the ghost-citation gate on
#: every digest that used them.  Readers opt in with ``include_legacy=True``;
#: see :func:`extract_citations` for why the opt-in is not global.
LEGACY_CITATION_RE = re.compile(r"\[m:([0-9a-fA-F]{4,40})\]")

#: Near-miss IDs: a bare ``m_…`` that is not wrapped in brackets, or a bracketed
#: token whose hex run is too short.  ``essay_fold.LOOSE_TAG_RE`` equivalent.
#: Deliberately NOT extended to the colon surface: flagging ``[m:…]`` as
#: malformed would refuse every edit to a legacy digest instead of migrating it.
LOOSE_TAG_RE = re.compile(r"(?<![0-9A-Za-z_])m_([0-9a-fA-F]{1,40})(?![0-9a-fA-F])")


def extract_citations(body: str, *, include_legacy: bool = False) -> list[str]:
    """Return the distinct ``m_<id>`` citations in ``body``, in first-seen order.

    ``include_legacy`` additionally recognises the ``[m:<id>]`` surface and
    normalises it to ``m_<id>``.  It is off by default, and callers enable it
    only where the gate is *resolution-only*:

    * The digest gate checks that each citation resolves against ``memories``.
      Recognising the legacy surface there arms a gate that was passing
      vacuously, and cannot refuse anything a resolvable citation would not
      already have refused.
    * The dossier gate additionally checks that each citation is *in scope*
      for the entity.  Legacy dossiers carry out-of-scope colon citations that
      predate the check, so recognising them there would hard-refuse the next
      edit to an already-published essay.  That path stays canonical-only.

    The parameter is a migration shim: once digests are regenerated in the
    canonical form, ``LEGACY_CITATION_RE`` and this flag can both be dropped.
    """
    seen: dict[str, None] = {}
    pattern = (
        re.compile(f"(?:{CITATION_RE.pattern})|(?:{LEGACY_CITATION_RE.pattern})")
        if include_legacy
        else CITATION_RE
    )
    for match in pattern.finditer(body or ""):
        # With the alternation exactly one group participates in the match.
        ident = next(group for group in match.groups() if group is not None)
        seen.setdefault(f"m_{ident}", None)
    return list(seen)


def find_bracket_tokens(body: str) -> list[str]:
    return BRACKET_TOKEN_RE.findall(body or "")


def find_loose_citations(body: str) -> list[str]:
    """Return ``m_…`` surfaces that are not properly bracketed citations."""
    text = body or ""
    bracketed: set[tuple[int, int]] = set()
    for match in CITATION_RE.finditer(text):
        bracketed.add((match.start(1), match.end(1)))
    loose: list[str] = []
    for match in LOOSE_TAG_RE.finditer(text):
        if (match.start(1), match.end(1)) in bracketed:
            continue
        loose.append(match.group(0))
    return loose


# ── Digest section constitution (§6.4, §7.1) ──────────────────────

#: The seven constitutional sections.  Matching is heading-prefix based so a
#: digest may title a section "Open threads / where-we-are".
REQUIRED_DIGEST_SECTIONS: tuple[str, ...] = (
    "Timeline",
    "Narrative",
    "Projects",
    "People",
    "Standing decisions",
    "Open threads",
    "Agent narrative",
)

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]*(\S.*?)[ \t]*$")


def digest_sections(text: str) -> dict[str, str]:
    """Map each required section name to its body text.

    A section absent from ``text`` is absent from the mapping; a present but
    empty section maps to ``""``.
    """
    lines = (text or "").splitlines()
    headings: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match:
            headings.append((index, len(match.group(1)), match.group(2)))

    found: dict[str, str] = {}
    for position, (index, level, title) in enumerate(headings):
        folded = title.casefold()
        for section in REQUIRED_DIGEST_SECTIONS:
            if section in found:
                continue
            if not folded.startswith(section.casefold()):
                continue
            end = len(lines)
            for next_index, next_level, _ in headings[position + 1:]:
                if next_level <= level:
                    end = next_index
                    break
            found[section] = "\n".join(lines[index + 1:end]).strip()
            break
    return found


def digest_section_errors(text: str) -> list[str]:
    """Return one error string per missing or emptied constitutional section."""
    sections = digest_sections(text)
    errors: list[str] = []
    for section in REQUIRED_DIGEST_SECTIONS:
        if section not in sections:
            errors.append(f"required section {section!r} is missing")
        elif not sections[section]:
            errors.append(f"required section {section!r} is empty")
    return errors


# ── Group roster block (§5.2) ─────────────────────────────────────

ROSTER_BEGIN_RE = re.compile(r"<!--\s*roster:begin[^>]*-->")
ROSTER_END_RE = re.compile(r"<!--\s*roster:end\s*-->")
ROSTER_BLOCK_RE = re.compile(
    r"<!--\s*roster:begin[^>]*-->.*?<!--\s*roster:end\s*-->",
    re.DOTALL,
)


def render_roster_block(members: Iterable[dict[str, Any]]) -> str:
    """Render the protected group-member roster with its content checksum.

    One canonical function so a whitespace change cannot silently invalidate
    every group dossier at once (§11, low risk).
    """
    rows = sorted(
        (
            (
                str(member.get("member_key") or ""),
                str(member.get("display_name") or member.get("member_key") or ""),
                str(member.get("role") or ""),
            )
            for member in members
        ),
        key=lambda row: row[0],
    )
    lines = [f"- {key}  {name}  — {role}" for key, name, role in rows]
    payload = "\n".join(lines)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    parts = [f"<!-- roster:begin sha256={digest} -->"]
    if payload:
        parts.append(payload)
    parts.append("<!-- roster:end -->")
    return "\n".join(parts)


def roster_block_of(body: str) -> str | None:
    match = ROSTER_BLOCK_RE.search(body or "")
    return match.group(0) if match else None


def touches_roster(body: str, fragment: str) -> bool:
    """True when ``fragment`` intersects the roster block inside ``body``."""
    if not fragment:
        return False
    if ROSTER_BEGIN_RE.search(fragment) or ROSTER_END_RE.search(fragment):
        return True
    block = roster_block_of(body)
    if block is None:
        return False
    start = 0
    while True:
        index = (body or "").find(fragment, start)
        if index < 0:
            return False
        block_start = (body or "").find(block)
        block_end = block_start + len(block)
        if index < block_end and index + len(fragment) > block_start:
            return True
        start = index + 1


# ── Shadow overlay + per-turn context (§3.6) ──────────────────────

#: Cap on refused-call records kept for one curation turn, plus a per-message
#: truncation, so a model looping on the same refusal cannot grow the audit
#: row without bound.
REJECTION_LOG_CAP = 20
REJECTION_TEXT_CHARS = 500


@dataclass
class CurationExecutionContext:
    """Authority, mode, and the shadow overlay for one curation turn.

    In ``shadow`` mode every mutation computes its exact post-operation state,
    runs the same validators as ``write`` mode, records an ``entity_events`` row
    whose ``event_type`` is suffixed ``_shadow``, and stores the result in a
    call-local overlay so a following shadow call composes against it.  The
    overlay is discarded when the turn ends; only ``_shadow`` audit rows
    persist.
    """

    mode: str = "shadow"
    batch: CurationBatch | None = None
    trigger_id: str = ""
    actor_node: str = ""
    groups_enabled: bool = False
    entities: dict[str, dict[str, Any]] = field(default_factory=dict)
    links: dict[str, set[str]] = field(default_factory=dict)
    essays: dict[str, dict[str, Any]] = field(default_factory=dict)
    group_members: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    digest_text: str | None = None
    intents: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    provisional_seq: int = 0
    #: Per-write-attempt audit for this turn (G-004).  Records outcomes only;
    #: it never participates in the decision to refuse a write.
    write_log: WriteAttemptLog = field(default_factory=WriteAttemptLog)
    #: Measurement handed up by the write path for the attempt in flight, as
    #: ``(measured_tokens, budget_tokens)``.  Consumed exactly once by the
    #: instrumentation wrapper; see :meth:`note_measurement`.
    pending_measurement: tuple[int | None, int | None] | None = None
    #: Additions refused at an artifact's token ceiling during this turn
    #: (T-001).  Held, not queued: the primary mechanism is a same-turn
    #: compress-and-retry, so whether an addition is really un-landed is only
    #: knowable at turn end.  See :meth:`record_ceiling_refusal`.
    ceiling_refusals: list[dict[str, Any]] = field(default_factory=list)

    @property
    def shadow(self) -> bool:
        return self.mode != "write"

    def record_intent(self, operation: str, detail: dict[str, Any]) -> None:
        self.intents.append({"operation": operation, **detail})

    def record_rejection(self, tool: str, error: str) -> None:
        """Record one refused tool call so the turn's audit row shows it.

        ``entity_events`` records what succeeded and ``details_json.tool_calls``
        records what was attempted; without this a refusal leaves no trace at
        all once the turn ends.
        """
        if len(self.rejections) >= REJECTION_LOG_CAP:
            return
        self.rejections.append(
            {"tool": tool, "error": error[:REJECTION_TEXT_CHARS]}
        )

    def note_measurement(
        self, measured_tokens: int | None, budget_tokens: int | None,
    ) -> None:
        """Hand the in-flight write's measured/budget tokens to the auditor.

        The measurement is taken deep inside the write path — after the
        candidate body is composed but before the ceiling check resolves it —
        while the attempt is classified by the wrapper that called that path.
        Rather than reshaping every return value, the write path drops the two
        numbers here and the wrapper picks them up.  Purely additive: nothing
        reads this to decide anything.
        """
        self.pending_measurement = (measured_tokens, budget_tokens)

    def take_measurement(self) -> tuple[int | None, int | None]:
        """Consume the pending measurement, leaving nothing behind.

        Consuming rather than peeking matters: a write path that returns before
        it measures anything (bad arguments, missing artifact) must not inherit
        the previous attempt's numbers.
        """
        pending = self.pending_measurement
        self.pending_measurement = None
        return pending if pending is not None else (None, None)

    def record_ceiling_refusal(
        self,
        *,
        target_artifact: str,
        tool: str,
        entity_key: str,
        arguments: dict[str, Any],
        measured_tokens: int | None,
        budget_tokens: int | None,
    ) -> None:
        """Hold one over-ceiling refusal for the turn-end queueing sweep.

        Only ceiling refusals are held.  A malformed edit — ``old_text`` that
        does not match, a protected roster patch — is not a pending addition:
        replaying it later would fail exactly the same way.  The distinction
        is made by the caller from the measurement, which is present only when
        the ceiling is what refused the write.
        """
        if len(self.ceiling_refusals) >= REJECTION_LOG_CAP:
            return
        self.ceiling_refusals.append({
            "target_artifact": target_artifact,
            "tool": tool,
            "entity_key": entity_key or "",
            "old_text": str(arguments.get("old_text") or ""),
            "new_text": str(arguments.get("new_text") or ""),
            "replace_all": bool(arguments.get("replace_all", False)),
            "reason": str(arguments.get("reason") or ""),
            "measured_tokens": measured_tokens,
            "budget_tokens": budget_tokens,
        })

    def next_provisional_key(self, entity_type: str, slug: str) -> str:
        """Deterministic provisional key a following shadow call can accept."""
        self.provisional_seq += 1
        return f"{entity_type}:{slug}#shadow{self.provisional_seq}"

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "tool_calls": self.tool_calls,
            "intents": [item["operation"] for item in self.intents],
            "provisional_entities": sorted(self.entities),
            "shadow_essays": sorted(self.essays),
            "digest_touched": self.digest_text is not None,
            "rejections": len(self.rejections),
            "write_attempts": self.write_log.summary(),
        }


# ── The update instruction (§2) ───────────────────────────────────

DIGEST_CONSTITUTION = """\
Your standing digest has seven sections: Timeline, Narrative, Projects,
People, Standing decisions & conventions, Open threads, and Agent
narrative. Maintain all seven; never empty a must-keep section.

Citation format — EXACT. A memory citation is the memory's 12-hex-char
ID wrapped as [m_<id>], with an UNDERSCORE after the m:

    correct:    [m_3f723ef2c694]
    wrong:      [m:3f723ef2c694]   (colon — unreadable to memory_get)
    wrong:      [3f723ef2c694]     (no m_ prefix)
    wrong:      m_3f723ef2c694     (no brackets)

The batch listing above prints every memory in exactly this form. Copy
the handle; never retype or reformat it. Only the canonical form is a
citation: anything else is prose that happens to contain hex, it will
not resolve for a future reader, and it silently bypasses the check
that citations point at real memories.

Citation-to-claim correspondence. A citation attaches to the specific
claim it follows, and must be a memory that supports THAT claim. A true
sentence carrying a citation to a different (even adjacent, even
same-project) memory is a defect: it looks verified and is not, and no
automated check can catch it. If a claim draws on three memories, cite
all three. If you cannot name the memory a claim came from, you are
recalling it rather than reading it — drop the claim or go read the
memory with memory_get.

Per-section discipline:
  Timeline      — append dated one-liners; consolidate ranges when long.
                  STRICTLY CHRONOLOGICAL, oldest first: every bullet's
                  date is >= the date of the bullet above it. Insert new
                  material at its date, not at the end. Before you
                  finish, read the dates top to bottom and fix any pair
                  that goes backwards.
  Narrative     — the compressed ARCS: what happened over time and what
                  it meant. Prose, not bullets. This section carries the
                  causal story the Timeline only indexes — why the work
                  turned when it did, what was learned, what a decision
                  overturned. Every arc cites the memories it rests on.
                  A Narrative that is a few lines long, or has no
                  citations, or stops months before the Timeline does,
                  is a DEFECT, not a compact section: it means arc
                  material was filed into Timeline and Open threads and
                  the synthesis was never written. If you find it in
                  that state, rebuild it from the Timeline you have.
                  Accrete arcs; compress by LIVENESS, not age. A
                  months-old arc that is still load-bearing compresses
                  AFTER a quiet recent one.
  Projects      — per-project record plus repo pointer; merge, don't
                  duplicate.
  People        — roster; merge entries, don't duplicate.
  Standing      — durable rulings ledger; grows monotonically.
  Open threads  — rewrite freely; this section is live state.
  Agent narrative — first-person, 250-300 words, regenerate when the
                  window's work changes the story.

Recent coverage. The most recent weeks are the most load-bearing and
the easiest to skip, because their memories arrive while the digest is
already near budget. Before finishing, check that the last two weeks
are actually represented — a dated span present in the batch but absent
from Timeline and Narrative is a coverage hole, not a compaction
decision. If budget forces a choice, consolidate OLD Timeline entries
into ranges to make room; never buy room by dropping recent work.

Under budget pressure, compact in this order:
  1. Timeline consolidates first.
  2. Narrative compresses second, by liveness.
  3. Must-keep sections are tightened last and never emptied.

An arc may GRADUATE out of Narrative only when all three hold:
  (i)   it is already compressed to conclusion-only;
  (ii)  its thread is resolved or quiet; and
  (iii) its conclusion is safe elsewhere — as a Standing decision or a
        resolvable [m_<id>] memory record.
Condition (iii) is a safety interlock. An insight living only in
Narrative prose does NOT graduate.

Use the space the material justifies; never pad or invent content to
approach the ceiling.\
"""

ESSAY_CONSTITUTION = """\
What an essay is FOR. The essay is the substantive curated artifact of
this whole system — the place the accumulated understanding of one
person, project, or event actually lives. The entity record itself is
only metadata: a key, a type, a display name, an identity note. It
holds no knowledge. Everything a future reader will know about this
entity, they will know because you wrote it here.

That purpose sets the bar. An essay exists to do two jobs no other
artifact does:

  1. RETRIEVAL. Months from now this essay is what gets read instead of
     re-reading hundreds of raw memories. It must carry the context a
     reader needs to act — what happened, in what order, why it went
     that way, what was decided and what was rejected, what is still
     open, and which memories to go read for detail. Specifics are the
     product: dates, names, versions, commit hashes, file paths,
     numbers, verdicts. Generalities that would be true of any entity
     of this type carry nothing.

  2. MERGING. Two entities that turn out to be the same must be
     recognisable as the same, and two that merely share a name must be
     distinguishable. So state the disambiguating facts explicitly:
     other names and handles this entity goes by, its role and
     relationships, when it first and last appears, what it is NOT to
     be confused with. A reader deciding whether to merge has only the
     essays to go on.

A thin essay is a failed essay. A body that restates the entity's name
and type, or that could have been written from the registry alone
without reading a single memory, is not a small essay — it is a missing
one. If the linked evidence is genuinely thin, say what the evidence
supports and stop; do not pad. But if fifty memories are linked and the
essay is a paragraph, the work was not done.

Every essay must observe the following rules.

Structural requirements by entity type:
  Person  — Identity, Timeline, Narrative, Reflection, Open threads.
            Identity carries the merge facts: aliases, handles, role,
            first and last appearance.
            Timeline states cited facts only (no interpretation), in
            strict chronological order, oldest first.
            Narrative interprets but anchors every claim to [m_<id>].
  Project — Goal, Decision arc, Current state, Open issues, Related.
            Decision arc is chronological and cited, and records
            rejected options with the reason, not only what was
            adopted.  Related enumerates people and dependencies
            explicitly, by entity key.
  Event   — What happened, Participants (entity keys), Significance,
            Aftermath. Participants explicitly names every entity
            involved, referenced by key.
  Group   — member roster block is protected: never edit it by hand.

Citation format — EXACT. A citation is the memory's 12-hex-char ID
wrapped as [m_<id>], with an UNDERSCORE after the m:

    correct:    [m_3f723ef2c694]
    wrong:      [m:3f723ef2c694]   (colon — unreadable to memory_get)
    wrong:      [3f723ef2c694]     (no m_ prefix)
    wrong:      m_3f723ef2c694     (no brackets)

Copy each handle exactly as the batch listing prints it; never retype
or reformat it. Every occurrence of a memory handle must use the fully
bracketed canonical form; never mention one as a bare m_<id>. A citation
may only reference a memory LINKED to this entity — an unlinked or
unresolvable citation refuses the whole write.

Citation discipline: every factual or interpretive claim must cite at
least one [m_<id>] memory reference.  Claims without citations are not
allowed — remove them, don't soften them.  The citation must support
the SPECIFIC claim it attaches to: a true sentence carrying a citation
to some other memory of the same project is a defect, because it reads
as verified and is not.  If a claim rests on several memories, cite
them all.

Evidence: write only what linked memories support.  No speculation.
Only ACTIVE entities are publishable — a write against a pending entity
is refused, so link the evidence now and write the essay on a later
turn once the registry shows it active.  When you do write, draw on ALL
the entity's linked memories, not only the ones in the current batch:
read the ones you have not seen with memory_get first.  An essay
composed from the batch alone reproduces the batch, which is the one
thing the essay is not for.

Merge: when entity_merge combines two entities, compare both essays'
evidence.  If they contradict, the newer memory ALWAYS prevails —
discard the older claim.  When an older claim is discarded, add a
brief note recording what was superseded and which newer memory
replaced it.  The merged essay must cite evidence from both sources,
with a merger note when the reconciliation is non-trivial.

Token budget: respect essay_token_budget (default 4000).  Over 80% →
compress before adding.  Over-ceiling writes are refused, not truncated.

Consistency: after updating an essay, check whether the entity's entry
in the digest (People or Projects section) needs a matching update.
The digest should reflect the current state of knowledge the essay
captures.\
"""

GROUP_CONSTITUTION = """\
Every group essay must observe the following rules.

When to create a group: a group represents a collection of entities that
share a relationship, purpose, or affiliation — a research team, a
working group, or the set of collaborators on a specific project.  If the
entity is an individual human, it is a person.  If it is a discrete effort
with deliverables and a timeline, it is a project.  If it is a collection
of related entities, it is a group.  Never create a group speculatively.

Structural requirements:
  Purpose     — why the group exists, with cited evidence.
  Members     — protected roster block; never edit it by hand.  Managed
                exclusively through entity_group_member_add / remove.
  Activity    — cited narrative of group-relevant events.
  Open threads — live state, rewritten freely each curation turn.

Membership: add a member only when a memory explicitly links that entity
to the group's purpose.  Remove only when evidence shows departure or
obsolescence.  Never add speculatively or by association.

Consistency: after updating a group essay, check the digest's Projects
or People sections to ensure they reflect the group's current state if
the group is mentioned there.\
"""

#: Inserted at ``[[GROUPS]]`` only once Phase 2 group tools are offered, so the
#: agent is never instructed to call an unavailable mutation tool (§2.2).
GROUPS_INSTRUCTION_BLOCK = """\
3a. Manage groups when the batch supplies real relationship evidence.
   Create a group only with a clear purpose (entity_group_create); add or
   remove members only through entity_group_member_add /
   entity_group_member_remove. Group activation is deterministic and not
   your decision. Update an active group's essay through essay_get /
   essay_edit; never hand-edit its protected roster block — a patch that
   touches it is refused.  See the group constitution below.\
"""

#: Named anchors replaced by string insertion, never ``str.format()`` — a
#: second out-of-repo consumer renders the same prompt file and a new
#: ``{placeholder}`` raises ``KeyError`` there (the Increment 3 convention).
_ANCHORS = (
    "[[PENDING]]",
    "[[BATCH]]",
    "[[REGISTRY]]",
    "[[BUDGETS]]",
    "[[GROUPS]]",
    "[[GROUP_CONSTITUTION]]",
    "[[CONSTITUTION]]",
    "[[ESSAY_CONSTITUTION]]",
)


UPDATE_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "curation"
    / "self_curation_update.txt"
)


def load_update_template(path: "str | Path | None" = None) -> str:
    """Read the update-instruction template from the prompt library."""
    return Path(path or UPDATE_TEMPLATE_PATH).read_text()


def render_update_instruction(
    template: str,
    *,
    batch_block: str,
    registry_block: str,
    budgets_block: str,
    groups_block: str = "",
    pending_block: str = "",
    constitution: str = DIGEST_CONSTITUTION,
    essay_constitution: str = ESSAY_CONSTITUTION,
    group_constitution: str = GROUP_CONSTITUTION,
) -> str:
    """Assemble the update instruction by replacing named anchors."""
    rendered = template
    # Drain-first (T-001): additions refused at a ceiling on an earlier turn
    # are presented ahead of the new batch, so the model spends its first
    # compaction on work already known to be owed.  An empty queue leaves no
    # section behind at all — the common case must cost nothing.
    pending_section = (
        "─── QUEUED ADDITIONS (LAND THESE FIRST) ───\n\n"
        f"{pending_block.strip()}\n\n"
        if pending_block.strip()
        else ""
    )
    # Keyed with its trailing newline so an empty queue leaves no blank line
    # where the section would have been.
    rendered = rendered.replace("[[PENDING]]\n", pending_section)
    replacements = {
        "[[PENDING]]": pending_section,
        "[[BATCH]]": batch_block.strip(),
        "[[REGISTRY]]": registry_block.strip(),
        "[[BUDGETS]]": budgets_block.strip(),
        "[[GROUPS]]": groups_block.strip(),
        "[[GROUP_CONSTITUTION]]": group_constitution.strip(),
        "[[CONSTITUTION]]": constitution.strip(),
        "[[ESSAY_CONSTITUTION]]": essay_constitution.strip(),
    }
    for anchor, value in replacements.items():
        rendered = rendered.replace(anchor, value)
    return rendered


def render_batch_block(
    batch: CurationBatch,
    rows: list[dict[str, Any]],
    *,
    prior_failed_ids: Iterable[str] = (),
) -> str:
    """Render the ``[BATCH]`` insertion from store-resolved rows.

    Each memory is listed in the *citation surface itself* — ``[m_<id>]``, not
    the bare hex ID.  The listing is the only place the model sees a concrete
    memory reference before it writes one, so a bare ID left it to invent a
    delimiter; agents that guessed ``[m:<id>]`` produced digests whose every
    citation was invisible to the ghost-citation gate.  Modelling the canonical
    form here is what makes the instruction in the constitutions copyable.
    """
    lines = [f"Formation reason: {batch.reason or 'unknown'}"]
    lines.append(f"New memories ({len(rows)}):")
    lines.append(
        "  (cite a memory by copying its [m_<id>] handle exactly as shown)"
    )
    for row in rows:
        entities = row.get("entities") or []
        entity_text = ", ".join(entities) if entities else "none"
        flag = "digest-candidate" if row.get("digest_candidate") else "db-only"
        lines.append(
            f"  [m_{row.get('id', '?')}]  {row.get('retrieval_key', '')!r}"
            f"  [{flag}] [entities: {entity_text}]"
        )
    if not rows:
        lines.append("  (none resolvable — the commit is durable regardless)")
    failed = [str(item) for item in prior_failed_ids if item]
    if failed:
        lines.append("")
        lines.append(
            "Prior failed curation work, carried forward (re-check these): "
            + ", ".join(failed)
        )
    return "\n".join(lines)


def render_budgets_block(
    *,
    digest_tokens: int,
    digest_ceiling: int,
    dossiers: list[tuple[str, int]],
    dossier_ceiling: int,
    degraded_groups: Iterable[str] = (),
    stale_groups: Iterable[str] = (),
    tokenizer_ok: bool = True,
) -> str:
    """Render the measured ``[BUDGETS]`` insertion (§6.6)."""
    digest_hard = (
        " (net contraction required before any addition)"
        if digest_tokens >= int(digest_ceiling * 0.85)
        else ""
    )
    lines = [
        f"Standing digest:  {digest_tokens} / {digest_ceiling} tokens"
        f"{digest_hard}"
    ]
    if not tokenizer_ok:
        lines.append(
            "WARNING: tiktoken is unavailable; sizes are word-heuristic "
            "estimates and no candidate can be certified against budget."
        )
    threshold = int(dossier_ceiling * 0.8)
    over = [(key, size) for key, size in dossiers if size >= threshold]
    if over:
        lines.append(
            f"Dossiers over 80% of their {dossier_ceiling}-token ceiling:"
        )
        for key, size in sorted(over, key=lambda item: -item[1]):
            hard = " (net contraction required before any addition)" if (
                size >= int(dossier_ceiling * 0.85)
            ) else ""
            lines.append(f"  {key}  {size} tokens{hard}")
    degraded = [str(item) for item in degraded_groups if item]
    if degraded:
        lines.append(
            "Degraded groups (fewer than 2 active members; merge or retire "
            "explicitly, nothing is auto-retired): " + ", ".join(degraded)
        )
    stale = [str(item) for item in stale_groups if item]
    if stale:
        lines.append(
            "Stale pending groups with no new bridge evidence — consider "
            "retiring or merging: " + ", ".join(stale)
        )
    return "\n".join(lines)


# ── Phase 3 agent-driven backfill (§9, "Phase 3") ─────────────────

#: Formation reason stamped on every backfill slice.  It flows into the
#: ``[BATCH]`` block ("Formation reason: backfill") so the model can tell an
#: old-history repair slice from a live post-commit batch, and into the
#: ``curation_turn`` audit row through ``details["formation_reason"]``.
CURATION_BACKFILL_REASON = "backfill"

#: Rows the backfill walk never treats as curation input.  Claim 40 keeps
#: parse-failure placeholders out of the live trigger path; the same record
#: must not re-enter curation through the back door.
BACKFILL_EXCLUDED_FORMATION_SOURCES = frozenset({"live-extraction-fallback"})


def slice_backfill_batches(
    rows: Iterable[tuple[str, str]],
    *,
    curated_ids: Iterable[str] = (),
    slice_size: int = 10,
    max_batches: int = 50,
) -> list[CurationBatch]:
    """Chunk an oldest-first memory walk into bounded backfill slices.

    ``rows`` is ``(memory_id, created_at)`` ordered oldest-first.  The walk has
    two distinct reactions to an already-curated memory, and the difference is
    what makes backfill resumable:

    * a curated memory encountered **before** any uncurated one is *skipped*.
      Otherwise the second invocation would halt on the prefix the first one
      just curated and every run after the first would be a no-op;
    * a curated memory encountered **after** collection has started *stops* the
      walk.  That row is the frontier between the uncurated backlog and the
      live-curated era, and backfill never leapfrogs it.

    The result is at most ``max_batches`` slices of at most ``slice_size``
    memories each — the safety ceiling that keeps a single invocation from
    curating the agent's entire history.  A trailing partial slice is kept; a
    remainder beyond ``max_batches`` is dropped and picked up by the next run.
    """
    if slice_size < 1 or max_batches < 1:
        return []
    curated = set(curated_ids)
    collected: list[tuple[str, str]] = []
    for memory_id, created_at in rows:
        if memory_id in curated:
            if collected:
                break          # frontier reached — stop, do not leapfrog
            continue           # already-backfilled prefix — skip, keep walking
        collected.append((memory_id, str(created_at or "")))
        if len(collected) >= slice_size * max_batches:
            break

    batches: list[CurationBatch] = []
    for start in range(0, len(collected), slice_size):
        window = collected[start:start + slice_size]
        if not window:
            continue
        batches.append(
            CurationBatch(
                reason=CURATION_BACKFILL_REASON,
                memory_ids=tuple(item[0] for item in window),
                entity_keys=(),
                formed_at=window[0][1],
            )
        )
        if len(batches) >= max_batches:
            break
    return batches


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "BACKFILL_EXCLUDED_FORMATION_SOURCES",
    "BRACKET_TOKEN_RE",
    "CITATION_RE",
    "ESSAY_CONSTITUTION",
    "CURATION_BACKFILL_REASON",
    "CURATION_FORBIDDEN_LINK_ARGS",
    "CurationBatch",
    "CurationExecutionContext",
    "WriteAttemptLog",
    "DIGEST_CONSTITUTION",
    "LEGACY_CITATION_RE",
    "LOOSE_TAG_RE",
    "REQUIRED_DIGEST_SECTIONS",
    "ROSTER_BLOCK_RE",
    "SELF_CURATION_ENTITY_TOOLS",
    "SELF_CURATION_GROUP_TOOLS",
    "SELF_CURATION_MUTATION_TOOLS",
    "SELF_CURATION_ONLY_TOOLS",
    "SELF_CURATION_READ_TOOLS",
    "canonical_dossier_hash",
    "curation_tool_names",
    "curation_window_key",
    "digest_section_errors",
    "digest_sections",
    "GROUPS_INSTRUCTION_BLOCK",
    "GROUP_CONSTITUTION",
    "UPDATE_TEMPLATE_PATH",
    "extract_citations",
    "find_bracket_tokens",
    "find_loose_citations",
    "load_update_template",
    "render_batch_block",
    "render_budgets_block",
    "render_roster_block",
    "render_update_instruction",
    "roster_block_of",
    "slice_backfill_batches",
    "touches_roster",
    "utcnow_iso",
]
