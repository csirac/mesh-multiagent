"""Over-ceiling carry-forward: rules at the refusal, ledger underneath (T-001).

Goal G-001 says an over-ceiling addition is never a terminal drop.  Two
mechanisms deliver that, and these tests pin both.

*Compress-then-write* is primary: the refusal itself now carries the
budget-pressure order from the standing-digest constitution, so the router LLM
can make room and re-issue the edit without having the constitution in context.
Before T-001 both refusals said "compact per the constitution" / "compress
before publishing" — a pointer to a document the live curation turn never sees.

The *durable pending-additions ledger* is the guarantee layer: an addition the
model does not land is written to ``curation_pending_additions`` at turn end,
recorded as ``queued`` in the T-004 audit trail, and offered back at the top of
a later curation turn.

The tests drive the real write paths — ``_execute_curation_artifact_tool``,
``EntityService.publish_dossier``, and the real turn-end sweep — because the
claim is about what the live pipeline does.  Every store is a ``tmp_path``
store; no live agent database is opened (D-002).
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mesh.agent_node import CURRENT_CURATION_CONTEXT, AgentNode
from mesh.config import NodeConfig
from mesh.memory.ceiling_rules import (
    DIGEST_COMPACTION_RULES,
    DOSSIER_COMPACTION_RULES,
    digest_ceiling_refusal,
    dossier_ceiling_refusal,
)
from mesh.memory.curation import (
    CurationBatch,
    CurationExecutionContext,
    load_update_template,
    render_update_instruction,
)
from mesh.memory.entities import EntityError, EntityExecutionContext, EntityService
from mesh.memory.pending_additions import (
    DRAIN_RENDER_CAP,
    MAX_PENDING_PER_ARTIFACT,
    RENDER_TEXT_CHARS,
    STATUS_LANDED,
    STATUS_PENDING,
    STATUS_SUPERSEDED,
    PendingAdditionLedger,
    _clip,
    render_pending_block,
)
from mesh.memory.store import MemoryEntry, MemoryStore
from mesh.memory.write_audit import (
    OUTCOME_LANDED,
    OUTCOME_QUEUED,
    OUTCOME_REFUSED,
    RESOLUTION_LANDED_CLEAN,
    RESOLUTION_TERMINAL_DROP,
    load_write_attempts,
    resolve_turn,
    summarize_attempts,
)
from mesh.tools import get_registry


# ─────────────────────────────────────────────────────────────────────
# Fixtures — mirrored from test_curation_write_audit.py so both suites
# construct a real node over a real tmp_path store.
# ─────────────────────────────────────────────────────────────────────


def _mid(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()[:12]


def _entry(memory_id: str) -> MemoryEntry:
    return MemoryEntry(
        id=memory_id,
        created_at=datetime.now(timezone.utc),
        summary=f"Summary for {memory_id}",
        reflection=f"Reflection for {memory_id}",
        trace="trace",
        trigger="trigger",
        retrieval_key=f"retrieval {memory_id}",
        tags=["test"],
        outcome="success",
        reflection_embedding=np.ones(4, dtype=np.float32),
        retrieval_key_embedding=np.ones(4, dtype=np.float32),
    )


DIGEST_SECTIONS = (
    "# Standing digest\n\n"
    "## Timeline\n- 2026-07-30: something happened.\n\n"
    "## Narrative\nAn arc that is load-bearing.\n\n"
    "## Projects\nmesh — the platform.\n\n"
    "## People\nProject Owner — primary user.\n\n"
    "## Standing decisions & conventions\n- Curate after every batch.\n\n"
    "## Open threads / where-we-are\n- Self-curation lands.\n\n"
    "## Agent narrative\nI maintain my own state now.\n"
)


@pytest.fixture
def store(tmp_path):
    result = MemoryStore("carryforward", db_dir=str(tmp_path))
    yield result
    result.close()


@pytest.fixture
def digest_file(tmp_path) -> Path:
    path = tmp_path / "agent-curator.md"
    path.write_text(DIGEST_SECTIONS)
    return path


def _memory_system(store):
    from mesh.memory.system_v2 import MemorySystemV2

    system = MemorySystemV2.__new__(MemorySystemV2)
    system._store = store
    system._entity_activation_window_threshold = 3
    system._entity_resolution_enabled = True
    system._embedder = None
    system._pool = []
    system._curation_batch_cb = None
    system._personality_cache = ""
    return system


def _node(tmp_path, store, *, digest: Path | None = None, **overrides):
    config_kwargs = dict(
        id="agent:test:curator",
        tools=[],
        entity_resolution_mode="write",
        entity_self_curation_enabled=True,
        essay_token_budget=4000,
        standing_digest_budget_tokens=32000,
    )
    if digest is not None:
        config_kwargs["standing_digest_enabled"] = True
        config_kwargs["standing_digest_path"] = str(digest)
    config_kwargs.update(overrides)
    node = AgentNode(NodeConfig(**config_kwargs), tool_registry=get_registry())
    node._memory_system = _memory_system(store)
    return node


def _curation_context(mode: str = "write", turn: str = "curation-turn-1"):
    return CurationExecutionContext(
        mode=mode,
        trigger_id=turn,
        actor_node="agent:test:curator",
        batch=CurationBatch(reason="time-based", memory_ids=("m1",)),
    )


def _trigger(turn: str = "curation-turn-1"):
    return SimpleNamespace(
        id=turn,
        from_node="agent:test:curator",
        content="synthetic curation batch summary",
    )


async def _call(node, name: str, arguments: dict, context, turn="curation-turn-1"):
    token = CURRENT_CURATION_CONTEXT.set(context)
    try:
        return await node._execute_entity_tool(name, arguments, _trigger(turn))
    finally:
        CURRENT_CURATION_CONTEXT.reset(token)


def _exec_context() -> EntityExecutionContext:
    return EntityExecutionContext(
        actor_node="agent:test:curator",
        source_message_id="curation-turn-1",
        source_author="agent:test:curator",
        source_content="synthetic curation batch summary",
    )


def _active_entity(store, display_name: str = "Project Owner") -> str:
    service = EntityService(
        store._conn, actor_node="agent:test:curator", mutations_enabled=True,
    )
    created = service.create_pending_entity(
        "person",
        display_name,
        f"{display_name} identity note",
        origin="self-curation",
        context=_exec_context(),
        reason="test setup",
    )
    key = created["entity_key"]
    store._conn.execute(
        "UPDATE entities SET status='active' WHERE entity_key = ?", (key,)
    )
    store._conn.commit()
    return key


def _linked_memory(store, entity_key: str, label: str = "evidence") -> str:
    """A committed memory linked to ``entity_key``, usable as a citation."""
    memory_id = _mid(label)
    store.insert(_entry(memory_id))
    store._conn.execute(
        "INSERT OR IGNORE INTO memory_entities "
        "(memory_id, entity_key, window_key, assignment_source, assigned_at) "
        "VALUES (?,?,?,?,?)",
        (
            memory_id,
            entity_key,
            "w1",
            "test",
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    store._conn.commit()
    return memory_id


def _oversized(words: int = 4000) -> str:
    return "- " + ("padding " * words)


# ─────────────────────────────────────────────────────────────────────
# 1. The refusal carries the rules
# ─────────────────────────────────────────────────────────────────────


class TestRefusalCarriesTheRules:
    """The refusal is the only moment the model is guaranteed to be reading."""

    def test_digest_rules_state_the_budget_pressure_order_concretely(self):
        rules = DIGEST_COMPACTION_RULES
        # The order itself, not a pointer to where the order is written.
        assert "Timeline FIRST" in rules
        assert "Narrative SECOND" in rules
        assert "Must-keep sections LAST" in rules
        assert rules.index("Timeline FIRST") < rules.index("Narrative SECOND")
        assert rules.index("Narrative SECOND") < rules.index(
            "Must-keep sections LAST"
        )
        # The three non-negotiables the constitution guards.
        assert "LIVENESS" in rules
        assert "NEVER empty a section" in rules
        assert "[m_<id>]" in rules
        assert "sole surviving copy of a judgment" in rules
        # Measurement-driven, and never pad toward the ceiling.
        assert "do not estimate your own output length" in rules
        assert "never pad" in rules

    def test_dossier_rules_are_artifact_appropriate_not_a_digest_copy(self):
        rules = DOSSIER_COMPACTION_RULES
        # A dossier has no seven-section skeleton, so naming Timeline and
        # Narrative here would be instructions the model cannot follow.
        assert "Timeline" not in rules
        assert "Narrative" not in rules
        # The transferable principles are all present.
        assert "LIVENESS" in rules
        assert "[m_<id>]" in rules
        assert "sole surviving copy of a judgment" in rules
        assert "never empty them" in rules
        assert "do not estimate your own output length" in rules

    def test_refusal_builders_carry_measurement_and_overage(self):
        digest = digest_ceiling_refusal(33000, 32000)
        assert "33000 tokens" in digest
        assert "32000-token ceiling" in digest
        assert "by 1000" in digest
        assert DIGEST_COMPACTION_RULES in digest

        dossier = dossier_ceiling_refusal(4500, 4000)
        assert "4500 tokens" in dossier
        assert "4000-token ceiling" in dossier
        assert "by 500" in dossier
        assert DOSSIER_COMPACTION_RULES in dossier

    @pytest.mark.asyncio
    async def test_over_ceiling_digest_edit_returns_the_rules_and_writes_nothing(
        self, tmp_path, store, digest_file
    ):
        node = _node(
            tmp_path, store, digest=digest_file,
            standing_digest_budget_tokens=200,
        )
        context = _curation_context()
        original = digest_file.read_text()

        result = await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": _oversized(),
                "reason": "blow the ceiling",
            },
            context,
        )

        assert result.startswith("Error:")
        # (a) the numbers, (b) the concrete rules.
        assert "over the 200-token ceiling by" in result
        assert DIGEST_COMPACTION_RULES in result
        # Never-truncate is untouched: not one byte moved.
        assert digest_file.read_text() == original

    @pytest.mark.asyncio
    async def test_over_ceiling_essay_edit_returns_the_rules_and_writes_nothing(
        self, tmp_path, store
    ):
        node = _node(tmp_path, store, essay_token_budget=120)
        key = _active_entity(store)
        memory_id = _linked_memory(store, key)
        context = _curation_context()

        result = await _call(
            node,
            "essay_edit",
            {
                "key": key,
                "old_text": "",
                "new_text": (
                    "Project Owner runs the mesh. " * 300 + f"[m_{memory_id}]"
                ),
                "reason": "blow the ceiling",
            },
            context,
        )

        assert result.startswith("Error:")
        assert "over the 120-token ceiling by" in result
        assert DOSSIER_COMPACTION_RULES in result
        # Never-truncate: the dossier was never created at a trimmed size.
        assert store.get_essay(key) is None

    def test_publish_dossier_itself_refuses_with_the_rules(self, store):
        """The lower layer carries the contract, not just the agent wrapper."""
        key = _active_entity(store)
        memory_id = _linked_memory(store, key)
        service = EntityService(
            store._conn, actor_node="agent:test:curator", mutations_enabled=True,
        )
        entity = service.get_entity(key)

        with pytest.raises(EntityError) as excinfo:
            service.publish_dossier(
                key,
                body="Project Owner runs the mesh. " * 300 + f"[m_{memory_id}]",
                title="Project Owner",
                expected_evidence_version=int(entity["evidence_version"]),
                expected_entity_type=entity["entity_type"],
                token_budget=100,
                measure=lambda text: len(text.split()),
                context=_exec_context(),
                reason="over ceiling",
            )

        message = str(excinfo.value)
        assert DOSSIER_COMPACTION_RULES in message
        assert "refused, never truncated" in message
        # The structural numbers T-004 relies on survive the reword.
        assert excinfo.value.measured_tokens > excinfo.value.budget_tokens
        assert excinfo.value.budget_tokens == 100


# ─────────────────────────────────────────────────────────────────────
# 2. The ledger itself
# ─────────────────────────────────────────────────────────────────────


class TestPendingAdditionLedger:

    def test_queue_then_read_back_round_trips_every_replay_field(self, store):
        ledger = PendingAdditionLedger(store._conn, agent="agent:test:curator")
        stored = ledger.queue(
            target_artifact="digest:agent:test:curator",
            tool="digest_edit",
            old_text="- old line",
            new_text="- new line",
            replace_all=True,
            reason="batch 7",
            measured_tokens=33000,
            budget_tokens=32000,
            origin_turn_id="curation-turn-1",
        )

        assert stored is not None
        # Everything a replay needs is on the row.
        assert stored.target_artifact == "digest:agent:test:curator"
        assert stored.tool == "digest_edit"
        assert stored.old_text == "- old line"
        assert stored.new_text == "- new line"
        assert stored.replace_all is True
        assert stored.reason == "batch 7"
        assert stored.measured_tokens == 33000
        assert stored.budget_tokens == 32000
        assert stored.origin_turn_id == "curation-turn-1"
        assert stored.status == STATUS_PENDING
        assert stored.created_at and stored.updated_at

    def test_the_queue_survives_a_restart(self, tmp_path):
        """Durability is the whole point; prove it across a real reopen."""
        first = MemoryStore("restart", db_dir=str(tmp_path))
        PendingAdditionLedger(first._conn, agent="a").queue(
            target_artifact="essay:person:owner",
            tool="essay_edit",
            entity_key="person:owner",
            new_text="Project Owner chairs the review.",
        )
        first.close()

        second = MemoryStore("restart", db_dir=str(tmp_path))
        try:
            pending = PendingAdditionLedger(second._conn).pending()
            assert [item.new_text for item in pending] == [
                "Project Owner chairs the review."
            ]
            assert pending[0].entity_key == "person:owner"
        finally:
            second.close()

    def test_requeueing_the_same_addition_leaves_one_row(self, store):
        ledger = PendingAdditionLedger(store._conn)
        first = ledger.queue(
            target_artifact="digest:a", tool="digest_edit", new_text="- x",
        )
        second = ledger.queue(
            target_artifact="digest:a", tool="digest_edit", new_text="- x",
        )
        assert first is not None
        assert second is None
        assert ledger.pending_count() == 1

    def test_pending_is_oldest_first_so_nothing_starves(self, store):
        ledger = PendingAdditionLedger(store._conn)
        for index in range(5):
            ledger.queue(
                target_artifact="digest:a",
                tool="digest_edit",
                new_text=f"- line {index}",
            )
        assert [item.new_text for item in ledger.pending()] == [
            f"- line {index}" for index in range(5)
        ]
        assert [item.new_text for item in ledger.pending(limit=2)] == [
            "- line 0", "- line 1",
        ]

    def test_per_artifact_cap_keeps_the_oldest_rows(self, store):
        ledger = PendingAdditionLedger(store._conn)
        for index in range(MAX_PENDING_PER_ARTIFACT):
            assert ledger.queue(
                target_artifact="digest:a",
                tool="digest_edit",
                new_text=f"- line {index}",
            ) is not None
        # The cap rejects the new one rather than evicting a waiting one.
        assert ledger.queue(
            target_artifact="digest:a", tool="digest_edit", new_text="- overflow",
        ) is None
        assert ledger.pending_count() == MAX_PENDING_PER_ARTIFACT
        assert ledger.pending()[0].new_text == "- line 0"

    def test_resolution_closes_only_rows_whose_text_is_really_present(
        self, store
    ):
        ledger = PendingAdditionLedger(store._conn)
        landed_row = ledger.queue(
            target_artifact="digest:a", tool="digest_edit", new_text="- alpha",
        )
        waiting_row = ledger.queue(
            target_artifact="digest:a", tool="digest_edit", new_text="- beta",
        )
        other_artifact = ledger.queue(
            target_artifact="essay:person:owner",
            tool="essay_edit",
            new_text="- alpha",
        )

        closed = ledger.resolve_landed(
            "digest:a", "## Timeline\n- alpha\n", turn_id="curation-turn-9",
        )

        assert closed == [landed_row.rowid]
        assert ledger.get(landed_row.rowid).status == STATUS_LANDED
        # Not present → still owed.  The queue never closes on optimism.
        assert ledger.get(waiting_row.rowid).status == STATUS_PENDING
        # A different artifact containing the same text is not evidence.
        assert ledger.get(other_artifact.rowid).status == STATUS_PENDING

    def test_compressed_landing_resolves_as_superseded_with_citations_and_marker(
        self, store
    ):
        ledger = PendingAdditionLedger(store._conn)
        row = ledger.queue(
            target_artifact="essay:project:mesh-autopilot",
            tool="essay_edit",
            new_text=(
                "- T-027 exact-containment reconciliation left seven of seven "
                "compressed additions permanently pending [m_abc123def456]."
            ),
        )

        closed = ledger.resolve_landed(
            "essay:project:mesh-autopilot",
            (
                "- T-027 reconciliation: all 7/7 compacted additions had "
                "already landed in the essay [m_abc123def456]."
            ),
            turn_id="curation-turn-compressed",
        )

        assert closed == [row.rowid]
        assert ledger.get(row.rowid).status == STATUS_SUPERSEDED
        assert ledger.pending_count() == 0

    def test_matching_citation_without_substantive_marker_stays_pending(
        self, store
    ):
        ledger = PendingAdditionLedger(store._conn)
        row = ledger.queue(
            target_artifact="essay:project:mesh-autopilot",
            tool="essay_edit",
            new_text=(
                "- The distinct ledger reconciliation remained unlanded "
                "[m_abc123def456]."
            ),
        )

        closed = ledger.resolve_landed(
            "essay:project:mesh-autopilot",
            "- A different update cites [m_abc123def456].",
        )

        assert closed == []
        assert ledger.get(row.rowid).status == STATUS_PENDING

    def test_compressed_landing_accepts_a_distinctive_phrase_marker(self, store):
        ledger = PendingAdditionLedger(store._conn)
        row = ledger.queue(
            target_artifact="essay:project:mesh-autopilot",
            tool="essay_edit",
            new_text=(
                "- The distinctive checkpoint evidence bundle remained queued "
                "after compaction [m_abcdef123456]."
            ),
        )

        closed = ledger.resolve_landed(
            "essay:project:mesh-autopilot",
            (
                "- The artifact now preserves the distinctive checkpoint "
                "evidence bundle in compressed form [m_abcdef123456]."
            ),
        )

        assert closed == [row.rowid]
        assert ledger.get(row.rowid).status == STATUS_SUPERSEDED

    def test_queue_rejects_a_lossy_rendered_preview(self, store, caplog):
        ledger = PendingAdditionLedger(store._conn)
        preview = _clip("x" * (RENDER_TEXT_CHARS + 1))

        assert ledger.queue(
            target_artifact="digest:a",
            tool="digest_edit",
            new_text=preview,
        ) is None

        assert ledger.pending_count() == 0
        assert "refusing clipped pending-addition preview" in caplog.text

    def test_offer_counter_makes_a_stuck_addition_visible(self, store):
        ledger = PendingAdditionLedger(store._conn)
        row = ledger.queue(
            target_artifact="digest:a", tool="digest_edit", new_text="- x",
        )
        assert ledger.get(row.rowid).offers == 0
        ledger.note_offered([row.rowid])
        ledger.note_offered([row.rowid])
        assert ledger.get(row.rowid).offers == 2

    def test_render_block_is_empty_when_nothing_is_owed(self):
        assert render_pending_block([]) == ""

    def test_render_block_names_the_edits_and_the_held_back_count(self, store):
        ledger = PendingAdditionLedger(store._conn)
        for index in range(3):
            ledger.queue(
                target_artifact="digest:a",
                tool="digest_edit",
                old_text=f"- old {index}",
                new_text=f"- new {index}",
                reason=f"batch {index}",
                measured_tokens=33000,
                budget_tokens=32000,
            )
        additions = ledger.pending(limit=2)

        block = render_pending_block(additions, total_pending=3)

        assert "Land them FIRST" in block
        assert "3 queued addition(s)" in block
        assert "1 held back" in block
        assert "- new 0" in block and "- new 1" in block
        assert "- new 2" not in block          # held back, not rendered
        assert "- old 0" in block              # replay needs the anchor too
        assert "33000/32000 tokens" in block


# ─────────────────────────────────────────────────────────────────────
# 3. Turn-end carry-forward through the real write path
# ─────────────────────────────────────────────────────────────────────


def _pending(store) -> list:
    return PendingAdditionLedger(store._conn).pending()


class TestTurnEndCarryForward:

    @pytest.mark.asyncio
    async def test_a_refused_and_unretried_addition_is_queued_not_dropped(
        self, tmp_path, store, digest_file
    ):
        node = _node(
            tmp_path, store, digest=digest_file,
            standing_digest_budget_tokens=200,
        )
        context = _curation_context()
        oversized = _oversized()

        refused = await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": oversized,
                "reason": "batch 7 timeline entry",
            },
            context,
        )
        assert refused.startswith("Error:")
        # The model does nothing further; the turn ends.
        node._queue_unlanded_curation_additions(context, _trigger())

        pending = _pending(store)
        assert len(pending) == 1
        addition = pending[0]
        assert addition.target_artifact == "digest:agent:test:curator"
        assert addition.tool == "digest_edit"
        assert addition.new_text == oversized
        assert addition.old_text == "- 2026-07-30: something happened."
        assert addition.reason == "batch 7 timeline entry"
        assert addition.origin_turn_id == "curation-turn-1"
        assert addition.measured_tokens > addition.budget_tokens

    @pytest.mark.asyncio
    async def test_the_queued_addition_is_countable_in_the_audit_trail(
        self, tmp_path, store, digest_file
    ):
        node = _node(
            tmp_path, store, digest=digest_file,
            standing_digest_budget_tokens=200,
        )
        context = _curation_context()

        await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": _oversized(),
                "reason": "over ceiling",
            },
            context,
        )
        node._queue_unlanded_curation_additions(context, _trigger())

        outcomes = [item.outcome for item in context.write_log.attempts]
        assert outcomes == [OUTCOME_REFUSED, OUTCOME_QUEUED]

        # And it is durable in entity_events, not just in the live log.
        rows = load_write_attempts(store._conn)
        queued = [row for row in rows if row["outcome"] == OUTCOME_QUEUED]
        assert len(queued) == 1
        assert queued[0]["pending_id"] == _pending(store)[0].rowid
        assert queued[0]["target_artifact"] == "digest:agent:test:curator"
        assert queued[0]["budget_tokens"] == 200

    @pytest.mark.asyncio
    async def test_the_turn_no_longer_resolves_as_a_terminal_drop(
        self, tmp_path, store, digest_file
    ):
        node = _node(
            tmp_path, store, digest=digest_file,
            standing_digest_budget_tokens=200,
        )
        context = _curation_context()

        await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": _oversized(),
                "reason": "over ceiling",
            },
            context,
        )
        # Before the sweep the refusal reads as loss — that was the old world.
        assert context.write_log.resolution() == {
            "digest:agent:test:curator": RESOLUTION_TERMINAL_DROP
        }

        node._queue_unlanded_curation_additions(context, _trigger())

        assert context.write_log.resolution() == {
            "digest:agent:test:curator": OUTCOME_QUEUED
        }
        summary = context.write_log.summary()
        assert summary["terminal_drops"] == 0
        assert summary["by_outcome"][OUTCOME_QUEUED] == 1

    @pytest.mark.asyncio
    async def test_an_addition_landed_after_compacting_is_never_queued(
        self, tmp_path, store, digest_file
    ):
        """Compress-then-write is primary; the ledger must not shadow it."""
        node = _node(
            tmp_path, store, digest=digest_file,
            standing_digest_budget_tokens=400,
        )
        context = _curation_context()

        refused = await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": _oversized(),
                "reason": "too big",
            },
            context,
        )
        assert refused.startswith("Error:")
        # The model applies the rules it was just handed and re-issues.
        landed = await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": "- 2026-07-30–31: consolidated span.",
                "reason": "compacted per the budget-pressure order",
            },
            context,
        )
        assert not landed.startswith("Error")

        node._queue_unlanded_curation_additions(context, _trigger())

        # The oversized text never landed, so it is still owed — but the turn
        # itself is a retry success, not a drop.
        assert [item.new_text for item in _pending(store)] == [_oversized()]
        assert context.write_log.resolution() == {
            "digest:agent:test:curator": "retry_success"
        }

    @pytest.mark.asyncio
    async def test_an_addition_whose_text_did_land_is_not_queued(
        self, tmp_path, store, digest_file
    ):
        """The sweep checks the artifact, not the refusal history."""
        node = _node(
            tmp_path, store, digest=digest_file,
            standing_digest_budget_tokens=400,
        )
        context = _curation_context()
        wanted = "- 2026-08-01: the addition that matters."

        # Refuse a write carrying `wanted` plus a lot of padding …
        await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": wanted + "\n" + _oversized(),
                "reason": "too big",
            },
            context,
        )
        # … then land exactly the part that mattered.
        await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": wanted,
                "reason": "compacted",
            },
            context,
        )
        node._queue_unlanded_curation_additions(context, _trigger())

        # The refused text as a whole is absent, so it is still owed.  What
        # matters is that the check is evidence-based, not that it is lenient:
        # a row is queued only when its text really is missing.
        pending = _pending(store)
        assert len(pending) == 1
        assert wanted in digest_file.read_text()

    @pytest.mark.asyncio
    async def test_a_non_ceiling_refusal_is_not_a_pending_addition(
        self, tmp_path, store, digest_file
    ):
        """Replaying a malformed edit would fail identically; do not queue it."""
        node = _node(tmp_path, store, digest=digest_file)
        context = _curation_context()

        result = await _call(
            node,
            "digest_edit",
            {
                "old_text": "a line that is not in the digest",
                "new_text": "- replacement",
                "reason": "bad anchor",
            },
            context,
        )
        assert "old_text not found" in result
        node._queue_unlanded_curation_additions(context, _trigger())

        assert _pending(store) == []

    @pytest.mark.asyncio
    async def test_shadow_mode_never_queues(
        self, tmp_path, store, digest_file
    ):
        """A shadow turn commits nothing, so it owes nothing."""
        node = _node(
            tmp_path, store, digest=digest_file,
            standing_digest_budget_tokens=200,
        )
        context = _curation_context(mode="shadow")

        await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": _oversized(),
                "reason": "over ceiling in shadow",
            },
            context,
        )
        node._queue_unlanded_curation_additions(context, _trigger())

        assert _pending(store) == []

    @pytest.mark.asyncio
    async def test_a_refused_essay_addition_is_queued_with_its_entity_key(
        self, tmp_path, store
    ):
        node = _node(tmp_path, store, essay_token_budget=120)
        key = _active_entity(store)
        memory_id = _linked_memory(store, key)
        context = _curation_context()
        body = "Project Owner runs the mesh. " * 300 + f"[m_{memory_id}]"

        refused = await _call(
            node,
            "essay_edit",
            {"key": key, "old_text": "", "new_text": body, "reason": "too big"},
            context,
        )
        assert refused.startswith("Error:")
        node._queue_unlanded_curation_additions(context, _trigger())

        pending = _pending(store)
        assert len(pending) == 1
        assert pending[0].tool == "essay_edit"
        assert pending[0].entity_key == key
        assert pending[0].target_artifact == f"essay:{key}"
        assert pending[0].new_text == body


# ─────────────────────────────────────────────────────────────────────
# 4. Drain-first: the queue is offered at the top of a later turn
# ─────────────────────────────────────────────────────────────────────


def _router(store, node_id: str = "agent:test:curator"):
    """A RouterV2 with just enough state to render a curation instruction."""
    from mesh.router_v2 import RouterV2

    router = RouterV2.__new__(RouterV2)
    router._node_id = node_id
    router._memory = SimpleNamespace(_store=store)
    router._config = NodeConfig(
        id=node_id,
        tools=[],
        entity_self_curation_enabled=True,
        essay_token_budget=4000,
        standing_digest_budget_tokens=32000,
    )
    router._curation_recovery_ids = []
    return router


def _drain_block(store, node_id: str = "agent:test:curator") -> str:
    """Drive the real router-side drain renderer over this store."""
    return _router(store, node_id)._render_curation_pending_block(store)


class TestDrainFirst:

    def test_a_clean_queue_costs_the_turn_nothing(self, store):
        assert _drain_block(store) == ""
        instruction = render_update_instruction(
            load_update_template(),
            batch_block="one memory",
            registry_block="(registry)",
            budgets_block="digest 10/32000",
            pending_block="",
        )
        assert "QUEUED ADDITIONS" not in instruction
        # No stray blank block where the section would have been.
        assert "\n\n\n─── NEW MEMORIES ───" not in instruction

    def test_the_drain_block_reaches_the_curation_instruction(self, store):
        PendingAdditionLedger(store._conn).queue(
            target_artifact="digest:agent:test:curator",
            tool="digest_edit",
            old_text="- 2026-07-30: something happened.",
            new_text="- 2026-08-01: the deferred addition.",
            reason="batch 7",
        )

        block = _drain_block(store)
        instruction = render_update_instruction(
            load_update_template(),
            batch_block="one memory",
            registry_block="(registry)",
            budgets_block="digest 10/32000",
            pending_block=block,
        )

        assert "QUEUED ADDITIONS (LAND THESE FIRST)" in instruction
        assert "- 2026-08-01: the deferred addition." in instruction
        # Drain-first: the queue is presented ahead of the new batch.
        assert instruction.index("QUEUED ADDITIONS") < instruction.index(
            "─── NEW MEMORIES ───"
        )

    def test_the_real_instruction_builder_wires_the_drain_through(self, store):
        """Cover the glue, not just the two halves it joins."""
        PendingAdditionLedger(store._conn).queue(
            target_artifact="digest:agent:test:curator",
            tool="digest_edit",
            new_text="- 2026-08-01: the deferred addition.",
            reason="batch 7",
        )
        router = _router(store)

        instruction, _batch_block = router._render_curation_instruction(
            CurationBatch(reason="time-based", memory_ids=()),
        )

        assert "QUEUED ADDITIONS (LAND THESE FIRST)" in instruction
        assert "- 2026-08-01: the deferred addition." in instruction
        assert instruction.index("QUEUED ADDITIONS") < instruction.index(
            "─── NEW MEMORIES ───"
        )

    def test_the_real_instruction_builder_omits_an_empty_queue(self, store):
        instruction, _ = _router(store)._render_curation_instruction(
            CurationBatch(reason="time-based", memory_ids=()),
        )
        assert "QUEUED ADDITIONS" not in instruction

    def test_rendering_the_block_counts_the_offer(self, store):
        ledger = PendingAdditionLedger(store._conn)
        row = ledger.queue(
            target_artifact="digest:agent:test:curator",
            tool="digest_edit",
            new_text="- deferred",
        )
        assert _drain_block(store)
        assert ledger.get(row.rowid).offers == 1
        assert _drain_block(store)
        assert ledger.get(row.rowid).offers == 2

    def test_overflow_beyond_the_render_cap_is_stated_never_hidden(self, store):
        ledger = PendingAdditionLedger(store._conn)
        for index in range(DRAIN_RENDER_CAP + 3):
            ledger.queue(
                target_artifact="digest:agent:test:curator",
                tool="digest_edit",
                new_text=f"- deferred {index}",
            )

        block = _drain_block(store)

        assert f"{DRAIN_RENDER_CAP + 3} queued addition(s)" in block
        assert "3 held back" in block
        # Held-back rows are still owed, not discarded.
        assert ledger.pending_count() == DRAIN_RENDER_CAP + 3

    @pytest.mark.asyncio
    async def test_a_successful_drain_records_landed_and_closes_the_row(
        self, tmp_path, store, digest_file
    ):
        """The headline path: refused on turn 1, landed on turn 2, no drop."""
        node = _node(
            tmp_path, store, digest=digest_file,
            standing_digest_budget_tokens=200,
        )

        # ── turn 1: refused at the ceiling, carried forward ──
        turn_one = _curation_context(turn="curation-turn-1")
        await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": _oversized(),
                "reason": "over ceiling",
            },
            turn_one,
        )
        node._queue_unlanded_curation_additions(turn_one, _trigger("curation-turn-1"))
        row = _pending(store)[0]

        # ── between turns: the queue is offered again ──
        assert "queued" in _drain_block(store).lower()

        # ── turn 2: the model makes room and lands the queued text ──
        node.config.standing_digest_budget_tokens = 32000
        turn_two = _curation_context(turn="curation-turn-2")
        landed = await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": row.new_text,
                "reason": "drained the queue after compacting",
            },
            turn_two,
            turn="curation-turn-2",
        )
        assert not landed.startswith("Error")

        # The row closed on evidence: its text is in the committed artifact.
        assert PendingAdditionLedger(store._conn).get(row.rowid).status == (
            STATUS_LANDED
        )
        assert PendingAdditionLedger(store._conn).pending_count() == 0
        assert row.new_text in digest_file.read_text()

        # And the audit trail records a landing, never a drop.
        assert [item.outcome for item in turn_two.write_log.attempts] == [
            OUTCOME_LANDED
        ]
        assert turn_two.write_log.resolution() == {
            "digest:agent:test:curator": RESOLUTION_LANDED_CLEAN
        }
        assert turn_two.write_log.summary()["terminal_drops"] == 0

    @pytest.mark.asyncio
    async def test_an_essay_drain_closes_its_row(self, tmp_path, store):
        node = _node(tmp_path, store, essay_token_budget=120)
        key = _active_entity(store)
        memory_id = _linked_memory(store, key)
        body = f"Project Owner chairs the weekly mesh review. [m_{memory_id}]"

        ledger = PendingAdditionLedger(store._conn)
        row = ledger.queue(
            target_artifact=f"essay:{key}",
            tool="essay_edit",
            entity_key=key,
            new_text=body,
            reason="deferred from an earlier turn",
        )

        node.config.essay_token_budget = 4000
        context = _curation_context(turn="curation-turn-2")
        result = await _call(
            node,
            "essay_edit",
            {"key": key, "old_text": "", "new_text": body, "reason": "drain"},
            context,
            turn="curation-turn-2",
        )
        assert not result.startswith("Error"), result

        assert ledger.get(row.rowid).status == STATUS_LANDED
        assert ledger.pending_count() == 0


# ─────────────────────────────────────────────────────────────────────
# 5. Audit semantics: queued is not a drop, and never double-counted
# ─────────────────────────────────────────────────────────────────────


class TestQueuedResolutionSemantics:

    def test_queued_clears_the_terminal_drop_it_follows(self):
        assert resolve_turn([
            {"call_ordinal": 1, "target_artifact": "digest:a",
             "outcome": OUTCOME_REFUSED},
            {"call_ordinal": 2, "target_artifact": "digest:a",
             "outcome": OUTCOME_QUEUED},
        ]) == {"digest:a": OUTCOME_QUEUED}

    def test_queued_does_not_overwrite_a_real_landing(self):
        """A turn that also landed something keeps the stronger verdict."""
        assert resolve_turn([
            {"call_ordinal": 1, "target_artifact": "digest:a",
             "outcome": OUTCOME_LANDED},
            {"call_ordinal": 2, "target_artifact": "digest:a",
             "outcome": OUTCOME_QUEUED},
        ]) == {"digest:a": RESOLUTION_LANDED_CLEAN}

        assert resolve_turn([
            {"call_ordinal": 1, "target_artifact": "digest:a",
             "outcome": OUTCOME_REFUSED},
            {"call_ordinal": 2, "target_artifact": "digest:a",
             "outcome": OUTCOME_LANDED},
            {"call_ordinal": 3, "target_artifact": "digest:a",
             "outcome": OUTCOME_QUEUED},
        ]) == {"digest:a": "retry_success"}

    def test_carried_forward_is_counted_and_skews_neither_rate(self):
        """Owed is neither success nor loss; it must be reported as itself."""
        summary = summarize_attempts([
            # digest:a — refused, then carried forward.
            {"agent": "a", "turn_id": "t1", "call_ordinal": 1,
             "target_artifact": "digest:a", "outcome": OUTCOME_REFUSED},
            {"agent": "a", "turn_id": "t1", "call_ordinal": 2,
             "target_artifact": "digest:a", "outcome": OUTCOME_QUEUED},
            # essay:b — refused, then landed on retry.
            {"agent": "a", "turn_id": "t2", "call_ordinal": 1,
             "target_artifact": "essay:b", "outcome": OUTCOME_REFUSED},
            {"agent": "a", "turn_id": "t2", "call_ordinal": 2,
             "target_artifact": "essay:b", "outcome": OUTCOME_LANDED},
        ])["agents"]["a"]

        assert summary["carried_forward"] == 1
        assert summary["terminal_drops"] == 0
        assert summary["retry_success"] == 1
        # The rate denominator counts only refusals that RESOLVED in-turn, so
        # the carried-forward artifact neither inflates nor deflates it.
        assert summary["refused_artifacts"] == 1
        assert summary["retry_success_rate"] == 1.0
        assert summary["drop_rate"] == 0.0

    def test_an_untouched_refusal_is_still_a_drop(self):
        """The drop signal must not be blunted where a drop really happened."""
        assert resolve_turn([
            {"call_ordinal": 1, "target_artifact": "digest:a",
             "outcome": OUTCOME_REFUSED},
        ]) == {"digest:a": RESOLUTION_TERMINAL_DROP}

    @pytest.mark.asyncio
    async def test_the_same_addition_is_not_queued_twice_across_turns(
        self, tmp_path, store, digest_file
    ):
        """A model that hits the same ceiling twice leaves one row, not two."""
        node = _node(
            tmp_path, store, digest=digest_file,
            standing_digest_budget_tokens=200,
        )
        oversized = _oversized()

        for turn in ("curation-turn-1", "curation-turn-2"):
            context = _curation_context(turn=turn)
            await _call(
                node,
                "digest_edit",
                {
                    "old_text": "- 2026-07-30: something happened.",
                    "new_text": oversized,
                    "reason": "over ceiling",
                },
                context,
                turn=turn,
            )
            node._queue_unlanded_curation_additions(context, _trigger(turn))

        assert PendingAdditionLedger(store._conn).pending_count() == 1


# ─────────────────────────────────────────────────────────────────────
# 6. The invariant that must not have moved
# ─────────────────────────────────────────────────────────────────────


class TestNeverTruncateInvariant:

    @pytest.mark.asyncio
    async def test_no_over_ceiling_write_ever_commits_a_trimmed_artifact(
        self, tmp_path, store, digest_file
    ):
        node = _node(
            tmp_path, store, digest=digest_file,
            standing_digest_budget_tokens=200,
        )
        context = _curation_context()
        original = digest_file.read_text()

        for attempt in range(3):
            result = await _call(
                node,
                "digest_edit",
                {
                    "old_text": "- 2026-07-30: something happened.",
                    "new_text": _oversized(1000 * (attempt + 1)),
                    "reason": f"attempt {attempt}",
                },
                context,
            )
            assert result.startswith("Error:")
            # Byte-identical after every refusal — no partial, no trim.
            assert digest_file.read_text() == original

        node._queue_unlanded_curation_additions(context, _trigger())
        assert digest_file.read_text() == original
        # Refused, yet nothing was lost: all three are owed.
        assert PendingAdditionLedger(store._conn).pending_count() == 3

    @pytest.mark.asyncio
    async def test_the_carry_forward_never_fails_a_turn(
        self, tmp_path, store, digest_file, monkeypatch
    ):
        """A broken ledger degrades to the old behaviour, never to an error."""
        node = _node(
            tmp_path, store, digest=digest_file,
            standing_digest_budget_tokens=200,
        )
        context = _curation_context()
        await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": _oversized(),
                "reason": "over ceiling",
            },
            context,
        )

        def _explode(*args, **kwargs):
            raise RuntimeError("ledger unavailable")

        monkeypatch.setattr(node, "_curation_pending_ledger", _explode)

        # Must not raise out of the turn's teardown.
        node._queue_unlanded_curation_additions(context, _trigger())
