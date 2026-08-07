"""Per-write-attempt curation instrumentation (T-004 / goal G-004).

The pipeline refuses over-ceiling writes to the standing digest and to entity
dossiers, and does not carry a refusal forward.  Before this instrumentation the
resulting information loss could only be *inferred* from refusal prose in the
``curation_turn`` trail.  These tests pin the property that makes it countable:
every write attempt records its own resolution, and the true retry-success and
terminal-drop rates fall out of a query over the event trail.

The tests drive the real write paths — ``AgentNode._execute_curation_artifact_tool``
and ``EntityService.publish_dossier`` — rather than asserting against a stub,
because the whole claim is about what the live write path records.  Every store
is a pytest ``tmp_path`` store; no live agent database is opened.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from mesh.agent_node import CURRENT_CURATION_CONTEXT, AgentNode
from mesh.config import NodeConfig
from mesh.memory.curation import CurationBatch, CurationExecutionContext
from mesh.memory.entities import EntityError, EntityExecutionContext, EntityService
from mesh.memory.store import MemoryEntry, MemoryStore
from mesh.memory.write_audit import (
    CURATION_WRITE_ATTEMPT_EVENT,
    OUTCOME_LANDED,
    OUTCOME_QUEUED,
    OUTCOME_REFUSED,
    OUTCOME_RETRIED,
    RESOLUTION_LANDED_CLEAN,
    RESOLUTION_RETRY_SUCCESS,
    RESOLUTION_TERMINAL_DROP,
    WRITE_ATTEMPT_CAP,
    WriteAttemptLog,
    load_write_attempts,
    report,
    resolve_turn,
    short_hash,
)
from mesh.tools import get_registry


# ─────────────────────────────────────────────────────────────────────
# Fixtures — mirrored from test_entity_self_curation.py so the two suites
# exercise the same construction of a real node over a tmp_path store.
# ─────────────────────────────────────────────────────────────────────


def _mid(label: str) -> str:
    """A realistic bare 12-hex memory ID, exactly as ``memories.id`` stores it."""
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
    result = MemoryStore("writeaudit", db_dir=str(tmp_path))
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


def _curation_context(mode: str = "write", **kwargs) -> CurationExecutionContext:
    return CurationExecutionContext(
        mode=mode,
        trigger_id="curation-turn-1",
        actor_node="agent:test:curator",
        batch=CurationBatch(reason="time-based", memory_ids=("m1",)),
        **kwargs,
    )


def _trigger():
    return SimpleNamespace(
        id="curation-turn-1",
        from_node="agent:test:curator",
        content="synthetic curation batch summary",
    )


async def _call(node, name: str, arguments: dict, context) -> str:
    token = CURRENT_CURATION_CONTEXT.set(context)
    try:
        return await node._execute_entity_tool(name, arguments, _trigger())
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


# ─────────────────────────────────────────────────────────────────────
# The log itself: ordinals, retry classification, turn resolution
# ─────────────────────────────────────────────────────────────────────


class TestWriteAttemptLog:

    def test_short_hash_is_stable_and_distinguishes_content(self):
        assert short_hash("alpha") == short_hash("alpha")
        assert short_hash("alpha") != short_hash("beta")
        assert len(short_hash("alpha")) == 12
        # None and "" are the same artifact state: "nothing there yet".
        assert short_hash(None) == short_hash("")

    def test_ordinals_are_sequential_within_a_turn(self):
        log = WriteAttemptLog()
        for _ in range(3):
            log.record(
                target_artifact="digest:a", tool="digest_edit", landed=True,
                before_hash="aa", after_hash="bb",
            )
        assert [a.call_ordinal for a in log.attempts] == [1, 2, 3]

    def test_first_refusal_is_refused_and_a_second_is_retried(self):
        log = WriteAttemptLog()
        first = log.record(
            target_artifact="digest:a", tool="digest_edit", landed=False,
            before_hash="aa", after_hash="aa",
        )
        second = log.record(
            target_artifact="digest:a", tool="digest_edit", landed=False,
            before_hash="aa", after_hash="aa",
        )
        assert first.outcome == OUTCOME_REFUSED
        assert first.is_retry is False
        assert second.outcome == OUTCOME_RETRIED
        assert second.is_retry is True

    def test_a_landed_retry_records_landed_and_stays_identifiable(self):
        """G-004 requires the successful retry to record ``landed``.

        ``is_retry`` is what keeps it separable from a first-try success, so
        retry-success is countable without re-deriving turn order.
        """
        log = WriteAttemptLog()
        log.record(
            target_artifact="digest:a", tool="digest_edit", landed=False,
            before_hash="aa", after_hash="aa",
        )
        landed = log.record(
            target_artifact="digest:a", tool="digest_edit", landed=True,
            before_hash="aa", after_hash="bb",
        )
        assert landed.outcome == OUTCOME_LANDED
        assert landed.is_retry is True

    def test_refusal_on_one_artifact_does_not_taint_another(self):
        log = WriteAttemptLog()
        log.record(
            target_artifact="digest:a", tool="digest_edit", landed=False,
            before_hash="aa", after_hash="aa",
        )
        other = log.record(
            target_artifact="essay:person:owner", tool="essay_edit", landed=True,
            before_hash="cc", after_hash="dd",
        )
        assert other.is_retry is False
        assert other.outcome == OUTCOME_LANDED

    def test_over_budget_is_derived_from_measured_versus_budget(self):
        log = WriteAttemptLog()
        over = log.record(
            target_artifact="digest:a", tool="digest_edit", landed=False,
            before_hash="aa", after_hash="aa",
            measured_tokens=500, budget_tokens=200,
        )
        under = log.record(
            target_artifact="essay:person:owner", tool="essay_edit", landed=True,
            before_hash="cc", after_hash="dd",
            measured_tokens=100, budget_tokens=200,
        )
        unknown = log.record(
            target_artifact="essay:person:bo", tool="essay_edit", landed=False,
            before_hash="ee", after_hash="ee",
        )
        assert over.over_budget is True
        assert under.over_budget is False
        # A refusal that is not a ceiling refusal must not claim to be one.
        assert unknown.over_budget is None

    def test_the_per_turn_cap_counts_overflow_rather_than_hiding_it(self):
        log = WriteAttemptLog()
        for _ in range(WRITE_ATTEMPT_CAP + 5):
            log.record(
                target_artifact="digest:a", tool="digest_edit", landed=False,
                before_hash="aa", after_hash="aa",
            )
        assert len(log.attempts) == WRITE_ATTEMPT_CAP
        assert log.overflow == 5

    def test_queued_is_reserved_and_never_counts_as_a_drop(self):
        """T-001 will emit ``queued``; the reporting query must already cope."""
        log = WriteAttemptLog()
        queued = log.record(
            target_artifact="digest:a", tool="digest_edit", landed=False,
            before_hash="aa", after_hash="aa", queued=True,
        )
        assert queued.outcome == OUTCOME_QUEUED
        assert log.resolution()["digest:a"] == OUTCOME_QUEUED
        assert log.summary()["terminal_drops"] == 0


class TestTurnResolution:

    def test_refuse_then_land_resolves_as_retry_success(self):
        log = WriteAttemptLog()
        log.record(
            target_artifact="digest:a", tool="digest_edit", landed=False,
            before_hash="aa", after_hash="aa",
        )
        log.record(
            target_artifact="digest:a", tool="digest_edit", landed=True,
            before_hash="aa", after_hash="bb",
        )
        assert log.resolution() == {"digest:a": RESOLUTION_RETRY_SUCCESS}
        assert log.summary()["retry_success"] == 1
        assert log.summary()["terminal_drops"] == 0

    def test_a_never_retried_refusal_resolves_as_a_terminal_drop(self):
        log = WriteAttemptLog()
        log.record(
            target_artifact="digest:a", tool="digest_edit", landed=False,
            before_hash="aa", after_hash="aa",
        )
        assert log.resolution() == {"digest:a": RESOLUTION_TERMINAL_DROP}
        assert log.summary()["terminal_drops"] == 1

    def test_a_clean_first_try_is_not_counted_as_a_retry(self):
        log = WriteAttemptLog()
        log.record(
            target_artifact="digest:a", tool="digest_edit", landed=True,
            before_hash="aa", after_hash="bb",
        )
        assert log.resolution() == {"digest:a": RESOLUTION_LANDED_CLEAN}
        assert log.summary()["retry_success"] == 0
        assert log.summary()["terminal_drops"] == 0

    def test_landing_then_being_refused_again_is_still_a_drop(self):
        """The last word in the turn decides — a later refusal loses content."""
        log = WriteAttemptLog()
        log.record(
            target_artifact="digest:a", tool="digest_edit", landed=True,
            before_hash="aa", after_hash="bb",
        )
        log.record(
            target_artifact="digest:a", tool="digest_edit", landed=False,
            before_hash="bb", after_hash="bb",
        )
        assert log.resolution() == {"digest:a": RESOLUTION_TERMINAL_DROP}

    def test_resolve_turn_sorts_by_ordinal_not_arrival_order(self):
        """A windowed query may hand rows back interleaved across turns."""
        shuffled = [
            {"target_artifact": "digest:a", "outcome": OUTCOME_LANDED,
             "call_ordinal": 2},
            {"target_artifact": "digest:a", "outcome": OUTCOME_REFUSED,
             "call_ordinal": 1},
        ]
        assert resolve_turn(shuffled) == {"digest:a": RESOLUTION_RETRY_SUCCESS}


# ─────────────────────────────────────────────────────────────────────
# The live write paths
# ─────────────────────────────────────────────────────────────────────


class TestDigestWriteInstrumentation:

    @pytest.mark.asyncio
    async def test_a_landed_digest_write_records_every_required_field(
        self, tmp_path, store, digest_file
    ):
        node = _node(tmp_path, store, digest=digest_file)
        context = _curation_context()
        before_text = digest_file.read_text()

        result = await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": "- 2026-07-30: something else happened.",
                "reason": "landed write",
            },
            context,
        )
        assert "Error" not in result, result

        assert len(context.write_log.attempts) == 1
        attempt = context.write_log.attempts[0]
        assert attempt.target_artifact == "digest:agent:test:curator"
        assert attempt.tool == "digest_edit"
        assert attempt.call_ordinal == 1
        assert attempt.outcome == OUTCOME_LANDED
        assert attempt.turn_id == "curation-turn-1"
        assert attempt.agent == "agent:test:curator"
        assert attempt.timestamp
        assert attempt.mode == "write"
        # measured-vs-budget comes from the real measurement, not a guess.
        assert attempt.measured_tokens is not None and attempt.measured_tokens > 0
        assert attempt.budget_tokens == 32000
        assert attempt.over_budget is False
        # The artifact really moved, and the hashes say so.
        assert attempt.before_hash == short_hash(before_text)
        assert attempt.after_hash == short_hash(digest_file.read_text())
        assert attempt.changed is True

    @pytest.mark.asyncio
    async def test_an_over_ceiling_refusal_records_refused_with_equal_hashes(
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
                "new_text": "- " + ("padding " * 4000),
                "reason": "blow the ceiling",
            },
            context,
        )
        assert "Error" in result, result
        # Refusal semantics are untouched: the bytes did not move.
        assert digest_file.read_text() == original

        attempt = context.write_log.attempts[0]
        assert attempt.outcome == OUTCOME_REFUSED
        assert attempt.before_hash == attempt.after_hash == short_hash(original)
        assert attempt.changed is False
        assert attempt.measured_tokens > attempt.budget_tokens
        assert attempt.budget_tokens == 200
        assert attempt.over_budget is True
        assert "ceiling" in attempt.detail

    @pytest.mark.asyncio
    async def test_compress_then_retry_records_landed_and_counts_as_retry_success(
        self, tmp_path, store, digest_file
    ):
        node = _node(
            tmp_path, store, digest=digest_file,
            standing_digest_budget_tokens=200,
        )
        context = _curation_context()

        refused = await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": "- " + ("padding " * 4000),
                "reason": "first, too big",
            },
            context,
        )
        assert "Error" in refused
        landed = await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": "- 2026-07-30: compressed.",
                "reason": "second, compressed",
            },
            context,
        )
        assert "Error" not in landed, landed

        outcomes = [a.outcome for a in context.write_log.attempts]
        assert outcomes == [OUTCOME_REFUSED, OUTCOME_LANDED]
        assert context.write_log.attempts[1].is_retry is True
        artifact = "digest:agent:test:curator"
        assert context.write_log.resolution() == {
            artifact: RESOLUTION_RETRY_SUCCESS
        }
        assert context.write_log.summary()["retry_success"] == 1
        assert context.write_log.summary()["terminal_drops"] == 0

    @pytest.mark.asyncio
    async def test_a_never_retried_refusal_is_countable_as_a_terminal_drop(
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
                "new_text": "- " + ("padding " * 4000),
                "reason": "dropped for good",
            },
            context,
        )
        artifact = "digest:agent:test:curator"
        assert context.write_log.resolution() == {
            artifact: RESOLUTION_TERMINAL_DROP
        }
        assert context.write_log.summary()["terminal_drops"] == 1

    @pytest.mark.asyncio
    async def test_a_non_ceiling_refusal_records_no_false_measurement(
        self, tmp_path, store, digest_file
    ):
        """``old_text`` not found returns before anything is measured.

        The measurement slot must not leak a previous attempt's numbers into
        this one, or a plain typo would be miscounted as a ceiling refusal.
        """
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
                "new_text": "- " + ("padding " * 4000),
                "reason": "ceiling refusal first",
            },
            context,
        )
        await _call(
            node,
            "digest_edit",
            {
                "old_text": "text that is definitely not in the digest",
                "new_text": "irrelevant",
                "reason": "not a ceiling problem",
            },
            context,
        )
        second = context.write_log.attempts[1]
        assert second.outcome == OUTCOME_RETRIED
        assert second.measured_tokens is None
        assert second.budget_tokens is None
        assert second.over_budget is None


class TestEssayWriteInstrumentation:

    @pytest.mark.asyncio
    async def test_a_landed_essay_write_records_the_entity_as_the_artifact(
        self, tmp_path, store
    ):
        key = _active_entity(store)
        node = _node(tmp_path, store)
        context = _curation_context()

        result = await _call(
            node,
            "essay_edit",
            {
                "key": key,
                "old_text": "",
                "new_text": "Project Owner is the primary user of the mesh.",
                "title": "Project Owner",
                "reason": "first dossier",
            },
            context,
        )
        assert "Error" not in result, result

        attempt = context.write_log.attempts[0]
        assert attempt.target_artifact == f"essay:{key}"
        assert attempt.tool == "essay_edit"
        assert attempt.outcome == OUTCOME_LANDED
        assert attempt.budget_tokens == 4000
        assert attempt.measured_tokens is not None
        assert attempt.over_budget is False
        assert attempt.changed is True

    @pytest.mark.asyncio
    async def test_an_over_ceiling_essay_refusal_carries_measured_and_budget(
        self, tmp_path, store
    ):
        """The ceiling ``EntityError`` carries its numbers structurally.

        Without that the auditor would have to scrape them from the refusal
        prose, which breaks as soon as the message is reworded.
        """
        key = _active_entity(store)
        node = _node(tmp_path, store, essay_token_budget=50)
        context = _curation_context()

        result = await _call(
            node,
            "essay_edit",
            {
                "key": key,
                "old_text": "",
                "new_text": "Project Owner " * 4000,
                "title": "Project Owner",
                "reason": "too big",
            },
            context,
        )
        assert "Error" in result, result
        assert store.get_essay(key) is None, "refused write must not commit"

        attempt = context.write_log.attempts[0]
        assert attempt.outcome == OUTCOME_REFUSED
        assert attempt.before_hash == attempt.after_hash
        assert attempt.measured_tokens > attempt.budget_tokens
        assert attempt.budget_tokens == 50
        assert attempt.over_budget is True

    def test_the_ceiling_error_exposes_the_numbers_directly(self, store):
        """Pinned at the source: publish_dossier's refusal is structured."""
        from mesh.llm import estimate_tokens

        key = _active_entity(store)
        service = EntityService(
            store._conn, actor_node="agent:test:curator", mutations_enabled=True,
        )
        entity = service.get_entity(key)
        with pytest.raises(EntityError) as excinfo:
            service.publish_dossier(
                key,
                body="Project Owner " * 4000,
                title="Project Owner",
                expected_evidence_version=int(entity["evidence_version"]),
                expected_entity_type=entity["entity_type"],
                token_budget=50,
                measure=estimate_tokens,
                context=_exec_context(),
                reason="too big",
            )
        assert getattr(excinfo.value, "measured_tokens", None) is not None
        assert excinfo.value.measured_tokens > 50
        assert excinfo.value.budget_tokens == 50


# ─────────────────────────────────────────────────────────────────────
# The durable trail and the reporting query
# ─────────────────────────────────────────────────────────────────────


class TestEventTrail:

    @pytest.mark.asyncio
    async def test_each_attempt_lands_as_its_own_event_row(
        self, tmp_path, store, digest_file
    ):
        node = _node(
            tmp_path, store, digest=digest_file,
            standing_digest_budget_tokens=200,
        )
        context = _curation_context()
        await _call(
            node, "digest_edit",
            {"old_text": "- 2026-07-30: something happened.",
             "new_text": "- " + ("padding " * 4000), "reason": "refused"},
            context,
        )
        await _call(
            node, "digest_edit",
            {"old_text": "- 2026-07-30: something happened.",
             "new_text": "- 2026-07-30: compressed.", "reason": "landed"},
            context,
        )

        rows = store._conn.execute(
            "SELECT event_type, actor_node, run_key, details_json "
            "FROM entity_events WHERE event_type = ? ORDER BY sequence",
            (CURATION_WRITE_ATTEMPT_EVENT,),
        ).fetchall()
        assert len(rows) == 2
        for event_type, actor_node, run_key, _details in rows:
            assert event_type == CURATION_WRITE_ATTEMPT_EVENT
            assert actor_node == "agent:test:curator"
            # run_key is the turn id — this is what makes turn-level
            # resolution a group-by rather than a reconstruction.
            assert run_key == "curation-turn-1"
        payloads = [json.loads(row[3]) for row in rows]
        assert [p["outcome"] for p in payloads] == [
            OUTCOME_REFUSED, OUTCOME_LANDED,
        ]
        assert [p["call_ordinal"] for p in payloads] == [1, 2]
        assert payloads[1]["is_retry"] is True

    @pytest.mark.asyncio
    async def test_the_trail_round_trips_through_the_reader(
        self, tmp_path, store, digest_file
    ):
        node = _node(
            tmp_path, store, digest=digest_file,
            standing_digest_budget_tokens=200,
        )
        context = _curation_context()
        await _call(
            node, "digest_edit",
            {"old_text": "- 2026-07-30: something happened.",
             "new_text": "- " + ("padding " * 4000), "reason": "refused"},
            context,
        )
        attempts = load_write_attempts(store._conn)
        assert len(attempts) == 1
        assert attempts[0]["agent"] == "agent:test:curator"
        assert attempts[0]["turn_id"] == "curation-turn-1"
        assert attempts[0]["outcome"] == OUTCOME_REFUSED
        assert resolve_turn(attempts) == {
            "digest:agent:test:curator": RESOLUTION_TERMINAL_DROP
        }

    def test_the_covering_indexes_exist(self, store):
        names = {
            row[0] for row in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_entity_events_run" in names
        assert "idx_entity_events_type_created" in names


def _insert_attempt(
    conn: sqlite3.Connection, *, agent: str, turn: str, ordinal: int,
    outcome: str, artifact: str, created_at: str,
) -> None:
    """Write one synthetic ``curation_write_attempt`` row."""
    details = {
        "target_artifact": artifact,
        "tool": "digest_edit",
        "call_ordinal": ordinal,
        "outcome": outcome,
        "is_retry": outcome in {OUTCOME_RETRIED},
        "before_hash": "aaaaaaaaaaaa",
        "after_hash": "aaaaaaaaaaaa" if outcome != OUTCOME_LANDED else "bbbbbbbbbbbb",
    }
    conn.execute(
        "INSERT INTO entity_events (event_type, actor_node, reason, run_key, "
        "details_json, created_at) VALUES (?, ?, '', ?, ?, ?)",
        (
            CURATION_WRITE_ATTEMPT_EVENT, agent, turn,
            json.dumps(details), created_at,
        ),
    )
    conn.commit()


class TestReportingQuery:

    def test_report_counts_a_small_synthetic_trail_correctly(self, store):
        conn = store._conn
        # bob, turn 1: refuse then land  -> one retry success
        _insert_attempt(conn, agent="bob", turn="t1", ordinal=1,
                        outcome=OUTCOME_REFUSED, artifact="digest:bob",
                        created_at="2026-08-01T00:00:00+00:00")
        _insert_attempt(conn, agent="bob", turn="t1", ordinal=2,
                        outcome=OUTCOME_LANDED, artifact="digest:bob",
                        created_at="2026-08-01T00:00:01+00:00")
        # bob, turn 2: refuse only -> one terminal drop
        _insert_attempt(conn, agent="bob", turn="t2", ordinal=1,
                        outcome=OUTCOME_REFUSED, artifact="digest:bob",
                        created_at="2026-08-02T00:00:00+00:00")
        # alice, turn 3: clean land -> neither
        _insert_attempt(conn, agent="alice", turn="t3", ordinal=1,
                        outcome=OUTCOME_LANDED, artifact="essay:person:x",
                        created_at="2026-08-02T00:00:00+00:00")

        result = report(conn)
        bob = result["agents"]["bob"]
        assert bob["attempts"] == 3
        assert bob["turns"] == 2
        assert bob["by_outcome"][OUTCOME_REFUSED] == 2
        assert bob["by_outcome"][OUTCOME_LANDED] == 1
        assert bob["retry_success"] == 1
        assert bob["terminal_drops"] == 1
        assert bob["refused_artifacts"] == 2
        assert bob["retry_success_rate"] == 0.5
        assert bob["drop_rate"] == 0.5

        alice = result["agents"]["alice"]
        assert alice["landed_clean"] == 1
        assert alice["retry_success"] == 0
        assert alice["terminal_drops"] == 0
        # No refusals means no rate, not a rate of zero.
        assert alice["retry_success_rate"] is None

        totals = result["totals"]
        assert totals["attempts"] == 4
        assert totals["retry_success"] == 1
        assert totals["terminal_drops"] == 1
        assert totals["drop_rate"] == 0.5

    def test_the_window_bounds_are_honoured(self, store):
        conn = store._conn
        _insert_attempt(conn, agent="bob", turn="t1", ordinal=1,
                        outcome=OUTCOME_REFUSED, artifact="digest:bob",
                        created_at="2026-08-01T00:00:00+00:00")
        _insert_attempt(conn, agent="bob", turn="t2", ordinal=1,
                        outcome=OUTCOME_REFUSED, artifact="digest:bob",
                        created_at="2026-08-05T00:00:00+00:00")

        windowed = report(conn, since="2026-08-03", until="2026-08-06")
        assert windowed["totals"]["attempts"] == 1
        # ``until`` is exclusive, so a row exactly on the bound is outside it.
        assert report(
            conn, since="2026-08-01", until="2026-08-05",
        )["totals"]["attempts"] == 1
        assert report(conn, agent="alice")["totals"]["attempts"] == 0

    def test_turns_are_resolved_independently(self, store):
        """A refusal in one turn must not make a later turn's land a retry."""
        conn = store._conn
        _insert_attempt(conn, agent="bob", turn="t1", ordinal=1,
                        outcome=OUTCOME_REFUSED, artifact="digest:bob",
                        created_at="2026-08-01T00:00:00+00:00")
        _insert_attempt(conn, agent="bob", turn="t2", ordinal=1,
                        outcome=OUTCOME_LANDED, artifact="digest:bob",
                        created_at="2026-08-02T00:00:00+00:00")
        bob = report(conn)["agents"]["bob"]
        assert bob["terminal_drops"] == 1
        assert bob["landed_clean"] == 1
        assert bob["retry_success"] == 0

    def test_a_malformed_details_payload_does_not_break_the_report(self, store):
        conn = store._conn
        conn.execute(
            "INSERT INTO entity_events (event_type, actor_node, reason, "
            "run_key, details_json, created_at) VALUES (?, 'bob', '', 't1', "
            "'not json', '2026-08-01T00:00:00+00:00')",
            (CURATION_WRITE_ATTEMPT_EVENT,),
        )
        conn.commit()
        result = report(conn)
        # The row is counted as an attempt but contributes no resolution.
        assert result["totals"]["attempts"] == 1
        assert result["totals"]["terminal_drops"] == 0


class TestEntityRecordInstrumentation:
    """The third artifact class G-004 names, alongside digest and dossier."""

    @pytest.mark.asyncio
    async def test_a_landed_entity_edit_records_the_entity_as_the_artifact(
        self, tmp_path, store
    ):
        key = _active_entity(store)
        node = _node(tmp_path, store)
        context = _curation_context()

        result = await _call(
            node,
            "entity_edit",
            {
                "entity_key": key,
                "operation": "update_details",
                "identity_note": "Project Owner runs the mesh.",
                "reason": "sharpen the identity note",
            },
            context,
        )
        assert "Error" not in result, result

        attempt = context.write_log.attempts[0]
        assert attempt.target_artifact == f"entity:{key}"
        assert attempt.tool == "entity_edit"
        assert attempt.outcome == OUTCOME_LANDED
        assert attempt.changed is True
        # No ceiling applies to a registry record, and the audit must not
        # invent one.
        assert attempt.measured_tokens is None
        assert attempt.budget_tokens is None
        assert attempt.over_budget is None

    @pytest.mark.asyncio
    async def test_a_refused_entity_edit_records_refused_with_equal_hashes(
        self, tmp_path, store
    ):
        key = _active_entity(store)
        node = _node(tmp_path, store)
        context = _curation_context()

        # Every curation entity mutation requires a non-empty reason.
        result = await _call(
            node,
            "entity_edit",
            {"entity_key": key, "operation": "update_details",
             "identity_note": "no reason given"},
            context,
        )
        assert "Error" in result, result

        attempt = context.write_log.attempts[0]
        assert attempt.target_artifact == f"entity:{key}"
        assert attempt.outcome == OUTCOME_REFUSED
        assert attempt.before_hash == attempt.after_hash
        assert attempt.changed is False
        assert context.write_log.resolution() == {
            f"entity:{key}": RESOLUTION_TERMINAL_DROP
        }

    @pytest.mark.asyncio
    async def test_refuse_then_fix_on_an_entity_is_a_retry_success(
        self, tmp_path, store
    ):
        key = _active_entity(store)
        node = _node(tmp_path, store)
        context = _curation_context()

        await _call(
            node, "entity_edit",
            {"entity_key": key, "operation": "update_details",
             "identity_note": "no reason"}, context,
        )
        landed = await _call(
            node, "entity_edit",
            {"entity_key": key, "operation": "update_details",
             "identity_note": "Project Owner runs the mesh.",
             "reason": "now with a reason"},
            context,
        )
        assert "Error" not in landed, landed
        assert [a.outcome for a in context.write_log.attempts] == [
            OUTCOME_REFUSED, OUTCOME_LANDED,
        ]
        assert context.write_log.resolution() == {
            f"entity:{key}": RESOLUTION_RETRY_SUCCESS
        }

    @pytest.mark.asyncio
    async def test_a_create_without_a_key_still_groups_within_the_turn(
        self, tmp_path, store
    ):
        """``entity_create`` has no key yet; the placeholder must be stable."""
        node = _node(tmp_path, store)
        context = _curation_context()

        for _ in range(2):
            await _call(
                node,
                "entity_create",
                {
                    "entity_type": "person",
                    "display_name": "Kaylee",
                    "identity_note": "A person the agent works with.",
                },
                context,
            )
        artifacts = {a.target_artifact for a in context.write_log.attempts}
        assert len(artifacts) == 1, artifacts
        assert context.write_log.attempts[1].is_retry is True

    @pytest.mark.asyncio
    async def test_entity_and_digest_writes_share_one_trail(
        self, tmp_path, store, digest_file
    ):
        key = _active_entity(store)
        node = _node(tmp_path, store, digest=digest_file)
        context = _curation_context()

        await _call(
            node, "entity_edit",
            {"entity_key": key, "operation": "update_details",
             "identity_note": "Project Owner runs the mesh.", "reason": "note"},
            context,
        )
        await _call(
            node, "digest_edit",
            {"old_text": "- 2026-07-30: something happened.",
             "new_text": "- 2026-07-30: something else.", "reason": "note"},
            context,
        )
        attempts = load_write_attempts(store._conn)
        assert [a["tool"] for a in attempts] == ["entity_edit", "digest_edit"]
        assert [a["call_ordinal"] for a in attempts] == [1, 2]
        assert {a["target_artifact"] for a in attempts} == {
            f"entity:{key}", "digest:agent:test:curator",
        }
        # One turn, two artifacts, both resolved independently.
        assert report(store._conn)["agents"]["agent:test:curator"][
            "landed_clean"
        ] == 2


class TestSemanticsUnchanged:
    """The instrumentation measures; it must not decide anything."""

    @pytest.mark.asyncio
    async def test_a_refused_write_still_leaves_the_artifact_byte_identical(
        self, tmp_path, store, digest_file
    ):
        node = _node(
            tmp_path, store, digest=digest_file,
            standing_digest_budget_tokens=200,
        )
        original = digest_file.read_text()
        result = await _call(
            node, "digest_edit",
            {"old_text": "- 2026-07-30: something happened.",
             "new_text": "- " + ("padding " * 4000), "reason": "refused"},
            _curation_context(),
        )
        assert "refused, never truncated" in result
        assert digest_file.read_text() == original

    @pytest.mark.asyncio
    async def test_an_auditor_failure_does_not_fail_the_write(
        self, tmp_path, store, digest_file, monkeypatch
    ):
        """Losing an audit row beats failing a write that would have landed."""
        node = _node(tmp_path, store, digest=digest_file)

        def _boom(*_args, **_kwargs):
            raise RuntimeError("audit backend exploded")

        monkeypatch.setattr(WriteAttemptLog, "record", _boom)
        result = await _call(
            node, "digest_edit",
            {"old_text": "- 2026-07-30: something happened.",
             "new_text": "- 2026-07-30: still lands.", "reason": "landed"},
            _curation_context(),
        )
        assert "Error" not in result, result
        assert "still lands" in digest_file.read_text()

    @pytest.mark.asyncio
    async def test_the_turn_summary_reaches_the_context_summary(
        self, tmp_path, store, digest_file
    ):
        node = _node(
            tmp_path, store, digest=digest_file,
            standing_digest_budget_tokens=200,
        )
        context = _curation_context()
        await _call(
            node, "digest_edit",
            {"old_text": "- 2026-07-30: something happened.",
             "new_text": "- " + ("padding " * 4000), "reason": "refused"},
            context,
        )
        summary = context.summary()["write_attempts"]
        assert summary["attempts"] == 1
        assert summary["terminal_drops"] == 1
        assert summary["by_outcome"][OUTCOME_REFUSED] == 1
