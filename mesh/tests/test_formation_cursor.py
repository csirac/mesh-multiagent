"""Tests for Memory Formation v3 cursor + multi-trigger orchestration.

Plan: docs/plans/memory-formation-v3-2026-04-27.md (rev 6, §5.2).

Covers tests 15-29:
- 15-22: Cursor advancement, no re-formation, gap detection, defer-tail trim,
  serialization, shutdown trigger, segmenter failure does not advance cursor.
- 23-24: Non-blocking startup formation + disconnect cancellation.
- 25: Persistent parse-failure fallback after threshold.
- 26: Embedding migration runs once.
- 27: Conversation summary still fires under v3 (window-drop short-circuit).
- 28-29: Token-pressure trigger fires + counter resets; no double-spawn while lock held.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from mesh.conversation_history import Turn
from mesh.memory.formation_v3 import LLMSegmenterV3, Segment
from mesh.memory.entities import EntityExecutionContext, EntityService
from mesh.memory.store import MemoryEntry, MemoryStore
from mesh.memory.system_v2 import MemorySystemV2


# ── Helpers ────────────────────────────────────────────────────────


def _turn(i: int, content: str = "", role: str = "user", from_node: str = "user:testuser",
          token_estimate: int | None = None) -> Turn:
    t = Turn(
        role=role,
        content=content or f"turn-{i} content",
        timestamp=datetime(2026, 4, 27, 10, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=i),
        from_node=from_node,
        seq_id=i,
    )
    if token_estimate is not None:
        t._token_estimate = token_estimate
    return t


class _FakeEmbedder:
    def __init__(self, dim: int = 8):
        self.dim = dim
        self.batch_inputs: list[list[str]] = []

    async def embed_batch_to_arrays(self, texts: list[str]) -> list[np.ndarray]:
        self.batch_inputs.append(list(texts))
        out = []
        for t in texts:
            rng = np.random.RandomState(abs(hash(t)) % 2**31)
            out.append(rng.randn(self.dim).astype(np.float32))
        return out

    async def embed(self, text):
        rng = np.random.RandomState(abs(hash(text)) % 2**31)
        return rng.randn(self.dim).astype(np.float32).tolist()

    async def embed_to_array(self, text):
        rng = np.random.RandomState(abs(hash(text)) % 2**31)
        return rng.randn(self.dim).astype(np.float32)

    async def embed_batch(self, texts):
        return [(await self.embed(t)) for t in texts]


class _FakeLLMClient:
    def __init__(self, responses: list[str] | None = None, side_effect=None):
        self._responses = list(responses) if responses else []
        self.side_effect = side_effect
        self.calls: list[dict] = []

    async def complete(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        if self.side_effect:
            res = self.side_effect(prompt, kwargs)
            if asyncio.iscoroutine(res):
                return await res
            return res
        if not self._responses:
            return ""
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]


def _mk_segs_json(specs: list[dict]) -> str:
    return json.dumps(specs)


def _seg_spec(start: int, end: int, **overrides) -> dict:
    base = {
        "topic_label": f"topic-{start}-{end}",
        "retrieval_key": f"retrieval key {start}-{end}",
        "summary": f"summary {start}-{end}",
        "reflection": f"reflection {start}-{end}",
        "trace": f"turn-{start} content",
        "tags": ["test"],
        "outcome": "success",
        "project": "",
        "event_date": "2026-04-27",
        "digest_candidate": True,
    }
    base.update(overrides)
    return base


def _build_sys(tmp_dir, llm, embedder, *, formation_v3_enabled=True, **kwargs):
    sys = MemorySystemV2(
        nickname=kwargs.pop("nickname", "test-agent"),
        llm_client=llm,
        formation_v3_enabled=formation_v3_enabled,
        formation_v3_window_size=kwargs.pop("formation_v3_window_size", 50),
        formation_v3_overlap=kwargs.pop("formation_v3_overlap", 5),
        formation_v3_defer_tail=kwargs.pop("formation_v3_defer_tail", 3),
        formation_v3_parse_failure_fallback_threshold=kwargs.pop(
            "formation_v3_parse_failure_fallback_threshold", 3,
        ),
        **kwargs,
    )
    sys._store = MemoryStore("test-agent", db_dir=tmp_dir)
    sys._embedder = embedder
    sys._formation_lock = asyncio.Lock()
    return sys


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ── Cursor / no-re-formation / gap / defer-tail / serialization ─────


class TestCursor:
    def test_cursor_starts_at_zero(self, tmp_dir):
        """Test 15: fresh DB returns (0, '')."""
        store = MemoryStore("fresh", db_dir=tmp_dir)
        idx, ts = store.get_formation_cursor()
        assert idx == 0
        assert ts == ""
        store.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("mode", "expected_links"),
        [("shadow", 0), ("write", 1)],
    )
    async def test_entity_mode_translates_with_shadow_non_writing_telemetry(
        self, tmp_dir, mode, expected_links
    ):
        entity = {
            "existing_keys": ["person:project-owner"],
            "new_entities": [],
            "unresolved": [],
        }
        llm = _FakeLLMClient(responses=[
            _mk_segs_json([_seg_spec(0, 0, worthwhile=True, entity=entity)])
        ])
        sys = _build_sys(
            tmp_dir,
            llm,
            _FakeEmbedder(),
            formation_v3_window_size=10,
            entity_resolution_mode=mode,
        )
        EntityService(
            sys._store._conn,
            mutations_enabled=True,
        ).create_user_named_entity(
            "person",
            "Project Owner",
            naming_surface="Project Owner",
            context=EntityExecutionContext(
                actor_node="agent:test",
                source_message_id="seed",
                source_author="user:testuser",
                source_content="Project Owner",
            ),
        )

        turns = [_turn(0)]
        assert await sys.form_un_formed(turns, "time-based") == 1
        assert sys._store._conn.execute(
            "SELECT COUNT(*) FROM memory_entities"
        ).fetchone()[0] == expected_links

        telemetry_path = sys.entity_formation_telemetry_path
        records = [
            json.loads(line)
            for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        ]
        assert len(records) == 1
        record = records[0]
        assert record["mode"] == mode
        assert record["contract_version"] == 2
        assert record["existing_links_made"] == 1
        assert record["serialized_registry_token_count"] > 0
        assert record["window_key"] == hashlib.sha256(
            (
                "0\x1f1\x1f2026-04-27T10:00:00Z"
                "\x1f2026-04-27T10:00:00Z"
            ).encode("utf-8")
        ).hexdigest()[:16]
        sys._store.close()

    @pytest.mark.asyncio
    async def test_write_mode_translates_proposal_and_unresolved(self, tmp_dir):
        entity = {
            "existing_keys": [],
            "new_entities": [{
                "entity_type": "project",
                "display_name": "Steel",
                "identity_note": "A newly observed framework.",
                "aliases": ["steel-framework"],
            }],
            "unresolved": [{
                "surface": "the team",
                "candidates": [],
                "reason": "not enough identity evidence",
            }],
        }
        llm = _FakeLLMClient(responses=[
            _mk_segs_json([_seg_spec(0, 0, worthwhile=True, entity=entity)])
        ])
        sys = _build_sys(
            tmp_dir,
            llm,
            _FakeEmbedder(),
            formation_v3_window_size=10,
            entity_resolution_mode="write",
        )

        assert await sys.form_un_formed([_turn(0)], "time-based") == 1
        assert sys._store._conn.execute(
            "SELECT entity_key, status FROM entities"
        ).fetchall() == [("project:steel", "pending")]
        assert sys._store._conn.execute(
            "SELECT entity_key FROM memory_entities"
        ).fetchall() == [("project:steel",)]
        events = sys._store._conn.execute(
            "SELECT event_type FROM entity_events ORDER BY sequence"
        ).fetchall()
        assert ("entity_created_pending",) in events
        assert ("entity_unresolved",) in events
        sys._store.close()

    @pytest.mark.asyncio
    async def test_cursor_advances_after_formation(self, tmp_dir):
        """Test 16: 50 turns + 2 worthwhile segments → cursor=50, ts matches turn[49]."""
        llm = _FakeLLMClient(responses=[
            _mk_segs_json([
                _seg_spec(0, 24, worthwhile=True),
                _seg_spec(25, 49, worthwhile=True),
            ])
        ])
        embedder = _FakeEmbedder()
        sys = _build_sys(tmp_dir, llm, embedder, formation_v3_window_size=80)
        turns = [_turn(i) for i in range(50)]
        n = await sys.form_un_formed(turns, "time-based")
        assert n == 2
        cursor_idx, last_ts = sys._store.get_formation_cursor()
        assert cursor_idx == 50
        assert last_ts == turns[49].timestamp.isoformat()
        sys._store.close()

    @pytest.mark.asyncio
    async def test_no_re_formation_on_repeat_call(self, tmp_dir):
        """Test 17: second call returns 0, no new entries / no LLM calls."""
        llm = _FakeLLMClient(responses=[
            _mk_segs_json([_seg_spec(0, 9, worthwhile=True)])
        ])
        embedder = _FakeEmbedder()
        sys = _build_sys(tmp_dir, llm, embedder, formation_v3_window_size=80)
        turns = [_turn(i) for i in range(10)]

        n1 = await sys.form_un_formed(turns, "time-based")
        n2 = await sys.form_un_formed(turns, "time-based")
        assert n1 >= 1
        assert n2 == 0
        # Only one LLM call (the first).
        assert len(llm.calls) == 1
        sys._store.close()

    @pytest.mark.asyncio
    async def test_startup_gap_detection(self, tmp_dir):
        """Test 18: pre-seed cursor at 30, history at 80 → form_un_formed forms gap."""
        embedder = _FakeEmbedder()
        llm = _FakeLLMClient(responses=[
            _mk_segs_json([_seg_spec(0, 49, worthwhile=True)])
        ])
        sys = _build_sys(tmp_dir, llm, embedder, formation_v3_window_size=80)
        # Pre-seed cursor.
        sys._store.set_formation_cursor(30, datetime.now(timezone.utc).isoformat())

        turns = [_turn(i) for i in range(80)]
        n = await sys.form_un_formed(turns, "startup")
        assert n >= 1
        cursor_idx, _ = sys._store.get_formation_cursor()
        assert cursor_idx == 80
        # Verify the un-formed slice was 30..80 (50 turns).
        assert "50" in llm.calls[0]["prompt"] or "49" in llm.calls[0]["prompt"]
        sys._store.close()

    @pytest.mark.asyncio
    async def test_time_based_respects_defer_tail(self, tmp_dir):
        """Test 19: time-based callback trims un-formed turns to those older
        than `defer_tail_seconds`. We exercise the trim logic directly by
        slicing history before calling form_un_formed (mirroring the timer loop).
        """
        embedder = _FakeEmbedder()
        llm = _FakeLLMClient(responses=[
            _mk_segs_json([_seg_spec(0, 34, worthwhile=True)])
        ])
        sys = _build_sys(tmp_dir, llm, embedder, formation_v3_window_size=80)

        now = datetime.now(timezone.utc)
        defer_secs = 300
        turns = []
        for i in range(40):
            t = Turn(
                role="user", content=f"t{i}", from_node="user:testuser",
                timestamp=now - timedelta(seconds=defer_secs * 2 + (40 - i) * 10),
                seq_id=i,
            )
            turns.append(t)
        # Make last 5 turns "fresh" (within defer_tail).
        for i, t in enumerate(turns[-5:]):
            t.timestamp = now - timedelta(seconds=10 + i)

        cutoff = now.timestamp() - defer_secs
        trimmed = [t for t in turns if t.timestamp.timestamp() <= cutoff]
        assert len(trimmed) == 35

        n = await sys.form_un_formed(trimmed, "time-based")
        assert n == 1
        cursor_idx, _ = sys._store.get_formation_cursor()
        assert cursor_idx == 35
        sys._store.close()

    @pytest.mark.asyncio
    async def test_concurrent_triggers_serialize(self, tmp_dir):
        """Test 20: while one form_un_formed runs, a second call no-ops on its slice."""
        gate = asyncio.Event()

        async def slow_response(prompt, kwargs):
            await gate.wait()
            return _mk_segs_json([_seg_spec(0, 9, worthwhile=True)])

        llm = _FakeLLMClient(side_effect=slow_response)
        embedder = _FakeEmbedder()
        sys = _build_sys(tmp_dir, llm, embedder, formation_v3_window_size=80)
        turns = [_turn(i) for i in range(10)]

        task1 = asyncio.create_task(sys.form_un_formed(turns, "time-based"))
        # Yield so task1 acquires the lock and is waiting on gate.
        await asyncio.sleep(0.05)
        # Second call should serialize on the lock; once task1 releases lock,
        # cursor will be at 10 so call 2 sees nothing un-formed.
        gate.set()
        n2 = await sys.form_un_formed(turns, "time-based")
        n1 = await task1
        assert n1 == 1
        assert n2 == 0
        sys._store.close()

    @pytest.mark.asyncio
    async def test_shutdown_handler_forms_un_formed(self, tmp_dir):
        """Test 21: pre-seed cursor at 20, history at 30; shutdown forms 20..30."""
        embedder = _FakeEmbedder()
        llm = _FakeLLMClient(responses=[
            _mk_segs_json([_seg_spec(0, 9, worthwhile=True)])
        ])
        sys = _build_sys(tmp_dir, llm, embedder, formation_v3_window_size=80)
        sys._store.set_formation_cursor(20, datetime.now(timezone.utc).isoformat())
        turns = [_turn(i) for i in range(30)]
        n = await sys.form_un_formed(turns, "shutdown")
        assert n == 1
        cursor_idx, _ = sys._store.get_formation_cursor()
        assert cursor_idx == 30
        sys._store.close()

    @pytest.mark.asyncio
    async def test_segmenter_failure_does_not_advance_cursor(self, tmp_dir):
        """Test 22: segmenter raises → cursor unchanged, no entries, retryable."""
        async def fail(prompt, kwargs):
            raise RuntimeError("seg failed")

        llm = _FakeLLMClient(side_effect=fail)
        embedder = _FakeEmbedder()
        sys = _build_sys(tmp_dir, llm, embedder, formation_v3_window_size=80)
        turns = [_turn(i) for i in range(10)]
        n = await sys.form_un_formed(turns, "time-based")
        assert n == 0
        cursor_idx, _ = sys._store.get_formation_cursor()
        assert cursor_idx == 0
        # Failure counter bumped.
        assert sys._parse_failure_count.get((0, 10)) == 1
        sys._store.close()


# ── Non-blocking startup tests (rev 4) ──────────────────────────────


class TestStartupNonBlocking:
    @pytest.mark.asyncio
    async def test_startup_does_not_block_mesh_ready(self, tmp_dir):
        """Test 23: long-running startup formation doesn't block return.

        We model `connect()` as kicking off the startup task and returning;
        verify the task is set, not done, and the call returns quickly.
        """
        slow_done = asyncio.Event()

        async def slow_response(prompt, kwargs):
            await asyncio.sleep(60)  # would take 60s if awaited
            slow_done.set()
            return _mk_segs_json([_seg_spec(0, 9, worthwhile=True)])

        llm = _FakeLLMClient(side_effect=slow_response)
        embedder = _FakeEmbedder()
        sys = _build_sys(tmp_dir, llm, embedder, formation_v3_window_size=80)
        turns = [_turn(i) for i in range(10)]

        # Simulate the agent_node connect() pattern: fire-and-forget task.
        t0 = time.monotonic()
        task = asyncio.create_task(sys.form_un_formed(turns, "startup"))
        await asyncio.sleep(0.05)  # give event loop a tick
        elapsed = time.monotonic() - t0

        assert elapsed < 1.0, f"startup task creation took {elapsed:.2f}s"
        assert not task.done(), "startup task completed inline (should be background)"
        assert not slow_done.is_set()
        # Cleanup.
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        sys._store.close()

    @pytest.mark.asyncio
    async def test_disconnect_cancels_in_flight_startup_task(self, tmp_dir):
        """Test 24: disconnect cancels long-running startup task cleanly."""
        async def slow_response(prompt, kwargs):
            await asyncio.sleep(60)
            return _mk_segs_json([_seg_spec(0, 9, worthwhile=True)])

        llm = _FakeLLMClient(side_effect=slow_response)
        embedder = _FakeEmbedder()
        sys = _build_sys(tmp_dir, llm, embedder, formation_v3_window_size=80)
        turns = [_turn(i) for i in range(10)]

        task = asyncio.create_task(sys.form_un_formed(turns, "startup"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()

        # Cursor should be unchanged.
        cursor_idx, _ = sys._store.get_formation_cursor()
        assert cursor_idx == 0
        sys._store.close()


# ── Parse-failure fallback (rev 5) ─────────────────────────────────


class TestParseFailureFallback:
    @pytest.mark.asyncio
    async def test_parse_failure_fallback_after_threshold(self, tmp_dir):
        """Test 25: 3 consecutive failures → fallback entry written, cursor advances."""
        async def fail(prompt, kwargs):
            raise json.JSONDecodeError("bad", "", 0)

        llm = _FakeLLMClient(side_effect=fail)
        embedder = _FakeEmbedder()
        sys = _build_sys(
            tmp_dir, llm, embedder,
            formation_v3_window_size=80,
            formation_v3_parse_failure_fallback_threshold=3,
        )
        turns = [_turn(i) for i in range(10)]

        # 1st and 2nd: cursor stays at 0, no entry.
        n1 = await sys.form_un_formed(turns, "time-based")
        assert n1 == 0
        n2 = await sys.form_un_formed(turns, "time-based")
        assert n2 == 0
        cursor_idx, _ = sys._store.get_formation_cursor()
        assert cursor_idx == 0

        # 3rd: writes fallback, cursor advances.
        n3 = await sys.form_un_formed(turns, "time-based")
        assert n3 == 1
        cursor_idx, _ = sys._store.get_formation_cursor()
        assert cursor_idx == 10

        # The placeholder is in pool with formation-fallback tag.
        assert any("formation-fallback" in e.tags for e in sys._pool)
        assert any("score:0" in e.tags for e in sys._pool)

        # Counter reset.
        assert (0, 10) not in sys._parse_failure_count

        # Next call: nothing un-formed (cursor at 10, history len 10).
        n4 = await sys.form_un_formed(turns, "time-based")
        assert n4 == 0
        sys._store.close()


# ── Embedding migration (rev 5) ────────────────────────────────────


class TestEmbeddingMigration:
    @pytest.mark.asyncio
    async def test_embedding_migration_runs_once(self, tmp_dir):
        """Test 26: re-embed once, idempotent on second call, no-op when v3 disabled."""
        llm = _FakeLLMClient()
        embedder = _FakeEmbedder()
        sys = _build_sys(tmp_dir, llm, embedder)

        # Pre-seed 5 v1-style entries.
        for i in range(5):
            entry = MemoryEntry(
                id=f"id{i}",
                created_at=datetime.now(timezone.utc),
                summary=f"sum-{i}",
                reflection="long-form reflection " + ("xxxx " * 20),
                trace="trace",
                trigger="trig",
                retrieval_key=f"rk-{i}",
                topic_label=f"topic-{i}",
                tags=["v1"],
                outcome="success",
                reflection_embedding=np.array([1.0] * 8, dtype=np.float32),
                retrieval_key_embedding=np.array([1.0] * 8, dtype=np.float32),
                weight=0.0,
                project="",
            )
            sys._store.insert(entry)

        sys._pool = sys._store.load()
        # Run migration.
        await sys._maybe_run_v3_embedding_migration()

        # Embedder should have been called once with all 5 targets.
        assert len(embedder.batch_inputs) == 1
        assert len(embedder.batch_inputs[0]) == 5
        # Targets are summary + " " + retrieval_key.
        assert "sum-0 rk-0" in embedder.batch_inputs[0]

        # migrations_complete row present.
        assert sys._store.is_migration_complete("v3_reflection_embedding")

        # Second call: no-op.
        await sys._maybe_run_v3_embedding_migration()
        assert len(embedder.batch_inputs) == 1  # no new call

        # With v3 disabled: no-op.
        sys2 = _build_sys(tmp_dir, llm, embedder, formation_v3_enabled=False, nickname="test-agent2")
        await sys2._maybe_run_v3_embedding_migration()
        sys._store.close()
        sys2._store.close()


# ── Window-drop short-circuit + summary still fires (rev 5) ─────────


class TestWindowDropShortCircuit:
    @pytest.mark.asyncio
    async def test_conversation_summary_still_fires_under_v3(self, tmp_dir):
        """Test 27: with v3 enabled, on_window_drop runs ONLY summary; with v3 disabled, runs all."""
        llm = _FakeLLMClient()
        embedder = _FakeEmbedder()

        # v3 enabled.
        sys_v3 = _build_sys(tmp_dir, llm, embedder, formation_v3_enabled=True)
        usc_calls = []
        async def usc(turns, count):
            usc_calls.append(count)
        sys_v3._update_conversation_summary = usc

        # Track formation/curation/segment_by_topic/reflect/new_project — must NOT be called.
        seg_called = []
        sys_v3._segment_by_topic = lambda turns: (seg_called.append(turns) or [])
        sys_v3._reflect_on_segment = AsyncMock()
        sys_v3._curate_active_map = AsyncMock()
        sys_v3._create_new_project_map = AsyncMock()
        sys_v3._run_formation_v3 = AsyncMock()

        turns = [_turn(i) for i in range(50)]
        await sys_v3.on_window_drop(turns)
        assert len(usc_calls) == 1
        assert seg_called == []
        sys_v3._reflect_on_segment.assert_not_called()
        sys_v3._curate_active_map.assert_not_called()
        sys_v3._create_new_project_map.assert_not_called()
        sys_v3._run_formation_v3.assert_not_called()
        sys_v3._store.close()

        # v3 disabled — legacy path runs.
        sys_legacy = _build_sys(tmp_dir, llm, embedder, formation_v3_enabled=False, nickname="legacy")
        usc_calls2 = []
        async def usc2(turns, count):
            usc_calls2.append(count)
        sys_legacy._update_conversation_summary = usc2
        seg_called2 = []
        sys_legacy._segment_by_topic = lambda turns: (seg_called2.append(turns) or [("misc", turns)])
        sys_legacy._reflect_on_segment = AsyncMock(return_value=None)
        await sys_legacy.on_window_drop(turns)
        assert len(usc_calls2) == 1
        assert len(seg_called2) == 1
        # Legacy reflect-on-segment IS called.
        sys_legacy._reflect_on_segment.assert_called()
        sys_legacy._store.close()


# ── Token-pressure trigger (rev 6) ─────────────────────────────────


class TestTokenPressure:
    @pytest.mark.asyncio
    async def test_token_pressure_trigger_fires_and_resets_counter(self, tmp_dir):
        """Test 28: counter accumulates, fires task when crossing threshold,
        resets on cursor advance, threshold=0 disables."""
        llm = _FakeLLMClient(responses=[
            _mk_segs_json([_seg_spec(0, 4, worthwhile=True)])
        ])
        embedder = _FakeEmbedder()
        sys = _build_sys(tmp_dir, llm, embedder, formation_v3_window_size=80)

        # Build a stub agent_node-shaped object that mirrors _v3_on_turn_appended
        # behaviour. We'll exercise the logic locally rather than spinning up
        # a full AgentNode.
        spawned_tasks: list[asyncio.Task] = []

        class StubNode:
            def __init__(self, sys):
                self._memory_system = sys
                self._uncommitted_token_count = 0
                self._token_pressure_task = None
                self._conv_history = MagicMock()
                self._conv_history.window = []
                self.config = MagicMock()
                self.config.memory_formation_token_threshold = 1000
                self.node_id = "agent:test:bot"

            def _v3_on_turn_appended(self, turn):
                if not (self._memory_system and getattr(self._memory_system, "_formation_v3_enabled", False)):
                    return
                threshold = int(self.config.memory_formation_token_threshold)
                if threshold <= 0:
                    return
                self._uncommitted_token_count += int(getattr(turn, "token_estimate", 0) or 0)
                if self._uncommitted_token_count < threshold:
                    return
                lock = self._memory_system._formation_lock
                if lock and lock.locked():
                    return
                async def _runner():
                    history = list(self._conv_history.window)
                    if not history:
                        return
                    await self._memory_system.form_un_formed(history, "token-pressure")
                self._token_pressure_task = asyncio.create_task(_runner())
                spawned_tasks.append(self._token_pressure_task)

        node = StubNode(sys)
        # Wire cursor-advance callback.
        def _reset():
            node._uncommitted_token_count = 0
        sys._on_cursor_advance = _reset

        # Append turns whose tokens sum to 600 → no spawn.
        node._conv_history.window = []
        t1 = _turn(0, token_estimate=600)
        node._conv_history.window.append(t1)
        node._v3_on_turn_appended(t1)
        assert node._token_pressure_task is None
        assert node._uncommitted_token_count == 600

        # Append another worth 800 → cumulative 1400, crosses threshold.
        t2 = _turn(1, token_estimate=800)
        node._conv_history.window.append(t2)
        node._v3_on_turn_appended(t2)
        assert node._token_pressure_task is not None
        assert spawned_tasks[-1] is node._token_pressure_task
        # Wait for task.
        await spawned_tasks[-1]

        # Counter reset by callback.
        assert node._uncommitted_token_count == 0

        # Append 500 more → counter at 500 < 1000, no new spawn.
        prev_task = node._token_pressure_task
        t3 = _turn(2, token_estimate=500)
        node._conv_history.window.append(t3)
        node._v3_on_turn_appended(t3)
        assert node._token_pressure_task is prev_task

        # Threshold=0 disables.
        node._uncommitted_token_count = 0
        node.config.memory_formation_token_threshold = 0
        prev_task2 = node._token_pressure_task
        t4 = _turn(3, token_estimate=100000)
        node._conv_history.window.append(t4)
        node._v3_on_turn_appended(t4)
        assert node._token_pressure_task is prev_task2
        sys._store.close()

    @pytest.mark.asyncio
    async def test_token_pressure_no_double_spawn_while_lock_held(self, tmp_dir):
        """Test 29: with lock held, token-pressure does not spawn; only after release."""
        gate = asyncio.Event()

        async def slow_response(prompt, kwargs):
            await gate.wait()
            return _mk_segs_json([_seg_spec(0, 4, worthwhile=True)])

        llm = _FakeLLMClient(side_effect=slow_response)
        embedder = _FakeEmbedder()
        sys = _build_sys(tmp_dir, llm, embedder, formation_v3_window_size=80)

        spawned_tasks: list[asyncio.Task] = []

        class StubNode:
            def __init__(self, sys):
                self._memory_system = sys
                self._uncommitted_token_count = 0
                self._token_pressure_task = None
                self._conv_history = MagicMock()
                self._conv_history.window = [_turn(i, token_estimate=10) for i in range(5)]
                self.config = MagicMock()
                self.config.memory_formation_token_threshold = 1000
                self.node_id = "agent:test:bot"

            def _v3_on_turn_appended(self, turn):
                threshold = int(self.config.memory_formation_token_threshold)
                if threshold <= 0:
                    return
                self._uncommitted_token_count += int(getattr(turn, "token_estimate", 0) or 0)
                if self._uncommitted_token_count < threshold:
                    return
                lock = self._memory_system._formation_lock
                if lock and lock.locked():
                    return
                async def _runner():
                    await self._memory_system.form_un_formed(
                        list(self._conv_history.window), "token-pressure",
                    )
                self._token_pressure_task = asyncio.create_task(_runner())
                spawned_tasks.append(self._token_pressure_task)

        node = StubNode(sys)
        def _reset():
            node._uncommitted_token_count = 0
        sys._on_cursor_advance = _reset

        # Hold the formation lock by spawning a fake formation.
        held_task = asyncio.create_task(
            sys.form_un_formed(list(node._conv_history.window), "time-based"),
        )
        await asyncio.sleep(0.05)  # let it acquire the lock
        assert sys._formation_lock.locked()

        # Push counter past threshold via 5 calls to the hook.
        for _ in range(5):
            t = _turn(0, token_estimate=1000)
            node._v3_on_turn_appended(t)
        # No spawn while lock held.
        assert spawned_tasks == []
        assert node._uncommitted_token_count >= 5000

        # Release the lock; held_task completes.
        gate.set()
        n_held = await held_task
        assert n_held >= 0
        # held_task's success advances the cursor and resets the counter.
        assert node._uncommitted_token_count == 0

        # Now another append crossing threshold should spawn.
        node._uncommitted_token_count = 0
        # Reset cursor so there's something to form (held_task already advanced).
        sys._store.set_formation_cursor(0, "")
        t = _turn(0, token_estimate=1500)
        node._v3_on_turn_appended(t)
        assert len(spawned_tasks) == 1
        await spawned_tasks[0]
        sys._store.close()


# ── Persist-failure fallback (rev 7 — code-review Issue 1) ──────────


class TestPersistFailureFallback:
    @pytest.mark.asyncio
    async def test_persist_failure_triggers_fallback_after_threshold(self, tmp_dir):
        """Test 30: Persist-side exceptions increment the failure counter and
        after 3 strikes the fallback path fires (cursor advances, placeholder
        entry written tagged formation-fallback)."""
        # Segmenter returns a valid single-segment response on every call.
        good_response = _mk_segs_json([_seg_spec(0, 10)])
        llm = _FakeLLMClient(responses=[good_response])
        embedder = _FakeEmbedder()
        sys = _build_sys(
            tmp_dir, llm, embedder,
            formation_v3_window_size=80,
            formation_v3_parse_failure_fallback_threshold=3,
        )
        turns = [_turn(i) for i in range(10)]

        # Patch _persist_v3_entries_atomic to raise. The fallback path
        # uses the store's insert_entry_and_advance_cursor directly, so it
        # is not affected by this monkey-patch.
        original_persist = sys._persist_v3_entries_atomic

        async def boom(**kwargs):
            raise RuntimeError("simulated DB write error")

        sys._persist_v3_entries_atomic = boom

        # 1st and 2nd calls: persist raises, cursor stays put.
        n1 = await sys.form_un_formed(turns, "time-based")
        assert n1 == 0
        cursor_idx, _ = sys._store.get_formation_cursor()
        assert cursor_idx == 0
        assert sys._parse_failure_count.get((0, 10)) == 1

        n2 = await sys.form_un_formed(turns, "time-based")
        assert n2 == 0
        cursor_idx, _ = sys._store.get_formation_cursor()
        assert cursor_idx == 0
        assert sys._parse_failure_count.get((0, 10)) == 2

        # 3rd call: counter hits threshold, fallback path fires (which uses
        # the store directly, not the patched _persist_v3_entries_atomic).
        n3 = await sys.form_un_formed(turns, "time-based")
        assert n3 == 1
        cursor_idx, _ = sys._store.get_formation_cursor()
        assert cursor_idx == 10

        # Placeholder entry is in the pool with formation-fallback tag.
        assert any("formation-fallback" in e.tags for e in sys._pool)

        # Counter reset.
        assert (0, 10) not in sys._parse_failure_count

        # Restore + verify subsequent formation runs cleanly with no
        # un-formed turns left.
        sys._persist_v3_entries_atomic = original_persist
        n4 = await sys.form_un_formed(turns, "time-based")
        assert n4 == 0
        sys._store.close()


# ── Atomic embedding migration (rev 7 — code-review Issue 2) ────────


class TestEmbeddingMigrationAtomicity:
    @pytest.mark.asyncio
    async def test_embedding_migration_atomic_on_interruption(self, tmp_dir):
        """Test 31: If the embedding migration is interrupted mid-transaction,
        the migration marker is NOT written and on retry the migration runs
        cleanly to completion (no half-applied state)."""
        llm = _FakeLLMClient()
        embedder = _FakeEmbedder()
        sys = _build_sys(tmp_dir, llm, embedder)

        # Pre-seed 5 v1-style entries.
        original_embeddings = []
        for i in range(5):
            emb = np.array([float(i)] * 8, dtype=np.float32)
            original_embeddings.append(emb)
            entry = MemoryEntry(
                id=f"id{i}",
                created_at=datetime.now(timezone.utc),
                summary=f"sum-{i}",
                reflection="long-form reflection " + ("xxxx " * 20),
                trace="trace",
                trigger="trig",
                retrieval_key=f"rk-{i}",
                topic_label=f"topic-{i}",
                tags=["v1"],
                outcome="success",
                reflection_embedding=emb,
                retrieval_key_embedding=emb,
                weight=0.0,
                project="",
            )
            sys._store.insert(entry)
        sys._pool = sys._store.load()

        # Force the bulk transaction to fail mid-way: monkey-patch the
        # store helper to raise after embeddings have been computed.
        original_bulk = sys._store.bulk_update_reflection_embeddings_and_mark_migration

        def failing_bulk(updates, migration_name):
            raise RuntimeError("simulated mid-transaction crash")

        sys._store.bulk_update_reflection_embeddings_and_mark_migration = failing_bulk

        # First migration attempt: fails.
        await sys._maybe_run_v3_embedding_migration()

        # Marker is NOT written.
        assert not sys._store.is_migration_complete("v3_reflection_embedding")

        # Embeddings are unchanged in the DB (rollback held).
        reloaded = sys._store.load()
        for i, e in enumerate(reloaded):
            np.testing.assert_array_equal(
                e.reflection_embedding, original_embeddings[i],
            )

        # Restore the real helper, retry: now succeeds atomically.
        sys._store.bulk_update_reflection_embeddings_and_mark_migration = original_bulk
        await sys._maybe_run_v3_embedding_migration()

        # Marker now present.
        assert sys._store.is_migration_complete("v3_reflection_embedding")

        # All 5 entries have new embeddings (not the original [i]*8 vectors).
        reloaded = sys._store.load()
        for i, e in enumerate(reloaded):
            assert not np.array_equal(e.reflection_embedding, original_embeddings[i]), \
                f"entry {i} embedding was not updated"

        sys._store.close()
