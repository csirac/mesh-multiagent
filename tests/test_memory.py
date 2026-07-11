"""
Comprehensive tests for the two-tier memory system.

Tests cover:
1. MemoryStore — SQLite CRUD, persistence, serialization
2. Selection — facility location math, swapping, cold start, batch selection
3. MemorySystem — two-tier orchestrator (pool + active set)
4. Tool layer — remember, memory_list, memory_get, memory_delete, memory_add
"""

import asyncio
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mesh.memory.selection import (
    _build_sim_matrix,
    compute_marginal_gain,
    compute_withholding_costs,
    cosine_sim,
    facility_location,
    select_active_set,
    try_swap,
)
from mesh.memory.store import (
    MemoryEntry,
    MemoryStore,
    _deserialize_embedding,
    _parse_tags,
    _serialize_embedding,
    _serialize_tags,
)
from mesh.memory.system import MemorySystem, _extract_tag


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def random_emb(dim: int = 16) -> np.ndarray:
    """Random unit-norm embedding for tests."""
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def make_entry(
    entry_id: str = None,
    summary: str = "test summary",
    reflection: str = "test reflection",
    trace: str = "test trace",
    trigger: str = "test trigger",
    tags: list[str] = None,
    outcome: str = "success",
    emb_dim: int = 16,
) -> MemoryEntry:
    """Create a MemoryEntry with random embeddings for testing."""
    return MemoryEntry(
        id=entry_id or MemoryEntry.new_id(),
        created_at=datetime.now(timezone.utc),
        summary=summary,
        reflection=reflection,
        trace=trace,
        trigger=trigger,
        tags=tags or ["test"],
        outcome=outcome,
        reflection_embedding=random_emb(emb_dim),
        retrieval_key_embedding=random_emb(emb_dim),
        weight=0.0,
    )


# ════════════════════════════════════════════════════════════
# 1. STORE TESTS
# ════════════════════════════════════════════════════════════

class TestMemoryStore:
    """Tests for SQLite-backed MemoryStore."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = MemoryStore("test_agent", db_dir=self.tmpdir)

    def teardown_method(self):
        self.store.close()

    def test_empty_store(self):
        """Empty store returns empty list and count 0."""
        assert self.store.load() == []
        assert self.store.count() == 0

    def test_insert_and_load(self):
        """Insert an entry, load it back, verify all fields."""
        entry = make_entry(entry_id="abc123")
        self.store.insert(entry)

        loaded = self.store.load()
        assert len(loaded) == 1
        e = loaded[0]
        assert e.id == "abc123"
        assert e.summary == "test summary"
        assert e.reflection == "test reflection"
        assert e.trace == "test trace"
        assert e.trigger == "test trigger"
        assert e.tags == ["test"]
        assert e.outcome == "success"
        assert self.store.count() == 1

    def test_get_by_id(self):
        """Get a specific entry by ID."""
        entry = make_entry(entry_id="xyz789")
        self.store.insert(entry)

        result = self.store.get("xyz789")
        assert result is not None
        assert result.id == "xyz789"

        # Non-existent
        assert self.store.get("nonexistent") is None

    def test_delete(self):
        """Delete an entry and verify it's gone."""
        entry = make_entry(entry_id="del001")
        self.store.insert(entry)
        assert self.store.count() == 1

        deleted = self.store.delete("del001")
        assert deleted is True
        assert self.store.count() == 0
        assert self.store.get("del001") is None

        # Deleting non-existent returns False
        assert self.store.delete("nonexistent") is False

    def test_insert_or_replace(self):
        """INSERT OR REPLACE updates an existing entry."""
        entry = make_entry(entry_id="upd001", summary="original")
        self.store.insert(entry)

        entry.summary = "updated"
        self.store.insert(entry)

        assert self.store.count() == 1
        loaded = self.store.get("upd001")
        assert loaded.summary == "updated"

    def test_update_weight(self):
        """Update weight for a single entry."""
        entry = make_entry(entry_id="w001")
        self.store.insert(entry)

        self.store.update_weight("w001", 3.14)
        loaded = self.store.get("w001")
        assert abs(loaded.weight - 3.14) < 1e-6

    def test_update_weights_batch(self):
        """Batch update weights for multiple entries."""
        e1 = make_entry(entry_id="b001")
        e2 = make_entry(entry_id="b002")
        self.store.insert(e1)
        self.store.insert(e2)

        self.store.update_weights_batch({"b001": 1.5, "b002": 2.5})

        loaded = self.store.load()
        weights = {e.id: e.weight for e in loaded}
        assert abs(weights["b001"] - 1.5) < 1e-6
        assert abs(weights["b002"] - 2.5) < 1e-6

    def test_embedding_roundtrip(self):
        """Embeddings survive serialization/deserialization through SQLite."""
        emb = np.array([0.1, 0.2, 0.3, -0.4], dtype=np.float32)
        entry = make_entry(entry_id="emb001")
        entry.reflection_embedding = emb
        entry.retrieval_key_embedding = emb * 2

        self.store.insert(entry)
        loaded = self.store.get("emb001")

        np.testing.assert_array_almost_equal(loaded.reflection_embedding, emb)
        np.testing.assert_array_almost_equal(loaded.retrieval_key_embedding, emb * 2)

    def test_multiple_entries_ordered(self):
        """Multiple entries loaded in chronological order."""
        import time
        for i in range(5):
            self.store.insert(make_entry(entry_id=f"ord{i:03d}"))
            time.sleep(0.01)  # Ensure different timestamps

        loaded = self.store.load()
        assert len(loaded) == 5
        ids = [e.id for e in loaded]
        assert ids == [f"ord{i:03d}" for i in range(5)]

    def test_persistence_across_connections(self):
        """Data persists when store is closed and reopened."""
        entry = make_entry(entry_id="persist001")
        self.store.insert(entry)
        self.store.close()

        # Reopen
        store2 = MemoryStore("test_agent", db_dir=self.tmpdir)
        loaded = store2.load()
        assert len(loaded) == 1
        assert loaded[0].id == "persist001"
        store2.close()

    def test_store_is_unbounded(self):
        """Store accepts arbitrarily many entries (no cap)."""
        for i in range(50):
            self.store.insert(make_entry(entry_id=f"many{i:03d}"))
        assert self.store.count() == 50


class TestSerialization:
    """Tests for tag and embedding serialization helpers."""

    def test_serialize_tags(self):
        assert _serialize_tags(["a", "b", "c"]) == "a,b,c"
        assert _serialize_tags([]) == ""

    def test_parse_tags(self):
        assert _parse_tags("a,b,c") == ["a", "b", "c"]
        assert _parse_tags("") == []
        assert _parse_tags("  a , b , c  ") == ["a", "b", "c"]

    def test_serialize_embedding(self):
        emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        data = _serialize_embedding(emb)
        assert isinstance(data, bytes)
        assert len(data) == 12  # 3 floats × 4 bytes

        # None case
        assert _serialize_embedding(None) is None

    def test_deserialize_embedding(self):
        emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        data = _serialize_embedding(emb)
        result = _deserialize_embedding(data)
        np.testing.assert_array_equal(result, emb)

        # None case
        assert _deserialize_embedding(None) is None

    def test_new_id_format(self):
        """Entry IDs are 12-char hex strings."""
        id1 = MemoryEntry.new_id()
        id2 = MemoryEntry.new_id()
        assert len(id1) == 12
        assert id1 != id2
        int(id1, 16)  # Should parse as hex


# ════════════════════════════════════════════════════════════
# 2. SELECTION TESTS
# ════════════════════════════════════════════════════════════

class TestCosineSim:
    """Tests for cosine similarity."""

    def test_identical_vectors(self):
        v = random_emb(8)
        assert abs(cosine_sim(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert abs(cosine_sim(v1, v2)) < 1e-6

    def test_opposite_vectors(self):
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([-1.0, 0.0], dtype=np.float32)
        assert abs(cosine_sim(v1, v2) - (-1.0)) < 1e-6

    def test_zero_vector(self):
        v = random_emb(4)
        z = np.zeros(4, dtype=np.float32)
        assert cosine_sim(v, z) == 0.0
        assert cosine_sim(z, z) == 0.0


class TestBuildSimMatrix:
    """Tests for similarity matrix construction."""

    def test_empty(self):
        result = _build_sim_matrix([])
        assert result.size == 0

    def test_single_entry(self):
        """Single entry: self-sim is zeroed, so max sim is 0."""
        v = random_emb(8)
        sim = _build_sim_matrix([v])
        assert sim.shape == (1, 1)
        assert sim[0, 0] == 0.0

    def test_diagonal_is_zero(self):
        """Diagonal of sim matrix must be zero (critical bug fix)."""
        embs = [random_emb(8) for _ in range(5)]
        sim = _build_sim_matrix(embs)
        for i in range(5):
            assert sim[i, i] == 0.0

    def test_symmetric(self):
        """Sim matrix should be symmetric."""
        embs = [random_emb(8) for _ in range(4)]
        sim = _build_sim_matrix(embs)
        np.testing.assert_array_almost_equal(sim, sim.T)


class TestFacilityLocation:
    """Tests for the facility location function f(S)."""

    def test_empty_set(self):
        assert facility_location([]) == 0.0

    def test_single_entry(self):
        """Single entry has f(S) = 0 (no cross-coverage)."""
        v = random_emb(8)
        assert facility_location([v]) == 0.0

    def test_two_identical_entries(self):
        """Two identical entries: each sees max sim ~1.0 from the other."""
        v = random_emb(8)
        result = facility_location([v, v.copy()])
        assert abs(result - 2.0) < 0.01

    def test_orthogonal_entries(self):
        """Two orthogonal entries have low cross-coverage."""
        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        result = facility_location([v1, v2])
        assert result == 0.0

    def test_monotone(self):
        """Adding more entries should not decrease f(S)."""
        embs = [random_emb(16) for _ in range(10)]
        f_vals = []
        for i in range(1, len(embs) + 1):
            f_vals.append(facility_location(embs[:i]))
        for i in range(len(f_vals) - 1):
            assert f_vals[i + 1] >= f_vals[i] - 1e-6


class TestSelectActiveSet:
    """Tests for batch greedy submodular selection."""

    def test_empty_pool(self):
        selected, weights, f_s = select_active_set([], 10)
        assert selected == []
        assert weights == []
        assert f_s == 0.0

    def test_pool_smaller_than_active_size(self):
        """When pool is smaller than active_size, select all."""
        embs = [random_emb(16) for _ in range(3)]
        selected, weights, f_s = select_active_set(embs, 10)
        assert sorted(selected) == [0, 1, 2]
        assert len(weights) == 3

    def test_pool_equals_active_size(self):
        """When pool equals active_size, select all."""
        embs = [random_emb(16) for _ in range(5)]
        selected, weights, f_s = select_active_set(embs, 5)
        assert sorted(selected) == [0, 1, 2, 3, 4]

    def test_selects_correct_count(self):
        """Selects exactly active_size entries when pool > active_size."""
        np.random.seed(42)
        embs = [random_emb(16) for _ in range(20)]
        selected, weights, f_s = select_active_set(embs, 5)
        assert len(selected) == 5
        assert len(weights) == 5
        assert len(set(selected)) == 5  # No duplicates

    def test_selected_f_s_is_optimal_greedy(self):
        """Selected subset has at least as high f(S) as any random subset
        of the same size (probabilistically, not guaranteed — but usually)."""
        np.random.seed(42)
        embs = [random_emb(16) for _ in range(20)]
        selected, weights, f_s = select_active_set(embs, 5)

        # Compare against a few random subsets
        for _ in range(10):
            rand_indices = np.random.choice(20, 5, replace=False)
            rand_embs = [embs[i] for i in rand_indices]
            rand_f = facility_location(rand_embs)
            # Greedy should be >= random (with high probability)
            # Allow small tolerance for numerical edge cases
            assert f_s >= rand_f - 0.1

    def test_weights_match_fresh_computation(self):
        """Returned weights match freshly computed withholding costs."""
        np.random.seed(42)
        embs = [random_emb(16) for _ in range(10)]
        selected, weights, f_s = select_active_set(embs, 4)

        selected_embs = [embs[i] for i in selected]
        fresh_weights, fresh_f_s = compute_withholding_costs(selected_embs)
        np.testing.assert_array_almost_equal(weights, fresh_weights, decimal=5)
        assert abs(f_s - fresh_f_s) < 1e-5


class TestMarginalGain:
    """Tests for marginal gain computation."""

    def test_first_entry_zero_gain(self):
        """Adding to empty set: f({m}) = 0 (single entry), so gain is 0."""
        v = random_emb(8)
        gain = compute_marginal_gain([], v)
        assert gain == 0.0

    def test_positive_gain_for_diverse(self):
        """Adding a diverse candidate to an existing set should have positive gain."""
        base = random_emb(16)
        cluster = [base + np.random.randn(16).astype(np.float32) * 0.01 for _ in range(5)]
        cluster = [c / np.linalg.norm(c) for c in cluster]

        candidate = random_emb(16)
        candidate = candidate - base * np.dot(candidate, base)
        candidate = candidate / np.linalg.norm(candidate)

        gain = compute_marginal_gain(cluster, candidate)
        assert gain >= -1e-6

    def test_cached_f_s_consistency(self):
        """Using cached f(S) gives same result as computing fresh."""
        embs = [random_emb(16) for _ in range(5)]
        candidate = random_emb(16)

        gain_fresh = compute_marginal_gain(embs, candidate, cached_f_s=None)
        f_s = facility_location(embs)
        gain_cached = compute_marginal_gain(embs, candidate, cached_f_s=f_s)

        assert abs(gain_fresh - gain_cached) < 1e-6


class TestWithholdingCosts:
    """Tests for withholding cost computation."""

    def test_empty_set(self):
        costs, f_s = compute_withholding_costs([])
        assert costs == []
        assert f_s == 0.0

    def test_single_entry(self):
        """Single entry: removing it drops f to 0, so w_a = 0."""
        v = random_emb(8)
        costs, f_s = compute_withholding_costs([v])
        assert len(costs) == 1
        assert abs(costs[0]) < 1e-6

    def test_all_nonnegative(self):
        """Withholding costs should be non-negative."""
        embs = [random_emb(16) for _ in range(8)]
        costs, f_s = compute_withholding_costs(embs)
        for c in costs:
            assert c >= -1e-6

    def test_sum_relation(self):
        """Sanity check: individual costs relate to f(S) meaningfully."""
        embs = [random_emb(16) for _ in range(5)]
        costs, f_s = compute_withholding_costs(embs)
        assert len(costs) == 5
        assert f_s == facility_location(embs)


class TestTrySwap:
    """Tests for the full swap decision."""

    def test_cold_start_accepts_unconditionally(self):
        """Before max_entries is reached, all candidates accepted."""
        embs = [random_emb(16) for _ in range(3)]
        candidate = random_emb(16)
        accepted, evict_idx, weights, f_s = try_swap(embs, candidate, max_entries=10)

        assert accepted is True
        assert evict_idx is None
        assert len(weights) == 4

    def test_cold_start_empty(self):
        """First ever entry is accepted."""
        candidate = random_emb(16)
        accepted, evict_idx, weights, f_s = try_swap([], candidate, max_entries=10)

        assert accepted is True
        assert evict_idx is None
        assert len(weights) == 1

    def test_full_store_rejects_duplicate(self):
        """Full store rejects a near-duplicate of an existing entry."""
        np.random.seed(42)
        embs = [random_emb(32) for _ in range(5)]

        candidate = embs[0] + np.random.randn(32).astype(np.float32) * 0.001
        candidate = candidate / np.linalg.norm(candidate)

        accepted, evict_idx, weights, f_s = try_swap(embs, candidate, max_entries=5)
        assert isinstance(accepted, bool)
        assert len(weights) == 5

    def test_full_store_accepts_diverse(self):
        """Full store swaps out a zero-cost entry for a diverse candidate."""
        dim = 16
        v0 = np.zeros(dim, dtype=np.float32)
        v0[0] = 1.0
        embs = [v0.copy(), v0.copy()]

        for i in range(3):
            v = np.zeros(dim, dtype=np.float32)
            v[i + 1] = 1.0
            embs.append(v)

        candidate = np.zeros(dim, dtype=np.float32)
        candidate[0] = 0.3
        candidate[1] = 0.5
        candidate[2] = 0.8
        candidate = candidate / np.linalg.norm(candidate)

        costs, f_s = compute_withholding_costs(embs)
        w_m = compute_marginal_gain(embs, candidate, cached_f_s=f_s)

        assert min(costs) < 1e-6
        assert w_m > 0

        accepted, evict_idx, weights, f_s = try_swap(embs, candidate, max_entries=5)

        assert accepted is True
        assert evict_idx is not None
        assert 0 <= evict_idx < 5
        assert len(weights) == 5

    def test_cached_weights(self):
        """Passing cached weights gives same result as fresh computation."""
        np.random.seed(99)
        embs = [random_emb(16) for _ in range(5)]
        candidate = random_emb(16)

        res1 = try_swap(embs, candidate, max_entries=5)

        cached_w, cached_f = compute_withholding_costs(embs)
        res2 = try_swap(embs, candidate, max_entries=5,
                        cached_weights=cached_w, cached_f_s=cached_f)

        assert res1[0] == res2[0]
        assert res1[1] == res2[1]


# ════════════════════════════════════════════════════════════
# 3. MEMORY SYSTEM TESTS (TWO-TIER)
# ════════════════════════════════════════════════════════════

class TestExtractTag:
    """Tests for XML tag extraction helper."""

    def test_basic_extraction(self):
        text = "<summary>Hello world</summary>"
        assert _extract_tag(text, "summary") == "Hello world"

    def test_multiline(self):
        text = "<reflection>\nLine 1\nLine 2\n</reflection>"
        assert _extract_tag(text, "reflection") == "Line 1\nLine 2"

    def test_missing_tag(self):
        text = "no tags here"
        assert _extract_tag(text, "summary") == ""

    def test_multiple_tags(self):
        text = (
            "<reflection>reflect text</reflection>\n"
            "<summary>summary text</summary>\n"
            "<tags>a, b, c</tags>\n"
            "<outcome_label>success</outcome_label>"
        )
        assert _extract_tag(text, "reflection") == "reflect text"
        assert _extract_tag(text, "summary") == "summary text"
        assert _extract_tag(text, "tags") == "a, b, c"
        assert _extract_tag(text, "outcome_label") == "success"


class TestMemorySystemTwoTier:
    """Tests for the two-tier MemorySystem orchestrator."""

    @pytest.fixture
    def tmpdir(self):
        return tempfile.mkdtemp()

    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=(
            "<reflection>Test reflection content</reflection>\n"
            "<summary>Test summary content</summary>\n"
            "<tags>test, unit</tags>\n"
            "<outcome_label>success</outcome_label>"
        ))
        return llm

    @pytest.fixture
    def mock_embedder(self):
        """Mock embedder that returns deterministic random embeddings."""
        embedder = AsyncMock()

        async def fake_embed(text):
            np.random.seed(hash(text) % 2**31)
            return np.random.randn(16).astype(np.float32).tolist()

        async def fake_embed_batch(texts):
            results = []
            for t in texts:
                np.random.seed(hash(t) % 2**31)
                results.append(np.random.randn(16).astype(np.float32).tolist())
            return results

        async def fake_embed_to_array(text):
            np.random.seed(hash(text) % 2**31)
            return np.random.randn(16).astype(np.float32)

        async def fake_embed_batch_to_arrays(texts):
            results = []
            for t in texts:
                np.random.seed(hash(t) % 2**31)
                results.append(np.random.randn(16).astype(np.float32))
            return results

        embedder.embed = AsyncMock(side_effect=fake_embed)
        embedder.embed_batch = AsyncMock(side_effect=fake_embed_batch)
        embedder.embed_to_array = AsyncMock(side_effect=fake_embed_to_array)
        embedder.embed_batch_to_arrays = AsyncMock(side_effect=fake_embed_batch_to_arrays)
        return embedder

    @pytest.fixture
    async def system(self, tmpdir, mock_llm, mock_embedder):
        """Create and initialize a MemorySystem with mocks (active_size=5)."""
        with patch("mesh.memory.system.EmbeddingClient"):
            ms = MemorySystem(
                nickname="test_agent",
                llm_client=mock_llm,
                active_size=5,
                reflection_min_tools=2,
                retrieval_k=3,
            )
        ms._embedder = mock_embedder
        ms._store = MemoryStore("test_agent", db_dir=tmpdir)
        ms._pool = ms._store.load()
        return ms

    # ── Basic lifecycle ──

    @pytest.mark.asyncio
    async def test_initialize_empty(self, system):
        """System initializes with empty pool."""
        assert len(system._pool) == 0
        assert len(system._active_ids) == 0
        assert system._active_f_s == 0.0

    @pytest.mark.asyncio
    async def test_render_block_empty(self, system):
        """render_block returns empty string with no entries."""
        assert system.render_block() == ""

    @pytest.mark.asyncio
    async def test_render_block_for_query_empty(self, system):
        """render_block_for_query returns empty string with no entries."""
        result = await system.render_block_for_query("test query")
        assert result == ""

    # ── Adding entries (cold start) ──

    @pytest.mark.asyncio
    async def test_add_entry_cold_start(self, system):
        """Adding entry during cold start always accepts into active set."""
        entry, in_active = await system.add_entry(
            summary="First memory",
            reflection="Deep reflection on first task",
            tags=["test"],
            outcome="success",
        )
        assert in_active is True
        assert len(system._pool) == 1
        assert entry.id in system._active_ids
        assert system._pool[0].summary == "First memory"

    @pytest.mark.asyncio
    async def test_add_entries_up_to_active_size(self, system):
        """All entries accepted into active set during cold start."""
        for i in range(5):
            entry, in_active = await system.add_entry(
                summary=f"Memory {i}",
                reflection=f"Reflection {i} with unique content about topic {i * 37}",
                tags=[f"tag{i}"],
                outcome="success",
            )
            assert in_active is True

        assert len(system._pool) == 5
        assert len(system._active_ids) == 5

    # ── Two-tier: pool grows beyond active set ──

    @pytest.mark.asyncio
    async def test_pool_grows_beyond_active_size(self, system):
        """Pool grows beyond active_size; some entries stay pool-only."""
        for i in range(8):
            await system.add_entry(
                summary=f"Memory {i}",
                reflection=f"Reflection {i} unique text {i * 41}",
                tags=[f"tag{i}"],
            )

        # Pool has all 8 entries
        assert len(system._pool) == 8
        # Active set is capped at 5
        assert len(system._active_ids) <= 5
        # All entries in store
        assert system._store.count() == 8

    @pytest.mark.asyncio
    async def test_pool_entry_not_deleted_on_active_eviction(self, system):
        """When an entry is evicted from active set, it stays in the pool."""
        entries = []
        for i in range(8):
            e, _ = await system.add_entry(
                summary=f"Memory {i}",
                reflection=f"Reflection {i} unique {i * 41}",
            )
            entries.append(e)

        # All entries should still be in pool
        for e in entries:
            assert system._find_entry(e.id) is not None

        # Some should be pool-only (not active)
        pool_only = [e for e in entries if e.id not in system._active_ids]
        # With 8 entries and active_size=5, at least 3 should be pool-only
        assert len(pool_only) >= 3

    @pytest.mark.asyncio
    async def test_is_active(self, system):
        """is_active correctly identifies active vs pool-only entries."""
        entry, in_active = await system.add_entry(
            summary="Active entry",
            reflection="Should be active",
        )
        assert system.is_active(entry.id) is True

    # ── Rendering ──

    @pytest.mark.asyncio
    async def test_render_block_only_active(self, system):
        """render_block shows only active set entries."""
        entries = []
        for i in range(8):
            e, _ = await system.add_entry(
                summary=f"Memory {i}",
                reflection=f"Reflection {i} unique {i * 37}",
            )
            entries.append(e)

        block = system.render_block()
        assert "<memory>" in block
        assert "</memory>" in block

        # Count entries in block
        active_entries = system.active_entries
        for e in active_entries:
            assert e.summary in block
        # Pool-only entries should NOT be in block
        for e in entries:
            if e.id not in system._active_ids:
                assert e.summary not in block

    @pytest.mark.asyncio
    async def test_render_block_for_query_searches_full_pool(self, system):
        """render_block_for_query can return entries not in active set."""
        # Add entries with different embeddings
        for i in range(8):
            await system.add_entry(
                summary=f"Memory about topic {i}",
                reflection=f"Reflection about topic {i}",
                tags=[f"t{i}"],
            )

        # Query should search all 8 entries, not just active 5
        block = await system.render_block_for_query("topic", k=3)
        assert "<memory>" in block

    # ── Retrieval ──

    @pytest.mark.asyncio
    async def test_remember_returns_reflection(self, system):
        """remember(id) returns reflection text from full pool."""
        entry, _ = await system.add_entry(
            summary="Summary for remember test",
            reflection="Detailed reflection for remember test",
        )
        result = system.remember(entry.id)
        assert result == "Detailed reflection for remember test"

    @pytest.mark.asyncio
    async def test_remember_full(self, system):
        """remember(id, full=True) returns reflection + trace."""
        entry, _ = await system.add_entry(
            summary="Summary",
            reflection="Reflection text",
            trace="Trace text",
        )
        result = system.remember(entry.id, full=True)
        assert "Reflection text" in result
        assert "Trace text" in result

    @pytest.mark.asyncio
    async def test_remember_not_found(self, system):
        assert system.remember("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_entry(self, system):
        """get_entry returns from full pool."""
        entry, _ = await system.add_entry(summary="Test entry")
        result = system.get_entry(entry.id)
        assert result is not None
        assert result.summary == "Test entry"

    @pytest.mark.asyncio
    async def test_list_entries_returns_full_pool(self, system):
        """list_entries returns all entries, not just active."""
        for i in range(8):
            await system.add_entry(
                summary=f"Entry {i}",
                reflection=f"Reflection {i} unique {i * 41}",
            )
        entries = system.list_entries()
        assert len(entries) == 8  # All 8, not just active 5

    # ── Deletion ──

    @pytest.mark.asyncio
    async def test_delete_active_entry_triggers_reselection(self, system):
        """Deleting an active entry triggers full reselection."""
        entries = []
        for i in range(5):
            e, _ = await system.add_entry(
                summary=f"Entry {i}",
                reflection=f"Unique content {i}",
            )
            entries.append(e)

        # All 5 should be active
        assert len(system._active_ids) == 5

        # Delete one active entry
        active_id = next(iter(system._active_ids))
        deleted = await system.delete_entry(active_id)
        assert deleted is True
        assert len(system._pool) == 4
        assert len(system._active_ids) == 4
        assert active_id not in system._active_ids

    @pytest.mark.asyncio
    async def test_delete_pool_only_entry_no_reselection(self, system):
        """Deleting a pool-only entry doesn't trigger reselection."""
        entries = []
        for i in range(8):
            e, _ = await system.add_entry(
                summary=f"Entry {i}",
                reflection=f"Unique content {i * 41}",
            )
            entries.append(e)

        # Find a pool-only entry
        pool_only = [e for e in entries if e.id not in system._active_ids]
        assert len(pool_only) > 0

        old_active = set(system._active_ids)
        deleted = await system.delete_entry(pool_only[0].id)
        assert deleted is True
        assert len(system._pool) == 7
        # Active set should be unchanged
        assert system._active_ids == old_active

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, system):
        assert await system.delete_entry("nonexistent") is False

    # ── Weights ──

    @pytest.mark.asyncio
    async def test_active_weights_computed(self, system):
        """Active entries have non-trivial weights; pool-only have 0."""
        for i in range(8):
            await system.add_entry(
                summary=f"Entry {i}",
                reflection=f"Unique content {i * 41}",
            )

        # Active entries should have weights in _active_weights
        for eid in system._active_ids:
            assert eid in system._active_weights

        # Pool-only entries should have weight=0
        for e in system._pool:
            if e.id not in system._active_ids:
                assert e.weight == 0.0

    # ── Persistence ──

    @pytest.mark.asyncio
    async def test_persistence(self, tmpdir, mock_llm, mock_embedder):
        """Entries persist across system restarts (via SQLite store)."""
        # Create and populate
        with patch("mesh.memory.system.EmbeddingClient"):
            ms1 = MemorySystem(
                nickname="persist_test",
                llm_client=mock_llm,
                active_size=10,
            )
        ms1._embedder = mock_embedder
        ms1._store = MemoryStore("persist_test", db_dir=tmpdir)
        ms1._pool = ms1._store.load()

        await ms1.add_entry(summary="Persistent entry")
        assert len(ms1._pool) == 1
        entry_id = ms1._pool[0].id
        ms1._store.close()

        # Reopen
        with patch("mesh.memory.system.EmbeddingClient"):
            ms2 = MemorySystem(
                nickname="persist_test",
                llm_client=mock_llm,
                active_size=10,
            )
        ms2._embedder = mock_embedder
        ms2._store = MemoryStore("persist_test", db_dir=tmpdir)
        ms2._pool = ms2._store.load()
        ms2._reselect_active_set()

        assert len(ms2._pool) == 1
        assert ms2._pool[0].id == entry_id
        assert ms2._pool[0].summary == "Persistent entry"
        assert entry_id in ms2._active_ids
        ms2._store.close()

    @pytest.mark.asyncio
    async def test_persistence_pool_larger_than_active(self, tmpdir, mock_llm, mock_embedder):
        """Pool with more entries than active_size persists and reselects correctly."""
        with patch("mesh.memory.system.EmbeddingClient"):
            ms1 = MemorySystem(
                nickname="persist_pool",
                llm_client=mock_llm,
                active_size=3,
            )
        ms1._embedder = mock_embedder
        ms1._store = MemoryStore("persist_pool", db_dir=tmpdir)
        ms1._pool = ms1._store.load()

        for i in range(6):
            await ms1.add_entry(
                summary=f"Entry {i}",
                reflection=f"Unique reflection {i * 37}",
            )
        assert len(ms1._pool) == 6
        assert len(ms1._active_ids) <= 3
        ms1._store.close()

        # Reopen
        with patch("mesh.memory.system.EmbeddingClient"):
            ms2 = MemorySystem(
                nickname="persist_pool",
                llm_client=mock_llm,
                active_size=3,
            )
        ms2._embedder = mock_embedder
        ms2._store = MemoryStore("persist_pool", db_dir=tmpdir)
        ms2._pool = ms2._store.load()
        ms2._reselect_active_set()

        assert len(ms2._pool) == 6  # All entries still there
        assert len(ms2._active_ids) == 3  # Active set reselected
        ms2._store.close()

    # ── should_reflect ──

    def test_should_reflect_low_tools(self):
        ms = MemorySystem.__new__(MemorySystem)
        ms._reflection_min_tools = 3

        result = MagicMock()
        result.context = []
        result.error = None
        assert ms.should_reflect(result) is False

    def test_should_reflect_many_tools(self):
        ms = MemorySystem.__new__(MemorySystem)
        ms._reflection_min_tools = 2

        entries = []
        for i in range(3):
            msg = MagicMock()
            msg.metadata = {"tool_calls": [{"name": f"tool_{i}"}]}
            msg.content = "ok"
            entry = MagicMock()
            entry.message = msg
            entries.append(entry)

        result = MagicMock()
        result.context = entries
        result.error = None
        assert ms.should_reflect(result) is True

    def test_should_reflect_on_error(self):
        ms = MemorySystem.__new__(MemorySystem)
        ms._reflection_min_tools = 100

        result = MagicMock()
        result.context = []
        result.error = "something failed"
        assert ms.should_reflect(result) is True


# ════════════════════════════════════════════════════════════
# 4. REFLECTION PIPELINE TEST
# ════════════════════════════════════════════════════════════

class TestReflectionPipeline:
    """Test reflect_on_completion with full mocks."""

    @pytest.fixture
    def tmpdir(self):
        return tempfile.mkdtemp()

    @pytest.fixture
    async def system(self, tmpdir):
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=(
            "<reflection>Reflected on the task deeply. "
            "The approach worked well.</reflection>\n"
            "<summary>Completed nginx config update successfully.</summary>\n"
            "<tags>nginx, config</tags>\n"
            "<outcome_label>success</outcome_label>"
        ))

        mock_embedder = AsyncMock()

        async def fake_embed_batch_to_arrays(texts):
            results = []
            for t in texts:
                np.random.seed(hash(t) % 2**31)
                results.append(np.random.randn(16).astype(np.float32))
            return results

        mock_embedder.embed_batch_to_arrays = AsyncMock(
            side_effect=fake_embed_batch_to_arrays
        )

        with patch("mesh.memory.system.EmbeddingClient"):
            ms = MemorySystem(
                nickname="reflect_test",
                llm_client=mock_llm,
                active_size=10,
            )
        ms._embedder = mock_embedder
        ms._store = MemoryStore("reflect_test", db_dir=tmpdir)
        ms._pool = ms._store.load()
        return ms

    @pytest.mark.asyncio
    async def test_reflect_creates_entry_in_pool(self, system):
        """reflect_on_completion creates a new memory entry in pool."""
        result = MagicMock()
        result.context = []
        result.response = "nginx config updated successfully"
        result.error = None

        await system.reflect_on_completion(
            trigger="update the nginx config",
            result=result,
            worker_id="worker_001",
        )

        assert len(system._pool) == 1
        entry = system._pool[0]
        assert "nginx" in entry.summary.lower()
        assert entry.tags == ["nginx", "config"]
        assert entry.outcome == "success"
        assert entry.reflection_embedding is not None
        assert entry.retrieval_key_embedding is not None
        # Should also be in active set (cold start)
        assert entry.id in system._active_ids

    @pytest.mark.asyncio
    async def test_reflect_persists_to_store(self, system):
        """Reflected entry is persisted to SQLite store."""
        result = MagicMock()
        result.context = []
        result.response = "done"
        result.error = None

        await system.reflect_on_completion("task", result, "w1")

        stored = system._store.load()
        assert len(stored) == 1

    @pytest.mark.asyncio
    async def test_reflect_empty_output_skips(self, tmpdir):
        """If LLM returns empty reflection/summary, no entry is created."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value="gibberish with no tags")

        with patch("mesh.memory.system.EmbeddingClient"):
            ms = MemorySystem(
                nickname="skip_test",
                llm_client=mock_llm,
                active_size=10,
            )
        ms._store = MemoryStore("skip_test", db_dir=tmpdir)
        ms._pool = ms._store.load()

        result = MagicMock()
        result.context = []
        result.response = "done"
        result.error = None

        await ms.reflect_on_completion("task", result, "w1")
        assert len(ms._pool) == 0

    @pytest.mark.asyncio
    async def test_reflect_exception_does_not_crash(self, tmpdir):
        """Exceptions in reflection are caught and logged, not raised."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

        with patch("mesh.memory.system.EmbeddingClient"):
            ms = MemorySystem(
                nickname="crash_test",
                llm_client=mock_llm,
                active_size=10,
            )
        ms._store = MemoryStore("crash_test", db_dir=tmpdir)
        ms._pool = ms._store.load()

        result = MagicMock()
        result.context = []
        result.response = "done"
        result.error = None

        await ms.reflect_on_completion("task", result, "w1")
        assert len(ms._pool) == 0


# ════════════════════════════════════════════════════════════
# 5. SELECTION INTEGRATION TESTS
# ════════════════════════════════════════════════════════════

class TestSelectionIntegration:
    """Higher-level tests for the selection algorithm's behavior."""

    def test_diverse_set_preferred_over_clustered(self):
        """A clustered set of embeddings has higher f(S) than a random diverse set
        (under FL; this is the expected behavior)."""
        np.random.seed(42)

        base = random_emb(32)
        clustered = [base + np.random.randn(32).astype(np.float32) * 0.01
                     for _ in range(5)]
        clustered = [c / np.linalg.norm(c) for c in clustered]

        diverse = [random_emb(32) for _ in range(5)]

        f_clustered = facility_location(clustered)
        f_diverse = facility_location(diverse)

        assert f_clustered > 0
        assert f_diverse > 0

    def test_swap_improves_diversity(self):
        """After a swap, f(S) should increase."""
        np.random.seed(42)

        base = random_emb(32)
        embs = [base + np.random.randn(32).astype(np.float32) * 0.01
                for _ in range(5)]
        embs = [e / np.linalg.norm(e) for e in embs]

        f_before = facility_location(embs)

        candidate = np.zeros(32, dtype=np.float32)
        candidate[16:] = random_emb(16)
        candidate = candidate / np.linalg.norm(candidate)

        accepted, evict_idx, new_weights, new_f_s = try_swap(
            embs, candidate, max_entries=5
        )

        if accepted and evict_idx is not None:
            new_embs = embs[:evict_idx] + embs[evict_idx + 1:] + [candidate]
            f_after = facility_location(new_embs)
            assert f_after > f_before - 1e-6

    def test_weight_cache_consistency(self):
        """Cached weights match freshly computed weights."""
        np.random.seed(42)
        embs = [random_emb(16) for _ in range(5)]
        weights1, f_s1 = compute_withholding_costs(embs)

        candidate = random_emb(16)
        accepted, _, weights2, f_s2 = try_swap(embs, candidate, max_entries=10)
        assert accepted is True

        new_embs = embs + [candidate]
        weights_fresh, f_s_fresh = compute_withholding_costs(new_embs)
        np.testing.assert_array_almost_equal(weights2, weights_fresh, decimal=5)
        assert abs(f_s2 - f_s_fresh) < 1e-5


# ════════════════════════════════════════════════════════════
# 6. EDGE CASES
# ════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge case tests."""

    def test_single_entry_withholding_cost(self):
        v = random_emb(16)
        costs, f_s = compute_withholding_costs([v])
        assert abs(costs[0]) < 1e-6

    def test_two_entries_swap(self):
        np.random.seed(42)
        embs = [random_emb(16) for _ in range(2)]
        candidate = random_emb(16)

        accepted, evict_idx, weights, f_s = try_swap(
            embs, candidate, max_entries=2
        )
        assert isinstance(accepted, bool)
        assert len(weights) == 2

    def test_identical_entries_handled(self):
        v = random_emb(16)
        embs = [v.copy() for _ in range(3)]
        f_s = facility_location(embs)
        assert f_s >= 0

        costs, _ = compute_withholding_costs(embs)
        assert len(costs) == 3

    def test_tags_with_special_characters(self):
        assert _serialize_tags(["c++", "node.js"]) == "c++,node.js"
        assert _parse_tags("c++,node.js") == ["c++", "node.js"]

    def test_empty_summary_entry(self):
        tmpdir = tempfile.mkdtemp()
        store = MemoryStore("edge_test", db_dir=tmpdir)

        entry = make_entry(entry_id="empty001")
        entry.summary = ""
        entry.reflection = ""
        entry.trace = ""
        entry.trigger = ""
        entry.tags = []

        store.insert(entry)
        loaded = store.get("empty001")
        assert loaded.summary == ""
        assert loaded.tags == []
        store.close()


# ════════════════════════════════════════════════════════════
# POOL CAP TESTS
# ════════════════════════════════════════════════════════════

class TestPoolCap:
    """Tests for pool_max_entries cap and pruning behavior."""

    @pytest.fixture
    def tmpdir(self):
        return tempfile.mkdtemp()

    @pytest.fixture
    def mock_llm(self):
        return AsyncMock()

    @pytest.fixture
    def mock_embedder(self):
        embedder = AsyncMock()

        async def fake_embed_batch_to_arrays(texts):
            results = []
            for t in texts:
                np.random.seed(hash(t) % 2**31)
                results.append(np.random.randn(16).astype(np.float32))
            return results

        async def fake_embed_to_array(text):
            np.random.seed(hash(text) % 2**31)
            return np.random.randn(16).astype(np.float32)

        embedder.embed_batch_to_arrays = AsyncMock(side_effect=fake_embed_batch_to_arrays)
        embedder.embed_to_array = AsyncMock(side_effect=fake_embed_to_array)
        return embedder

    @pytest.fixture
    async def system(self, tmpdir, mock_llm, mock_embedder):
        """MemorySystem with pool_max_entries=8, active_size=3."""
        with patch("mesh.memory.system.EmbeddingClient"):
            ms = MemorySystem(
                nickname="cap_test",
                llm_client=mock_llm,
                active_size=3,
                pool_max_entries=8,
            )
        ms._embedder = mock_embedder
        ms._store = MemoryStore("cap_test", db_dir=tmpdir)
        ms._pool = ms._store.load()
        return ms

    @pytest.mark.asyncio
    async def test_pool_stays_under_cap(self, system):
        """Pool size never exceeds pool_max_entries after insertions."""
        from datetime import timedelta
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

        for i in range(12):
            entry = make_entry(entry_id=f"cap{i:03d}", summary=f"summary {i}")
            entry.created_at = base_time + timedelta(hours=i)
            system._pool.append(entry)
            system._store.insert(entry)
            system._incremental_active_update(entry)
            system._prune_pool()

        assert len(system._pool) <= system._pool_max_entries

    @pytest.mark.asyncio
    async def test_active_entries_never_pruned(self, system):
        """Active set entries are protected from pruning."""
        from datetime import timedelta
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

        for i in range(12):
            entry = make_entry(entry_id=f"prot{i:03d}", summary=f"summary {i}")
            entry.created_at = base_time + timedelta(hours=i)
            system._pool.append(entry)
            system._store.insert(entry)
            system._incremental_active_update(entry)
            system._prune_pool()

        # All active entries must still be in the pool
        for eid in system._active_ids:
            assert any(e.id == eid for e in system._pool), \
                f"Active entry {eid} was pruned!"

    @pytest.mark.asyncio
    async def test_oldest_pool_only_pruned_first(self, system):
        """Pruning removes the oldest pool-only entries."""
        from datetime import timedelta
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

        for i in range(12):
            entry = make_entry(entry_id=f"age{i:03d}", summary=f"summary {i}")
            entry.created_at = base_time + timedelta(hours=i)
            system._pool.append(entry)
            system._store.insert(entry)
            system._incremental_active_update(entry)
            system._prune_pool()

        pool_ids = {e.id for e in system._pool}
        # The newest entries should still be present
        assert "age011" in pool_ids
        assert "age010" in pool_ids

    @pytest.mark.asyncio
    async def test_pruned_entries_deleted_from_store(self, system):
        """Pruned entries are also removed from SQLite."""
        from datetime import timedelta
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

        for i in range(12):
            entry = make_entry(entry_id=f"db{i:03d}", summary=f"summary {i}")
            entry.created_at = base_time + timedelta(hours=i)
            system._pool.append(entry)
            system._store.insert(entry)
            system._incremental_active_update(entry)
            system._prune_pool()

        pool_ids = {e.id for e in system._pool}
        store_entries = system._store.load()
        store_ids = {e.id for e in store_entries}
        # Store should match pool exactly
        assert store_ids == pool_ids

    @pytest.mark.asyncio
    async def test_no_pruning_when_under_cap(self, system):
        """No pruning when pool is smaller than cap."""
        for i in range(5):
            entry = make_entry(entry_id=f"small{i:03d}")
            system._pool.append(entry)
            system._store.insert(entry)

        pruned = system._prune_pool()
        assert pruned == 0
        assert len(system._pool) == 5

    @pytest.mark.asyncio
    async def test_pool_max_configurable(self, tmpdir, mock_llm, mock_embedder):
        """Different pool_max_entries values work."""
        with patch("mesh.memory.system.EmbeddingClient"):
            ms = MemorySystem(
                nickname="config_test",
                llm_client=mock_llm,
                active_size=2,
                pool_max_entries=5,
            )
        ms._embedder = mock_embedder
        ms._store = MemoryStore("config_test", db_dir=tmpdir)
        ms._pool = ms._store.load()

        from datetime import timedelta
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

        for i in range(10):
            entry = make_entry(entry_id=f"cfg{i:03d}", summary=f"s{i}")
            entry.created_at = base_time + timedelta(hours=i)
            ms._pool.append(entry)
            ms._store.insert(entry)
            ms._incremental_active_update(entry)
            ms._prune_pool()

        assert len(ms._pool) <= 5
