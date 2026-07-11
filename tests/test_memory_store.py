"""Tests for mesh/memory/store.py — SQLite persistence layer."""

import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pytest

from mesh.memory.store import (
    MemoryEntry,
    MemoryStore,
    _serialize_tags,
    _parse_tags,
    _serialize_embedding,
    _deserialize_embedding,
)


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    """Temporary directory for test databases."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def store(tmp_dir):
    """Fresh MemoryStore in a temp directory."""
    s = MemoryStore("test-agent", db_dir=tmp_dir)
    yield s
    s.close()


def _make_entry(
    entry_id: str = "abc123",
    summary: str = "Test summary",
    reflection: str = "Test reflection",
    trace: str = "[TOOL] bash(ls)",
    trigger: str = "Fix the bug",
    tags: list[str] | None = None,
    outcome: str = "success",
    dim: int = 8,
) -> MemoryEntry:
    """Create a MemoryEntry with random embeddings."""
    rng = np.random.RandomState(hash(entry_id) % 2**31)
    return MemoryEntry(
        id=entry_id,
        created_at=datetime.now(timezone.utc),
        summary=summary,
        reflection=reflection,
        trace=trace,
        trigger=trigger,
        tags=tags or ["test"],
        outcome=outcome,
        reflection_embedding=rng.randn(dim).astype(np.float32),
        retrieval_key_embedding=rng.randn(dim).astype(np.float32),
        weight=0.5,
    )


# ── Tag serialization ──────────────────────────────────────────

class TestTagSerialization:
    def test_roundtrip(self):
        tags = ["nginx", "tls", "config"]
        assert _parse_tags(_serialize_tags(tags)) == tags

    def test_empty_list(self):
        assert _serialize_tags([]) == ""
        assert _parse_tags("") == []

    def test_single_tag(self):
        assert _parse_tags(_serialize_tags(["debug"])) == ["debug"]

    def test_whitespace_handling(self):
        assert _parse_tags("a , b , c") == ["a", "b", "c"]

    def test_empty_segments(self):
        assert _parse_tags("a,,b") == ["a", "b"]


# ── Embedding serialization ────────────────────────────────────

class TestEmbeddingSerialization:
    def test_roundtrip(self):
        emb = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        blob = _serialize_embedding(emb)
        restored = _deserialize_embedding(blob, dim=4)
        np.testing.assert_allclose(emb, restored)

    def test_none(self):
        assert _serialize_embedding(None) is None
        assert _deserialize_embedding(None) is None

    def test_preserves_dtype(self):
        emb = np.array([0.1, 0.2], dtype=np.float32)
        restored = _deserialize_embedding(_serialize_embedding(emb), dim=2)
        assert restored.dtype == np.float32


# ── MemoryEntry ─────────────────────────────────────────────────

class TestMemoryEntry:
    def test_new_id_unique(self):
        ids = {MemoryEntry.new_id() for _ in range(100)}
        assert len(ids) == 100

    def test_new_id_length(self):
        assert len(MemoryEntry.new_id()) == 12

    def test_default_fields(self):
        entry = MemoryEntry(
            id="test",
            created_at=datetime.now(timezone.utc),
            summary="s",
            reflection="r",
            trace="t",
            trigger="q",
        )
        assert entry.tags == []
        assert entry.outcome == "success"
        assert entry.reflection_embedding is None
        assert entry.retrieval_key_embedding is None
        assert entry.weight == 0.0


# ── MemoryStore CRUD ────────────────────────────────────────────

class TestMemoryStoreCRUD:
    def test_insert_and_load(self, store):
        entry = _make_entry()
        store.insert(entry)
        loaded = store.load()
        assert len(loaded) == 1
        assert loaded[0].id == entry.id
        assert loaded[0].summary == entry.summary
        assert loaded[0].reflection == entry.reflection
        assert loaded[0].trace == entry.trace
        assert loaded[0].trigger == entry.trigger
        assert loaded[0].tags == entry.tags
        assert loaded[0].outcome == entry.outcome

    def test_insert_preserves_embeddings(self, store):
        entry = _make_entry()
        store.insert(entry)
        loaded = store.load()
        np.testing.assert_allclose(
            loaded[0].reflection_embedding, entry.reflection_embedding, atol=1e-6
        )
        np.testing.assert_allclose(
            loaded[0].retrieval_key_embedding, entry.retrieval_key_embedding, atol=1e-6
        )

    def test_insert_preserves_weight(self, store):
        entry = _make_entry()
        entry.weight = 1.234
        store.insert(entry)
        loaded = store.load()
        assert loaded[0].weight == pytest.approx(1.234, abs=1e-6)

    def test_delete_existing(self, store):
        entry = _make_entry()
        store.insert(entry)
        assert store.delete(entry.id) is True
        assert store.load() == []

    def test_delete_nonexistent(self, store):
        assert store.delete("nonexistent") is False

    def test_get_existing(self, store):
        entry = _make_entry()
        store.insert(entry)
        got = store.get(entry.id)
        assert got is not None
        assert got.id == entry.id
        assert got.summary == entry.summary

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent") is None

    def test_count_empty(self, store):
        assert store.count() == 0

    def test_count_after_inserts(self, store):
        for i in range(5):
            store.insert(_make_entry(entry_id=f"entry_{i}"))
        assert store.count() == 5

    def test_count_after_delete(self, store):
        store.insert(_make_entry("a"))
        store.insert(_make_entry("b"))
        store.delete("a")
        assert store.count() == 1

    def test_insert_or_replace(self, store):
        """INSERT OR REPLACE should update existing entry."""
        entry = _make_entry()
        store.insert(entry)
        entry.summary = "Updated summary"
        store.insert(entry)
        loaded = store.load()
        assert len(loaded) == 1
        assert loaded[0].summary == "Updated summary"

    def test_list_all_alias(self, store):
        store.insert(_make_entry("a"))
        store.insert(_make_entry("b"))
        assert len(store.list_all()) == 2

    def test_load_order_by_created_at(self, store):
        """Entries should be ordered by created_at."""
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        e1 = _make_entry("e1")
        e1.created_at = now - timedelta(hours=2)
        e2 = _make_entry("e2")
        e2.created_at = now - timedelta(hours=1)
        e3 = _make_entry("e3")
        e3.created_at = now
        # Insert out of order
        store.insert(e3)
        store.insert(e1)
        store.insert(e2)
        loaded = store.load()
        assert [e.id for e in loaded] == ["e1", "e2", "e3"]


# ── Weight updates ──────────────────────────────────────────────

class TestWeightUpdates:
    def test_update_weight(self, store):
        entry = _make_entry()
        entry.weight = 0.0
        store.insert(entry)
        store.update_weight(entry.id, 2.5)
        loaded = store.load()
        assert loaded[0].weight == pytest.approx(2.5, abs=1e-6)

    def test_update_weights_batch(self, store):
        entries = [_make_entry(f"e{i}") for i in range(3)]
        for e in entries:
            store.insert(e)
        weights = {"e0": 1.0, "e1": 2.0, "e2": 3.0}
        store.update_weights_batch(weights)
        loaded = store.load()
        loaded_weights = {e.id: e.weight for e in loaded}
        assert loaded_weights["e0"] == pytest.approx(1.0, abs=1e-6)
        assert loaded_weights["e1"] == pytest.approx(2.0, abs=1e-6)
        assert loaded_weights["e2"] == pytest.approx(3.0, abs=1e-6)


# ── Persistence across reopens ──────────────────────────────────

class TestPersistence:
    def test_data_survives_close_reopen(self, tmp_dir):
        """Data persists after closing and reopening the store."""
        store1 = MemoryStore("persist-test", db_dir=tmp_dir)
        store1.insert(_make_entry("persist1"))
        store1.insert(_make_entry("persist2"))
        store1.close()

        store2 = MemoryStore("persist-test", db_dir=tmp_dir)
        loaded = store2.load()
        store2.close()

        assert len(loaded) == 2
        assert {e.id for e in loaded} == {"persist1", "persist2"}

    def test_db_file_created(self, tmp_dir):
        store = MemoryStore("filecheck", db_dir=tmp_dir)
        store.close()
        assert os.path.exists(os.path.join(tmp_dir, "filecheck.db"))

    def test_wal_mode(self, tmp_dir):
        """Database should use WAL journal mode."""
        store = MemoryStore("waltest", db_dir=tmp_dir)
        cursor = store._conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        store.close()
        assert mode == "wal"


# ── Edge cases ──────────────────────────────────────────────────

class TestEdgeCases:
    def test_unicode_content(self, store):
        entry = _make_entry()
        entry.summary = "Fixed the nginx config for 日本語 site"
        entry.reflection = "The encoding was wrong — needed UTF-8 🔧"
        store.insert(entry)
        loaded = store.load()
        assert loaded[0].summary == entry.summary
        assert loaded[0].reflection == entry.reflection

    def test_empty_strings(self, store):
        entry = _make_entry()
        entry.trace = ""
        entry.trigger = ""
        store.insert(entry)
        loaded = store.load()
        assert loaded[0].trace == ""
        assert loaded[0].trigger == ""

    def test_many_tags(self, store):
        entry = _make_entry()
        entry.tags = [f"tag{i}" for i in range(50)]
        store.insert(entry)
        loaded = store.load()
        assert len(loaded[0].tags) == 50

    def test_large_embedding(self, store):
        """Test with realistic 1536-dim embeddings."""
        entry = _make_entry(dim=8)
        # Replace with full-size embeddings
        entry.reflection_embedding = np.random.randn(1536).astype(np.float32)
        entry.retrieval_key_embedding = np.random.randn(1536).astype(np.float32)
        store.insert(entry)
        loaded = store.load()
        np.testing.assert_allclose(
            loaded[0].reflection_embedding, entry.reflection_embedding, atol=1e-6
        )


# ── Personality ────────────────────────────────────────────────

class TestPersonality:
    def test_get_empty(self, store):
        """New store has no personality."""
        assert store.get_personality() == ""

    def test_set_and_get(self, store):
        store.set_personality("I'm Bob, a grumpy sysadmin.")
        assert store.get_personality() == "I'm Bob, a grumpy sysadmin."

    def test_overwrite(self, store):
        store.set_personality("Version 1")
        store.set_personality("Version 2")
        assert store.get_personality() == "Version 2"

    def test_updated_at(self, store):
        assert store.personality_updated_at() is None
        store.set_personality("Something")
        ts = store.personality_updated_at()
        assert ts is not None
        # Should be a valid ISO timestamp
        datetime.fromisoformat(ts)

    def test_persists_across_reopen(self, tmp_dir):
        """Personality survives close and reopen."""
        s1 = MemoryStore("personality-test", db_dir=tmp_dir)
        s1.set_personality("Persistent personality")
        s1.close()

        s2 = MemoryStore("personality-test", db_dir=tmp_dir)
        assert s2.get_personality() == "Persistent personality"
        s2.close()

    def test_singleton_constraint(self, store):
        """Only one personality row exists (singleton)."""
        store.set_personality("First")
        store.set_personality("Second")
        cursor = store._conn.execute("SELECT COUNT(*) FROM personality")
        assert cursor.fetchone()[0] == 1
