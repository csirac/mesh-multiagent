"""Tests for the interpretive essay storage layer (Phase 1).

Covers: schema creation, migration on existing DB, CRUD round-trip,
citation tracking, essay_edit exact-string-replacement semantics
including no-match and multiple-match failure modes, and tool-level
tests for essay_edit atomicity, citations, and cross-references.
"""

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pytest

from mesh.memory.store import MemoryStore


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def store(tmp_dir):
    s = MemoryStore("test-agent", db_dir=tmp_dir)
    yield s
    s.close()


# ── Schema creation ─────────────────────────────────────────────

class TestEssaySchema:
    def test_essays_table_exists(self, store):
        """The essays table is created on store init."""
        cursor = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='essays'"
        )
        assert cursor.fetchone() is not None

    def test_essays_table_columns(self, store):
        """Essays table has the expected columns."""
        cursor = store._conn.execute("PRAGMA table_info(essays)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            "entity_key", "title", "body", "citations",
            "cross_refs", "patch_count", "created_at", "updated_at",
            "curated_version", "verified_hash", "verified_at",
        }
        assert expected == columns

    def test_migration_on_existing_db(self, tmp_dir):
        """Opening a store on an existing DB without the essays table
        creates it via the CREATE TABLE IF NOT EXISTS."""
        db_path = os.path.join(tmp_dir, "legacy.db")
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                summary TEXT NOT NULL,
                reflection TEXT NOT NULL,
                trace TEXT NOT NULL,
                trigger_text TEXT NOT NULL,
                retrieval_key TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL,
                outcome TEXT NOT NULL,
                reflection_embedding BLOB,
                retrieval_key_embedding BLOB,
                weight REAL NOT NULL DEFAULT 0.0,
                topic_label TEXT DEFAULT '',
                project TEXT DEFAULT '',
                digest_candidate INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """)
        conn.commit()
        conn.close()

        s = MemoryStore("legacy", db_dir=tmp_dir)
        cursor = s._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='essays'"
        )
        assert cursor.fetchone() is not None
        s.close()


# ── CRUD round-trip ──────────────────────────────────────────────

class TestEssayCRUD:
    def test_create_and_get(self, store):
        store.create_essay(
            "person:kaylee",
            body="Kaylee is a key person.",
            title="Kaylee",
            citations=["m_abc123", "m_def456"],
            cross_refs=["project:fishing"],
        )
        essay = store.get_essay("person:kaylee")
        assert essay is not None
        assert essay["entity_key"] == "person:kaylee"
        assert essay["title"] == "Kaylee"
        assert essay["body"] == "Kaylee is a key person."
        assert essay["citations"] == ["m_abc123", "m_def456"]
        assert essay["cross_refs"] == ["project:fishing"]
        assert essay["patch_count"] == 0

    def test_get_nonexistent(self, store):
        assert store.get_essay("person:nobody") is None

    def test_create_duplicate_raises(self, store):
        store.create_essay("person:alice", body="Alice essay.")
        with pytest.raises(sqlite3.IntegrityError):
            store.create_essay("person:alice", body="Duplicate.")

    def test_create_defaults(self, store):
        """Create with minimal args uses defaults."""
        store.create_essay("event:test", body="Test event.")
        essay = store.get_essay("event:test")
        assert essay["title"] == ""
        assert essay["citations"] == []
        assert essay["cross_refs"] == []
        assert essay["patch_count"] == 0

    def test_list_essays_empty(self, store):
        assert store.list_essays() == []

    def test_list_essays_multiple(self, store):
        store.create_essay("person:alice", body="Alice.")
        store.create_essay("project:mesh", body="Mesh.")
        store.create_essay("event:deploy", body="Deploy.")
        essays = store.list_essays()
        assert len(essays) == 3
        keys = [e["entity_key"] for e in essays]
        assert keys == sorted(keys)
        assert "body" not in essays[0]

    def test_update_essay(self, store):
        store.create_essay("person:bob", body="Bob v1.", title="Bob")
        ok = store.update_essay("person:bob", body="Bob v2.", title="Robert")
        assert ok is True
        essay = store.get_essay("person:bob")
        assert essay["body"] == "Bob v2."
        assert essay["title"] == "Robert"
        assert essay["patch_count"] == 1

    def test_update_preserves_fields(self, store):
        """Update without title/citations preserves existing values."""
        store.create_essay(
            "person:x", body="V1.",
            title="X", citations=["m_001"], cross_refs=["person:y"],
        )
        store.update_essay("person:x", body="V2.")
        essay = store.get_essay("person:x")
        assert essay["body"] == "V2."
        assert essay["title"] == "X"
        assert essay["citations"] == ["m_001"]
        assert essay["cross_refs"] == ["person:y"]

    def test_update_nonexistent(self, store):
        ok = store.update_essay("person:ghost", body="Nope.")
        assert ok is False

    def test_delete_essay(self, store):
        store.create_essay("person:temp", body="Temporary.")
        assert store.delete_essay("person:temp") is True
        assert store.get_essay("person:temp") is None

    def test_delete_nonexistent(self, store):
        assert store.delete_essay("person:ghost") is False


# ── essay_edit (exact string replacement) ────────────────────────

class TestEssayEdit:
    def test_simple_replacement(self, store):
        store.create_essay("person:a", body="Hello world. Hello again.")
        ok, msg = store.essay_edit("person:a", "world", "earth")
        assert ok is True
        assert "1 replacement" in msg
        essay = store.get_essay("person:a")
        assert essay["body"] == "Hello earth. Hello again."
        assert essay["patch_count"] == 1

    def test_no_match(self, store):
        store.create_essay("person:b", body="Some content.")
        ok, msg = store.essay_edit("person:b", "nonexistent", "replacement")
        assert ok is False
        assert "not found" in msg

    def test_multiple_match_without_replace_all(self, store):
        store.create_essay("person:c", body="foo bar foo baz foo")
        ok, msg = store.essay_edit("person:c", "foo", "qux")
        assert ok is False
        assert "3 locations" in msg

    def test_multiple_match_with_replace_all(self, store):
        store.create_essay("person:d", body="foo bar foo baz foo")
        ok, msg = store.essay_edit("person:d", "foo", "qux", replace_all=True)
        assert ok is True
        assert "3 replacements" in msg
        essay = store.get_essay("person:d")
        assert essay["body"] == "qux bar qux baz qux"

    def test_edit_nonexistent_essay(self, store):
        ok, msg = store.essay_edit("person:ghost", "old", "new")
        assert ok is False
        assert "No essay found" in msg

    def test_edit_increments_patch_count(self, store):
        store.create_essay("person:e", body="Version one.")
        store.essay_edit("person:e", "one", "two")
        store.essay_edit("person:e", "two", "three")
        essay = store.get_essay("person:e")
        assert essay["patch_count"] == 2
        assert essay["body"] == "Version three."


# ── Citation tracking ────────────────────────────────────────────

class TestEssayCitations:
    def test_update_citations(self, store):
        store.create_essay("person:f", body="Essay.", citations=["m_001"])
        ok = store.update_essay_citations(
            "person:f",
            citations=["m_001", "m_002", "m_003"],
            cross_refs=["project:x"],
        )
        assert ok is True
        essay = store.get_essay("person:f")
        assert essay["citations"] == ["m_001", "m_002", "m_003"]
        assert essay["cross_refs"] == ["project:x"]
        assert essay["patch_count"] == 0  # metadata-only, no patch_count bump

    def test_update_citations_preserves_cross_refs(self, store):
        store.create_essay(
            "person:g", body="Essay.",
            citations=["m_001"], cross_refs=["project:y"],
        )
        store.update_essay_citations("person:g", citations=["m_002"])
        essay = store.get_essay("person:g")
        assert essay["citations"] == ["m_002"]
        assert essay["cross_refs"] == ["project:y"]

    def test_update_citations_nonexistent(self, store):
        ok = store.update_essay_citations("person:ghost", citations=["m_001"])
        assert ok is False

    def test_citations_json_roundtrip(self, store):
        """Citations and cross_refs survive JSON serialization."""
        refs = ["m_" + str(i) for i in range(20)]
        xrefs = ["person:a", "project:b", "event:c"]
        store.create_essay("person:h", body="Big.", citations=refs, cross_refs=xrefs)
        essay = store.get_essay("person:h")
        assert essay["citations"] == refs
        assert essay["cross_refs"] == xrefs


# ── Timestamps ───────────────────────────────────────────────────

class TestEssayTimestamps:
    def test_created_at_set(self, store):
        store.create_essay("person:ts", body="Timestamp test.")
        essay = store.get_essay("person:ts")
        assert essay["created_at"] is not None
        datetime.fromisoformat(essay["created_at"])

    def test_updated_at_changes(self, store):
        store.create_essay("person:ts2", body="V1.")
        e1 = store.get_essay("person:ts2")
        store.essay_edit("person:ts2", "V1", "V2")
        e2 = store.get_essay("person:ts2")
        assert e2["updated_at"] >= e1["updated_at"]


# ── Tool-level tests (public essay_edit function) ─────────────────

class TestEssayEditToolAtomicity:
    """Test the public essay_edit tool function for atomicity (Finding 2).

    Verifies: failed edits mutate nothing; successful edits increment
    patch_count exactly once regardless of title/citations/cross_refs.
    """

    @pytest.fixture(autouse=True)
    def setup_memory_system(self, store):
        """Patch _memory_system so the tool function can find the store."""
        from mesh import tool_implementations
        mock_system = type("FakeSystem", (), {"_store": store})()
        with patch.object(tool_implementations, "_memory_system", mock_system):
            self.store = store
            yield

    def test_failed_edit_mutates_nothing(self):
        """A failed edit with title must leave
        patch_count 0 and title unchanged."""
        from mesh.tool_implementations import essay_edit
        self.store.create_essay("person:test", body="Original body.", title="OldTitle")
        result = essay_edit(
            key="person:test",
            old_text="NONEXISTENT",
            new_text="whatever",
            title="NewTitle",
        )
        assert "Error" in result
        essay = self.store.get_essay("person:test")
        assert essay["patch_count"] == 0
        assert essay["title"] == "OldTitle"
        assert essay["body"] == "Original body."

    def test_successful_edit_with_title_increments_once(self):
        """A successful title+body edit must leave patch_count at exactly 1."""
        from mesh.tool_implementations import essay_edit
        self.store.create_essay("person:test2", body="Hello world.", title="OldTitle")
        result = essay_edit(
            key="person:test2",
            old_text="world",
            new_text="earth",
            title="NewTitle",
        )
        assert "Error" not in result
        essay = self.store.get_essay("person:test2")
        assert essay["patch_count"] == 1
        assert essay["title"] == "NewTitle"
        assert essay["body"] == "Hello earth."

    def test_failed_edit_with_citations_mutates_nothing(self):
        """Failed edit must not update citations either."""
        from mesh.tool_implementations import essay_edit
        self.store.create_essay(
            "person:test3", body="Content.",
            citations=["m_old"], cross_refs=["project:old"],
        )
        result = essay_edit(
            key="person:test3",
            old_text="MISSING",
            new_text="new",
            citations='["m_new"]',
            cross_refs='["project:new"]',
        )
        assert "Error" in result
        essay = self.store.get_essay("person:test3")
        assert essay["patch_count"] == 0
        assert essay["citations"] == ["m_old"]
        assert essay["cross_refs"] == ["project:old"]

    def test_ambiguous_match_mutates_nothing(self):
        """Multiple matches without replace_all must fail atomically."""
        from mesh.tool_implementations import essay_edit
        self.store.create_essay("person:test4", body="foo bar foo", title="T")
        result = essay_edit(
            key="person:test4",
            old_text="foo",
            new_text="baz",
            title="NewT",
        )
        assert "Error" in result
        essay = self.store.get_essay("person:test4")
        assert essay["patch_count"] == 0
        assert essay["title"] == "T"
        assert essay["body"] == "foo bar foo"


class TestEssayEditToolCitations:
    """Test the public essay_edit tool for citations/cross_refs (Finding 3)."""

    @pytest.fixture(autouse=True)
    def setup_memory_system(self, store):
        from mesh import tool_implementations
        mock_system = type("FakeSystem", (), {"_store": store})()
        with patch.object(tool_implementations, "_memory_system", mock_system):
            self.store = store
            yield

    def test_create_with_citations(self):
        """essay_edit creation path accepts citations and cross_refs."""
        from mesh.tool_implementations import essay_edit
        result = essay_edit(
            key="person:new",
            old_text="",
            new_text="New essay body.",
            title="New Person",
            citations='["m_001", "m_002"]',
            cross_refs='["project:alpha"]',
        )
        assert "created" in result
        essay = self.store.get_essay("person:new")
        assert essay["citations"] == ["m_001", "m_002"]
        assert essay["cross_refs"] == ["project:alpha"]

    def test_edit_updates_citations(self):
        """essay_edit on existing essay updates citations atomically."""
        from mesh.tool_implementations import essay_edit
        self.store.create_essay(
            "person:cit", body="Some text.",
            citations=["m_old"], cross_refs=["project:old"],
        )
        result = essay_edit(
            key="person:cit",
            old_text="Some",
            new_text="New",
            citations='["m_new1", "m_new2"]',
            cross_refs='["project:new", "person:other"]',
        )
        assert "Error" not in result
        essay = self.store.get_essay("person:cit")
        assert essay["body"] == "New text."
        assert essay["citations"] == ["m_new1", "m_new2"]
        assert essay["cross_refs"] == ["project:new", "person:other"]
        assert essay["patch_count"] == 1

    def test_edit_without_citations_preserves_existing(self):
        """Omitting citations/cross_refs from edit preserves existing values."""
        from mesh.tool_implementations import essay_edit
        self.store.create_essay(
            "person:keep", body="Keep this.",
            citations=["m_keep"], cross_refs=["project:keep"],
        )
        result = essay_edit(
            key="person:keep",
            old_text="Keep",
            new_text="Change",
        )
        assert "Error" not in result
        essay = self.store.get_essay("person:keep")
        assert essay["body"] == "Change this."
        assert essay["citations"] == ["m_keep"]
        assert essay["cross_refs"] == ["project:keep"]

    def test_invalid_citations_json_returns_error(self):
        """Bad JSON in citations parameter returns an error, mutates nothing."""
        from mesh.tool_implementations import essay_edit
        self.store.create_essay("person:badjson", body="Body.")
        result = essay_edit(
            key="person:badjson",
            old_text="Body",
            new_text="New",
            citations="not valid json",
        )
        assert "Error" in result
        essay = self.store.get_essay("person:badjson")
        assert essay["patch_count"] == 0
        assert essay["body"] == "Body."

    def test_invalid_cross_refs_json_returns_error(self):
        """Bad JSON in cross_refs returns an error, mutates nothing."""
        from mesh.tool_implementations import essay_edit
        self.store.create_essay("person:badjson2", body="Body.")
        result = essay_edit(
            key="person:badjson2",
            old_text="Body",
            new_text="New",
            cross_refs="{not an array}",
        )
        assert "Error" in result
        essay = self.store.get_essay("person:badjson2")
        assert essay["patch_count"] == 0

    def test_citations_not_array_returns_error(self):
        """citations must be a JSON array, not a string or object."""
        from mesh.tool_implementations import essay_edit
        self.store.create_essay("person:notarr", body="Body.")
        result = essay_edit(
            key="person:notarr",
            old_text="Body",
            new_text="New",
            citations='"just a string"',
        )
        assert "Error" in result
        assert "array" in result
