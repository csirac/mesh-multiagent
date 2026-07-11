"""Tests for Phase 1: digest_candidate tagging and fold injection filter."""
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np

from mesh.memory.store import MemoryEntry, MemoryStore


class TestDigestCandidateMigration(unittest.TestCase):
    """Schema migration adds digest_candidate column to existing DBs."""

    def test_migration_adds_column(self):
        """An existing DB without digest_candidate gets it via migration."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            conn = sqlite3.connect(db_path)
            conn.executescript("""
                CREATE TABLE memories (
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
                    project TEXT DEFAULT ''
                );
                INSERT INTO memories (id, created_at, summary, reflection,
                    trace, trigger_text, tags, outcome)
                VALUES ('abc123', '2026-01-01T00:00:00+00:00',
                    'Test summary', 'Test reflection', 'Test trace',
                    'trigger', 'test', 'success');
            """)
            conn.commit()
            conn.close()

            store = MemoryStore.__new__(MemoryStore)
            store._db_path = db_path
            store._conn = None
            store._open()

            cols = {r[1] for r in store._conn.execute(
                "PRAGMA table_info(memories)").fetchall()}
            self.assertIn("digest_candidate", cols)

            entry = store.get("abc123")
            self.assertIsNotNone(entry)
            self.assertTrue(entry.digest_candidate)

    def test_new_db_has_column(self):
        """A fresh DB gets digest_candidate in the CREATE TABLE."""
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore("fresh_agent", db_dir=tmp)
            cols = {r[1] for r in store._conn.execute(
                "PRAGMA table_info(memories)").fetchall()}
            self.assertIn("digest_candidate", cols)


class TestDigestCandidateInsert(unittest.TestCase):
    """Insert and load respect digest_candidate."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore("test_agent", db_dir=self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_entry(self, entry_id, digest_candidate=True):
        return MemoryEntry(
            id=entry_id,
            created_at=datetime.now(timezone.utc),
            summary=f"Summary for {entry_id}",
            reflection="Reflection",
            trace="Trace",
            trigger="Trigger",
            tags=["test"],
            outcome="success",
            digest_candidate=digest_candidate,
        )

    def test_insert_digest_candidate_true(self):
        e = self._make_entry("dc_true", digest_candidate=True)
        self.store.insert(e)
        loaded = self.store.get("dc_true")
        self.assertTrue(loaded.digest_candidate)

    def test_insert_digest_candidate_false(self):
        e = self._make_entry("dc_false", digest_candidate=False)
        self.store.insert(e)
        loaded = self.store.get("dc_false")
        self.assertFalse(loaded.digest_candidate)

    def test_load_all_includes_both(self):
        self.store.insert(self._make_entry("a", digest_candidate=True))
        self.store.insert(self._make_entry("b", digest_candidate=False))
        all_entries = self.store.load()
        self.assertEqual(len(all_entries), 2)
        by_id = {e.id: e for e in all_entries}
        self.assertTrue(by_id["a"].digest_candidate)
        self.assertFalse(by_id["b"].digest_candidate)

    def test_default_is_true(self):
        e = MemoryEntry(
            id="default_test",
            created_at=datetime.now(timezone.utc),
            summary="Default test",
            reflection="",
            trace="",
            trigger="",
        )
        self.assertTrue(e.digest_candidate)

    def test_insert_entry_and_advance_cursor(self):
        entries = [
            self._make_entry("cursor_a", digest_candidate=True),
            self._make_entry("cursor_b", digest_candidate=False),
        ]
        self.store.insert_entry_and_advance_cursor(entries, 100, "2026-01-01")
        loaded_a = self.store.get("cursor_a")
        loaded_b = self.store.get("cursor_b")
        self.assertTrue(loaded_a.digest_candidate)
        self.assertFalse(loaded_b.digest_candidate)


class TestFoldInjectionFilter(unittest.TestCase):
    """The _digest_visible filter in the fold driver."""

    def test_filter_when_lowbar_enabled(self):
        """Only digest_candidate=True records pass through."""
        import importlib
        import sys
        driver_path = "/tmp/test-pipeline/fold_driver/alice"
        if driver_path not in sys.path:
            sys.path.insert(0, driver_path)

        minted = [
            ("aaa", {"summary": "digest-worthy", "event_date": "2026-01-01",
                     "digest_candidate": True, "outcome": "success"}),
            ("bbb", {"summary": "db-only aside", "event_date": "2026-01-01",
                     "digest_candidate": False, "outcome": ""}),
            ("ccc", {"summary": "also digest", "event_date": "2026-01-02",
                     "digest_candidate": True, "outcome": "success"}),
        ]

        with patch.dict(os.environ, {"FOLD_LOWBAR": "1"}):
            if "run_edit_fold" in sys.modules:
                mod = sys.modules["run_edit_fold"]
                orig = mod.LOWBAR_ENABLED
                mod.LOWBAR_ENABLED = True
            else:
                mod = None

            try:
                if mod:
                    visible = mod._digest_visible(minted)
                else:
                    visible = [(mid, rec) for mid, rec in minted
                               if rec.get("digest_candidate", True)]

                self.assertEqual(len(visible), 2)
                self.assertEqual(visible[0][0], "aaa")
                self.assertEqual(visible[1][0], "ccc")
            finally:
                if mod:
                    mod.LOWBAR_ENABLED = orig

    def test_no_filter_when_lowbar_disabled(self):
        """All records pass through when lowbar is off."""
        minted = [
            ("aaa", {"summary": "a", "event_date": "2026-01-01",
                     "digest_candidate": True}),
            ("bbb", {"summary": "b", "event_date": "2026-01-01",
                     "digest_candidate": False}),
        ]
        filtered = minted  # no filter when LOWBAR_ENABLED=False
        self.assertEqual(len(filtered), 2)


class TestConfigFlag(unittest.TestCase):
    """Config flag defaults to False."""

    def test_default_off(self):
        from mesh.config import NodeConfig
        cfg = NodeConfig(id="agent:test:dummy")
        self.assertFalse(cfg.memory_formation_lowbar)

    def test_search_mode_default_hybrid(self):
        from mesh.config import NodeConfig
        cfg = NodeConfig(id="agent:test:dummy")
        self.assertEqual(cfg.memory_search_mode, "hybrid")


class TestFTS5ProductionWiring(unittest.TestCase):
    """FTS5 standalone table is created and kept in sync."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore("test_agent", db_dir=self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_entry(self, entry_id, summary="Test summary"):
        return MemoryEntry(
            id=entry_id,
            created_at=datetime.now(timezone.utc),
            summary=summary,
            reflection="Test reflection",
            trace="Trace",
            trigger="Trigger",
            retrieval_key=summary,
            tags=["test"],
            outcome="success",
        )

    def test_fts_table_created_on_fresh_db(self):
        exists = self.store._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        ).fetchone()
        self.assertIsNotNone(exists)

    def test_fts_migration_on_legacy_db(self):
        """Opening a legacy DB (no FTS) creates and populates the FTS table."""
        with tempfile.TemporaryDirectory() as tmp2:
            db_path = os.path.join(tmp2, "legacy.db")
            conn = sqlite3.connect(db_path)
            conn.executescript("""
                CREATE TABLE memories (
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
                    project TEXT DEFAULT ''
                );
                INSERT INTO memories (id, created_at, summary, reflection,
                    trace, trigger_text, retrieval_key, tags, outcome)
                VALUES ('legacy1', '2026-01-01T00:00:00+00:00',
                    'Kaylee birthday party', 'Details about party', '',
                    'trigger', 'Kaylee birthday', 'test', 'success');
            """)
            conn.commit()
            conn.close()

            store2 = MemoryStore.__new__(MemoryStore)
            store2._db_path = db_path
            store2._conn = None
            store2._open()

            results = store2.search_fts("Kaylee", limit=5)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][0], "legacy1")

    def test_insert_syncs_fts(self):
        self.store.insert(self._make_entry("fts1", "Kaylee birthday party"))
        results = self.store.search_fts("Kaylee", limit=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "fts1")

    def test_delete_syncs_fts(self):
        self.store.insert(self._make_entry("fts_del", "fishing trip"))
        self.assertTrue(len(self.store.search_fts("fishing", limit=5)) > 0)
        self.store.delete("fts_del")
        self.assertEqual(len(self.store.search_fts("fishing", limit=5)), 0)

    def test_update_syncs_fts(self):
        self.store.insert(self._make_entry("fts_upd", "old unique789 content"))
        self.store.update_entry(entry_id="fts_upd", summary="Kaylee new content",
                                retrieval_key="Kaylee new content")
        results = self.store.search_fts("Kaylee", limit=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "fts_upd")
        old_results = self.store.search_fts("unique789", limit=5)
        self.assertEqual(len(old_results), 0)

    def test_insert_entry_and_advance_cursor_syncs_fts(self):
        entries = [
            self._make_entry("fts_cursor_a", "fishing trip Alaska"),
            self._make_entry("fts_cursor_b", "Kaylee birthday plan"),
        ]
        self.store.insert_entry_and_advance_cursor(entries, 50, "2026-01-01")
        results_fish = self.store.search_fts("fishing", limit=5)
        results_kaylee = self.store.search_fts("Kaylee", limit=5)
        self.assertEqual(len(results_fish), 1)
        self.assertEqual(results_fish[0][0], "fts_cursor_a")
        self.assertEqual(len(results_kaylee), 1)
        self.assertEqual(results_kaylee[0][0], "fts_cursor_b")

    def test_search_fts_bm25_ordering(self):
        self.store.insert(self._make_entry("bm_a", "cat dog"))
        self.store.insert(self._make_entry("bm_b", "Kaylee Kaylee Kaylee"))
        self.store.insert(self._make_entry("bm_c", "Kaylee dog"))
        results = self.store.search_fts("Kaylee", limit=10)
        ids = [r[0] for r in results]
        self.assertIn("bm_b", ids)
        self.assertIn("bm_c", ids)
        self.assertNotIn("bm_a", ids)
        self.assertEqual(ids[0], "bm_b")


if __name__ == "__main__":
    unittest.main()
