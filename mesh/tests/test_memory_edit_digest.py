"""Tests for memory_edit, digest_edit, memory_delete, and FTS/hybrid search."""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pytest

from mesh.memory.store import MemoryEntry, MemoryStore, _SENTINEL
from mesh.memory.system_v2 import _reciprocal_rank_fusion


# ── Helpers ──────────────────────────────────────────────────

def _make_entry(
    id: str = "test_entry_01",
    summary: str = "User participated in a charity walk raising $250",
    reflection: str = "The user mentioned a charity walk",
    retrieval_key: str = "charity walk $250 fundraiser",
    tags: list[str] | None = None,
) -> MemoryEntry:
    dim = 1536
    rng = np.random.RandomState(42)
    return MemoryEntry(
        id=id,
        created_at=datetime.now(timezone.utc),
        summary=summary,
        reflection=reflection,
        trace="trace",
        trigger="trigger",
        retrieval_key=retrieval_key,
        tags=tags or ["charity"],
        outcome="success",
        reflection_embedding=rng.randn(dim).astype(np.float32),
        retrieval_key_embedding=rng.randn(dim).astype(np.float32),
        weight=0.5,
    )


# ── MemoryStore.update_entry tests ───────────────────────────

class TestStoreUpdateEntry:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.store = MemoryStore("test_agent", db_dir=self._tmpdir)
        self.entry = _make_entry()
        self.store.insert(self.entry)

    def teardown_method(self):
        self.store.close()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_update_summary_preserves_id(self):
        ok = self.store.update_entry(
            "test_entry_01", summary="Corrected: raised $300"
        )
        assert ok
        updated = self.store.get("test_entry_01")
        assert updated is not None
        assert updated.id == "test_entry_01"
        assert updated.summary == "Corrected: raised $300"
        assert updated.reflection == "The user mentioned a charity walk"

    def test_update_multiple_fields(self):
        ok = self.store.update_entry(
            "test_entry_01",
            summary="New summary",
            retrieval_key="new key",
            outcome="partial",
        )
        assert ok
        updated = self.store.get("test_entry_01")
        assert updated.summary == "New summary"
        assert updated.retrieval_key == "new key"
        assert updated.outcome == "partial"

    def test_update_tags(self):
        ok = self.store.update_entry(
            "test_entry_01", tags=["corrected", "charity"]
        )
        assert ok
        updated = self.store.get("test_entry_01")
        assert "corrected" in updated.tags

    def test_update_nonexistent_returns_false(self):
        ok = self.store.update_entry("nonexistent_id", summary="nope")
        assert not ok

    def test_update_no_fields_returns_exists_check(self):
        ok = self.store.update_entry("test_entry_01")
        assert ok
        ok2 = self.store.update_entry("nonexistent_id")
        assert not ok2

    def test_update_embeddings(self):
        new_emb = np.ones(1536, dtype=np.float32)
        ok = self.store.update_entry(
            "test_entry_01",
            reflection_embedding=new_emb,
        )
        assert ok
        updated = self.store.get("test_entry_01")
        assert np.allclose(updated.reflection_embedding, new_emb)

    def test_sentinel_skips_embedding_update(self):
        old = self.store.get("test_entry_01")
        old_emb = old.reflection_embedding.copy()
        ok = self.store.update_entry(
            "test_entry_01",
            summary="Changed summary only",
            reflection_embedding=_SENTINEL,
        )
        assert ok
        updated = self.store.get("test_entry_01")
        assert updated.summary == "Changed summary only"
        assert np.allclose(updated.reflection_embedding, old_emb)


# ── digest_edit tests ────────────────────────────────────────

class TestDigestEdit:
    def test_exact_replacement(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("The Art Cube gallery was fabricated.\nOther content.\n")
            path = f.name
        try:
            from mesh.tool_implementations import digest_edit
            import mesh.tool_implementations as ti
            orig_resolve = ti._resolve_digest_path
            ti._resolve_digest_path = lambda: path
            try:
                result = digest_edit(
                    "The Art Cube gallery was fabricated.",
                    "The Art Cube gallery opening was attended.",
                )
                assert "updated successfully" in result
                assert "1 replacement" in result
                content = open(path).read()
                assert "was attended" in content
                assert "fabricated" not in content
            finally:
                ti._resolve_digest_path = orig_resolve
        finally:
            os.unlink(path)

    def test_not_found(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Some digest content.\n")
            path = f.name
        try:
            from mesh.tool_implementations import digest_edit
            import mesh.tool_implementations as ti
            orig_resolve = ti._resolve_digest_path
            ti._resolve_digest_path = lambda: path
            try:
                result = digest_edit("nonexistent text", "replacement")
                assert "not found" in result
            finally:
                ti._resolve_digest_path = orig_resolve
        finally:
            os.unlink(path)

    def test_ambiguous_match(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("foo bar\nfoo bar\n")
            path = f.name
        try:
            from mesh.tool_implementations import digest_edit
            import mesh.tool_implementations as ti
            orig_resolve = ti._resolve_digest_path
            ti._resolve_digest_path = lambda: path
            try:
                result = digest_edit("foo bar", "baz")
                assert "matches 2 locations" in result
            finally:
                ti._resolve_digest_path = orig_resolve
        finally:
            os.unlink(path)

    def test_replace_all(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("foo bar\nfoo bar\n")
            path = f.name
        try:
            from mesh.tool_implementations import digest_edit
            import mesh.tool_implementations as ti
            orig_resolve = ti._resolve_digest_path
            ti._resolve_digest_path = lambda: path
            try:
                result = digest_edit("foo bar", "baz", replace_all=True)
                assert "2 replacements" in result
                content = open(path).read()
                assert content == "baz\nbaz\n"
            finally:
                ti._resolve_digest_path = orig_resolve
        finally:
            os.unlink(path)

    def test_no_config(self):
        from mesh.tool_implementations import digest_edit
        import mesh.tool_implementations as ti
        orig_resolve = ti._resolve_digest_path
        ti._resolve_digest_path = lambda: None
        try:
            result = digest_edit("old", "new")
            assert "no standing_digest_path" in result
        finally:
            ti._resolve_digest_path = orig_resolve


# ── digest_get tests ─────────────────────────────────────────

class TestDigestGet:
    def test_reads_content(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("## Timeline\n\nSome events.\n")
            path = f.name
        try:
            from mesh.tool_implementations import digest_get
            import mesh.tool_implementations as ti
            orig_resolve = ti._resolve_digest_path
            ti._resolve_digest_path = lambda: path
            try:
                result = digest_get()
                assert "## Timeline" in result
                assert "Some events" in result
            finally:
                ti._resolve_digest_path = orig_resolve
        finally:
            os.unlink(path)

    def test_missing_file(self):
        from mesh.tool_implementations import digest_get
        import mesh.tool_implementations as ti
        orig_resolve = ti._resolve_digest_path
        ti._resolve_digest_path = lambda: "/tmp/nonexistent_digest_12345.md"
        try:
            result = digest_get()
            assert "not found" in result
        finally:
            ti._resolve_digest_path = orig_resolve


class TestCorrectionToolReachability:
    """Correction tools must be both registered and exposed to fleet agents."""

    def test_correction_tools_are_registered_and_allowlisted(self):
        import mesh.tool_implementations  # noqa: F401 - populates the global registry
        from mesh.router_v2 import ROUTER_TOOL_NAMES
        from mesh.tools import get_registry

        correction_tools = {"memory_edit", "digest_get", "digest_edit"}
        registry = get_registry()
        for name in correction_tools:
            definition = registry.get(name)
            assert definition is not None, f"{name} is not registered"
            assert callable(definition.handler), f"{name} has no callable handler"

        assert correction_tools <= ROUTER_TOOL_NAMES

        # Public releases deliberately ship no fleet-specific mesh.yaml. The
        # generic example is an operator choice; registration and router
        # exposure are the portable contract.


# ── FTS5 full-text search tests ─────────────────────────────

class TestStoreFTS:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.store = MemoryStore("test_fts", db_dir=self._tmpdir)

    def teardown_method(self):
        self.store.close()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _insert(self, id: str, summary: str, reflection: str = "",
                retrieval_key: str = ""):
        dim = 1536
        rng = np.random.RandomState(hash(id) % (2**31))
        self.store.insert(MemoryEntry(
            id=id,
            created_at=datetime.now(timezone.utc),
            summary=summary,
            reflection=reflection or summary,
            trace="t", trigger="t",
            retrieval_key=retrieval_key or summary,
            tags=["test"],
            outcome="success",
            reflection_embedding=rng.randn(dim).astype(np.float32),
            retrieval_key_embedding=rng.randn(dim).astype(np.float32),
            weight=0.5,
        ))

    def test_fts_basic_search(self):
        self._insert("a", "User loves green socks")
        self._insert("b", "User prefers blue hats")
        results = self.store.search_fts("green socks")
        assert len(results) >= 1
        assert results[0][0] == "a"

    def test_fts_no_match(self):
        self._insert("a", "User loves green socks")
        results = self.store.search_fts("quantum entanglement")
        assert len(results) == 0

    def test_fts_synced_after_update(self):
        self._insert("a", "User loves green socks", reflection="nothing",
                     retrieval_key="nothing")
        self.store.update_entry("a", summary="User loves red shoes")
        results = self.store.search_fts("red shoes")
        assert len(results) >= 1
        assert results[0][0] == "a"
        old = self.store.search_fts("green socks")
        assert len(old) == 0

    def test_fts_synced_after_delete(self):
        self._insert("a", "User loves green socks")
        self.store.delete("a")
        results = self.store.search_fts("green socks")
        assert len(results) == 0

    def test_fts_populated_on_open(self):
        self._insert("a", "quantum computing breakthrough")
        self.store.close()
        store2 = MemoryStore("test_fts", db_dir=self._tmpdir)
        try:
            results = store2.search_fts("quantum computing")
            assert len(results) >= 1
            assert results[0][0] == "a"
        finally:
            store2.close()

    def test_fts_multiple_results_ranked(self):
        self._insert("a", "charity walk fundraiser $250")
        self._insert("b", "charity gala dinner $500")
        self._insert("c", "morning jog in the park")
        results = self.store.search_fts("charity")
        ids = [r[0] for r in results]
        assert "a" in ids
        assert "b" in ids
        assert "c" not in ids


# ── Reciprocal rank fusion tests ────────────────────────────

class TestRRF:
    def test_single_ranking(self):
        ranking = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        merged = _reciprocal_rank_fusion([ranking])
        ids = [eid for eid, _ in merged]
        assert ids == ["a", "b", "c"]

    def test_two_rankings_merge(self):
        r1 = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        r2 = [("c", 5.0), ("b", 3.0), ("d", 1.0)]
        merged = _reciprocal_rank_fusion([r1, r2])
        ids = [eid for eid, _ in merged]
        assert "b" in ids[:2] or "c" in ids[:2]
        assert "d" in ids

    def test_empty_rankings(self):
        merged = _reciprocal_rank_fusion([[], []])
        assert merged == []

    def test_disjoint_rankings(self):
        r1 = [("a", 0.9)]
        r2 = [("b", 5.0)]
        merged = _reciprocal_rank_fusion([r1, r2])
        ids = [eid for eid, _ in merged]
        assert set(ids) == {"a", "b"}
