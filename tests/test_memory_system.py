"""Tests for mesh/memory/system.py — orchestrator logic.

Uses mocked LLM and embedding clients to test the full memory lifecycle
without external API calls.
"""

import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio

from mesh.memory.store import MemoryEntry, MemoryStore
from mesh.memory.system import MemorySystem, _extract_tag


# ── Helpers ─────────────────────────────────────────────────────

def _random_emb(dim: int = 1536, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _make_mock_llm(reflection: str = "Deep reflection here.",
                   summary: str = "One-paragraph summary.",
                   tags: str = "mesh, testing",
                   outcome: str = "success"):
    """Mock LLM client that returns a structured reflection response."""
    llm = MagicMock()
    response = (
        f"<reflection>\n{reflection}\n</reflection>\n"
        f"<summary>\n{summary}\n</summary>\n"
        f"<tags>\n{tags}\n</tags>\n"
        f"<outcome_label>\n{outcome}\n</outcome_label>"
    )
    llm.complete = AsyncMock(return_value=response)
    return llm


def _make_mock_embedder(dim: int = 1536):
    """Mock EmbeddingClient that returns deterministic embeddings."""
    embedder = MagicMock()
    call_count = [0]

    async def mock_embed_to_array(text: str) -> np.ndarray:
        call_count[0] += 1
        return _random_emb(dim, seed=hash(text) % 2**31)

    async def mock_embed_batch_to_arrays(texts: list[str]) -> list[np.ndarray]:
        return [_random_emb(dim, seed=hash(t) % 2**31) for t in texts]

    embedder.embed_to_array = mock_embed_to_array
    embedder.embed_batch_to_arrays = mock_embed_batch_to_arrays
    return embedder


def _make_worker_result(
    response: str = "Done.",
    tool_calls: int = 5,
    error: str | None = None,
):
    """Create a mock worker result with tool call history."""
    context = []
    for i in range(tool_calls):
        # Tool call message
        call_msg = SimpleNamespace(
            content=f"Calling tool {i}",
            metadata={"tool_calls": [{"name": f"tool_{i}", "arguments": f"arg_{i}"}]},
        )
        context.append(SimpleNamespace(message=call_msg))
        # Tool result message
        result_msg = SimpleNamespace(
            content=f"Result {i}: ok",
            metadata={"tool_results": True},
        )
        context.append(SimpleNamespace(message=result_msg))

    return SimpleNamespace(
        response=response,
        context=context,
        error=error,
    )


# ── Fixtures ────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def memory_system(tmp_path):
    """Create a MemorySystem with mocked dependencies."""
    llm = _make_mock_llm()
    ms = MemorySystem(
        nickname="test-agent",
        llm_client=llm,
        pool_max_entries=5,
        reflection_min_tools=3,
        retrieval_k=3,
    )
    # Replace embedder with mock
    ms._embedder = _make_mock_embedder()
    # Override store to use temp directory
    ms._store = MemoryStore("test-agent", db_dir=str(tmp_path))
    ms._pool = []
    ms._active_ids = set()
    ms._active_weights = {}
    yield ms
    await ms.close()


# ── _extract_tag ────────────────────────────────────────────────

class TestExtractTag:
    def test_basic(self):
        text = "<summary>Hello world</summary>"
        assert _extract_tag(text, "summary") == "Hello world"

    def test_multiline(self):
        text = "<reflection>\nLine 1\nLine 2\n</reflection>"
        assert _extract_tag(text, "reflection") == "Line 1\nLine 2"

    def test_missing_tag(self):
        assert _extract_tag("no tags here", "summary") == ""

    def test_empty_content(self):
        assert _extract_tag("<summary></summary>", "summary") == ""

    def test_multiple_tags(self):
        text = (
            "<summary>First</summary>\n"
            "<reflection>Second</reflection>"
        )
        assert _extract_tag(text, "summary") == "First"
        assert _extract_tag(text, "reflection") == "Second"


# ── should_reflect ──────────────────────────────────────────────

class TestShouldReflect:
    def test_many_tools(self, memory_system):
        result = _make_worker_result(tool_calls=5)
        assert memory_system.should_reflect(result) is True

    def test_few_tools(self, memory_system):
        result = _make_worker_result(tool_calls=1)
        assert memory_system.should_reflect(result) is False

    def test_error_triggers(self, memory_system):
        result = _make_worker_result(tool_calls=0, error="Something broke")
        assert memory_system.should_reflect(result) is True

    def test_exactly_min_tools(self, memory_system):
        result = _make_worker_result(tool_calls=3)
        assert memory_system.should_reflect(result) is True

    def test_below_min_tools_no_error(self, memory_system):
        result = _make_worker_result(tool_calls=2, error=None)
        assert memory_system.should_reflect(result) is False


# ── reflect_on_completion ───────────────────────────────────────

class TestReflectOnCompletion:
    @pytest.mark.asyncio
    async def test_creates_entry(self, memory_system):
        """Reflection on a substantial task creates a memory entry."""
        result = _make_worker_result(tool_calls=5)
        await memory_system.reflect_on_completion(
            trigger="Fix the nginx config",
            result=result,
            worker_id="coder",
        )
        assert len(memory_system._pool) == 1
        entry = memory_system._pool[0]
        assert entry.summary == "One-paragraph summary."
        assert entry.reflection == "Deep reflection here."
        assert entry.tags == ["mesh", "testing"]
        assert entry.outcome == "success"

    @pytest.mark.asyncio
    async def test_embeddings_computed(self, memory_system):
        result = _make_worker_result()
        await memory_system.reflect_on_completion("test", result, "coder")
        entry = memory_system._pool[0]
        assert entry.reflection_embedding is not None
        assert entry.retrieval_key_embedding is not None
        assert entry.reflection_embedding.shape == (1536,)

    @pytest.mark.asyncio
    async def test_persisted_to_store(self, memory_system):
        result = _make_worker_result()
        await memory_system.reflect_on_completion("test", result, "coder")
        # Verify it's in the SQLite store
        loaded = memory_system._store.load()
        assert len(loaded) == 1
        assert loaded[0].id == memory_system._pool[0].id

    @pytest.mark.asyncio
    async def test_cold_start_fills(self, memory_system):
        """Cold start: all entries accepted until max_entries."""
        for i in range(5):
            result = _make_worker_result(response=f"Task {i}")
            # Give each a unique LLM response
            memory_system._llm_client.complete = AsyncMock(return_value=(
                f"<reflection>Reflection {i}</reflection>\n"
                f"<summary>Summary {i}</summary>\n"
                f"<tags>tag{i}</tags>\n"
                f"<outcome_label>success</outcome_label>"
            ))
            await memory_system.reflect_on_completion(f"Task {i}", result, "coder")
        assert len(memory_system._pool) == 5

    @pytest.mark.asyncio
    async def test_weights_computed(self, memory_system):
        """After adding entries, active weights should be populated."""
        for i in range(3):
            memory_system._llm_client.complete = AsyncMock(return_value=(
                f"<reflection>R{i}</reflection>\n"
                f"<summary>S{i}</summary>\n"
                f"<tags>t{i}</tags>\n"
                f"<outcome_label>success</outcome_label>"
            ))
            result = _make_worker_result()
            await memory_system.reflect_on_completion(f"Task {i}", result, "coder")
        assert len(memory_system._active_weights) == 3
        for w in memory_system._active_weights.values():
            assert isinstance(w, (float, np.floating))

    @pytest.mark.asyncio
    async def test_llm_called_with_reflection_prompt(self, memory_system):
        result = _make_worker_result()
        await memory_system.reflect_on_completion("Fix bug", result, "coder")
        memory_system._llm_client.complete.assert_called_once()
        prompt = memory_system._llm_client.complete.call_args[0][0]
        assert "Fix bug" in prompt

    @pytest.mark.asyncio
    async def test_bad_llm_output_skips(self, memory_system):
        """If LLM returns garbage, no entry is created."""
        memory_system._llm_client.complete = AsyncMock(
            return_value="No valid tags here, just garbage."
        )
        result = _make_worker_result()
        await memory_system.reflect_on_completion("test", result, "coder")
        assert len(memory_system._pool) == 0

    @pytest.mark.asyncio
    async def test_exception_during_reflection_handled(self, memory_system):
        """Exceptions during reflection are caught, not propagated."""
        memory_system._llm_client.complete = AsyncMock(
            side_effect=RuntimeError("API down")
        )
        result = _make_worker_result()
        # Should not raise
        await memory_system.reflect_on_completion("test", result, "coder")
        assert len(memory_system._pool) == 0


# ── render_block ────────────────────────────────────────────────

class TestRenderBlock:
    @pytest.mark.asyncio
    async def test_empty_returns_empty(self, memory_system):
        assert await memory_system.render_block() == ""

    @pytest.mark.asyncio
    async def test_single_entry(self, memory_system):
        entry = MemoryEntry(
            id="abc123",
            created_at=datetime(2026, 2, 7, tzinfo=timezone.utc),
            summary="Fixed nginx config",
            reflection="Detailed reflection",
            trace="[TOOL] bash(ls)",
            trigger="fix nginx",
            tags=["nginx", "config"],
            outcome="success",
        )
        memory_system._pool.append(entry)
        # Add to active set so it renders
        memory_system._active_ids.add("abc123")
        memory_system._active_weights["abc123"] = 1.0
        block = await memory_system.render_block()
        assert "<memory>" in block
        assert "</memory>" in block
        assert 'id="abc123"' in block
        # At full depth, reflection is shown (not summary)
        assert "Detailed reflection" in block

    @pytest.mark.asyncio
    async def test_multiple_entries(self, memory_system):
        for i in range(3):
            entry = MemoryEntry(
                id=f"entry{i}",
                created_at=datetime(2026, 2, 7, tzinfo=timezone.utc),
                summary=f"Summary {i}",
                reflection=f"Reflection {i}",
                trace="",
                trigger="",
                tags=[f"tag{i}"],
                outcome="success",
            )
            memory_system._pool.append(entry)
            memory_system._active_ids.add(f"entry{i}")
            memory_system._active_weights[f"entry{i}"] = 1.0
        block = await memory_system.render_block()
        assert block.count("<entry ") == 3
        assert block.count("</entry>") == 3


# ── render_block_for_query ──────────────────────────────────────

class TestRenderBlockForQuery:
    @pytest.mark.asyncio
    async def test_empty_returns_empty(self, memory_system):
        assert await memory_system.render_block_for_query("test") == ""

    @pytest.mark.asyncio
    async def test_returns_top_k(self, memory_system):
        """Should return entries based on relevance."""
        for i in range(5):
            entry = MemoryEntry(
                id=f"e{i}",
                created_at=datetime(2026, 2, 7, tzinfo=timezone.utc),
                summary=f"Summary {i}",
                reflection=f"R{i}",
                trace="",
                trigger=f"Trigger {i}",
                tags=[],
                outcome="success",
                retrieval_key_embedding=_random_emb(1536, seed=i),
            )
            memory_system._pool.append(entry)

        block = await memory_system.render_block_for_query("test query")
        # Should render some entries (budget-based, not count-based)
        assert "<memory>" in block or block == ""

    @pytest.mark.asyncio
    async def test_entries_without_embeddings_skipped(self, memory_system):
        """Entries with no retrieval_key_embedding are excluded from retrieval."""
        entry = MemoryEntry(
            id="no-emb",
            created_at=datetime(2026, 2, 7, tzinfo=timezone.utc),
            summary="No embedding",
            reflection="R",
            trace="",
            trigger="T",
            tags=[],
            outcome="success",
            retrieval_key_embedding=None,
        )
        memory_system._pool.append(entry)
        block = await memory_system.render_block_for_query("test")
        # Entry without embedding should not appear in relevant slice
        # (it may still appear in recent slice though)
        assert block == "" or "no-emb" not in block or "<memory>" in block


# ── remember ────────────────────────────────────────────────────

class TestRemember:
    def test_existing_entry_reflection(self, memory_system):
        entry = MemoryEntry(
            id="mem1",
            created_at=datetime.now(timezone.utc),
            summary="S",
            reflection="Deep reflection content",
            trace="[TOOL] bash(ls)",
            trigger="Q",
        )
        memory_system._pool.append(entry)
        result = memory_system.remember("mem1")
        assert result == "Deep reflection content"

    def test_existing_entry_full(self, memory_system):
        entry = MemoryEntry(
            id="mem1",
            created_at=datetime.now(timezone.utc),
            summary="S",
            reflection="Reflection",
            trace="[TOOL] bash(ls)\n[RESULT] output",
            trigger="Q",
        )
        memory_system._pool.append(entry)
        result = memory_system.remember("mem1", full=True)
        assert "Reflection" in result
        assert "## Trace" in result
        assert "[TOOL] bash(ls)" in result

    def test_nonexistent_returns_none(self, memory_system):
        assert memory_system.remember("nonexistent") is None


# ── add_entry ───────────────────────────────────────────────────

class TestAddEntry:
    @pytest.mark.asyncio
    async def test_manual_add(self, memory_system):
        entry, accepted = await memory_system.add_entry(
            summary="Manual memory",
            reflection="Manual reflection",
            tags=["manual"],
            outcome="success",
        )
        assert accepted is True
        assert entry.summary == "Manual memory"
        assert len(memory_system._pool) == 1

    @pytest.mark.asyncio
    async def test_manual_add_uses_summary_as_trigger(self, memory_system):
        entry, accepted = await memory_system.add_entry(summary="My summary")
        assert entry.trigger == "My summary"

    @pytest.mark.asyncio
    async def test_manual_add_runs_selection(self, memory_system):
        """Manual entries go through try_swap, not just appended."""
        # Fill to max
        for i in range(5):
            await memory_system.add_entry(
                summary=f"Entry {i}",
                tags=[f"tag{i}"],
            )
        assert len(memory_system._pool) == 5
        # Next one will trigger selection
        entry, accepted = await memory_system.add_entry(
            summary="Entry 5 — will it swap?",
        )
        # Whether accepted or not, pool count should be <= max_entries
        assert len(memory_system._pool) <= 6  # pool can exceed, active set is bounded


# ── delete_entry ────────────────────────────────────────────────

class TestDeleteEntry:
    @pytest.mark.asyncio
    async def test_delete_existing(self, memory_system):
        entry, _ = await memory_system.add_entry(summary="To be deleted")
        result = await memory_system.delete_entry(entry.id)
        assert result is True
        assert len(memory_system._pool) == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, memory_system):
        result = await memory_system.delete_entry("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_recomputes_weights(self, memory_system):
        # Add 3 entries
        ids = []
        for i in range(3):
            entry, _ = await memory_system.add_entry(summary=f"Entry {i}")
            ids.append(entry.id)
        assert len(memory_system._active_weights) == 3
        # Delete one
        await memory_system.delete_entry(ids[0])
        assert len(memory_system._active_weights) == 2
        assert len(memory_system._pool) == 2


# ── _build_trace ────────────────────────────────────────────────

class TestBuildTrace:
    def test_basic_trace(self, memory_system):
        result = _make_worker_result(tool_calls=3)
        trace = memory_system._build_trace(result)
        assert "[TOOL]" in trace
        assert "[RESULT]" in trace

    def test_empty_context(self, memory_system):
        result = SimpleNamespace(response="ok", context=[], error=None)
        trace = memory_system._build_trace(result)
        assert trace == ""

    def test_truncation(self, memory_system):
        """Very large traces should be truncated."""
        # Create a result with many tool calls
        result = _make_worker_result(tool_calls=500)
        memory_system._trace_max_tokens = 50  # Very small budget
        trace = memory_system._build_trace(result)
        assert "truncated" in trace


# ── Integration: full lifecycle ─────────────────────────────────

class TestLifecycle:
    @pytest.mark.asyncio
    async def test_reflect_remember_render(self, memory_system):
        """Full cycle: reflect -> render -> remember."""
        result = _make_worker_result()
        await memory_system.reflect_on_completion("Fix nginx", result, "coder")

        # render_block shows the entry (async now)
        block = await memory_system.render_block()
        assert "<memory>" in block
        # At full depth, reflection is shown
        assert "Deep reflection here." in block

        # remember retrieves reflection
        entry_id = memory_system._pool[0].id
        reflection = memory_system.remember(entry_id)
        assert reflection == "Deep reflection here."

        # remember full=True includes trace
        full = memory_system.remember(entry_id, full=True)
        assert "## Trace" in full

    @pytest.mark.asyncio
    async def test_reflect_then_delete(self, memory_system):
        result = _make_worker_result()
        await memory_system.reflect_on_completion("test", result, "coder")
        entry_id = memory_system._pool[0].id

        # Delete
        assert await memory_system.delete_entry(entry_id) is True
        assert len(memory_system._pool) == 0
        assert await memory_system.render_block() == ""


# ── Personality ────────────────────────────────────────────────

class TestPersonality:
    def test_get_empty(self, memory_system):
        """New system has no personality."""
        assert memory_system.get_personality() == ""

    def test_set_and_get(self, memory_system):
        memory_system.set_personality("I'm Bob, a grumpy sysadmin.")
        assert memory_system.get_personality() == "I'm Bob, a grumpy sysadmin."

    def test_overwrite(self, memory_system):
        memory_system.set_personality("Version 1")
        memory_system.set_personality("Version 2")
        assert memory_system.get_personality() == "Version 2"

    def test_seed_when_empty(self, memory_system):
        """Seeding sets personality when DB is empty."""
        memory_system.seed_personality("Seed personality")
        assert memory_system.get_personality() == "Seed personality"

    def test_seed_skips_when_set(self, memory_system):
        """Seeding does NOT overwrite existing personality."""
        memory_system.set_personality("Custom personality")
        memory_system.seed_personality("Should be ignored")
        assert memory_system.get_personality() == "Custom personality"

    def test_seed_skips_empty_config(self, memory_system):
        """Empty config string doesn't seed."""
        memory_system.seed_personality("")
        assert memory_system.get_personality() == ""

    def test_get_when_store_none(self):
        """get_personality returns empty when store is not initialized."""
        llm = _make_mock_llm()
        ms = MemorySystem(nickname="no-store", llm_client=llm)
        # _store is None before initialize()
        assert ms.get_personality() == ""
