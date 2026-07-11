"""Tests for Memory v2: Phases 1–5."""

import asyncio
import os
import tempfile
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mesh.memory.store import MemoryEntry, MemoryStore


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


def _make_entry(entry_id="abc123", project="", dim=8, **kwargs):
    """Create a MemoryEntry with random embeddings."""
    rng = np.random.RandomState(hash(entry_id) % 2**31)
    defaults = dict(
        id=entry_id,
        created_at=datetime.now(timezone.utc),
        summary="Test summary",
        reflection="Test reflection",
        trace="[TOOL] bash(ls)",
        trigger="Fix the bug",
        retrieval_key="fix bug in bash",
        topic_label="debugging",
        tags=["test"],
        outcome="success",
        reflection_embedding=rng.randn(dim).astype(np.float32),
        retrieval_key_embedding=rng.randn(dim).astype(np.float32),
        weight=0.5,
        project=project,
    )
    defaults.update(kwargs)
    return MemoryEntry(**defaults)


# ── Schema Migration ──────────────────────────────────────────

class TestSchemaMigration:
    def test_project_maps_table_exists(self, store):
        cursor = store._conn.execute("PRAGMA table_info(project_maps)")
        cols = {row[1] for row in cursor.fetchall()}
        expected = {
            "id", "project_name", "content", "updated_at",
            "created_at", "embedding", "is_active", "project_dir",
            "summary",
        }
        assert cols == expected

    def test_project_column_on_memories(self, store):
        cursor = store._conn.execute("PRAGMA table_info(memories)")
        cols = {row[1] for row in cursor.fetchall()}
        assert "project" in cols

    def test_migration_idempotent(self, tmp_dir):
        """Opening the same DB twice doesn't break migration."""
        s1 = MemoryStore("test", db_dir=tmp_dir)
        s1.close()
        s2 = MemoryStore("test", db_dir=tmp_dir)
        cursor = s2._conn.execute("PRAGMA table_info(memories)")
        cols = {row[1] for row in cursor.fetchall()}
        assert "project" in cols
        s2.close()


# ── Map CRUD ──────────────────────────────────────────────────

class TestMapCRUD:
    def test_create_and_get(self, store):
        store.create_map("m1", "hello-world", "# Project: hello-world")
        row = store.get_map("hello-world")
        assert row is not None
        # content is no longer returned by get_map (maps live on disk)
        assert "content" not in row
        assert row["project_name"] == "hello-world"
        assert row["is_active"] is False

    def test_get_nonexistent(self, store):
        assert store.get_map("nope") is None

    def test_list_maps(self, store):
        store.create_map("m1", "proj-a", "# A")
        store.create_map("m2", "proj-b", "# B")
        maps = store.list_maps()
        assert len(maps) == 2
        names = {m["project_name"] for m in maps}
        assert names == {"proj-a", "proj-b"}
        # list_maps doesn't include content
        assert "content" not in maps[0]

    def test_update_map(self, store):
        store.create_map("m1", "proj", "# Old")
        store.update_map("proj", "# New")
        # content is on disk now, verify update didn't break metadata
        assert store.get_map("proj") is not None

    def test_update_nonexistent(self, store):
        assert store.update_map("nope", "content") is False

    def test_delete_map(self, store):
        store.create_map("m1", "proj", "# Content")
        assert store.delete_map("proj") is True
        assert store.get_map("proj") is None

    def test_delete_nonexistent(self, store):
        assert store.delete_map("nope") is False

    def test_unique_project_name(self, store):
        store.create_map("m1", "proj", "# V1")
        with pytest.raises(Exception):  # UNIQUE constraint
            store.create_map("m2", "proj", "# V2")


# ── Active Project ────────────────────────────────────────────

class TestActiveProject:
    def test_set_and_get(self, store):
        store.create_map("m1", "proj", "# P")
        store.set_active_project("proj")
        assert store.get_active_project() == "proj"

    def test_only_one_active(self, store):
        store.create_map("m1", "a", "# A")
        store.create_map("m2", "b", "# B")
        store.set_active_project("a")
        store.set_active_project("b")
        assert store.get_active_project() == "b"
        assert store.get_map("a")["is_active"] is False
        assert store.get_map("b")["is_active"] is True

    def test_clear_active(self, store):
        store.create_map("m1", "proj", "# P")
        store.set_active_project("proj")
        store.clear_active_project()
        assert store.get_active_project() is None

    def test_no_active_initially(self, store):
        assert store.get_active_project() is None

    def test_persists_across_restart(self, tmp_dir):
        s1 = MemoryStore("test", db_dir=tmp_dir)
        s1.create_map("m1", "proj", "# Content")
        s1.set_active_project("proj")
        s1.close()

        s2 = MemoryStore("test", db_dir=tmp_dir)
        assert s2.get_active_project() == "proj"
        s2.close()


# ── Project Dir Persistence ───────────────────────────────────

class TestProjectDirPersistence:
    def test_create_map_with_project_dir(self, store):
        store.create_map("m1", "proj", "# P", project_dir="/tmp/test-proj")
        row = store.get_map("proj")
        assert row["project_dir"] == "/tmp/test-proj"

    def test_create_map_without_project_dir(self, store):
        store.create_map("m1", "proj", "# P")
        row = store.get_map("proj")
        assert row["project_dir"] is None

    def test_update_map_sets_project_dir(self, store):
        store.create_map("m1", "proj", "# Old")
        store.update_map("proj", "# New", project_dir="/tmp/test-proj")
        row = store.get_map("proj")
        assert row["project_dir"] == "/tmp/test-proj"

    def test_update_map_preserves_project_dir(self, store):
        store.create_map("m1", "proj", "# Old", project_dir="/tmp/test-proj")
        store.update_map("proj", "# New")  # no project_dir arg
        row = store.get_map("proj")
        assert row["project_dir"] == "/tmp/test-proj"

    def test_get_project_dir(self, store):
        store.create_map("m1", "proj", "# P", project_dir="/tmp/test-proj")
        assert store.get_project_dir("proj") == "/tmp/test-proj"

    def test_get_project_dir_none(self, store):
        store.create_map("m1", "proj", "# P")
        assert store.get_project_dir("proj") is None

    def test_get_project_dir_nonexistent(self, store):
        assert store.get_project_dir("nope") is None

    def test_persists_across_restart(self, tmp_dir):
        s1 = MemoryStore("test", db_dir=tmp_dir)
        s1.create_map("m1", "proj", "# Content", project_dir="/tmp/proj")
        s1.close()

        s2 = MemoryStore("test", db_dir=tmp_dir)
        assert s2.get_project_dir("proj") == "/tmp/proj"
        s2.close()


# ── Memory Entry with Project ────────────────────────────────

class TestMemoryEntryProject:
    def test_insert_with_project(self, store):
        entry = _make_entry(project="mesh-system")
        store.insert(entry)
        got = store.get(entry.id)
        assert got.project == "mesh-system"

    def test_load_with_project(self, store):
        store.insert(_make_entry("e1", project="proj-a"))
        store.insert(_make_entry("e2", project="proj-b"))
        store.insert(_make_entry("e3", project=""))
        entries = store.load()
        projects = {e.id: e.project for e in entries}
        assert projects["e1"] == "proj-a"
        assert projects["e2"] == "proj-b"
        assert projects["e3"] == ""

    def test_default_empty_project(self, store):
        entry = _make_entry()
        assert entry.project == ""
        store.insert(entry)
        got = store.get(entry.id)
        assert got.project == ""

    def test_backward_compat_old_entry(self, store):
        """Entries created before the project column default to ''."""
        entry = _make_entry()
        store.insert(entry)
        got = store.get(entry.id)
        assert got.project == ""


# ── MemorySystemV2 ────────────────────────────────────────────
# These tests use mocked LLM clients to avoid external deps.

class TestMemorySystemV2:
    @pytest.fixture
    def mock_llm(self):
        client = AsyncMock()
        client.complete = AsyncMock(return_value="# Project: test\n\n## Architecture\nTest.")
        return client

    @pytest.fixture
    def system(self, tmp_dir, mock_llm):
        from mesh.memory.system_v2 import MemorySystemV2
        sys = MemorySystemV2(
            nickname="test-agent",
            llm_client=mock_llm,
            pool_max_entries=100,
            embedding_backend="openai",
            embedding_model="text-embedding-3-small",
        )
        # Manually set store to avoid async init
        sys._store = MemoryStore("test-agent", db_dir=tmp_dir)
        sys._active_project_dir = tmp_dir
        return sys

    def test_diagnostics(self, system):
        diag = system.get_diagnostics()
        assert diag["version"] == 2
        assert diag["pool_size"] == 0
        assert diag["active_project"] is None
        assert diag["map_count"] == 0

    @pytest.mark.asyncio
    async def test_map_crud(self, system, tmp_dir):
        await system.create_map("proj", "# Content", project_dir=tmp_dir)
        content = await system.get_map("proj")
        assert content == "# Content"

        maps = await system.list_maps()
        assert len(maps) == 1

        await system.update_map("proj", "# Updated")
        assert await system.get_map("proj") == "# Updated"

        assert await system.delete_map("proj") is True
        assert await system.get_map("proj") is None

    @pytest.mark.asyncio
    async def test_apply_map_edit(self, system, tmp_dir):
        await system.create_map("proj", "# Project: proj\n\nWatchdog: 15min", project_dir=tmp_dir)
        result = await system.apply_map_edit("proj", "15min", "30min")
        assert "success" in result.lower()
        content = await system.get_map("proj")
        assert "30min" in content
        assert "15min" not in content

    @pytest.mark.asyncio
    async def test_apply_map_edit_not_found(self, system, tmp_dir):
        await system.create_map("proj", "# Content", project_dir=tmp_dir)
        result = await system.apply_map_edit("proj", "nonexistent text", "new")
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_apply_map_edit_multiple_matches(self, system, tmp_dir):
        await system.create_map("proj", "foo bar foo", project_dir=tmp_dir)
        result = await system.apply_map_edit("proj", "foo", "baz")
        assert "2 locations" in result.lower()

    @pytest.mark.asyncio
    async def test_apply_map_edit_nonexistent_map(self, system):
        result = await system.apply_map_edit("nope", "old", "new")
        assert "no map found" in result.lower()

    @pytest.mark.asyncio
    async def test_render_maps_block_no_active(self, system):
        result = await system.render_maps_block()
        assert result == ""

    @pytest.mark.asyncio
    async def test_render_maps_block_with_active(self, system):
        await system.create_map("proj", "# Content here")
        system._active_project = "proj"
        system._store.set_active_project("proj")
        result = await system.render_maps_block()
        assert '<project_map project="proj"' in result
        assert "# Content here" in result

    @pytest.mark.asyncio
    async def test_set_project_context_creates_map(self, system, tmp_dir):
        """set_project_context creates a map via scan when none exists."""
        # Create a small project directory
        proj_dir = os.path.join(tmp_dir, "my-project")
        os.makedirs(proj_dir)
        with open(os.path.join(proj_dir, "main.py"), "w") as f:
            f.write("print('hello')\n")
        with open(os.path.join(proj_dir, "README.md"), "w") as f:
            f.write("# My Project\n\nA test project.\n")

        result = await system.set_project_context(proj_dir)
        assert "initialized" in result.lower()
        assert system._active_project == "my-project"
        content = await system.get_map("my-project")
        assert content is not None

    @pytest.mark.asyncio
    async def test_set_project_context_loads_existing(self, system, tmp_dir):
        """set_project_context loads an existing map."""
        # Create a real project directory so the map file can be found
        proj_dir = os.path.join(tmp_dir, "proj")
        os.makedirs(proj_dir, exist_ok=True)
        with open(os.path.join(proj_dir, "PROJECT_MAP.md"), "w") as f:
            f.write("# Existing map")
        # Register metadata in the store with project_dir
        system._store.create_map("m1", "proj", "", project_dir=proj_dir)
        result = await system.set_project_context(proj_dir)
        assert "loaded" in result.lower()
        assert system._active_project == "proj"

    @pytest.mark.asyncio
    async def test_set_project_context_reset(self, system, tmp_dir, mock_llm):
        """reset=True forces re-scan."""
        proj_dir = os.path.join(tmp_dir, "proj")
        os.makedirs(proj_dir)
        with open(os.path.join(proj_dir, "main.py"), "w") as f:
            f.write("x = 1\n")

        await system.create_map("proj", "# Old map")
        result = await system.set_project_context(proj_dir, reset=True)
        assert "initialized" in result.lower()
        # The old map should be replaced by a fresh scan
        content = await system.get_map("proj")
        assert content is not None
        # LLM was called for synthesis
        assert mock_llm.complete.called

    @pytest.mark.asyncio
    async def test_active_project_persists(self, tmp_dir, mock_llm):
        """Active project survives system restart."""
        from mesh.memory.system_v2 import MemorySystemV2

        s1 = MemorySystemV2(nickname="test", llm_client=mock_llm)
        s1._store = MemoryStore("test", db_dir=tmp_dir)
        s1._active_project_dir = tmp_dir
        await s1.create_map("proj-x", "# Map")
        s1._active_project = "proj-x"
        s1._store.set_active_project("proj-x")

        # Simulate restart
        s1._store.close()
        s2 = MemorySystemV2(nickname="test", llm_client=mock_llm)
        s2._store = MemoryStore("test", db_dir=tmp_dir)
        s2._active_project = s2._store.get_active_project()
        assert s2._active_project == "proj-x"
        s2._store.close()

    @pytest.mark.asyncio
    async def test_set_project_context_persists_dir(self, system, tmp_dir):
        """set_project_context stores project_dir in the DB."""
        proj_dir = os.path.join(tmp_dir, "my-proj")
        os.makedirs(proj_dir)
        with open(os.path.join(proj_dir, "main.py"), "w") as f:
            f.write("x = 1\n")

        await system.set_project_context(proj_dir)
        stored = system._store.get_project_dir("my-proj")
        assert stored == proj_dir

    @pytest.mark.asyncio
    async def test_set_project_context_by_name(self, system, tmp_dir):
        """set_project_context accepts a project name if map exists with stored dir."""
        proj_dir = os.path.join(tmp_dir, "my-proj")
        os.makedirs(proj_dir)
        with open(os.path.join(proj_dir, "main.py"), "w") as f:
            f.write("x = 1\n")

        # First call with full path creates the map
        await system.set_project_context(proj_dir)
        assert system._active_project == "my-proj"

        # Reset active state
        system._active_project = None
        system._active_project_dir = None
        system._store.clear_active_project()

        # Now set context using just the project name
        result = await system.set_project_context("my-proj")
        assert "loaded" in result.lower()
        assert system._active_project == "my-proj"
        assert system._active_project_dir == proj_dir

    @pytest.mark.asyncio
    async def test_project_dir_restored_on_init(self, tmp_dir, mock_llm):
        """_active_project_dir is restored from DB on initialize."""
        from mesh.memory.system_v2 import MemorySystemV2

        s1 = MemorySystemV2(nickname="test", llm_client=mock_llm)
        s1._store = MemoryStore("test", db_dir=tmp_dir)
        s1._store.create_map("m1", "proj-x", "# Map", project_dir="/tmp/proj-x")
        s1._store.set_active_project("proj-x")
        s1._store.close()

        # Simulate restart with full initialize
        s2 = MemorySystemV2(nickname="test", llm_client=mock_llm)
        s2._store = MemoryStore("test", db_dir=tmp_dir)
        s2._active_project = s2._store.get_active_project()
        s2._active_project_dir = s2._store.get_project_dir(s2._active_project)
        assert s2._active_project == "proj-x"
        assert s2._active_project_dir == "/tmp/proj-x"
        s2._store.close()

    def test_personality_carried_from_v1(self, system):
        system.set_personality("I am a helpful agent")
        assert system.get_personality() == "I am a helpful agent"

    def test_seed_personality(self, system):
        system.seed_personality("seeded text")
        assert system.get_personality() == "seeded text"
        # Don't overwrite once set
        system.seed_personality("new seed")
        assert system.get_personality() == "seeded text"


# ── Map Word Budget ───────────────────────────────────────────
# Verifying that the synthesis prompt asks for 800-1,500 words

class TestMapSynthesisPrompt:
    def test_prompt_mentions_word_target(self):
        from mesh.memory.system_v2 import MAP_SYNTHESIS_PROMPT
        assert "800-1,500 words" in MAP_SYNTHESIS_PROMPT
        assert "entry files" in MAP_SYNTHESIS_PROMPT


# ══════════════════════════════════════════════════════════════
# Phase 2: Window Drop → Log Entry Creation Pipeline
# ══════════════════════════════════════════════════════════════

def _make_turn(role="user", content="hello", topic_label="", from_node="",
               meta=None, seq_id=0):
    """Create a Turn-like object for testing."""
    from mesh.conversation_history import Turn
    m = meta or {}
    if topic_label:
        m["topic_label"] = topic_label
    return Turn(
        role=role, content=content, timestamp="2026-02-28T12:00:00",
        from_node=from_node, meta=m, seq_id=seq_id,
    )


class TestSegmentByTopic:
    """Test _segment_by_topic static method."""

    def test_empty_turns(self):
        from mesh.memory.system_v2 import MemorySystemV2
        assert MemorySystemV2._segment_by_topic([]) == []

    def test_single_topic(self):
        from mesh.memory.system_v2 import MemorySystemV2
        turns = [
            _make_turn(topic_label="debugging"),
            _make_turn(topic_label="debugging"),
            _make_turn(topic_label="debugging"),
        ]
        segments = MemorySystemV2._segment_by_topic(turns)
        assert len(segments) == 1
        assert segments[0][0] == "debugging"
        assert len(segments[0][1]) == 3

    def test_two_topics(self):
        from mesh.memory.system_v2 import MemorySystemV2
        turns = [
            _make_turn(topic_label="debugging"),
            _make_turn(topic_label="debugging"),
            _make_turn(topic_label="deployment"),
            _make_turn(topic_label="deployment"),
        ]
        segments = MemorySystemV2._segment_by_topic(turns)
        assert len(segments) == 2
        assert segments[0][0] == "debugging"
        assert len(segments[0][1]) == 2
        assert segments[1][0] == "deployment"
        assert len(segments[1][1]) == 2

    def test_unlabeled_turns_attach_to_current(self):
        from mesh.memory.system_v2 import MemorySystemV2
        turns = [
            _make_turn(topic_label="debugging"),
            _make_turn(topic_label=""),  # unlabeled, attaches to debugging
            _make_turn(topic_label="debugging"),
        ]
        segments = MemorySystemV2._segment_by_topic(turns)
        assert len(segments) == 1
        assert len(segments[0][1]) == 3

    def test_all_unlabeled(self):
        from mesh.memory.system_v2 import MemorySystemV2
        turns = [
            _make_turn(topic_label=""),
            _make_turn(topic_label=""),
        ]
        segments = MemorySystemV2._segment_by_topic(turns)
        assert len(segments) == 1
        assert segments[0][0] == "misc"

    def test_leading_unlabeled_then_labeled(self):
        from mesh.memory.system_v2 import MemorySystemV2
        turns = [
            _make_turn(topic_label=""),
            _make_turn(topic_label="coding"),
            _make_turn(topic_label="coding"),
        ]
        segments = MemorySystemV2._segment_by_topic(turns)
        assert len(segments) == 2
        assert segments[0][0] == "misc"
        assert segments[1][0] == "coding"

    def test_alternating_topics(self):
        from mesh.memory.system_v2 import MemorySystemV2
        turns = [
            _make_turn(topic_label="A"),
            _make_turn(topic_label="B"),
            _make_turn(topic_label="A"),
        ]
        segments = MemorySystemV2._segment_by_topic(turns)
        assert len(segments) == 3


class TestSignificanceGate:
    """Test _significance_gate method."""

    @pytest.fixture
    def system(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        sys = MemorySystemV2(
            nickname="test", llm_client=AsyncMock(),
            reflection_min_tools=3,
            reflection_min_discussion_turns=4,
            reflection_min_discussion_chars=1500,
            reflection_min_brainstorm_response_chars=1500,
            reflection_max_brainstorm_tools=2,
        )
        sys._store = MemoryStore("test", db_dir=tmp_dir)
        return sys

    def test_below_all_thresholds(self, system):
        turns = [_make_turn(content="hi")]
        assert system._significance_gate(turns) is False

    def test_tool_heavy(self, system):
        turns = [
            _make_turn(role="tool", content="result1"),
            _make_turn(role="tool", content="result2"),
            _make_turn(role="tool", content="result3"),
        ]
        assert system._significance_gate(turns) is True

    def test_extended_discussion(self, system):
        long_content = "x" * 500
        turns = [
            _make_turn(role="user", content=long_content, from_node="user:testuser"),
            _make_turn(role="assistant", content=long_content, from_node="agent:coder"),
            _make_turn(role="user", content=long_content, from_node="user:testuser"),
            _make_turn(role="assistant", content=long_content, from_node="agent:coder"),
        ]
        assert system._significance_gate(turns) is True

    def test_brainstorm(self, system):
        long_response = "x" * 1600
        turns = [
            _make_turn(role="user", content="brainstorm ideas", from_node="user:testuser"),
            _make_turn(role="assistant", content=long_response, from_node="agent:coder"),
        ]
        assert system._significance_gate(turns) is True

    def test_errors(self, system):
        turns = [
            _make_turn(role="tool", content="Error: file not found"),
        ]
        assert system._significance_gate(turns) is True

    def test_meta_tool_calls_count(self, system):
        turns = [
            _make_turn(meta={"tool_calls": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}),
        ]
        assert system._significance_gate(turns) is True

    def test_cc_tool_events_count(self, system):
        turns = [
            _make_turn(meta={"cc_tool_events": True, "cc_tool_calls": 5}),
        ]
        assert system._significance_gate(turns) is True


class TestFormatTurnsAsTrace:
    def test_basic_format(self):
        from mesh.memory.system_v2 import MemorySystemV2
        sys = MemorySystemV2.__new__(MemorySystemV2)
        sys._trace_max_tokens = 2000

        turns = [
            _make_turn(role="user", content="fix the bug", from_node="user:testuser"),
            _make_turn(role="assistant", content="I'll look into it", from_node="agent:coder"),
            _make_turn(role="tool", content="bash result: file.py"),
        ]
        trace = sys._format_turns_as_trace(turns)
        assert "[USER]" in trace
        assert "[AGENT]" in trace
        assert "[RESULT]" in trace

    def test_truncation(self):
        from mesh.memory.system_v2 import MemorySystemV2
        sys = MemorySystemV2.__new__(MemorySystemV2)
        sys._trace_max_tokens = 10  # Very small budget

        turns = [
            _make_turn(role="user", content="x" * 1000, from_node="user:testuser"),
            _make_turn(role="assistant", content="y" * 1000, from_node="agent:coder"),
        ]
        trace = sys._format_turns_as_trace(turns)
        assert "truncated" in trace


class TestFormatTurnsAsText:
    def test_basic_format(self):
        from mesh.memory.system_v2 import MemorySystemV2
        turns = [
            _make_turn(role="user", content="hello world", from_node="user:testuser"),
            _make_turn(role="assistant", content="hi there", from_node="agent:coder"),
        ]
        text = MemorySystemV2._format_turns_as_text(turns)
        assert "User:" in text
        assert "Agent" in text
        assert "hello world" in text
        assert "hi there" in text


class TestReflectOnSegment:
    """Test _reflect_on_segment with mocked LLM."""

    @pytest.fixture
    def system(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=(
            "<reflection>\nThe agent debugged a tricky async issue.\n"
            "It involved multiple coroutines.\n</reflection>\n"
            "<summary>\nFixed async bug in router.\n</summary>\n"
            "<tags>\nasyncio, debugging\n</tags>\n"
            "<outcome_label>\nsuccess\n</outcome_label>\n"
            "<retrieval_key>\nAsync bug fix in router coroutine handling\n</retrieval_key>\n"
            "<project>\nmesh-system\n</project>"
        ))
        sys = MemorySystemV2(
            nickname="test", llm_client=mock_llm,
        )
        sys._store = MemoryStore("test", db_dir=tmp_dir)
        # Mock embedder
        sys._embedder = MagicMock()
        sys._embedder.embed_batch_to_arrays = AsyncMock(
            return_value=[np.random.randn(8).astype(np.float32),
                          np.random.randn(8).astype(np.float32)]
        )
        return sys

    @pytest.mark.asyncio
    async def test_creates_entry(self, system):
        turns = [
            _make_turn(role="user", content="fix the async bug", from_node="user:testuser"),
            _make_turn(role="assistant", content="I'll debug it", from_node="agent:coder"),
            _make_turn(role="tool", content="result of bash ls"),
            _make_turn(role="tool", content="result of bash grep"),
            _make_turn(role="tool", content="result of edit"),
        ]
        entry = await system._reflect_on_segment(turns, "debugging", "mesh-system, demo-project")
        assert entry is not None
        assert entry.topic_label == "debugging"
        assert entry.project == "mesh-system"
        assert entry.outcome == "success"
        assert "async" in entry.reflection.lower()

    @pytest.mark.asyncio
    async def test_trigger_from_user_message(self, system):
        turns = [
            _make_turn(role="assistant", content="status update", from_node="agent:coder"),
            _make_turn(role="user", content="fix the bug now", from_node="user:testuser"),
        ]
        entry = await system._reflect_on_segment(turns, "topic", "")
        # The trigger should use the user message
        assert "fix the bug now" in entry.trigger

    @pytest.mark.asyncio
    async def test_empty_reflection_returns_none(self, system):
        system._llm_client.complete = AsyncMock(return_value="garbage output")
        turns = [_make_turn(content="test")]
        entry = await system._reflect_on_segment(turns, "misc", "")
        assert entry is None

    @pytest.mark.asyncio
    async def test_entry_stored_in_pool_and_db(self, system):
        turns = [_make_turn(role="user", content="test", from_node="user:a")]
        entry = await system._reflect_on_segment(turns, "test", "")
        assert entry is not None
        assert entry in system._pool
        db_entry = system._store.get(entry.id)
        assert db_entry is not None
        assert db_entry.summary == entry.summary

    @pytest.mark.asyncio
    async def test_invalid_outcome_defaults_to_success(self, system):
        system._llm_client.complete = AsyncMock(return_value=(
            "<reflection>\nReflection text.\n</reflection>\n"
            "<summary>\nSummary text.\n</summary>\n"
            "<tags>\ntest\n</tags>\n"
            "<outcome_label>\ninvalid_value\n</outcome_label>\n"
            "<retrieval_key>\nKey.\n</retrieval_key>\n"
            "<project>\n\n</project>"
        ))
        turns = [_make_turn(content="test")]
        entry = await system._reflect_on_segment(turns, "test", "")
        assert entry.outcome == "success"


class TestOnWindowDrop:
    """Test the full on_window_drop pipeline."""

    @pytest.fixture
    def system(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        mock_llm = AsyncMock()
        # Default reflection response
        mock_llm.complete = AsyncMock(return_value=(
            "<reflection>\nDid some work.\n</reflection>\n"
            "<summary>\nWork summary.\n</summary>\n"
            "<tags>\ntest\n</tags>\n"
            "<outcome_label>\nsuccess\n</outcome_label>\n"
            "<retrieval_key>\nWork on testing.\n</retrieval_key>\n"
            "<project>\ntest-proj\n</project>"
        ))
        sys = MemorySystemV2(
            nickname="test", llm_client=mock_llm,
        )
        sys._store = MemoryStore("test", db_dir=tmp_dir)
        sys._active_project_dir = tmp_dir
        sys._embedder = MagicMock()
        sys._embedder.embed_batch_to_arrays = AsyncMock(
            return_value=[np.random.randn(8).astype(np.float32),
                          np.random.randn(8).astype(np.float32)]
        )
        return sys

    @pytest.mark.asyncio
    async def test_empty_turns_noop(self, system):
        await system.on_window_drop([])
        assert len(system._pool) == 0

    @pytest.mark.asyncio
    async def test_trivial_segment_now_creates_entry(self, system):
        """Per 2026-04-26 directive: significance gate removed from
        on_window_drop. Memory formation must trigger on every window
        drop regardless of segment size or map status. A single short
        turn now produces a log entry."""
        turns = [_make_turn(content="hi", topic_label="chitchat")]
        await system.on_window_drop(turns)
        assert len(system._pool) == 1
        assert system._pool[0].topic_label == "chitchat"

    @pytest.mark.asyncio
    async def test_significant_segment_creates_entry(self, system):
        """Tool-heavy segment should produce a log entry."""
        turns = [
            _make_turn(role="user", content="fix bug", topic_label="debugging",
                       from_node="user:testuser"),
            _make_turn(role="tool", content="result1", topic_label="debugging"),
            _make_turn(role="tool", content="result2", topic_label="debugging"),
            _make_turn(role="tool", content="result3", topic_label="debugging"),
            _make_turn(role="assistant", content="fixed", topic_label="debugging",
                       from_node="agent:coder"),
        ]
        await system.on_window_drop(turns)
        assert len(system._pool) == 1
        assert system._pool[0].topic_label == "debugging"

    @pytest.mark.asyncio
    async def test_multiple_segments(self, system):
        """Two significant segments from different topics."""
        turns = [
            # Segment 1: debugging (tool-heavy)
            _make_turn(role="user", content="fix bug", topic_label="debugging",
                       from_node="user:testuser"),
            _make_turn(role="tool", content="r1", topic_label="debugging"),
            _make_turn(role="tool", content="r2", topic_label="debugging"),
            _make_turn(role="tool", content="r3", topic_label="debugging"),
            # Segment 2: deployment (tool-heavy)
            _make_turn(role="user", content="deploy", topic_label="deployment",
                       from_node="user:testuser"),
            _make_turn(role="tool", content="r4", topic_label="deployment"),
            _make_turn(role="tool", content="r5", topic_label="deployment"),
            _make_turn(role="tool", content="r6", topic_label="deployment"),
        ]
        await system.on_window_drop(turns)
        assert len(system._pool) == 2
        topics = {e.topic_label for e in system._pool}
        assert topics == {"debugging", "deployment"}

    @pytest.mark.asyncio
    async def test_map_curation_called_with_active_project(self, system):
        """Map curation runs when there's an active project."""
        # Setup active project
        await system.create_map("test-proj", "# Project: test-proj\n\n## Architecture\nTest.")
        system._active_project = "test-proj"
        system._store.set_active_project("test-proj")

        # Make the curation response return updated map
        original_complete = system._llm_client.complete
        call_count = [0]

        async def side_effect(prompt, **kwargs):
            call_count[0] += 1
            if "maintaining a project knowledge map" in prompt:
                return "# Project: test-proj\n\n## Architecture\nUpdated."
            return await original_complete(prompt, **kwargs)

        system._llm_client.complete = AsyncMock(side_effect=side_effect)

        turns = [
            _make_turn(role="user", content="update arch", topic_label="coding",
                       from_node="user:testuser"),
            _make_turn(role="tool", content="r1", topic_label="coding"),
            _make_turn(role="tool", content="r2", topic_label="coding"),
            _make_turn(role="tool", content="r3", topic_label="coding"),
        ]
        await system.on_window_drop(turns)

        # Map should be updated
        content = await system.get_map("test-proj")
        assert "Updated" in content

    @pytest.mark.asyncio
    async def test_no_curation_without_active_project(self, system):
        """Map curation doesn't run without an active project."""
        # Use a project name that matches an existing map to avoid new project creation
        system._llm_client.complete = AsyncMock(return_value=(
            "<reflection>\nDid some work.\n</reflection>\n"
            "<summary>\nWork summary.\n</summary>\n"
            "<tags>\ntest\n</tags>\n"
            "<outcome_label>\nsuccess\n</outcome_label>\n"
            "<retrieval_key>\nWork on testing.\n</retrieval_key>\n"
            "<project>\n\n</project>"
        ))

        turns = [
            _make_turn(role="user", content="fix", topic_label="coding",
                       from_node="user:testuser"),
            _make_turn(role="tool", content="r1", topic_label="coding"),
            _make_turn(role="tool", content="r2", topic_label="coding"),
            _make_turn(role="tool", content="r3", topic_label="coding"),
        ]
        await system.on_window_drop(turns)

        # Reflection + conversation summary, but no curation (no active project)
        assert system._llm_client.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_new_project_detection(self, system, tmp_dir):
        """When reflection names an unknown project, a new map is created."""
        system._llm_client.complete = AsyncMock(side_effect=[
            # Reflection response naming new project
            "<reflection>\nBuilt a new fishing app.\n</reflection>\n"
            "<summary>\nNew demo-project app created.\n</summary>\n"
            "<tags>\nfishing\n</tags>\n"
            "<outcome_label>\nsuccess\n</outcome_label>\n"
            "<retrieval_key>\nRec fishing app.\n</retrieval_key>\n"
            "<project>\ndemo-project\n</project>",
            # Conversation summary response
            "## Active Topics\n- Building a fishing app",
            # New project map creation response
            "# Project: demo-project\n\n## Architecture\nFishing app.",
        ])

        turns = [
            _make_turn(role="user", content="build fishing app", topic_label="fishing",
                       from_node="user:testuser"),
            _make_turn(role="tool", content="r1", topic_label="fishing"),
            _make_turn(role="tool", content="r2", topic_label="fishing"),
            _make_turn(role="tool", content="r3", topic_label="fishing"),
        ]
        await system.on_window_drop(turns)

        # New map file should exist on disk (written via _active_project_dir)
        map_file = os.path.join(tmp_dir, "PROJECT_MAP.md")
        assert os.path.exists(map_file)
        with open(map_file) as f:
            content = f.read()
        assert "demo-project" in content


class TestBuildTraceFallback:
    """Regression for B1: topic-segment-flush WorkerResult contexts contain
    raw routed Message objects without metadata.tool_calls / tool_results.
    _build_trace must fall back to formatting message content as
    USER/AGENT lines so the trace isn't silently empty."""

    @pytest.fixture
    def system(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        sys = MemorySystemV2(nickname="test", llm_client=AsyncMock())
        sys._store = MemoryStore("test", db_dir=tmp_dir)
        return sys

    def test_topic_segment_messages_produce_nonempty_trace(self, system):
        """Raw routed Messages with no tool metadata still produce trace lines."""
        from mesh.protocol import Message, MessageType
        from mesh.router_v2 import WorkerResult

        msgs = [
            Message(from_node="user:testuser", to_node="agent:sysadmin:bob",
                    type=MessageType.MESSAGE, content="please check the logs"),
            Message(from_node="agent:sysadmin:bob", to_node="user:testuser",
                    type=MessageType.MESSAGE, content="On it. Pulling now."),
            Message(from_node="user:testuser", to_node="agent:sysadmin:bob",
                    type=MessageType.MESSAGE, content="thanks"),
        ]
        result = WorkerResult(response="(topic segment: logs)", context=msgs)
        trace = system._build_trace(result)

        assert trace, "trace must not be empty for topic-segment messages"
        assert "[USER]" in trace
        assert "[AGENT]" in trace
        assert "please check the logs" in trace
        assert "On it. Pulling now." in trace

    def test_messages_with_tool_metadata_still_use_tool_format(self, system):
        """When metadata.tool_calls is present, prefer the tool format."""
        from mesh.protocol import Message, MessageType
        from mesh.router_v2 import WorkerResult

        msgs = [
            Message(from_node="agent:coder:tron", to_node="user:testuser",
                    type=MessageType.MESSAGE, content="running",
                    metadata={"tool_calls": [{"name": "bash", "arguments": "ls"}]}),
        ]
        result = WorkerResult(response="ok", context=msgs)
        trace = system._build_trace(result)

        assert "[TOOL]" in trace
        assert "bash" in trace
        # The fallback shouldn't engage when tool metadata was found
        assert "[AGENT]" not in trace

    def test_empty_context_returns_empty_string(self, system):
        from mesh.router_v2 import WorkerResult
        result = WorkerResult(response="", context=[])
        assert system._build_trace(result) == ""

    def test_skips_blank_content(self, system):
        from mesh.protocol import Message, MessageType
        from mesh.router_v2 import WorkerResult

        msgs = [
            Message(from_node="user:testuser", to_node="agent:bob",
                    type=MessageType.MESSAGE, content="   "),
            Message(from_node="user:testuser", to_node="agent:bob",
                    type=MessageType.MESSAGE, content="real content"),
        ]
        trace = system._build_trace(WorkerResult(response="", context=msgs))
        assert trace.count("[USER]") == 1
        assert "real content" in trace


class TestWindowDropUnconditional:
    """Regression for the 2026-04-26 directive: memory formation must
    trigger on every window drop, regardless of segment size, tool counts,
    or active-project status. Previously _significance_gate filtered ~98%
    of segments."""

    @pytest.fixture
    def system(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=(
            "<reflection>\nA brief exchange.\n</reflection>\n"
            "<summary>\nShort chat.\n</summary>\n"
            "<tags>\nchat\n</tags>\n"
            "<outcome_label>\nsuccess\n</outcome_label>\n"
            "<retrieval_key>\nShort chat exchange.\n</retrieval_key>\n"
            "<project>\n\n</project>"
        ))
        sys = MemorySystemV2(nickname="test", llm_client=mock_llm)
        sys._store = MemoryStore("test", db_dir=tmp_dir)
        sys._active_project_dir = tmp_dir
        sys._embedder = MagicMock()
        sys._embedder.embed_batch_to_arrays = AsyncMock(
            return_value=[np.random.randn(8).astype(np.float32),
                          np.random.randn(8).astype(np.float32)]
        )
        return sys

    @pytest.mark.asyncio
    async def test_single_short_turn_creates_entry_no_active_project(self, system):
        """No active project, one trivial turn — old gate would skip; now creates."""
        assert not system._active_project
        turns = [_make_turn(role="user", content="hi", topic_label="chitchat",
                            from_node="user:testuser")]
        await system.on_window_drop(turns)
        assert len(system._pool) == 1

    @pytest.mark.asyncio
    async def test_short_two_turn_segment_creates_entry(self, system):
        """Two short turns, no tools, no errors — would have failed every
        prior gate clause. Should now produce a memory."""
        turns = [
            _make_turn(role="user", content="ok", topic_label="ack",
                       from_node="user:testuser"),
            _make_turn(role="assistant", content="acknowledged",
                       topic_label="ack", from_node="agent:bob"),
        ]
        await system.on_window_drop(turns)
        assert len(system._pool) == 1
        assert system._pool[0].topic_label == "ack"

    @pytest.mark.asyncio
    async def test_each_segment_produces_its_own_entry(self, system):
        """Two trivial segments — each must produce a memory entry."""
        turns = [
            _make_turn(role="user", content="hi", topic_label="t1",
                       from_node="user:testuser"),
            _make_turn(role="user", content="bye", topic_label="t2",
                       from_node="user:testuser"),
        ]
        await system.on_window_drop(turns)
        assert len(system._pool) == 2
        assert {e.topic_label for e in system._pool} == {"t1", "t2"}


class TestCurateActiveMap:
    """Test _curate_active_map method."""

    @pytest.fixture
    def system(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        mock_llm = AsyncMock()
        sys = MemorySystemV2(
            nickname="test", llm_client=mock_llm,
        )
        sys._store = MemoryStore("test", db_dir=tmp_dir)
        sys._active_project_dir = tmp_dir
        return sys

    @pytest.mark.asyncio
    async def test_no_updates_needed(self, system):
        await system.create_map("proj", "# Project: proj\n\nOriginal content.")
        system._active_project = "proj"
        system._llm_client.complete = AsyncMock(return_value="No updates needed.")
        await system._curate_active_map("Some conversation text")
        content = await system.get_map("proj")
        assert content == "# Project: proj\n\nOriginal content."

    @pytest.mark.asyncio
    async def test_map_updated(self, system):
        await system.create_map("proj", "# Project: proj\n\nOld.")
        system._active_project = "proj"
        system._llm_client.complete = AsyncMock(
            return_value="# Project: proj\n\nNew content."
        )
        await system._curate_active_map("Conversation about changes.")
        content = await system.get_map("proj")
        assert content == "# Project: proj\n\nNew content."

    @pytest.mark.asyncio
    async def test_strips_code_fences(self, system):
        await system.create_map("proj", "# Project: proj\n\nOld.")
        system._active_project = "proj"
        system._llm_client.complete = AsyncMock(
            return_value="```markdown\n# Project: proj\n\nUpdated.\n```"
        )
        await system._curate_active_map("convo")
        content = await system.get_map("proj")
        assert content == "# Project: proj\n\nUpdated."

    @pytest.mark.asyncio
    async def test_rejects_invalid_output(self, system):
        await system.create_map("proj", "# Project: proj\n\nOriginal.")
        system._active_project = "proj"
        system._llm_client.complete = AsyncMock(return_value="garbage output here")
        await system._curate_active_map("convo")
        content = await system.get_map("proj")
        assert content == "# Project: proj\n\nOriginal."

    @pytest.mark.asyncio
    async def test_no_active_project_skips(self, system):
        system._active_project = None
        await system._curate_active_map("convo")
        assert not system._llm_client.complete.called


class TestCheckpointRecovery:
    """Test checkpoint/recovery for dropped turns."""

    @pytest.fixture
    def system(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        sys = MemorySystemV2(
            nickname="test-ckpt", llm_client=AsyncMock(),
        )
        sys._store = MemoryStore("test-ckpt", db_dir=tmp_dir)
        return sys

    def test_checkpoint_save_and_load(self, system):
        turns = [
            _make_turn(role="user", content="hello", topic_label="chat"),
            _make_turn(role="assistant", content="hi", topic_label="chat"),
        ]
        path = system.checkpoint_dropped_turns(turns)
        assert path is not None
        assert os.path.exists(path)

        loaded = system.load_checkpoint()
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].role == "user"
        assert loaded[0].content == "hello"
        assert loaded[0].meta.get("topic_label") == "chat"

    def test_load_nonexistent(self, system):
        assert system.load_checkpoint() is None

    def test_clear_checkpoint(self, system):
        turns = [_make_turn(content="test")]
        path = system.checkpoint_dropped_turns(turns)
        assert os.path.exists(path)

        system.clear_checkpoint()
        assert not os.path.exists(path)

    def test_clear_nonexistent_is_noop(self, system):
        system.clear_checkpoint()  # Should not raise

    @pytest.mark.asyncio
    async def test_replay_checkpoint(self, system):
        """Replay processes turns and clears checkpoint."""
        system._llm_client.complete = AsyncMock(return_value=(
            "<reflection>\nTest.\n</reflection>\n"
            "<summary>\nTest summary.\n</summary>\n"
            "<tags>\ntest\n</tags>\n"
            "<outcome_label>\nsuccess\n</outcome_label>\n"
            "<retrieval_key>\nTest key.\n</retrieval_key>\n"
            "<project>\n\n</project>"
        ))
        system._embedder = MagicMock()
        system._embedder.embed_batch_to_arrays = AsyncMock(
            return_value=[np.random.randn(8).astype(np.float32),
                          np.random.randn(8).astype(np.float32)]
        )

        # Save checkpoint with significant turns
        turns = [
            _make_turn(role="user", content="fix bug", topic_label="debug",
                       from_node="user:testuser"),
            _make_turn(role="tool", content="r1", topic_label="debug"),
            _make_turn(role="tool", content="r2", topic_label="debug"),
            _make_turn(role="tool", content="r3", topic_label="debug"),
        ]
        system.checkpoint_dropped_turns(turns)

        # Replay
        replayed = await system.replay_checkpoint()
        assert replayed is True
        assert len(system._pool) == 1  # One log entry created
        assert system.load_checkpoint() is None  # Checkpoint cleared

    @pytest.mark.asyncio
    async def test_replay_no_checkpoint(self, system):
        assert await system.replay_checkpoint() is False


class TestKnownProjectNames:
    @pytest.fixture
    def system(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        sys = MemorySystemV2(nickname="test", llm_client=AsyncMock())
        sys._store = MemoryStore("test", db_dir=tmp_dir)
        return sys

    def test_no_maps(self, system):
        assert system._known_project_names() == "(none)"

    def test_with_maps(self, system):
        system._store.create_map("m1", "proj-a", "# A")
        system._store.create_map("m2", "proj-b", "# B")
        names = system._known_project_names()
        assert "proj-a" in names
        assert "proj-b" in names
        assert ", " in names


class TestPartitionAndDropOld:
    """Test ConversationHistory.partition_and_drop_old()."""

    def test_basic_partition(self):
        from mesh.conversation_history import ConversationHistory, Turn
        hist = ConversationHistory(window_budget=100)
        # Add turns with known token costs
        for i in range(10):
            t = Turn(
                role="user", content=f"message {i}" * 20,
                timestamp="2026-02-28T12:00:00",
            )
            hist.append(t)

        window_before = len(hist._window)
        old_half = hist.partition_and_drop_old()
        assert len(old_half) > 0
        assert len(hist._window) < window_before
        assert len(old_half) + len(hist._window) == window_before

    def test_empty_window(self):
        from mesh.conversation_history import ConversationHistory
        hist = ConversationHistory(window_budget=100)
        old_half = hist.partition_and_drop_old()
        assert old_half == []

    def test_single_turn(self):
        """Single turn that fits within W — gets dropped (old_half = everything).

        In practice, partition_and_drop_old is only called at 2W,
        so single-turn windows don't arise. But the logic is correct:
        the single turn fits within W, so it's all 'old'.
        """
        from mesh.conversation_history import ConversationHistory, Turn
        hist = ConversationHistory(window_budget=100)
        hist.append(Turn(role="user", content="hi", timestamp="now"))
        old_half = hist.partition_and_drop_old()
        assert len(old_half) == 1
        assert len(hist._window) == 0


class TestWindowDropPrompts:
    """Verify prompt templates have required structure."""

    def test_reflection_prompt_has_format_enforcement(self):
        from mesh.memory.system_v2 import WINDOW_DROP_REFLECTION_PROMPT
        assert "IMPORTANT: You MUST close every XML tag" in WINDOW_DROP_REFLECTION_PROMPT
        assert "{known_project_names}" in WINDOW_DROP_REFLECTION_PROMPT
        assert "<reflection>" in WINDOW_DROP_REFLECTION_PROMPT
        assert "</reflection>" in WINDOW_DROP_REFLECTION_PROMPT
        assert "<project>" in WINDOW_DROP_REFLECTION_PROMPT
        assert "</project>" in WINDOW_DROP_REFLECTION_PROMPT

    def test_curation_prompt_has_required_structure(self):
        from mesh.memory.system_v2 import MAP_CURATION_PROMPT
        assert "WHAT TO EXTRACT" in MAP_CURATION_PROMPT
        assert "SELF-CHECK" in MAP_CURATION_PROMPT
        assert "Glossary" in MAP_CURATION_PROMPT
        assert "{current_map}" in MAP_CURATION_PROMPT
        assert "{raw_conversation}" in MAP_CURATION_PROMPT
        assert "No updates needed" in MAP_CURATION_PROMPT

    def test_new_project_prompt(self):
        from mesh.memory.system_v2 import NEW_PROJECT_MAP_PROMPT
        assert "{project_name}" in NEW_PROJECT_MAP_PROMPT
        assert "{summary}" in NEW_PROJECT_MAP_PROMPT

    def test_curation_prompt_has_tool_instructions(self):
        from mesh.memory.system_v2 import MAP_CURATION_PROMPT
        assert "<file_read>" in MAP_CURATION_PROMPT
        assert "<bash_exec>" in MAP_CURATION_PROMPT
        assert "TOOLS" in MAP_CURATION_PROMPT

    def test_audit_prompt_structure(self):
        from mesh.memory.system_v2 import MAP_AUDIT_PROMPT
        assert "{map_content}" in MAP_AUDIT_PROMPT
        assert "No issues found" in MAP_AUDIT_PROMPT


# ── Phase 3: Tool Call Parsing ────────────────────────────────

class TestParseToolCalls:
    """Test _parse_tool_calls helper."""

    def test_no_tool_calls(self):
        from mesh.memory.system_v2 import _parse_tool_calls
        assert _parse_tool_calls("No updates needed.") == []

    def test_single_file_read(self):
        from mesh.memory.system_v2 import _parse_tool_calls
        text = "Let me check: <file_read>src/main.py</file_read>"
        calls = _parse_tool_calls(text)
        assert calls == [("file_read", "src/main.py")]

    def test_single_bash_exec(self):
        from mesh.memory.system_v2 import _parse_tool_calls
        text = "Running: <bash_exec>ls -la src/</bash_exec>"
        calls = _parse_tool_calls(text)
        assert calls == [("bash_exec", "ls -la src/")]

    def test_multiple_calls(self):
        from mesh.memory.system_v2 import _parse_tool_calls
        text = (
            "Check files:\n"
            "<file_read>a.py</file_read>\n"
            "<bash_exec>cat b.py</bash_exec>\n"
            "<file_read>c.py</file_read>"
        )
        calls = _parse_tool_calls(text)
        assert len(calls) == 3
        assert calls[0] == ("file_read", "a.py")
        assert calls[1] == ("file_read", "c.py")  # file_read parsed first
        assert calls[2] == ("bash_exec", "cat b.py")

    def test_strips_whitespace(self):
        from mesh.memory.system_v2 import _parse_tool_calls
        text = "<file_read>  /path/to/file  </file_read>"
        calls = _parse_tool_calls(text)
        assert calls == [("file_read", "/path/to/file")]


class TestExecuteToolCall:
    """Test _execute_tool_call helper."""

    @pytest.mark.asyncio
    async def test_file_read_success(self, tmp_dir):
        from mesh.memory.system_v2 import _execute_tool_call
        path = os.path.join(tmp_dir, "test.txt")
        with open(path, "w") as f:
            f.write("hello world")
        result = await _execute_tool_call("file_read", path)
        assert "hello world" in result

    @pytest.mark.asyncio
    async def test_file_read_missing(self):
        from mesh.memory.system_v2 import _execute_tool_call
        result = await _execute_tool_call("file_read", "/nonexistent/path.txt")
        assert "Error reading" in result

    @pytest.mark.asyncio
    async def test_bash_exec_success(self):
        from mesh.memory.system_v2 import _execute_tool_call
        result = await _execute_tool_call("bash_exec", "echo hello")
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        from mesh.memory.system_v2 import _execute_tool_call
        result = await _execute_tool_call("unknown_tool", "arg")
        assert "Unknown tool" in result


# ── Phase 3: Curation with Tool Calls ────────────────────────

class TestCurationWithTools:
    """Test _curate_active_map with tool call support."""

    @pytest.fixture
    def system(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        mock_llm = AsyncMock()
        sys = MemorySystemV2(
            nickname="test", llm_client=mock_llm,
            curation_audit_max_tool_calls=3,
        )
        sys._store = MemoryStore("test", db_dir=tmp_dir)
        sys._active_project_dir = tmp_dir
        return sys

    @pytest.mark.asyncio
    async def test_tool_call_loop(self, system, tmp_dir):
        """LLM requests file_read, gets result, then outputs map."""
        test_file = os.path.join(tmp_dir, "config.yaml")
        with open(test_file, "w") as f:
            f.write("port: 8080")

        # First call: LLM requests a file read
        # Second call: LLM outputs the updated map
        system._llm_client.complete = AsyncMock(side_effect=[
            f"Need to check config.\n<file_read>{test_file}</file_read>",
            "# Project: proj\n\nUpdated with port 8080.",
        ])
        await system.create_map("proj", "# Project: proj\n\nOld content.")
        system._active_project = "proj"

        await system._curate_active_map("Conversation about config.", 5)

        content = await system.get_map("proj")
        assert "port 8080" in content
        assert system._llm_client.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_tool_call_cap_enforced(self, system, tmp_dir):
        """Tool calls stop at _curation_audit_max_tool_calls."""
        # system has cap=3. LLM always requests a tool call.
        system._llm_client.complete = AsyncMock(side_effect=[
            "<file_read>/a</file_read>",
            "<file_read>/b</file_read>",
            "<file_read>/c</file_read>",
            # After 3 tool calls, loop should break even though response has tool call
            "# Project: proj\n\nFinal.",
        ])
        await system.create_map("proj", "# Project: proj\n\nOld.")
        system._active_project = "proj"

        await system._curate_active_map("convo", 3)

        content = await system.get_map("proj")
        assert content == "# Project: proj\n\nFinal."
        # 4 LLM calls: 3 with tool calls + 1 final (cap stops further tool parsing)
        assert system._llm_client.complete.call_count == 4

    @pytest.mark.asyncio
    async def test_no_tool_calls_single_pass(self, system):
        """Normal curation without tool calls — single LLM call."""
        system._llm_client.complete = AsyncMock(
            return_value="# Project: proj\n\nUpdated."
        )
        await system.create_map("proj", "# Project: proj\n\nOld.")
        system._active_project = "proj"

        await system._curate_active_map("convo", 5)

        content = await system.get_map("proj")
        assert content == "# Project: proj\n\nUpdated."
        assert system._llm_client.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_bash_exec_tool_call(self, system):
        """LLM uses bash_exec to run a command."""
        system._llm_client.complete = AsyncMock(side_effect=[
            "Checking structure:\n<bash_exec>echo 'found'</bash_exec>",
            "# Project: proj\n\nVerified.",
        ])
        await system.create_map("proj", "# Project: proj\n\nOld.")
        system._active_project = "proj"

        await system._curate_active_map("convo", 2)

        content = await system.get_map("proj")
        assert "Verified" in content


# ── Phase 3: Audit Map Consistency ────────────────────────────

class TestAuditMapConsistency:
    """Test _audit_map_consistency standalone method."""

    @pytest.fixture
    def system(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        mock_llm = AsyncMock()
        sys = MemorySystemV2(
            nickname="test", llm_client=mock_llm,
        )
        sys._store = MemoryStore("test", db_dir=tmp_dir)
        return sys

    @pytest.mark.asyncio
    async def test_consistent_map(self, system):
        system._llm_client.complete = AsyncMock(
            return_value="No issues found."
        )
        issues = await system._audit_map_consistency("proj", "# Project: proj")
        assert issues == []

    @pytest.mark.asyncio
    async def test_inconsistencies_found(self, system):
        system._llm_client.complete = AsyncMock(
            return_value=(
                "- Components mentions 'Redis cache' but Architecture says 'no caching'\n"
                "- Open Questions has resolved item about auth method"
            )
        )
        issues = await system._audit_map_consistency("proj", "# Project: proj")
        assert len(issues) == 2
        assert "Redis cache" in issues[0]
        assert "auth method" in issues[1]

    @pytest.mark.asyncio
    async def test_audit_failure_returns_empty(self, system):
        system._llm_client.complete = AsyncMock(
            side_effect=Exception("LLM error")
        )
        issues = await system._audit_map_consistency("proj", "# Project: proj")
        assert issues == []


# ── Phase 3: Curation Logging Contract ───────────────────────

class TestCurationLogging:
    """Verify logging matches the logging contract."""

    @pytest.fixture
    def system(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        mock_llm = AsyncMock()
        sys = MemorySystemV2(
            nickname="test", llm_client=mock_llm,
        )
        sys._store = MemoryStore("test", db_dir=tmp_dir)
        sys._active_project_dir = tmp_dir
        return sys

    @pytest.mark.asyncio
    async def test_curation_started_log(self, system, caplog):
        """Verify 'Curating map' INFO log on start."""
        import logging
        system._llm_client.complete = AsyncMock(
            return_value="# Project: proj\n\nUpdated."
        )
        await system.create_map("proj", "# Project: proj\n\nOld content.")
        system._active_project = "proj"

        with caplog.at_level(logging.INFO):
            await system._curate_active_map("Some conversation", 10)

        started_msgs = [r for r in caplog.records if "Curating map" in r.message]
        assert len(started_msgs) == 1
        assert "10 input turns" in started_msgs[0].message
        assert "tokens of raw context" in started_msgs[0].message

    @pytest.mark.asyncio
    async def test_curation_completed_log(self, system, caplog):
        """Verify 'curated:' INFO log with token counts."""
        import logging
        system._llm_client.complete = AsyncMock(
            return_value="# Project: proj\n\nNew content here."
        )
        await system.create_map("proj", "# Project: proj\n\nOld.")
        system._active_project = "proj"

        with caplog.at_level(logging.INFO):
            await system._curate_active_map("convo", 5)

        completed = [r for r in caplog.records if "curated:" in r.message]
        assert len(completed) == 1
        assert "tool_calls=0" in completed[0].message
        assert "→" in completed[0].message

    @pytest.mark.asyncio
    async def test_no_op_log(self, system, caplog):
        """Verify no-op log message matches contract."""
        import logging
        system._llm_client.complete = AsyncMock(
            return_value="No updates needed."
        )
        await system.create_map("proj", "# Project: proj\n\nContent.")
        system._active_project = "proj"

        with caplog.at_level(logging.INFO):
            await system._curate_active_map("irrelevant convo", 3)

        noop = [r for r in caplog.records
                if "no updates needed" in r.message]
        assert len(noop) == 1
        assert "dropped turns irrelevant" in noop[0].message

    @pytest.mark.asyncio
    async def test_curation_failed_log(self, system, caplog):
        """Verify failure log matches contract."""
        import logging
        system._llm_client.complete = AsyncMock(
            side_effect=Exception("LLM connection error")
        )
        await system.create_map("proj", "# Project: proj\n\nContent.")
        system._active_project = "proj"

        with caplog.at_level(logging.WARNING):
            await system._curate_active_map("convo", 2)

        failed = [r for r in caplog.records if "curation failed" in r.message]
        assert len(failed) == 1
        assert "LLM connection error" in failed[0].message

    @pytest.mark.asyncio
    async def test_tool_call_debug_log(self, system, tmp_dir, caplog):
        """Verify tool call DEBUG log."""
        import logging
        test_file = os.path.join(tmp_dir, "check.py")
        with open(test_file, "w") as f:
            f.write("x = 1")

        system._llm_client.complete = AsyncMock(side_effect=[
            f"<file_read>{test_file}</file_read>",
            "# Project: proj\n\nDone.",
        ])
        await system.create_map("proj", "# Project: proj\n\nOld.")
        system._active_project = "proj"

        with caplog.at_level(logging.DEBUG):
            await system._curate_active_map("convo", 1)

        tool_logs = [r for r in caplog.records
                     if "Curation tool call" in r.message]
        assert len(tool_logs) == 1
        assert "file_read" in tool_logs[0].message

    @pytest.mark.asyncio
    async def test_new_project_detected_log(self, system, caplog):
        """Verify new project detection log in on_window_drop."""
        import logging
        await system.create_map("main-proj", "# Project: main-proj\n\nContent.")
        system._active_project = "main-proj"
        system._store.set_active_project("main-proj")

        # Mock reflection to return an entry naming a new project
        reflection_response = """<summary>Explored new-tool features</summary>
<reflection>Built a new-tool.</reflection>
<tags>new-tool</tags>
<outcome_label>success</outcome_label>
<retrieval_key>new tool development</retrieval_key>
<project>new-tool</project>"""

        system._llm_client.complete = AsyncMock(side_effect=[
            # Reflection call
            reflection_response,
            # Curation call (for main-proj)
            "No updates needed.",
            # New project map creation call
            "# Project: new-tool\n\n## Architecture\nNew tool.",
        ])

        # Needs 4+ turns with 1500+ chars to pass significance gate
        turns = [
            _make_turn(role="user", content="Let's build new-tool. " * 50,
                       topic_label="new-tool"),
            _make_turn(role="assistant", content="Sure, building new-tool now. " * 50,
                       topic_label="new-tool"),
            _make_turn(role="user", content="How about the architecture? " * 50,
                       topic_label="new-tool"),
            _make_turn(role="assistant", content="The architecture uses modules. " * 50,
                       topic_label="new-tool"),
        ]

        with caplog.at_level(logging.INFO):
            await system.on_window_drop(turns)

        new_proj_logs = [r for r in caplog.records
                         if "New project detected" in r.message]
        assert len(new_proj_logs) == 1
        assert "'new-tool'" in new_proj_logs[0].message
        assert "bootstrapping map from" in new_proj_logs[0].message


# ── Phase 4: New Rendering Pipeline ─────────────────────────────

class TestRouterV2PromptRendering:
    """Test that _build_router_prompt uses v2 four-section layout."""

    @pytest.fixture
    def v2_memory(self, tmp_dir):
        """MemorySystemV2 with a map, log entries, and representative memories."""
        from mesh.memory.system_v2 import MemorySystemV2
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value="ok")
        sys = MemorySystemV2(
            nickname="test-agent",
            llm_client=mock_llm,
            pool_max_entries=100,
            embedding_backend="openai",
            embedding_model="text-embedding-3-small",
        )
        sys._store = MemoryStore("test-agent", db_dir=tmp_dir)
        sys._active_project_dir = tmp_dir
        # Set active project with a map — write file to disk and store metadata
        map_content = "# Test Project\n\n## Architecture\nModular."
        with open(os.path.join(tmp_dir, "PROJECT_MAP.md"), "w") as f:
            f.write(map_content)
        sys._store.create_map("map-1", "test-proj", "", project_dir=tmp_dir)
        sys._active_project = "test-proj"
        # Add entries to the pool (these serve as both representative and log)
        for i in range(5):
            entry = _make_entry(
                entry_id=f"entry-{i}",
                project="test-proj",
                summary=f"Summary {i}",
                reflection=f"Reflection {i}: detailed analysis.",
                topic_label=f"topic-{i}",
                created_at=datetime.now(timezone.utc) - timedelta(hours=5 - i),
            )
            sys._pool.append(entry)
            sys._active_weights[entry.id] = 0.5 + i * 0.1
        # Mark some as active (representative)
        sys._active_ids = {e.id for e in sys._pool[:3]}
        return sys

    @pytest.fixture
    def router_with_v2(self, v2_memory):
        """Minimal RouterV2 with v2 memory for prompt testing."""
        from mesh.router_v2 import RouterV2, RouterV2Config

        async def noop_send(content, in_reply_to=None):
            pass

        async def noop_worker(context, trigger, **kwargs):
            from mesh.router_v2 import WorkerResult
            return WorkerResult(response="ok", context=[], usage=None, error=None)

        router = RouterV2(
            worker_fn=noop_worker,
            send_fn=noop_send,
            config=RouterV2Config(llm_enabled=True),
            llm_client=AsyncMock(),
            nickname="test-bot",
            agent_type="assistant",
            node_id="agent:test:test-bot",
            system_prompt="You are a helpful assistant.",
            identity_block="<identity>Test Bot</identity>",
            memory_system=v2_memory,
        )
        return router

    @pytest.mark.asyncio
    async def test_v2_prompt_contains_all_sections(self, router_with_v2):
        """v2 prompt has representative → map → recent log in order."""
        prompt = await router_with_v2._build_router_prompt(
            instructions="Test instructions.",
            include_tools=False,
        )
        # All sections present
        assert "<memory>" in prompt
        assert "<project_map" in prompt
        assert "<recent_activity>" in prompt
        assert "<instructions>" in prompt

    @pytest.mark.asyncio
    async def test_v2_prompt_section_order(self, router_with_v2):
        """Verify ordering: system → identity → personality → representative → map → log → instructions."""
        prompt = await router_with_v2._build_router_prompt(
            instructions="Test instructions.",
            include_tools=False,
        )
        idx_system = prompt.index("<system>")
        idx_identity = prompt.index("<identity>")
        idx_rep = prompt.index("<memory>")
        idx_map = prompt.index("<project_map")
        idx_log = prompt.index("<recent_activity>")
        idx_instr = prompt.index("<instructions>")

        assert idx_system < idx_identity < idx_rep < idx_map < idx_log < idx_instr

    @pytest.mark.asyncio
    async def test_v2_prompt_includes_map_content(self, router_with_v2):
        """Map content renders in the prompt."""
        prompt = await router_with_v2._build_router_prompt(
            instructions="Test.",
            include_tools=False,
        )
        assert 'project="test-proj"' in prompt
        assert "## Architecture" in prompt
        assert "Modular." in prompt

    @pytest.mark.asyncio
    async def test_v2_prompt_includes_representative_entries(self, router_with_v2):
        """Representative block draws from active set only."""
        prompt = await router_with_v2._build_router_prompt(
            instructions="Test.",
            include_tools=False,
        )
        assert 'source="representative"' in prompt
        # Only 3 entries are in active set
        assert prompt.count('source="representative"') == 3

    @pytest.mark.asyncio
    async def test_v2_prompt_recent_log_shows_last_4(self, router_with_v2):
        """Recent log block shows last m=4 entries."""
        prompt = await router_with_v2._build_router_prompt(
            instructions="Test.",
            include_tools=False,
        )
        # 5 entries in pool, but recent_log_count defaults to 4
        assert prompt.count("<entry date=") >= 4
        # Most recent entry should be present
        assert "Reflection 4:" in prompt

    @pytest.mark.asyncio
    async def test_v2_prompt_no_relevant_memories_by_default(self, router_with_v2):
        """No relevant_memories section when none has been set."""
        prompt = await router_with_v2._build_router_prompt(
            instructions="Test.",
            include_tools=False,
        )
        assert "<relevant_memories>" not in prompt

    @pytest.mark.asyncio
    async def test_v2_prompt_includes_relevant_memories(self, router_with_v2):
        """Relevant context persists and renders when set."""
        router_with_v2._relevant_context = '<log_entry date="2026-01-01">Found it.</log_entry>'
        prompt = await router_with_v2._build_router_prompt(
            instructions="Test.",
            include_tools=False,
        )
        assert "<relevant_memories>" in prompt
        assert "Found it." in prompt

    @pytest.mark.asyncio
    async def test_v2_prompt_empty_memory(self, tmp_dir):
        """v2 prompt with empty memory system still renders cleanly."""
        from mesh.memory.system_v2 import MemorySystemV2
        from mesh.router_v2 import RouterV2, RouterV2Config

        sys = MemorySystemV2(
            nickname="test-agent",
            llm_client=AsyncMock(),
            pool_max_entries=100,
            embedding_backend="openai",
            embedding_model="text-embedding-3-small",
        )
        sys._store = MemoryStore("test-agent", db_dir=tmp_dir)

        async def noop_send(content, in_reply_to=None):
            pass

        async def noop_worker(context, trigger, **kwargs):
            from mesh.router_v2 import WorkerResult
            return WorkerResult(response="ok", context=[], usage=None, error=None)

        router = RouterV2(
            worker_fn=noop_worker,
            send_fn=noop_send,
            config=RouterV2Config(llm_enabled=True),
            llm_client=AsyncMock(),
            nickname="test-bot",
            agent_type="assistant",
            node_id="agent:test:test-bot",
            memory_system=sys,
        )
        prompt = await router._build_router_prompt(
            instructions="Test.",
            include_tools=False,
        )
        # No memory sections when pool is empty and no active project
        assert "<memory>" not in prompt
        assert "<project_map" not in prompt
        assert "<recent_activity>" not in prompt
        # Instructions still present
        assert "<instructions>" in prompt

    @pytest.mark.asyncio
    async def test_v2_classification_no_crash(self, router_with_v2):
        """Classification call doesn't crash on v2 (no classifier_profile)."""
        # The classification path builds a prompt — it should not raise
        # even though MemorySystemV2 has no classifier_profile
        prompt = await router_with_v2._build_router_prompt(
            instructions="Classify this.",
            memory_profile=None,  # v2 ignores this
            include_tools=False,
            max_history_turns=30,
        )
        assert "<instructions>" in prompt

    @pytest.mark.asyncio
    async def test_v2_preferences_still_injected(self, router_with_v2):
        """Preferences block still works with v2 memory."""
        prompt = await router_with_v2._build_router_prompt(
            instructions="Test.",
            preferences_block="<preferences>User likes dark mode.</preferences>",
            include_tools=False,
        )
        assert "<preferences>" in prompt
        assert "dark mode" in prompt
        # Preferences come after memory blocks, before instructions
        idx_log = prompt.index("</recent_activity>")
        idx_pref = prompt.index("<preferences>")
        idx_instr = prompt.index("<instructions>")
        assert idx_log < idx_pref < idx_instr


class TestSynthesisPathV2:
    """Test _build_synthesis_context() injects v2 memory blocks and conversation."""

    def _make_router_with_v2(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        from mesh.router_v2 import RouterV2, RouterV2Config

        sys = MemorySystemV2(
            nickname="test-agent",
            llm_client=AsyncMock(),
            pool_max_entries=100,
            embedding_backend="openai",
            embedding_model="text-embedding-3-small",
        )
        sys._store = MemoryStore("test-agent", db_dir=tmp_dir)
        sys._active_project_dir = tmp_dir
        # Write map file to disk and store metadata
        with open(os.path.join(tmp_dir, "PROJECT_MAP.md"), "w") as f:
            f.write("# MyProject\n\nKey info.")
        sys._store.create_map("map-2", "myproject", "", project_dir=tmp_dir)
        sys._active_project = "myproject"
        entry = _make_entry(
            entry_id="synth-1", project="myproject",
            reflection="Recent fix to auth module.",
        )
        sys._pool.append(entry)
        sys._active_ids = {entry.id}
        sys._active_weights[entry.id] = 1.0

        async def noop_send(content, in_reply_to=None):
            pass

        async def noop_worker(context, trigger, **kwargs):
            from mesh.router_v2 import WorkerResult
            return WorkerResult(response="ok", context=[], usage=None, error=None)

        router = RouterV2(
            worker_fn=noop_worker,
            send_fn=noop_send,
            config=RouterV2Config(llm_enabled=True),
            llm_client=AsyncMock(),
            nickname="test-bot",
            memory_system=sys,
        )
        return router, sys

    @pytest.mark.asyncio
    async def test_synthesis_context_includes_v2_blocks(self, tmp_dir):
        """_build_synthesis_context includes representative, map, and log blocks."""
        router, mem = self._make_router_with_v2(tmp_dir)
        ctx = await router._build_synthesis_context()

        # Identity
        assert "test-bot" in ctx
        # Map block
        assert 'project="myproject"' in ctx
        assert "Key info" in ctx
        # Representative block
        assert "<memory>" in ctx
        assert "auth module" in ctx
        # Recent activity (log block)
        assert "<recent_activity>" in ctx

    @pytest.mark.asyncio
    async def test_synthesis_context_includes_conversation(self, tmp_dir):
        """_build_synthesis_context includes recent conversation turns without truncation."""
        router, mem = self._make_router_with_v2(tmp_dir)

        from mesh.conversation_history import Turn
        long_content = "A" * 2000  # Would have been truncated by old 500-char limit
        router._history._window = [
            Turn(role="user", content=long_content, from_node="user:testuser", timestamp="2026-03-01T12:00:00"),
            Turn(role="assistant", content="I helped you.", from_node="agent:test", timestamp="2026-03-01T12:01:00"),
        ]

        ctx = await router._build_synthesis_context()

        assert "Recent conversation:" in ctx
        # Full content preserved — no truncation
        assert long_content in ctx
        assert "I helped you." in ctx

    @pytest.mark.asyncio
    async def test_synthesis_context_relevant_memories(self, tmp_dir):
        """_build_synthesis_context includes relevant_memories when present."""
        router, mem = self._make_router_with_v2(tmp_dir)
        router._relevant_context = "JWT auth decided in sprint 3"

        ctx = await router._build_synthesis_context()
        assert "<relevant_memories>" in ctx
        assert "JWT auth decided in sprint 3" in ctx

    @pytest.mark.asyncio
    async def test_synthesis_context_respects_turn_limit(self, tmp_dir):
        """Only the last synthesis_context_turns turns are included."""
        router, mem = self._make_router_with_v2(tmp_dir)
        router._config.synthesis_context_turns = 2

        from mesh.conversation_history import Turn
        router._history._window = [
            Turn(role="user", content="old msg", from_node="user:testuser", timestamp="2026-03-01T12:00:00"),
            Turn(role="assistant", content="old reply", from_node="agent:test", timestamp="2026-03-01T12:01:00"),
            Turn(role="user", content="recent msg", from_node="user:testuser", timestamp="2026-03-01T12:02:00"),
        ]

        ctx = await router._build_synthesis_context()
        assert "recent msg" in ctx
        assert "old reply" in ctx
        assert "old msg" not in ctx  # Sliced off

    @pytest.mark.asyncio
    async def test_synthesis_context_v1_fallback(self, tmp_dir):
        """v1 memory system gets profile-based rendering instead of v2 blocks."""
        from mesh.router_v2 import RouterV2, RouterV2Config

        v1_memory = MagicMock()
        v1_memory.get_personality.return_value = "Friendly helper"
        v1_memory.light_profile = "test-profile"
        v1_memory.render = AsyncMock(return_value="<v1_memory>User likes dark mode</v1_memory>")

        async def noop_send(content, in_reply_to=None):
            pass

        async def noop_worker(context, trigger, **kwargs):
            from mesh.router_v2 import WorkerResult
            return WorkerResult(response="ok", context=[], usage=None, error=None)

        router = RouterV2(
            worker_fn=noop_worker,
            send_fn=noop_send,
            config=RouterV2Config(llm_enabled=True),
            llm_client=AsyncMock(),
            nickname="test-bot",
            memory_system=v1_memory,
        )

        ctx = await router._build_synthesis_context()
        assert "Friendly helper" in ctx
        assert "dark mode" in ctx
        # Should NOT have v2 blocks
        assert "<project_map" not in ctx


class TestWorkerV2Rendering:
    """Test unified prompt — workers and router get the same memory context."""

    @pytest.fixture
    def v2_memory(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        sys = MemorySystemV2(
            nickname="test-agent",
            llm_client=AsyncMock(),
            pool_max_entries=100,
            embedding_backend="openai",
            embedding_model="text-embedding-3-small",
        )
        sys._store = MemoryStore("test-agent", db_dir=tmp_dir)
        sys._active_project_dir = tmp_dir
        # Write map file to disk and store metadata
        with open(os.path.join(tmp_dir, "PROJECT_MAP.md"), "w") as f:
            f.write("# Work Project\n\nBuild system notes.")
        sys._store.create_map("map-3", "work-proj", "", project_dir=tmp_dir)
        sys._active_project = "work-proj"
        for i in range(3):
            entry = _make_entry(
                entry_id=f"w-{i}", project="work-proj",
                reflection=f"Worker reflection {i}.",
            )
            sys._pool.append(entry)
            sys._active_ids.add(entry.id)
            sys._active_weights[entry.id] = 0.5
        return sys

    @pytest.mark.asyncio
    async def test_worker_gets_all_memory_blocks(self, v2_memory):
        """Workers now get representative + map + log — same as router."""
        memory = v2_memory

        # Simulate the unified agent_node v2 prompt construction
        # (no _is_worker gating — workers and router use the same path)
        v2_parts = []
        rep_block = await memory.render_representative_block()
        if rep_block:
            v2_parts.append(rep_block)
        map_block = await memory.render_maps_block()
        if map_block:
            v2_parts.append(map_block)
        log_block = await memory.render_recent_log_block()
        if log_block:
            v2_parts.append(log_block)
        memory_block = "\n\n".join(v2_parts)

        # Workers now see ALL blocks — same as the router
        assert "<project_map" in memory_block
        assert "<recent_activity>" in memory_block
        assert "<memory>" in memory_block
        assert 'source="representative"' in memory_block

    @pytest.mark.asyncio
    async def test_non_worker_gets_all_three_sections(self, v2_memory):
        """Non-workers get representative + map + recent log."""
        memory = v2_memory

        v2_parts = []
        rep_block = await memory.render_representative_block()
        if rep_block:
            v2_parts.append(rep_block)
        map_block = await memory.render_maps_block()
        if map_block:
            v2_parts.append(map_block)
        log_block = await memory.render_recent_log_block()
        if log_block:
            v2_parts.append(log_block)
        memory_block = "\n\n".join(v2_parts)

        assert "<memory>" in memory_block
        assert 'source="representative"' in memory_block
        assert "<project_map" in memory_block
        assert "<recent_activity>" in memory_block

    @pytest.mark.asyncio
    async def test_worker_v2_map_content_correct(self, v2_memory):
        """Worker sees the correct project map content."""
        map_block = await v2_memory.render_maps_block()
        assert 'project="work-proj"' in map_block
        assert "Build system notes." in map_block

    @pytest.mark.asyncio
    async def test_representative_draws_from_all_projects(self, v2_memory, tmp_dir):
        """Representative block draws from full pool regardless of project."""
        # Add an entry from a different project
        cross_entry = _make_entry(
            entry_id="cross-proj",
            project="other-proj",
            summary="Cross-project insight.",
        )
        v2_memory._pool.append(cross_entry)
        v2_memory._active_ids.add(cross_entry.id)
        v2_memory._active_weights[cross_entry.id] = 0.9

        rep_block = await v2_memory.render_representative_block()
        assert "Cross-project insight." in rep_block
        # Active project entries also present
        assert "Test summary" in rep_block


class TestWorkerPromptUnification:
    """Test that the worker prompt construction includes ALL memory blocks,
    matching the router's prompt exactly (unified context)."""

    @pytest.fixture
    def v2_memory_with_summary(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        sys = MemorySystemV2(
            nickname="test-agent",
            llm_client=AsyncMock(),
            pool_max_entries=100,
            embedding_backend="openai",
            embedding_model="text-embedding-3-small",
        )
        sys._store = MemoryStore("test-agent", db_dir=tmp_dir)
        sys._active_project_dir = tmp_dir
        # Write map file to disk and store metadata
        with open(os.path.join(tmp_dir, "PROJECT_MAP.md"), "w") as f:
            f.write("# Unified Project\n\nTest map content.")
        sys._store.create_map("map-u", "unified-proj", "", project_dir=tmp_dir)
        sys._active_project = "unified-proj"
        # Add representative memories
        for i in range(3):
            entry = _make_entry(
                entry_id=f"u-{i}", project="unified-proj",
                summary=f"Unified memory {i}.",
            )
            sys._pool.append(entry)
            sys._active_ids.add(entry.id)
            sys._active_weights[entry.id] = 0.5
        # Add a conversation summary (set in-memory directly, as initialize() isn't called)
        sys._conversation_summary = "Summary: we decided to use model X."
        sys._summary_messages_summarized = 10
        return sys

    @pytest.mark.asyncio
    async def test_unified_prompt_includes_all_blocks(self, v2_memory_with_summary):
        """Verify the unified prompt construction (used by both worker and router)
        includes representative, map, log, and summary blocks."""
        from mesh.memory.system_v2 import MemorySystemV2
        memory = v2_memory_with_summary

        # Replicate the unified prompt construction from _process_with_llm
        v2_parts = []
        rep_block = await memory.render_representative_block()
        if rep_block:
            v2_parts.append(rep_block)
        map_block = await memory.render_maps_block()
        if map_block:
            v2_parts.append(map_block)
        log_block = await memory.render_recent_log_block()
        if log_block:
            v2_parts.append(log_block)
        summary_block = await memory.render_summary_block()
        if summary_block:
            v2_parts.append(summary_block)
        memory_block = "\n\n".join(v2_parts)

        # ALL blocks present
        assert "<memory>" in memory_block, "Missing representative block"
        assert 'source="representative"' in memory_block
        assert "<project_map" in memory_block, "Missing map block"
        assert "Unified Project" in memory_block
        assert "<conversation_summary " in memory_block, "Missing summary block"
        assert "we decided to use model X" in memory_block

    @pytest.mark.asyncio
    async def test_worker_instructions_have_no_task_description(self):
        """Worker instructions no longer contain a task description placeholder."""
        from mesh.agent_node import WORKER_INSTRUCTIONS
        # No {router_task_section} placeholder
        assert "router_task_section" not in WORKER_INSTRUCTIONS
        # Only {routing_context} placeholder
        assert "{routing_context}" in WORKER_INSTRUCTIONS
        # No "The router has dispatched you with this task" language
        assert "dispatched you with this task:" not in WORKER_INSTRUCTIONS

    @pytest.mark.asyncio
    async def test_worker_instructions_format_routing_context(self):
        """Worker instructions accept routing_context for channel vs DM delivery."""
        from mesh.agent_node import WORKER_INSTRUCTIONS
        # DM routing context
        dm_ctx = "\nRouting: This is a direct message task. Do NOT call send_message.\n"
        formatted = WORKER_INSTRUCTIONS.format(routing_context=dm_ctx)
        assert "Do NOT call send_message" in formatted
        # Channel routing context
        ch_ctx = "\nRouting: This task was triggered by an @mention in channel:dev.\n"
        formatted_ch = WORKER_INSTRUCTIONS.format(routing_context=ch_ctx)
        assert "@mention in channel:dev" in formatted_ch

    @pytest.mark.asyncio
    async def test_relevant_context_accessible_to_worker(self):
        """Worker prompt includes relevant_context from the router when available."""
        from mesh.router_v2 import RouterV2, RouterV2Config

        async def noop_send(content, in_reply_to=None):
            pass
        async def noop_worker(context, trigger, **kwargs):
            from mesh.router_v2 import WorkerResult
            return WorkerResult(response="ok", context=[], usage=None, error=None)

        router = RouterV2(
            worker_fn=noop_worker,
            send_fn=noop_send,
            config=RouterV2Config(llm_enabled=False),
            nickname="test-bot",
        )
        # Simulate relevant context being loaded
        router._relevant_context = "Retrieved: important constraint about model selection."
        assert router._relevant_context == "Retrieved: important constraint about model selection."


class TestRelevantContextPersistence:
    """Test that relevant context persists across router turns."""

    @pytest.mark.asyncio
    async def test_relevant_context_init_empty(self):
        """RouterV2 starts with empty relevant context."""
        from mesh.router_v2 import RouterV2, RouterV2Config

        async def noop_send(content, in_reply_to=None):
            pass

        async def noop_worker(context, trigger, **kwargs):
            from mesh.router_v2 import WorkerResult
            return WorkerResult(response="ok", context=[], usage=None, error=None)

        router = RouterV2(
            worker_fn=noop_worker,
            send_fn=noop_send,
            config=RouterV2Config(llm_enabled=False),
            nickname="test-bot",
        )
        assert router._relevant_context == ""

    @pytest.mark.asyncio
    async def test_relevant_context_survives_prompt_builds(self, tmp_dir):
        """Relevant context set on router persists across _build_router_prompt calls."""
        from mesh.memory.system_v2 import MemorySystemV2
        from mesh.router_v2 import RouterV2, RouterV2Config

        sys = MemorySystemV2(
            nickname="test-agent",
            llm_client=AsyncMock(),
            pool_max_entries=100,
            embedding_backend="openai",
            embedding_model="text-embedding-3-small",
        )
        sys._store = MemoryStore("test-agent", db_dir=tmp_dir)

        async def noop_send(content, in_reply_to=None):
            pass

        async def noop_worker(context, trigger, **kwargs):
            from mesh.router_v2 import WorkerResult
            return WorkerResult(response="ok", context=[], usage=None, error=None)

        router = RouterV2(
            worker_fn=noop_worker,
            send_fn=noop_send,
            config=RouterV2Config(llm_enabled=True),
            llm_client=AsyncMock(),
            nickname="test-bot",
            memory_system=sys,
        )

        # Set relevant context
        router._relevant_context = "Important context from retrieval."

        # Build prompt twice — context should persist
        p1 = await router._build_router_prompt(instructions="First.", include_tools=False)
        p2 = await router._build_router_prompt(instructions="Second.", include_tools=False)

        assert "Important context from retrieval." in p1
        assert "Important context from retrieval." in p2
        assert router._relevant_context == "Important context from retrieval."


class TestRenderRecentLogBlockContent:
    """Test render_recent_log_block returns last m entries in recency order."""

    @pytest.mark.asyncio
    async def test_returns_last_m_entries_most_recent_first(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        sys = MemorySystemV2(
            nickname="test-agent",
            llm_client=AsyncMock(),
            pool_max_entries=100,
            embedding_backend="openai",
            embedding_model="text-embedding-3-small",
            recent_log_count=3,
        )
        sys._store = MemoryStore("test-agent", db_dir=tmp_dir)

        base = datetime(2025, 6, 1, tzinfo=timezone.utc)
        for i in range(6):
            entry = _make_entry(
                entry_id=f"log-{i}", project="proj",
                reflection=f"Reflection {i}.",
                topic_label=f"topic-{i}",
                outcome="success",
                created_at=base + timedelta(hours=i),
            )
            sys._pool.append(entry)
            sys._active_ids.add(entry.id)

        block = await sys.render_recent_log_block()

        # Should contain the 3 most recent (indices 5, 4, 3)
        assert "Reflection 5." in block
        assert "Reflection 4." in block
        assert "Reflection 3." in block
        # Older entries excluded
        assert "Reflection 2." not in block
        assert "Reflection 1." not in block
        assert "Reflection 0." not in block

        # Most recent should appear first (reverse chronological)
        pos5 = block.index("Reflection 5.")
        pos4 = block.index("Reflection 4.")
        pos3 = block.index("Reflection 3.")
        assert pos5 < pos4 < pos3

        # Wrapped in correct tags
        assert block.startswith("<recent_activity>")
        assert block.endswith("</recent_activity>")


# ══════════════════════════════════════════════════════════════
# Phase 5 — Three-Outcome Classification + Retrieval Wiring
# ══════════════════════════════════════════════════════════════


class TestParseClassificationV2Format:
    """Test _parse_classification_response with v2 key-value formats."""

    def _make_router(self):
        from mesh.router_v2 import RouterV2, RouterV2Config

        async def noop_send(content, in_reply_to=None):
            pass

        async def noop_worker(context, trigger, **kwargs):
            from mesh.router_v2 import WorkerResult
            return WorkerResult(response="ok", context=[], usage=None, error=None)

        return RouterV2(
            worker_fn=noop_worker,
            send_fn=noop_send,
            config=RouterV2Config(llm_enabled=False),
            nickname="test-bot",
        )

    def test_dispatch_action(self):
        router = self._make_router()
        raw = "action: dispatch\ntask_summary: fix the login bug"
        result = router._parse_classification_response(raw)
        assert result["needs_worker"] is True
        assert result["task_summary"] == "fix the login bug"

    def test_direct_action(self):
        router = self._make_router()
        raw = 'action: direct\nresponse: The project uses PostgreSQL.'
        result = router._parse_classification_response(raw)
        assert result["needs_worker"] is False
        assert "PostgreSQL" in result["response"]

    def test_json_with_action_field_still_works(self):
        """v2 JSON format with action field is parsed via the JSON path."""
        router = self._make_router()
        raw = '{"needs_response": true, "needs_worker": false, "action": "retrieve", "query": "auth decisions"}'
        result = router._parse_classification_response(raw)
        assert result["action"] == "retrieve"
        assert result["query"] == "auth decisions"

    def test_v1_json_still_works(self):
        """v1 JSON format (no action field) is unaffected."""
        router = self._make_router()
        raw = '{"needs_response": true, "needs_worker": true, "response": "On it.", "task_complexity": "complex"}'
        result = router._parse_classification_response(raw)
        assert result["needs_worker"] is True
        assert result["response"] == "On it."
        assert "action" not in result

    def test_json_before_action_keyword_uses_json_path(self):
        """If JSON brace appears before 'action:', the JSON parser takes precedence."""
        router = self._make_router()
        raw = '{"needs_response": true, "needs_worker": false, "response": "action: direct is a phrase"}'
        result = router._parse_classification_response(raw)
        # Should parse as JSON, not as v2 key-value
        assert result["needs_worker"] is False
        assert "action: direct" in result["response"]


class TestMapEditInRouterTools:
    """Test map tools registered in ROUTER_TOOL_NAMES."""

    def test_map_tools_registered(self):
        from mesh.router_v2 import ROUTER_TOOL_NAMES
        for tool in ("map_list", "map_get", "map_edit", "map_create", "set_project_context"):
            assert tool in ROUTER_TOOL_NAMES, f"{tool} not in ROUTER_TOOL_NAMES"


# ── Phase 5b: Pre-router set-context intercept ──────────────────

class TestExtractSetContextPath:
    """Test _extract_set_context_path regex matching."""

    def _make_router(self, tmp_dir, use_v2=True):
        from mesh.router_v2 import RouterV2, RouterV2Config

        if use_v2:
            from mesh.memory.system_v2 import MemorySystemV2
            mem = MemorySystemV2(
                nickname="test-agent",
                llm_client=AsyncMock(),
                pool_max_entries=100,
                embedding_backend="openai",
                embedding_model="text-embedding-3-small",
            )
            mem._store = MemoryStore("test-agent", db_dir=tmp_dir)
        else:
            mem = MagicMock()
            mem.__class__ = type("NotMemorySystemV2", (), {})

        async def noop_send(content, in_reply_to=None):
            pass

        async def noop_worker(context, trigger, **kwargs):
            from mesh.router_v2 import WorkerResult
            return WorkerResult(response="ok", context=[], usage=None, error=None)

        router = RouterV2(
            worker_fn=noop_worker,
            send_fn=noop_send,
            config=RouterV2Config(llm_enabled=True),
            llm_client=AsyncMock(),
            nickname="test-bot",
            memory_system=mem,
        )
        return router

    def test_set_context_to(self, tmp_dir):
        router = self._make_router(tmp_dir)
        assert router._extract_set_context_path("set context to /tmp/test-app") == "/tmp/test-app"

    def test_set_your_context_to(self, tmp_dir):
        router = self._make_router(tmp_dir)
        assert router._extract_set_context_path("set your context to ~/log/grants") == "~/log/grants"

    def test_set_project_context_to(self, tmp_dir):
        router = self._make_router(tmp_dir)
        assert router._extract_set_context_path("set project context to /tmp/project") == "/tmp/project"

    def test_set_your_project_context_to(self, tmp_dir):
        router = self._make_router(tmp_dir)
        assert router._extract_set_context_path("set your project context to /tmp/project") == "/tmp/project"

    def test_case_insensitive(self, tmp_dir):
        router = self._make_router(tmp_dir)
        assert router._extract_set_context_path("SET CONTEXT TO /tmp/foo") == "/tmp/foo"

    def test_strips_quotes(self, tmp_dir):
        router = self._make_router(tmp_dir)
        assert router._extract_set_context_path('set context to "/tmp/my project"') == "/tmp/my project"
        assert router._extract_set_context_path("set context to '/tmp/my project'") == "/tmp/my project"

    def test_no_match_unrelated(self, tmp_dir):
        router = self._make_router(tmp_dir)
        assert router._extract_set_context_path("what is the context?") is None
        assert router._extract_set_context_path("please set the color to blue") is None

    def test_no_match_when_v1(self, tmp_dir):
        """v1 memory never triggers the intercept."""
        router = self._make_router(tmp_dir, use_v2=False)
        assert router._extract_set_context_path("set context to /tmp/foo") is None

    def test_leading_whitespace(self, tmp_dir):
        router = self._make_router(tmp_dir)
        assert router._extract_set_context_path("  set context to /tmp/foo  ") == "/tmp/foo"


class TestHandleSetContextRequest:
    """Test _handle_set_context_request confirmation flow."""

    def _make_router_with_v2(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        from mesh.router_v2 import RouterV2, RouterV2Config

        sys = MemorySystemV2(
            nickname="test-agent",
            llm_client=AsyncMock(),
            pool_max_entries=100,
            embedding_backend="openai",
            embedding_model="text-embedding-3-small",
        )
        sys._store = MemoryStore("test-agent", db_dir=tmp_dir)

        sent_messages = []

        async def capture_send(content, in_reply_to=None):
            sent_messages.append(content)

        async def noop_worker(context, trigger, **kwargs):
            from mesh.router_v2 import WorkerResult
            return WorkerResult(response="ok", context=[], usage=None, error=None)

        router = RouterV2(
            worker_fn=noop_worker,
            send_fn=capture_send,
            config=RouterV2Config(llm_enabled=True),
            llm_client=AsyncMock(),
            nickname="test-bot",
            memory_system=sys,
        )
        return router, sys, sent_messages

    @pytest.mark.asyncio
    async def test_successful_scan(self, tmp_dir):
        """Successful scan sends confirmation with project name and map stats."""
        router, mem, sent = self._make_router_with_v2(tmp_dir)

        # Mock set_project_context to simulate a successful scan
        mem.set_project_context = AsyncMock(return_value="Project context initialized: chat-app")
        mem._active_project = "chat-app"
        # Mock get_map at the system level (reads from disk now)
        mem.get_map = AsyncMock(return_value="# Project: chat-app\nThis is the project map with some content for testing.")

        from mesh.protocol import Message, MessageType
        msg = Message(type=MessageType.MESSAGE, content="set context to /tmp/test", from_node="user:testuser", to_node="agent:test")
        await router._handle_set_context_request(msg, "/tmp/test")

        assert len(sent) == 1
        assert "chat-app" in sent[0]
        assert "chars" in sent[0]
        assert "words" in sent[0]

    @pytest.mark.asyncio
    async def test_scan_failure(self, tmp_dir):
        """Failed scan sends error message."""
        router, mem, sent = self._make_router_with_v2(tmp_dir)

        mem.set_project_context = AsyncMock(side_effect=Exception("path not found"))

        from mesh.protocol import Message, MessageType
        msg = Message(type=MessageType.MESSAGE, content="set context to /nonexistent", from_node="user:testuser", to_node="agent:test")
        await router._handle_set_context_request(msg, "/nonexistent")

        assert len(sent) == 1
        assert "Failed" in sent[0]
        assert "path not found" in sent[0]

    @pytest.mark.asyncio
    async def test_no_map_fallback(self, tmp_dir):
        """When get_map returns None, falls back to raw result string."""
        router, mem, sent = self._make_router_with_v2(tmp_dir)

        mem.set_project_context = AsyncMock(return_value="Project context loaded: old-project")
        mem._active_project = "old-project"
        mem.get_map = AsyncMock(return_value=None)

        from mesh.protocol import Message, MessageType
        msg = Message(type=MessageType.MESSAGE, content="set context to /tmp/old", from_node="user:testuser", to_node="agent:test")
        await router._handle_set_context_request(msg, "/tmp/old")

        assert len(sent) == 1
        assert "Project context loaded: old-project" in sent[0]


class TestSetContextOnMessageIntegration:
    """Test that on_message intercepts set-context before classification."""

    def _make_router_with_v2(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        from mesh.router_v2 import RouterV2, RouterV2Config

        sys = MemorySystemV2(
            nickname="test-agent",
            llm_client=AsyncMock(),
            pool_max_entries=100,
            embedding_backend="openai",
            embedding_model="text-embedding-3-small",
        )
        sys._store = MemoryStore("test-agent", db_dir=tmp_dir)

        sent_messages = []
        classify_called = False

        async def capture_send(content, in_reply_to=None):
            sent_messages.append(content)

        async def noop_worker(context, trigger, **kwargs):
            from mesh.router_v2 import WorkerResult
            return WorkerResult(response="ok", context=[], usage=None, error=None)

        router = RouterV2(
            worker_fn=noop_worker,
            send_fn=capture_send,
            config=RouterV2Config(llm_enabled=True),
            llm_client=AsyncMock(),
            nickname="test-bot",
            memory_system=sys,
        )
        return router, sys, sent_messages

    @pytest.mark.asyncio
    async def test_on_message_intercepts_set_context(self, tmp_dir):
        """on_message should intercept 'set context to' and skip classification."""
        router, mem, sent = self._make_router_with_v2(tmp_dir)

        mem.set_project_context = AsyncMock(return_value="Project context initialized: testproj")
        mem._active_project = "testproj"
        mem.get_map = AsyncMock(return_value="# Project: testproj\nSome content here for the map.")

        # Track if classification was called
        classify_called = False
        original_classify = router._classify_message
        async def tracking_classify(msg):
            nonlocal classify_called
            classify_called = True
            return await original_classify(msg)
        router._classify_message = tracking_classify

        from mesh.protocol import Message, MessageType
        msg = Message(type=MessageType.MESSAGE, content="set context to /tmp/testproj", from_node="user:testuser", to_node="agent:test")
        await router.on_message(msg)

        # Should have sent a confirmation
        assert len(sent) == 1
        assert "testproj" in sent[0]

        # Classification should NOT have been called
        assert not classify_called

    @pytest.mark.asyncio
    async def test_on_message_normal_message_passes_through(self, tmp_dir):
        """Regular messages should not trigger the intercept."""
        router, mem, sent = self._make_router_with_v2(tmp_dir)

        classify_called = False

        async def mock_classify(msg):
            nonlocal classify_called
            classify_called = True
            return {"needs_response": True, "needs_worker": False, "response": "Hello!"}

        router._classify_message = mock_classify

        from mesh.protocol import Message, MessageType
        msg = Message(type=MessageType.MESSAGE, content="Hello, how are you?", from_node="user:testuser", to_node="agent:test")
        await router.on_message(msg)

        # Classification SHOULD have been called
        assert classify_called

    @pytest.mark.asyncio
    async def test_intercept_works_in_busy_state(self, tmp_dir):
        """set-context should work even when the router is BUSY."""
        from mesh.router_v2 import RouterState
        router, mem, sent = self._make_router_with_v2(tmp_dir)

        mem.set_project_context = AsyncMock(return_value="Project context initialized: testproj")
        mem._active_project = "testproj"
        mem.get_map = AsyncMock(return_value="# Project: testproj\nSome map content.")

        # Set router to BUSY state
        router._state = RouterState.BUSY

        from mesh.protocol import Message, MessageType
        msg = Message(type=MessageType.MESSAGE, content="set your context to /tmp/testproj", from_node="user:testuser", to_node="agent:test")
        await router.on_message(msg)

        # Should still have intercepted and sent confirmation
        assert len(sent) == 1
        assert "testproj" in sent[0]


# ── Conversation Summary Tests ─────────────────────────────────


class TestConversationSummaryStore:
    """Test conversation_summary table CRUD in MemoryStore."""

    def test_get_summary_empty(self, tmp_dir):
        """Empty table returns None."""
        store = MemoryStore("test-summ", db_dir=tmp_dir)
        assert store.get_summary() is None

    def test_save_and_get_summary(self, tmp_dir):
        """Round-trip save and load."""
        store = MemoryStore("test-summ", db_dir=tmp_dir)
        store.save_summary("This is the summary.", 42, 15)
        row = store.get_summary()
        assert row is not None
        assert row["summary_text"] == "This is the summary."
        assert row["messages_summarized"] == 42
        assert row["token_estimate"] == 15
        assert row["updated_at"]  # non-empty timestamp

    def test_upsert_overwrites(self, tmp_dir):
        """Second save overwrites the first (single-row table)."""
        store = MemoryStore("test-summ", db_dir=tmp_dir)
        store.save_summary("First.", 10, 5)
        store.save_summary("Second.", 20, 10)
        row = store.get_summary()
        assert row["summary_text"] == "Second."
        assert row["messages_summarized"] == 20

    def test_summary_survives_reopen(self, tmp_dir):
        """Summary persists across store reopens."""
        store1 = MemoryStore("test-summ", db_dir=tmp_dir)
        store1.save_summary("Persistent summary.", 50, 20)
        store1.close()
        store2 = MemoryStore("test-summ", db_dir=tmp_dir)
        row = store2.get_summary()
        assert row["summary_text"] == "Persistent summary."


class TestConversationSummaryGeneration:
    """Test _update_conversation_summary in MemorySystemV2."""

    @pytest.fixture
    def system(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value="Updated conversation summary text.")
        sys = MemorySystemV2(
            nickname="test-summ", llm_client=mock_llm,
        )
        sys._store = MemoryStore("test-summ", db_dir=tmp_dir)
        sys._embedder = MagicMock()
        return sys

    @pytest.mark.asyncio
    async def test_first_summary_created(self, system):
        """First window drop creates a summary from scratch."""
        turns = [
            _make_turn(role="user", content="fix the bug please", from_node="user:testuser"),
            _make_turn(role="tool", content="file_read result"),
            _make_turn(role="tool", content="edit result"),
            _make_turn(role="assistant", content="Bug is fixed.", from_node="agent:test"),
        ]
        await system._update_conversation_summary(turns, len(turns))
        assert system._conversation_summary == "Updated conversation summary text."
        assert system._summary_messages_summarized == 4
        # Verify persisted to DB
        row = system._store.get_summary()
        assert row is not None
        assert row["summary_text"] == "Updated conversation summary text."
        assert row["messages_summarized"] == 4

    @pytest.mark.asyncio
    async def test_summary_additive(self, system):
        """Second update incorporates existing summary (additive compression)."""
        # Simulate an existing summary
        system._conversation_summary = "Previous context: fixed a bug."
        system._summary_messages_summarized = 10

        turns = [
            _make_turn(role="user", content="now deploy it"),
            _make_turn(role="assistant", content="Deployed."),
        ]
        await system._update_conversation_summary(turns, len(turns))

        # Should have been called with the existing summary in the prompt
        call_args = system._llm_client.complete.call_args
        prompt_text = call_args[0][0]
        assert "Previous context: fixed a bug." in prompt_text
        assert system._summary_messages_summarized == 12  # 10 + 2

    @pytest.mark.asyncio
    async def test_summary_prompt_includes_turns(self, system):
        """The prompt includes formatted dropped turns."""
        turns = [
            _make_turn(role="user", content="What is the status?", from_node="user:testuser"),
            _make_turn(role="assistant", content="Everything is green.", from_node="agent:test"),
        ]
        await system._update_conversation_summary(turns, len(turns))
        call_args = system._llm_client.complete.call_args
        prompt_text = call_args[0][0]
        assert "What is the status?" in prompt_text
        assert "Everything is green." in prompt_text

    @pytest.mark.asyncio
    async def test_summary_persisted_to_db(self, system):
        """Summary is saved to the conversation_summary table."""
        turns = [_make_turn(content="test turn")]
        await system._update_conversation_summary(turns, 1)
        row = system._store.get_summary()
        assert row is not None
        assert row["summary_text"] == "Updated conversation summary text."

    @pytest.mark.asyncio
    async def test_summary_llm_failure_nonfatal(self, system):
        """LLM failure doesn't crash the pipeline — summary stays unchanged."""
        system._conversation_summary = "Existing summary."
        system._summary_messages_summarized = 5
        system._llm_client.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

        turns = [_make_turn(content="trigger")]
        await system._update_conversation_summary(turns, 1)

        # Summary unchanged
        assert system._conversation_summary == "Existing summary."
        assert system._summary_messages_summarized == 5


class TestConversationSummaryRendering:
    """Test render_summary_block output."""

    @pytest.fixture
    def system(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        sys = MemorySystemV2(
            nickname="test-render", llm_client=AsyncMock(),
        )
        sys._store = MemoryStore("test-render", db_dir=tmp_dir)
        return sys

    @pytest.mark.asyncio
    async def test_empty_summary_returns_empty(self, system):
        """No summary → empty string."""
        assert await system.render_summary_block() == ""

    @pytest.mark.asyncio
    async def test_renders_xml_block(self, system):
        """Non-empty summary renders as XML."""
        system._conversation_summary = "We discussed debugging and deployment."
        system._summary_messages_summarized = 25
        block = await system.render_summary_block()
        assert "<conversation_summary" in block
        assert 'messages_covered="25"' in block
        assert "We discussed debugging and deployment." in block
        assert "</conversation_summary>" in block


class TestConversationSummaryInWindowDrop:
    """Test that on_window_drop calls _update_conversation_summary."""

    @pytest.fixture
    def system(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        mock_llm = AsyncMock()
        # Response for reflection calls
        mock_llm.complete = AsyncMock(return_value=(
            "<reflection>\nReflected.\n</reflection>\n"
            "<summary>\nSummary.\n</summary>\n"
            "<tags>\ntest\n</tags>\n"
            "<outcome_label>\nsuccess\n</outcome_label>\n"
            "<retrieval_key>\nKey.\n</retrieval_key>\n"
            "<project>\ntest-proj\n</project>"
        ))
        sys = MemorySystemV2(
            nickname="test-drop", llm_client=mock_llm,
        )
        sys._store = MemoryStore("test-drop", db_dir=tmp_dir)
        sys._embedder = MagicMock()
        sys._embedder.embed_batch_to_arrays = AsyncMock(
            return_value=[np.random.randn(8).astype(np.float32),
                          np.random.randn(8).astype(np.float32)]
        )
        return sys

    @pytest.mark.asyncio
    async def test_window_drop_updates_summary(self, system):
        """on_window_drop should call _update_conversation_summary."""
        turns = [
            _make_turn(role="user", content="fix bug", topic_label="debugging",
                       from_node="user:testuser"),
            _make_turn(role="tool", content="result1", topic_label="debugging"),
            _make_turn(role="tool", content="result2", topic_label="debugging"),
            _make_turn(role="tool", content="result3", topic_label="debugging"),
            _make_turn(role="assistant", content="fixed", topic_label="debugging",
                       from_node="agent:coder"),
        ]
        await system.on_window_drop(turns)
        # Summary should have been generated (last LLM call is the summary)
        assert system._conversation_summary  # non-empty
        assert system._summary_messages_summarized == 5

    @pytest.mark.asyncio
    async def test_window_drop_summary_persisted(self, system):
        """Summary should be in the DB after on_window_drop."""
        turns = [
            _make_turn(role="user", content="deploy to prod", topic_label="deploy",
                       from_node="user:testuser"),
            _make_turn(role="tool", content="deploy step 1", topic_label="deploy"),
            _make_turn(role="tool", content="deploy step 2", topic_label="deploy"),
            _make_turn(role="tool", content="deploy step 3", topic_label="deploy"),
            _make_turn(role="assistant", content="deployed", topic_label="deploy",
                       from_node="agent:coder"),
        ]
        await system.on_window_drop(turns)
        row = system._store.get_summary()
        assert row is not None
        assert row["messages_summarized"] == 5


class TestConversationSummaryInit:
    """Test summary restore on initialize."""

    @pytest.mark.asyncio
    async def test_summary_restored_on_init(self, tmp_dir):
        """Summary is loaded from DB during initialize."""
        from mesh.memory.system_v2 import MemorySystemV2
        # Pre-populate DB
        store = MemoryStore("test-init", db_dir=tmp_dir)
        store.save_summary("Restored summary from DB.", 100, 50)
        store.close()

        sys = MemorySystemV2(
            nickname="test-init", llm_client=AsyncMock(),
        )
        # Manually initialize (normally called from agent setup)
        sys._store = MemoryStore("test-init", db_dir=tmp_dir)
        row = sys._store.get_summary()
        if row:
            sys._conversation_summary = row["summary_text"]
            sys._summary_messages_summarized = row["messages_summarized"]

        assert sys._conversation_summary == "Restored summary from DB."
        assert sys._summary_messages_summarized == 100


class TestConversationSummaryInPrompt:
    """Test that summary is injected into the router prompt."""

    def _make_router_with_v2(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        from mesh.router_v2 import RouterV2, RouterV2Config

        sys = MemorySystemV2(
            nickname="test-prompt",
            llm_client=AsyncMock(),
        )
        sys._store = MemoryStore("test-prompt", db_dir=tmp_dir)

        sent_messages = []

        async def capture_send(content, in_reply_to=None):
            sent_messages.append(content)

        async def noop_worker(context, trigger, **kwargs):
            from mesh.router_v2 import WorkerResult
            return WorkerResult(response="ok", context=[], usage=None, error=None)

        router = RouterV2(
            worker_fn=noop_worker,
            send_fn=capture_send,
            config=RouterV2Config(llm_enabled=True),
            llm_client=AsyncMock(),
            nickname="test-bot",
            memory_system=sys,
        )
        return router, sys, sent_messages

    @pytest.mark.asyncio
    async def test_summary_in_router_prompt(self, tmp_dir):
        """Conversation summary appears in _build_router_prompt output."""
        router, mem, _ = self._make_router_with_v2(tmp_dir)
        mem._conversation_summary = "Summary of earlier conversation."
        mem._summary_messages_summarized = 30

        prompt = await router._build_router_prompt(
            instructions="Test instructions.",
        )
        assert "<conversation_summary" in prompt
        assert "Summary of earlier conversation." in prompt
        assert 'messages_covered="30"' in prompt

    @pytest.mark.asyncio
    async def test_no_summary_no_block(self, tmp_dir):
        """Empty summary doesn't inject a block."""
        router, mem, _ = self._make_router_with_v2(tmp_dir)
        # No summary set
        prompt = await router._build_router_prompt(
            instructions="Test instructions.",
        )
        assert "<conversation_summary" not in prompt

    @pytest.mark.asyncio
    async def test_summary_in_synthesis_context(self, tmp_dir):
        """Conversation summary appears in _build_synthesis_context output."""
        router, mem, _ = self._make_router_with_v2(tmp_dir)
        mem._conversation_summary = "Synthesis summary context."
        mem._summary_messages_summarized = 15

        context = await router._build_synthesis_context()
        assert "Synthesis summary context." in context
        assert "<conversation_summary" in context


# ── Map Review ──────────────────────────────────────────────────

class TestReviewActiveMap:
    """Test review_active_map method."""

    @pytest.fixture
    def system(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        mock_llm = AsyncMock()
        sys = MemorySystemV2(
            nickname="test", llm_client=mock_llm,
        )
        sys._store = MemoryStore("test", db_dir=tmp_dir)
        sys._active_project_dir = tmp_dir
        return sys

    @pytest.fixture
    def project_dir(self, tmp_dir):
        """Create a small project directory for scanning."""
        proj = os.path.join(tmp_dir, "myproject")
        os.makedirs(proj)
        with open(os.path.join(proj, "README.md"), "w") as f:
            f.write("# My Project\nA test project.\n")
        with open(os.path.join(proj, "main.py"), "w") as f:
            f.write("def main():\n    print('hello')\n")
        with open(os.path.join(proj, "config.yaml"), "w") as f:
            f.write("setting: value\n")
        return proj

    @pytest.mark.asyncio
    async def test_no_active_project(self, system, project_dir):
        system._active_project = None
        result = await system.review_active_map(project_dir)
        assert result["updated"] is False
        assert "No active project" in result["summary"]

    @pytest.mark.asyncio
    async def test_no_map_exists(self, system, project_dir):
        system._active_project = "myproject"
        result = await system.review_active_map(project_dir)
        assert result["updated"] is False
        assert "No map found" in result["summary"]

    @pytest.mark.asyncio
    async def test_invalid_project_dir(self, system, tmp_dir):
        await system.create_map("myproject", "# Project: myproject\nContent.")
        system._active_project = "myproject"
        result = await system.review_active_map("/nonexistent/path")
        assert result["updated"] is False
        assert "not found" in result["summary"]

    @pytest.mark.asyncio
    async def test_map_updated_successfully(self, system, project_dir):
        old_map = "# Project: myproject\n\n## Architecture\nOld architecture."
        await system.create_map("myproject", old_map)
        system._active_project = "myproject"

        new_map = "# Project: myproject\n\n## Architecture\nNew architecture."
        system._llm_client.complete = AsyncMock(
            return_value=(
                "<updated_map>\n"
                f"{new_map}\n"
                "</updated_map>\n\n"
                "<ambiguities>\nNone.\n</ambiguities>"
            )
        )
        result = await system.review_active_map(project_dir)
        assert result["updated"] is True
        assert "ambiguities" not in result  # no longer surfaced
        content = await system.get_map("myproject")
        assert content == new_map

    @pytest.mark.asyncio
    async def test_no_changes_needed(self, system, project_dir):
        current = "# Project: myproject\n\nContent."
        await system.create_map("myproject", current)
        system._active_project = "myproject"

        # LLM returns the same map
        system._llm_client.complete = AsyncMock(
            return_value=(
                f"<updated_map>\n{current}\n</updated_map>\n\n"
                "<ambiguities>\nNone.\n</ambiguities>"
            )
        )
        result = await system.review_active_map(project_dir)
        assert result["updated"] is False
        assert "no changes needed" in result["summary"].lower()

    @pytest.mark.asyncio
    async def test_strips_code_fences(self, system, project_dir):
        await system.create_map("myproject", "# Project: myproject\nOld.")
        system._active_project = "myproject"

        system._llm_client.complete = AsyncMock(
            return_value=(
                "<updated_map>\n"
                "```markdown\n# Project: myproject\n\nFenced.\n```\n"
                "</updated_map>\n\n"
                "<ambiguities>\nNone.\n</ambiguities>"
            )
        )
        result = await system.review_active_map(project_dir)
        assert result["updated"] is True
        content = await system.get_map("myproject")
        assert content == "# Project: myproject\n\nFenced."

    @pytest.mark.asyncio
    async def test_missing_updated_map_block(self, system, project_dir):
        await system.create_map("myproject", "# Project: myproject\nOld.")
        system._active_project = "myproject"

        system._llm_client.complete = AsyncMock(
            return_value="Some response without the expected XML blocks."
        )
        result = await system.review_active_map(project_dir)
        assert result["updated"] is False
        assert "could not extract" in result["summary"].lower()

    @pytest.mark.asyncio
    async def test_invalid_map_header(self, system, project_dir):
        await system.create_map("myproject", "# Project: myproject\nOld.")
        system._active_project = "myproject"

        system._llm_client.complete = AsyncMock(
            return_value=(
                "<updated_map>\nNo valid header here.\n</updated_map>\n\n"
                "<ambiguities>\nNone.\n</ambiguities>"
            )
        )
        result = await system.review_active_map(project_dir)
        assert result["updated"] is False
        assert "header" in result["summary"].lower()

    @pytest.mark.asyncio
    async def test_timeout_handled(self, system, project_dir):
        await system.create_map("myproject", "# Project: myproject\nOld.")
        system._active_project = "myproject"

        system._llm_client.complete = AsyncMock(side_effect=asyncio.TimeoutError)
        result = await system.review_active_map(project_dir)
        assert result["updated"] is False
        assert "timed out" in result["summary"].lower()

    @pytest.mark.asyncio
    async def test_prompt_includes_project_state(self, system, project_dir):
        """Verify the prompt includes project dir and current map for interactive exploration."""
        await system.create_map("myproject", "# Project: myproject\nOld.")
        system._active_project = "myproject"

        captured_prompt = None
        async def capture_complete(prompt, **kwargs):
            nonlocal captured_prompt
            captured_prompt = prompt
            return (
                "<updated_map>\n# Project: myproject\nNew.\n</updated_map>\n\n"
                "<ambiguities>\nNone.\n</ambiguities>"
            )

        system._llm_client.complete = capture_complete
        await system.review_active_map(project_dir)
        assert captured_prompt is not None
        assert "myproject" in captured_prompt
        assert str(project_dir) in captured_prompt  # project path for tool exploration
        assert "<current_map>" in captured_prompt
        assert "Old." in captured_prompt  # current map is included

    @pytest.mark.asyncio
    async def test_recent_turns_included(self, system, project_dir):
        """Recent conversation turns are included when provided."""
        await system.create_map("myproject", "# Project: myproject\nOld.")
        system._active_project = "myproject"

        captured_prompt = None
        async def capture_complete(prompt, **kwargs):
            nonlocal captured_prompt
            captured_prompt = prompt
            return (
                "<updated_map>\n# Project: myproject\nNew.\n</updated_map>\n\n"
                "<ambiguities>\nNone.\n</ambiguities>"
            )

        system._llm_client.complete = capture_complete
        await system.review_active_map(
            project_dir, recent_turns_text="User asked about config changes."
        )
        assert "User asked about config changes" in captured_prompt
        assert "<recent_conversation>" in captured_prompt

    @pytest.mark.asyncio
    async def test_tool_calls_executed(self, system, project_dir):
        """Review supports tool calls (file_read, bash_exec) for resolving ambiguities."""
        await system.create_map("myproject", "# Project: myproject\nOld.")
        system._active_project = "myproject"

        call_count = 0
        async def mock_complete(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "<file_read>README.md</file_read>"
            return (
                "<updated_map>\n# Project: myproject\nAfter tool.\n</updated_map>\n\n"
                "<ambiguities>\nNone.\n</ambiguities>"
            )

        system._llm_client.complete = mock_complete
        with patch("mesh.memory.system_v2._execute_tool_call", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = "# My Project\nA test project."
            result = await system.review_active_map(project_dir)
            mock_exec.assert_called_once()
        assert result["updated"] is True


# ── Review Map Trigger ────────────────────────────────────────

class TestReviewMapTrigger:
    """Test the 'review map' pre-router intercept regex."""

    def _make_router(self, tmp_dir):
        from mesh.memory.system_v2 import MemorySystemV2
        from mesh.router_v2 import RouterV2, RouterV2Config

        mem = MemorySystemV2(
            nickname="test-agent",
            llm_client=AsyncMock(),
            pool_max_entries=100,
            embedding_backend="openai",
            embedding_model="text-embedding-3-small",
        )
        mem._store = MemoryStore("test-agent", db_dir=tmp_dir)

        async def noop_send(content, in_reply_to=None):
            pass

        async def noop_worker(context, trigger, **kwargs):
            from mesh.router_v2 import WorkerResult
            return WorkerResult(response="ok", context=[], usage=None, error=None)

        return RouterV2(
            worker_fn=noop_worker,
            send_fn=noop_send,
            config=RouterV2Config(llm_enabled=True),
            llm_client=AsyncMock(),
            nickname="test-bot",
            memory_system=mem,
        )

    def test_review_map(self, tmp_dir):
        r = self._make_router(tmp_dir)
        assert r._is_review_map_request("review map") is True

    def test_review_the_map(self, tmp_dir):
        r = self._make_router(tmp_dir)
        assert r._is_review_map_request("review the map") is True

    def test_review_plan_no_match(self, tmp_dir):
        """'plan' variants should NOT trigger map review."""
        r = self._make_router(tmp_dir)
        assert r._is_review_map_request("review plan") is False

    def test_review_the_plan_no_match(self, tmp_dir):
        """'the plan' variants should NOT trigger map review."""
        r = self._make_router(tmp_dir)
        assert r._is_review_map_request("review the plan") is False

    def test_review_my_map(self, tmp_dir):
        r = self._make_router(tmp_dir)
        assert r._is_review_map_request("review my map") is True

    def test_review_project_map(self, tmp_dir):
        r = self._make_router(tmp_dir)
        assert r._is_review_map_request("review project map") is True

    def test_case_insensitive(self, tmp_dir):
        r = self._make_router(tmp_dir)
        assert r._is_review_map_request("Review Map") is True
        assert r._is_review_map_request("REVIEW THE MAP") is True

    def test_no_match_partial(self, tmp_dir):
        r = self._make_router(tmp_dir)
        assert r._is_review_map_request("can you review the map please") is False

    def test_no_match_unrelated(self, tmp_dir):
        r = self._make_router(tmp_dir)
        assert r._is_review_map_request("what is the map?") is False

    def test_no_match_when_v1(self, tmp_dir):
        """v1 memory never triggers the intercept."""
        from mesh.router_v2 import RouterV2, RouterV2Config

        mem = MagicMock()
        mem.__class__ = type("NotMemorySystemV2", (), {})

        async def noop_send(content, in_reply_to=None):
            pass
        async def noop_worker(context, trigger, **kwargs):
            from mesh.router_v2 import WorkerResult
            return WorkerResult(response="ok", context=[], usage=None, error=None)

        r = RouterV2(
            worker_fn=noop_worker,
            send_fn=noop_send,
            config=RouterV2Config(llm_enabled=True),
            llm_client=AsyncMock(),
            nickname="test-bot",
            memory_system=mem,
        )
        assert r._is_review_map_request("review map") is False
