"""Tests for memory retrieval redesign C1 — config, store filter, TOC builder."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

from mesh.config import NodeConfig
from mesh.memory.store import MemoryEntry, MemoryStore
from mesh.memory.system_v2 import MemorySystemV2, TocEntry


# ── Helpers ──────────────────────────────────────────────────────────

def _make_entry(
    id: str,
    project: str = "",
    retrieval_key: str = "",
    summary: str = "summary",
    tags: list[str] | None = None,
    created_at: datetime | None = None,
    retrieval_key_embedding: np.ndarray | None = None,
    reflection_embedding: np.ndarray | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        id=id,
        created_at=created_at or datetime.now(timezone.utc),
        summary=summary,
        reflection="reflection",
        trace="trace",
        trigger="trigger",
        retrieval_key=retrieval_key,
        project=project,
        tags=tags or [],
        outcome="success",
        retrieval_key_embedding=retrieval_key_embedding,
        reflection_embedding=reflection_embedding,
    )


def _make_embedding(seed: int, dim: int = 1536) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


@dataclass
class FakeTurn:
    content: str = ""
    meta: dict = field(default_factory=dict)


class FakeConvHistory:
    def __init__(self, tool_results: list[tuple[str, FakeTurn]] | None = None):
        self._tool_results = tool_results or []
        self.window: list[FakeTurn] = []

    def iter_tool_results(self, tool_name: str):
        return [t for name, t in self._tool_results if name == tool_name]


# ── Config defaults ──────────────────────────────────────────────────

def test_node_config_retrieval_redesign_defaults_off():
    cfg = NodeConfig(id="test")
    assert cfg.memory_retrieval_redesign_enabled is False
    assert cfg.memory_toc_size == 30
    assert cfg.memory_toc_ranking == "cosine"
    assert cfg.memory_get_payload_max_chars == 6000
    assert cfg.memory_search_default_k == 5


# ── Store: list_by_project ───────────────────────────────────────────

def test_store_list_by_project_filters(tmp_path):
    store = MemoryStore("test_proj_filter", db_dir=str(tmp_path))
    emb = _make_embedding(0)
    for i, proj in enumerate(["A", "A", "A", "B", ""]):
        e = _make_entry(
            f"m_{i:04d}", project=proj,
            retrieval_key_embedding=emb, reflection_embedding=emb,
        )
        e.project = proj
        store.insert(e)

    result_a = store.list_by_project("A")
    assert len(result_a) == 4  # 3 project=A + 1 project=""

    result_a_strict = store.list_by_project("A", include_project_empty=False)
    assert len(result_a_strict) == 3

    result_b = store.list_by_project("B")
    assert len(result_b) == 2  # 1 project=B + 1 project=""

    result_none = store.list_by_project(None)
    assert len(result_none) == 5

    store.close()


# ── build_toc ────────────────────────────────────────────────────────

@pytest.fixture
def memory_system(tmp_path, monkeypatch):
    """Create a MemorySystemV2 with a mock LLM client and pre-populated pool."""
    class FakeLLM:
        async def chat(self, *a, **kw):
            return "ok"
    sys = MemorySystemV2(
        nickname="test_toc",
        llm_client=FakeLLM(),
        embedding_backend="openai",
        embedding_model="text-embedding-3-small",
    )
    # Manually set up store and pool without calling initialize()
    sys._store = MemoryStore("test_toc", db_dir=str(tmp_path))
    sys._pool = []
    sys._active_project = None
    return sys


def test_build_toc_defaults_to_all_projects(memory_system):
    # New default (2026-07-09): project=None → full pool across all
    # projects, even when an active project is set.
    now = datetime.now(timezone.utc)
    emb = _make_embedding(42)
    entries = []
    for i in range(10):
        proj = "projA" if i < 5 else "projB"
        e = _make_entry(
            f"m_{i:04d}", project=proj,
            retrieval_key=f"task {i}",
            created_at=now - timedelta(hours=i),
            retrieval_key_embedding=emb,
        )
        memory_system._store.insert(e)
        entries.append(e)
    memory_system._pool = entries
    memory_system._active_project = "projA"

    toc = asyncio.run(
        memory_system.build_toc(k=30)
    )
    # All 10 entries from both projects are candidates now.
    assert len(toc) == 10
    assert {t.project for t in toc} == {"projA", "projB"}


def test_build_toc_explicit_project_scopes(memory_system):
    # Explicit project=<name> still scopes to that project.
    now = datetime.now(timezone.utc)
    emb = _make_embedding(42)
    entries = []
    for i in range(10):
        proj = "projA" if i < 5 else "projB"
        e = _make_entry(
            f"m_{i:04d}", project=proj,
            retrieval_key=f"task {i}",
            created_at=now - timedelta(hours=i),
            retrieval_key_embedding=emb,
        )
        memory_system._store.insert(e)
        entries.append(e)
    memory_system._pool = entries

    toc = asyncio.run(
        memory_system.build_toc(k=30, project="projA")
    )
    # Only projA entries (5) — no project="" entries in this dataset
    assert len(toc) == 5
    for t in toc:
        assert t.project == "projA"


def test_build_toc_size_capped_at_k(memory_system):
    now = datetime.now(timezone.utc)
    entries = []
    for i in range(100):
        e = _make_entry(
            f"m_{i:04d}", project="",
            retrieval_key=f"task {i}",
            created_at=now - timedelta(minutes=i),
        )
        memory_system._store.insert(e)
        entries.append(e)
    memory_system._pool = entries

    toc = asyncio.run(
        memory_system.build_toc(k=30)
    )
    assert len(toc) == 30


def test_build_toc_ranks_by_retrieval_key_similarity(memory_system):
    now = datetime.now(timezone.utc)
    emb_a = _make_embedding(1)
    emb_b = _make_embedding(2)
    emb_c = _make_embedding(3)

    entries = [
        _make_entry("m_aaa", retrieval_key="deploy kubernetes cluster",
                     created_at=now - timedelta(hours=3),
                     retrieval_key_embedding=emb_a),
        _make_entry("m_bbb", retrieval_key="fix CSS layout bug",
                     created_at=now - timedelta(hours=2),
                     retrieval_key_embedding=emb_b),
        _make_entry("m_ccc", retrieval_key="write unit tests for auth",
                     created_at=now - timedelta(hours=1),
                     retrieval_key_embedding=emb_c),
    ]
    for e in entries:
        memory_system._store.insert(e)
    memory_system._pool = entries

    # Monkeypatch _get_query_embedding to return emb_a (matches m_aaa)
    async def fake_embed(query):
        return emb_a
    memory_system._get_query_embedding = fake_embed

    toc = asyncio.run(
        memory_system.build_toc(query_text="deploy kubernetes cluster")
    )
    assert len(toc) == 3
    assert toc[0].id == "m_aaa"


def test_build_toc_falls_back_to_summary_when_retrieval_key_empty(memory_system):
    now = datetime.now(timezone.utc)
    e = _make_entry(
        "m_old", retrieval_key="",
        summary="A very long summary about debugging memory leaks in production servers " * 3,
        created_at=now,
    )
    memory_system._store.insert(e)
    memory_system._pool = [e]

    toc = asyncio.run(
        memory_system.build_toc()
    )
    assert len(toc) == 1
    assert toc[0].retrieval_key == e.summary[:150]


def test_build_toc_recency_fallback_when_no_query(memory_system):
    now = datetime.now(timezone.utc)
    entries = []
    for i in range(5):
        e = _make_entry(
            f"m_{i:04d}", retrieval_key=f"task {i}",
            created_at=now - timedelta(hours=4 - i),  # m_0004 is newest
        )
        memory_system._store.insert(e)
        entries.append(e)
    memory_system._pool = entries

    toc = asyncio.run(
        memory_system.build_toc(query_text=None)
    )
    assert toc[0].id == "m_0004"
    assert toc[-1].id == "m_0000"


# ── dedup_toc_against_window ─────────────────────────────────────────

def test_dedup_toc_against_window_marks_fetched(memory_system):
    toc = [
        TocEntry(id="m_a3c1", retrieval_key="task A", project="p"),
        TocEntry(id="m_b7d4", retrieval_key="task B", project="p"),
    ]
    call_turn = FakeTurn(
        content="",
        meta={"trace_block": "tool_call", "tool_call_id": "call_1",
              "tool_args": {"id": "m_a3c1"}},
    )
    result_turn = FakeTurn(
        content="memory content here",
        meta={"tool_call_id": "call_1"},
    )
    conv = FakeConvHistory(tool_results=[("memory_get", result_turn)])
    conv.window = [call_turn, result_turn]

    memory_system.dedup_toc_against_window(toc, conv)
    assert toc[0].already_in_context is True
    assert toc[0].truncated_in_context is False
    assert toc[1].already_in_context is False


def test_dedup_toc_against_window_truncated_flag(memory_system):
    toc = [
        TocEntry(id="m_a3c1", retrieval_key="task A", project="p"),
    ]
    call_turn = FakeTurn(
        content="",
        meta={"trace_block": "tool_call", "tool_call_id": "call_1",
              "tool_args": {"id": "m_a3c1"}},
    )
    result_turn = FakeTurn(
        content="truncated...",
        meta={"tool_call_id": "call_1", "truncated": True},
    )
    conv = FakeConvHistory(tool_results=[("memory_get", result_turn)])
    conv.window = [call_turn, result_turn]

    memory_system.dedup_toc_against_window(toc, conv)
    assert toc[0].already_in_context is True
    assert toc[0].truncated_in_context is True


def test_dedup_toc_against_window_includes_search_results(memory_system):
    toc = [
        TocEntry(id="m_abc12345", retrieval_key="task X", project="p"),
        TocEntry(id="m_def67890", retrieval_key="task Y", project="p"),
    ]
    result_turn = FakeTurn(
        content="Found entries:\n**ID**: m_abc12345 — something\n**ID**: m_def67890 — another",
        meta={},
    )
    conv = FakeConvHistory(tool_results=[("memory_search", result_turn)])
    conv.window = []

    memory_system.dedup_toc_against_window(toc, conv)
    assert toc[0].already_in_context is True
    assert toc[1].already_in_context is True


# ── render_toc_block ─────────────────────────────────────────────────

def test_render_toc_block_xml_format(memory_system):
    memory_system._active_project = "sp26-221"
    toc = [
        TocEntry(id="m_a3c1", retrieval_key="deploy k8s", project="sp26-221"),
        TocEntry(id="m_b7d4", retrieval_key="fix auth bug", project="sp26-221",
                 already_in_context=True),
        TocEntry(id="m_e2f0", retrieval_key="write tests", project="sp26-221",
                 already_in_context=True, truncated_in_context=True),
    ]
    result = memory_system.render_toc_block(toc)
    assert '<memory_toc count="3" project="sp26-221">' in result
    assert "  m_a3c1: deploy k8s" in result
    assert "  m_b7d4 [already in context]: fix auth bug" in result
    assert "  m_e2f0 [already in context (truncated)]: write tests" in result
    assert result.endswith("</memory_toc>")


def test_render_toc_block_empty(memory_system):
    assert memory_system.render_toc_block([]) == ""


# ══════════════════════════════════════════════════════════════════════
# C2 — memory_get / memory_search tool refactor
# ══════════════════════════════════════════════════════════════════════


class FakeMemorySystem:
    """Minimal mock of MemorySystemV2 for tool-level tests."""

    def __init__(self, entries=None, payload_max_chars=6000, active_project=None):
        self._pool = entries or []
        self._payload_max_chars = payload_max_chars
        self._active_project = active_project
        self._store = FakeStore(self._pool)
        self._query_embedding_cache = {}

    def get_entry(self, entry_id):
        for e in self._pool:
            if e.id == entry_id:
                return e
        return None

    def list_entries(self):
        return list(self._pool)

    def is_active(self, entry_id):
        return False

    async def search_block(self, query, k=5, project=None, tag=None, mode="hybrid"):
        # New default (2026-07-09): None/"" → all projects; explicit
        # project name scopes.
        if project == "":
            project = None

        candidates = self._pool[:]
        if project:
            candidates = [e for e in candidates if e.project == project or e.project == ""]
        if tag:
            candidates = [e for e in candidates if tag in e.tags]

        if not candidates:
            return ""

        parts = [f'<memory_search_results query="{query[:80]}" k="{k}">']
        for e in candidates[:k]:
            date_str = e.created_at.strftime("%Y-%m-%d")
            parts.append(
                f'<entry id="{e.id}" date="{date_str}" project="{e.project}" '
                f'similarity="0.900">'
            )
            parts.append(f"**ID**: {e.id}")
            parts.append(f"**Retrieval key**: {e.retrieval_key or '(none)'}")
            parts.append(f"**Summary**: {e.summary}")
            parts.append("</entry>")
        parts.append("</memory_search_results>")
        return "\n".join(parts)

    async def render_block_for_query(self, query, k=None, tag=None):
        return "legacy block"


class FakeStore:
    def __init__(self, pool):
        self._pool = pool

    def list_by_project(self, project, include_project_empty=True):
        results = []
        for e in self._pool:
            if e.project == project:
                results.append(e)
            elif include_project_empty and e.project == "":
                results.append(e)
        return results


# ── memory_get tests ────────────────────────────────────────────────


def test_memory_get_includes_retrieval_key_and_project():
    import mesh.tool_implementations as ti
    entry = _make_entry(
        "m_test01", project="proj-x", retrieval_key="deploy nginx cluster",
    )
    fake_sys = FakeMemorySystem(entries=[entry])
    old = ti._memory_system
    try:
        ti._memory_system = fake_sys
        result = ti.memory_get("m_test01")
        assert "**Project**: proj-x" in result
        assert "**Retrieval key**: deploy nginx cluster" in result
        assert "**ID**: m_test01" in result
    finally:
        ti._memory_system = old


def test_memory_get_pre_v3_entry_shows_none_marker():
    import mesh.tool_implementations as ti
    entry = _make_entry("m_old01", project="", retrieval_key="")
    fake_sys = FakeMemorySystem(entries=[entry])
    old = ti._memory_system
    try:
        ti._memory_system = fake_sys
        result = ti.memory_get("m_old01")
        assert "**Retrieval key**: (none — pre-v3 entry)" in result
        assert "**Project**: (none)" in result
    finally:
        ti._memory_system = old


def test_memory_get_truncates_at_payload_cap():
    import mesh.tool_implementations as ti
    entry = _make_entry("m_big01", project="proj-y", retrieval_key="big task")
    entry.trace = "X" * 20000  # Very long trace
    fake_sys = FakeMemorySystem(entries=[entry], payload_max_chars=500)
    old = ti._memory_system
    try:
        ti._memory_system = fake_sys
        result = ti.memory_get("m_big01")
        assert len(result) < 600  # 500 + truncation notice
        assert "[truncated:" in result
    finally:
        ti._memory_system = old


def test_memory_get_not_found():
    import mesh.tool_implementations as ti
    fake_sys = FakeMemorySystem(entries=[])
    old = ti._memory_system
    try:
        ti._memory_system = fake_sys
        result = ti.memory_get("m_nonexist")
        assert "No memory entry found" in result
    finally:
        ti._memory_system = old


# ── memory_search tests ─────────────────────────────────────────────


def test_memory_search_default_searches_all_projects():
    # New default (2026-07-09): omitting project searches all projects,
    # even when an active project is set.
    import mesh.tool_implementations as ti
    entries = [
        _make_entry("m_a", project="proj-x", retrieval_key="task A"),
        _make_entry("m_b", project="proj-y", retrieval_key="task B"),
    ]
    fake_sys = FakeMemorySystem(entries=entries, active_project="proj-x")
    old = ti._memory_system
    try:
        ti._memory_system = fake_sys
        result = asyncio.run(
            ti.memory_search("find task")
        )
        assert "m_a" in result
        assert "m_b" in result
    finally:
        ti._memory_system = old


def test_memory_search_explicit_empty_searches_all_projects():
    import mesh.tool_implementations as ti
    entries = [
        _make_entry("m_a", project="proj-x", retrieval_key="task A"),
        _make_entry("m_b", project="proj-y", retrieval_key="task B"),
    ]
    fake_sys = FakeMemorySystem(entries=entries, active_project="proj-x")
    old = ti._memory_system
    try:
        ti._memory_system = fake_sys
        result = asyncio.run(
            ti.memory_search("find task", project="")
        )
        assert "m_a" in result
        assert "m_b" in result
    finally:
        ti._memory_system = old


def test_memory_search_explicit_project_filters():
    import mesh.tool_implementations as ti
    entries = [
        _make_entry("m_a", project="proj-x", retrieval_key="task A"),
        _make_entry("m_b", project="proj-y", retrieval_key="task B"),
        _make_entry("m_c", project="", retrieval_key="task C"),
    ]
    fake_sys = FakeMemorySystem(entries=entries, active_project="proj-x")
    old = ti._memory_system
    try:
        ti._memory_system = fake_sys
        result = asyncio.run(
            ti.memory_search("find task", project="proj-y")
        )
        assert "m_b" in result
        assert "m_c" in result  # project="" included
        assert "m_a" not in result
    finally:
        ti._memory_system = old


def test_memory_search_tag_filter_still_works():
    import mesh.tool_implementations as ti
    entries = [
        _make_entry("m_a", project="proj-x", retrieval_key="task A", tags=["deploy"]),
        _make_entry("m_b", project="proj-x", retrieval_key="task B", tags=["bugfix"]),
    ]
    fake_sys = FakeMemorySystem(entries=entries, active_project="proj-x")
    old = ti._memory_system
    try:
        ti._memory_system = fake_sys
        result = asyncio.run(
            ti.memory_search("find task", tag="deploy")
        )
        assert "m_a" in result
        assert "m_b" not in result
    finally:
        ti._memory_system = old


# ── search_block integration (on real MemorySystemV2) ───────────────


def test_search_block_uses_reflection_embedding(memory_system):
    """Verify search_block scores against reflection_embedding, not just retrieval_key_embedding."""
    now = datetime.now(timezone.utc)
    # Create two entries with DIFFERENT reflection vs retrieval_key embeddings
    refl_emb_a = _make_embedding(100)
    rk_emb_a = _make_embedding(200)
    refl_emb_b = _make_embedding(300)
    rk_emb_b = _make_embedding(400)

    entries = [
        _make_entry("m_aaa", retrieval_key="task A", created_at=now - timedelta(hours=2),
                    retrieval_key_embedding=rk_emb_a, reflection_embedding=refl_emb_a),
        _make_entry("m_bbb", retrieval_key="task B", created_at=now - timedelta(hours=1),
                    retrieval_key_embedding=rk_emb_b, reflection_embedding=refl_emb_b),
    ]
    for e in entries:
        memory_system._store.insert(e)
    memory_system._pool = entries

    # Mock embedding to return refl_emb_a — should rank m_aaa first
    async def fake_embed(query):
        return refl_emb_a
    memory_system._get_query_embedding = fake_embed

    result = asyncio.run(
        memory_system.search_block("test query", k=5)
    )
    assert "m_aaa" in result
    # m_aaa should appear before m_bbb
    assert result.index("m_aaa") < result.index("m_bbb")


# ── C3 — TOC injection into router prompt ──────────────────────────

from unittest.mock import AsyncMock, MagicMock, patch
from mesh.router_v2 import RouterV2, RouterV2Config, WorkerResult


def _make_toc_entries(n: int = 5) -> list[TocEntry]:
    return [
        TocEntry(
            id=f"m_{i:04d}",
            retrieval_key=f"key-{i}",
            project="proj",
            tags=[],
            score=1.0 - i * 0.1,
        )
        for i in range(n)
    ]


def _make_router(*, toc_enabled: bool, toc_entries: list[TocEntry] | None = None):
    """Build a minimal RouterV2 with a mocked MemorySystemV2."""
    sent: list[dict] = []

    async def _send_fn(content, in_reply_to=None):
        sent.append({"content": content, "in_reply_to": in_reply_to})

    async def _worker_fn(context, trigger, **kwargs):
        return WorkerResult(response="Done.", context=[], usage=None, error=None)

    config = RouterV2Config(
        llm_enabled=False,
        history_persist=False,
        memory_retrieval_redesign_enabled=toc_enabled,
        memory_toc_size=30,
    )

    entries = toc_entries if toc_entries is not None else _make_toc_entries(5)

    mock_memory = MagicMock()
    mock_memory.__class__ = type("MemorySystemV2", (), {})
    # Make isinstance check work
    with patch("mesh.router_v2.MemorySystemV2", type(mock_memory)):
        pass
    mock_memory.get_personality = MagicMock(return_value=None)
    mock_memory.build_toc = AsyncMock(return_value=entries)
    mock_memory.dedup_toc_against_window = MagicMock(return_value=entries)
    mock_memory.render_toc_block = MagicMock(
        return_value=f'<memory_toc count="{len(entries)}" project="proj">\n'
        + "\n".join(f"  {e.id}: {e.retrieval_key}" for e in entries)
        + "\n</memory_toc>"
        if entries else ""
    )
    mock_memory.render_representative_block = AsyncMock(
        return_value="<memory>\nactive-set content\n</memory>"
    )
    mock_memory.render_maps_block = AsyncMock(
        return_value="<project_map>\nmap content\n</project_map>"
    )
    mock_memory.render_relevant_maps_block = AsyncMock(
        return_value='<project_map project="test" relevance="0.85">\nmap content\n</project_map>'
    )
    mock_memory.render_recent_log_block = AsyncMock(
        return_value="<recent_activity>\nlog content\n</recent_activity>"
    )
    mock_memory.render_summary_block = AsyncMock(return_value=None)

    router = RouterV2(
        worker_fn=_worker_fn,
        send_fn=_send_fn,
        config=config,
        nickname="test-bot",
        agent_type="test",
        node_id="agent:test:test-bot",
        memory_system=mock_memory,
    )
    # Patch isinstance check inside router
    router._memory = mock_memory

    return router, mock_memory, sent


def _build_prompt(router) -> str:
    """Call _build_router_prompt synchronously."""
    return asyncio.run(
        router._build_router_prompt("test instructions")
    )


def test_router_prompt_injects_toc_when_flag_on():
    """C3: When memory_retrieval_redesign_enabled, prompt contains <memory_toc>."""
    router, mock_mem, _ = _make_router(toc_enabled=True)

    # Patch isinstance to recognise our mock
    orig_isinstance = __builtins__["isinstance"] if isinstance(__builtins__, dict) else getattr(__builtins__, "isinstance")
    from mesh.memory.system_v2 import MemorySystemV2 as RealMemV2

    with patch("mesh.router_v2.MemorySystemV2", new=type(mock_mem)):
        # Monkeypatch the isinstance check in router_v2
        import mesh.router_v2 as rv2_mod
        old_msv2 = rv2_mod.MemorySystemV2
        rv2_mod.MemorySystemV2 = type(mock_mem)
        try:
            prompt = _build_prompt(router)
        finally:
            rv2_mod.MemorySystemV2 = old_msv2

    assert '<memory_toc count="5"' in prompt
    mock_mem.build_toc.assert_awaited_once()
    mock_mem.dedup_toc_against_window.assert_called_once()
    mock_mem.render_toc_block.assert_called_once()


def test_router_prompt_omits_active_set_when_toc_on():
    """C3: When TOC mode is on, the old <memory> representative block is absent."""
    router, mock_mem, _ = _make_router(toc_enabled=True)

    import mesh.router_v2 as rv2_mod
    old_msv2 = rv2_mod.MemorySystemV2
    rv2_mod.MemorySystemV2 = type(mock_mem)
    try:
        prompt = _build_prompt(router)
    finally:
        rv2_mod.MemorySystemV2 = old_msv2

    assert "<memory>" not in prompt
    assert "active-set content" not in prompt
    mock_mem.render_representative_block.assert_not_awaited()


def test_router_prompt_keeps_map_and_recent_log_with_toc():
    """C3: Map and recent log are preserved even when TOC is on."""
    router, mock_mem, _ = _make_router(toc_enabled=True)

    import mesh.router_v2 as rv2_mod
    old_msv2 = rv2_mod.MemorySystemV2
    rv2_mod.MemorySystemV2 = type(mock_mem)
    try:
        prompt = _build_prompt(router)
    finally:
        rv2_mod.MemorySystemV2 = old_msv2

    assert "<project_map" in prompt
    assert "<recent_activity>" in prompt


def test_router_prompt_falls_back_to_active_set_when_flag_off():
    """C3: When flag is off, the old <memory> block is used, not <memory_toc>."""
    router, mock_mem, _ = _make_router(toc_enabled=False)

    import mesh.router_v2 as rv2_mod
    old_msv2 = rv2_mod.MemorySystemV2
    rv2_mod.MemorySystemV2 = type(mock_mem)
    try:
        prompt = _build_prompt(router)
    finally:
        rv2_mod.MemorySystemV2 = old_msv2

    assert "<memory>" in prompt
    assert "active-set content" in prompt
    assert "<memory_toc" not in prompt
    mock_mem.render_representative_block.assert_awaited_once()
    mock_mem.build_toc.assert_not_awaited()
