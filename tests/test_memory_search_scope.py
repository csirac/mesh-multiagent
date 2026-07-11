"""Unit tests for memory search/TOC project-scoping defaults.

Behavior change (2026-07-09): `search_block` and `build_toc` with
project=None now search the FULL memory pool across all projects
(previously they defaulted to the agent's active project, which hid
rows from other projects — e.g. only 203/1534 rows searchable).

Covers:
  (a) project=None returns cross-project hits,
  (b) explicit project=<name> still scopes (plus project-empty rows,
      the existing include_project_empty=True semantics),
  (c) explicit project="" returns all projects (legacy escape hatch,
      equivalent to the new default).

No network: the embedding client is stubbed with fixed vectors.
"""
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# EmbeddingClient constructs AsyncOpenAI in __init__; ensure a dummy key
# so construction never fails. The embedder is replaced with a stub below,
# so no network calls are ever made.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

from mesh.memory.store import MemoryEntry, MemoryStore  # noqa: E402
from mesh.memory.system_v2 import MemorySystemV2  # noqa: E402

DIM = 8

# Query embedding: unit vector along axis 0.
QUERY_EMB = np.zeros(DIM, dtype=np.float32)
QUERY_EMB[0] = 1.0


def _unit(axis0: float, axis1: float) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = axis0
    v[1] = axis1
    return v / np.linalg.norm(v)


# (id, project, similarity-controlling embedding)
_ROWS = [
    ("aaa111111111", "personal", _unit(0.95, 0.31)),      # top hit, other project
    ("bbb222222222", "personal-journaling", _unit(0.90, 0.44)),
    ("ccc333333333", "pipeline", _unit(0.60, 0.80)),       # active project
    ("ddd444444444", "pipeline", _unit(0.50, 0.87)),
    ("eee555555555", "", _unit(0.40, 0.92)),               # project-empty row
]


class _StubEmbedder:
    """Embedding client stub: fixed query vector, no network."""

    async def embed_to_array(self, text: str) -> np.ndarray:
        return QUERY_EMB.copy()


def _make_system(tmp_path) -> MemorySystemV2:
    memory = MemorySystemV2(nickname="scope-test", llm_client=MagicMock())
    memory._store = MemoryStore("scope-test", db_dir=str(tmp_path))
    memory._embedder = _StubEmbedder()

    now = datetime.now(timezone.utc)
    for i, (eid, project, emb) in enumerate(_ROWS):
        memory._store.insert(MemoryEntry(
            id=eid,
            created_at=now - timedelta(hours=i),
            summary=f"summary {eid}",
            reflection=f"reflection {eid}",
            trace="",
            trigger="",
            retrieval_key=f"key {eid}",
            project=project,
            reflection_embedding=emb.copy(),
            retrieval_key_embedding=emb.copy(),
        ))
    memory._pool = memory._store.load()
    # Simulate an agent whose active project is 'pipeline' — the old
    # default would have hidden 'personal*' rows from search.
    memory._active_project = "pipeline"
    return memory


def _entry_ids(rendered: str) -> list[str]:
    return re.findall(r'<entry id="([^"]+)"', rendered)


def _entry_projects(rendered: str) -> set[str]:
    return set(re.findall(r'<entry[^>]* project="([^"]*)"', rendered))


# ── search_block ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_default_is_all_projects(tmp_path):
    """(a) project=None searches the full pool across all projects."""
    memory = _make_system(tmp_path)
    rendered = await memory.search_block("query", k=5)
    ids = _entry_ids(rendered)
    assert set(ids) == {r[0] for r in _ROWS}
    projects = _entry_projects(rendered)
    assert {"personal", "personal-journaling", "pipeline", ""} <= projects
    # Highest-similarity row (other project) ranks first.
    assert ids[0] == "aaa111111111"
    memory._store.close()


@pytest.mark.asyncio
async def test_search_explicit_project_still_scopes(tmp_path):
    """(b) project=<name> scopes to that project + project-empty rows."""
    memory = _make_system(tmp_path)
    rendered = await memory.search_block("query", k=5, project="pipeline")
    ids = set(_entry_ids(rendered))
    assert ids == {"ccc333333333", "ddd444444444", "eee555555555"}
    assert "aaa111111111" not in ids
    memory._store.close()


@pytest.mark.asyncio
async def test_search_empty_string_means_all_projects(tmp_path):
    """(c) project="" (legacy escape hatch) equals the new default."""
    memory = _make_system(tmp_path)
    rendered_empty = await memory.search_block("query", k=5, project="")
    rendered_none = await memory.search_block("query", k=5, project=None)
    assert _entry_ids(rendered_empty) == _entry_ids(rendered_none)
    assert set(_entry_ids(rendered_empty)) == {r[0] for r in _ROWS}
    memory._store.close()


# ── build_toc ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_toc_default_is_all_projects(tmp_path):
    """(a) build_toc project=None covers all projects, not just active."""
    memory = _make_system(tmp_path)
    toc = await memory.build_toc(query_text="query", k=10)
    ids = {e.id for e in toc}
    assert ids == {r[0] for r in _ROWS}
    assert {e.project for e in toc} >= {"personal", "personal-journaling", "pipeline"}
    memory._store.close()


@pytest.mark.asyncio
async def test_toc_explicit_project_still_scopes(tmp_path):
    """(b) build_toc project=<name> scopes + keeps project-empty rows."""
    memory = _make_system(tmp_path)
    toc = await memory.build_toc(query_text="query", k=10, project="personal")
    ids = {e.id for e in toc}
    assert ids == {"aaa111111111", "eee555555555"}
    memory._store.close()


@pytest.mark.asyncio
async def test_toc_empty_string_means_all_projects(tmp_path):
    """(c) build_toc project="" returns all projects."""
    memory = _make_system(tmp_path)
    toc = await memory.build_toc(query_text="query", k=10, project="")
    assert {e.id for e in toc} == {r[0] for r in _ROWS}
    memory._store.close()
