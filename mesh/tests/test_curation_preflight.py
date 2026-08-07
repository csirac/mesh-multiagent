"""T-010 regressions for deterministic curation-write refusals.

Every test uses a temporary ``MemoryStore``.  No live agent database is read
or written.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mesh.agent_node import CURRENT_CURATION_CONTEXT, AgentNode
from mesh.config import NodeConfig
from mesh.memory.curation import CurationBatch, CurationExecutionContext
from mesh.memory.entities import EntityExecutionContext, EntityService
from mesh.memory.pending_additions import PendingAdditionLedger
from mesh.memory.store import MemoryEntry, MemoryStore
from mesh.memory.system_v2 import MemorySystemV2
from mesh.memory.write_audit import OUTCOME_QUEUED
from mesh.tools import get_registry


def _mid(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()[:12]


def _cite(label: str) -> str:
    return f"[m_{_mid(label)}]"


def _entry(memory_id: str, *, project: str = "") -> MemoryEntry:
    return MemoryEntry(
        id=memory_id,
        created_at=datetime.now(timezone.utc),
        summary=f"Summary for {memory_id}",
        reflection=f"Reflection for {memory_id}",
        trace="trace",
        trigger="trigger",
        retrieval_key=f"retrieval {memory_id}",
        tags=["test"],
        outcome="success",
        reflection_embedding=np.ones(4, dtype=np.float32),
        retrieval_key_embedding=np.ones(4, dtype=np.float32),
        project=project,
    )


def _context() -> EntityExecutionContext:
    return EntityExecutionContext(
        actor_node="agent:test:curator",
        source_message_id="curation-turn-1",
        source_author="agent:test:curator",
        source_content="synthetic curation summary",
        curation_turn_id="curation-turn-1",
    )


def _curation_context() -> CurationExecutionContext:
    return CurationExecutionContext(
        mode="write",
        trigger_id="curation-turn-1",
        actor_node="agent:test:curator",
        batch=CurationBatch(reason="time-based", memory_ids=("m1",)),
    )


def _active(
    service: EntityService, name: str = "Project Owner", entity_type: str = "person",
) -> str:
    result = service.create_user_named_entity(
        entity_type,
        name,
        naming_surface=name,
        context=EntityExecutionContext(
            actor_node="agent:test:curator",
            source_message_id="user-message",
            source_author="user:testuser",
            source_content=f"Please name {name}.",
        ),
        reason="test setup",
    )
    return str(result["entity"]["entity_key"])


@pytest.fixture
def store(tmp_path):
    result = MemoryStore("curation-preflight", db_dir=str(tmp_path))
    yield result
    result.close()


@pytest.fixture
def service(store):
    return EntityService(
        store._conn,
        actor_node="agent:test:curator",
        mutations_enabled=True,
    )


def _memory_system(store) -> MemorySystemV2:
    system = MemorySystemV2.__new__(MemorySystemV2)
    system._store = store
    system._entity_activation_window_threshold = 3
    system._entity_resolution_enabled = True
    system._pool = []
    return system


def _node(tmp_path, store, *, digest: Path | None = None) -> AgentNode:
    kwargs = dict(
        id="agent:test:curator",
        tools=[],
        entity_resolution_mode="write",
        entity_self_curation_enabled=True,
    )
    if digest is not None:
        kwargs.update(
            standing_digest_enabled=True,
            standing_digest_path=str(digest),
        )
    node = AgentNode(NodeConfig(**kwargs), tool_registry=get_registry())
    node._memory_system = _memory_system(store)
    return node


async def _call(node: AgentNode, name: str, arguments: dict, context) -> str:
    token = CURRENT_CURATION_CONTEXT.set(context)
    try:
        return await node._execute_entity_tool(
            name,
            arguments,
            SimpleNamespace(
                id="curation-turn-1",
                from_node="agent:test:curator",
                content="synthetic curation summary",
            ),
        )
    finally:
        CURRENT_CURATION_CONTEXT.reset(token)


@pytest.mark.asyncio
async def test_pending_entity_is_rejected_before_dossier_composition(
    tmp_path, store, service,
):
    pending = service.create_pending_entity(
        "person", "Pending", "test pending entity", origin="self-curation",
        context=_context(), reason="test setup",
    )["entity_key"]

    node = _node(tmp_path, store)
    context = _curation_context()
    result = await _call(
        node,
        "essay_edit",
        {
            "key": str(pending),
            "old_text": "unused",
            "new_text": "Pending dossier text.",
            "reason": "pending preflight regression",
        },
        context,
    )

    assert result.startswith("Error: preflight:"), result
    assert "entity not yet active" in result
    assert store.get_essay(str(pending)) is None
    assert context.write_log.attempts == []


@pytest.mark.asyncio
async def test_unresolvable_citation_is_rejected_before_dossier_composition(
    tmp_path, store, service,
):
    key = _active(service)

    node = _node(tmp_path, store)
    context = _curation_context()
    result = await _call(
        node,
        "essay_edit",
        {
            "key": key,
            "old_text": "unused",
            "new_text": "Claim with a missing source. [m_deadbeef]",
            "reason": "citation preflight regression",
        },
        context,
    )

    assert result.startswith("Error: preflight:"), result
    assert "unresolvable citation [m_deadbeef]" in result
    assert context.write_log.attempts == []
    assert store.get_essay(key) is None


@pytest.mark.asyncio
async def test_linked_cross_project_citation_passes_preflight_and_commit(
    tmp_path, store, service,
):
    key = _active(service, "Mesh Autopilot", "project")
    foreign = _mid("foreign")
    store.insert(_entry(foreign, project="other-project"))
    service.link_memory(
        foreign, key, window_key="window-1", context=_context(), reason="setup",
    )

    node = _node(tmp_path, store)
    context = _curation_context()
    body = f"Claim with linked cross-project evidence. [m_{foreign}]"
    result = await _call(
        node,
        "essay_edit",
        {
            "key": key,
            "old_text": "unused",
            "new_text": body,
            "reason": "linked cross-project preflight regression",
        },
        context,
    )

    assert not result.startswith("Error"), result
    assert store.get_essay(key)["body"] == body


@pytest.mark.asyncio
async def test_unlinked_cross_project_citation_is_rejected_before_dossier_composition(
    tmp_path, store, service,
):
    key = _active(service, "Mesh Autopilot", "project")
    foreign = _mid("foreign")
    store.insert(_entry(foreign, project="other-project"))

    node = _node(tmp_path, store)
    context = _curation_context()
    result = await _call(
        node,
        "essay_edit",
        {
            "key": key,
            "old_text": "unused",
            "new_text": f"Claim with unlinked evidence. [m_{foreign}]",
            "reason": "unlinked cross-project preflight regression",
        },
        context,
    )

    assert store.get_essay(key) is None
    assert result.startswith("Error: preflight:"), result
    assert f"out-of-scope citation [m_{foreign}]" in result
    assert context.write_log.attempts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("artifact", ["essay", "digest"])
async def test_stale_anchor_is_refreshed_once_then_carried_forward(
    tmp_path, store, service, artifact,
):
    digest = tmp_path / "digest.md"
    digest.write_text(
        "# Standing digest\n\n"
        "## Timeline\n- Current event.\n\n"
        "## Narrative\nNarrative.\n\n"
        "## Projects\nProjects.\n\n"
        "## People\nPeople.\n\n"
        "## Standing decisions & conventions\n- Decision.\n\n"
        "## Open threads / where-we-are\n- Thread.\n\n"
        "## Agent narrative\nState.\n"
    )
    key = _active(service)
    memory_id = _mid("source")
    store.insert(_entry(memory_id))
    service.link_memory(
        memory_id, key, window_key="window-1", context=_context(), reason="setup",
    )
    service.publish_dossier(
        key,
        body=f"Existing dossier evidence. [m_{memory_id}]",
        title="Project Owner",
        token_budget=4000,
        measure=lambda text: len(text.split()),
        context=_context(),
        reason="seed",
    )
    node = _node(tmp_path, store, digest=digest)
    context = _curation_context()
    arguments = {
        "old_text": "anchor removed by another writer",
        "new_text": f"Durable follow-up. [m_{memory_id}]",
        "reason": "stale anchor regression",
    }
    if artifact == "essay":
        arguments["key"] = key
        tool = "essay_edit"
        target = f"essay:{key}"
    else:
        tool = "digest_edit"
        target = "digest:agent:test:curator"
        arguments["new_text"] = "Durable follow-up."

    first = await _call(node, tool, arguments, context)
    assert first.startswith("Error: stale old_text anchor"), first
    assert "fresh_artifact" in first
    assert "retry exactly once" in first

    second = await _call(node, tool, arguments, context)
    assert second.startswith("Error: stale old_text anchor"), second
    queued = PendingAdditionLedger(store._conn, agent="agent:test:curator").pending()
    assert len(queued) == 1
    assert queued[0].target_artifact == target
    assert queued[0].new_text == arguments["new_text"]
    assert context.write_log.resolution()[target] == OUTCOME_QUEUED


@pytest.mark.asyncio
async def test_unknown_link_target_is_rejected_before_embedding(store):
    class FailingEmbedder:
        async def embed_batch_to_arrays(self, _texts):
            raise AssertionError("embedding must not run for an unknown entity key")

    system = _memory_system(store)
    system._embedder = FailingEmbedder()
    memory_id = _mid("link-source")
    store.insert(_entry(memory_id))

    with pytest.raises(ValueError, match=r"unknown entity key 'person:missing'"):
        await system.correct_entity_link(
            memory_id=memory_id,
            reason="invalid target regression",
            context=_context(),
            add_entity_key="person:missing",
            memory_patch={"summary": "This change must not embed."},
        )

    assert store.get(memory_id).summary == f"Summary for {memory_id}"
