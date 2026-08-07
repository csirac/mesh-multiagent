"""Fail-first tests for entity/group/digest self-curation.

One test per numbered claim in ``docs/plans/entity-self-curation.md`` §9
("Fail-first tests", claims 1-41).  Each test name carries its claim number so a
failure maps straight back to the spec line it violates.

The tests deliberately drive the real code paths — ``AgentNode._execute_entity_tool``
for tool dispatch, ``EntityService.publish_dossier`` for commit-time validation,
``RouterV2``'s queue for delivery semantics — rather than asserting against
mocks, because most of these claims are about authority and atomicity, which a
mock cannot exhibit.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from mesh.agent_node import CURRENT_CURATION_CONTEXT, AgentNode
from mesh.config import (
    NodeConfig,
    resolve_self_curation_mode,
    validate_self_curation_enrollment,
)
from mesh.llm import estimate_tokens
from mesh.memory.curation import (
    CurationBatch,
    CurationExecutionContext,
    SELF_CURATION_GROUP_TOOLS,
    SELF_CURATION_MUTATION_TOOLS,
    SELF_CURATION_READ_TOOLS,
    curation_tool_names,
    digest_section_errors,
    extract_citations,
    find_bracket_tokens,
    render_roster_block,
    roster_block_of,
)
from mesh.memory.entities import (
    EntityError,
    EntityExecutionContext,
    EntityService,
)
from mesh.memory.store import MemoryEntry, MemoryStore
from mesh.protocol import Message, MessageType
from mesh.router_v2 import RouterV2, RouterV2Config
from mesh.tools import get_registry


# ─────────────────────────────────────────────────────────────────────
# Fixtures and helpers
# ─────────────────────────────────────────────────────────────────────


def _mid(label: str) -> str:
    """A realistic bare 12-hex memory ID, exactly as ``memories.id`` stores it.

    The stored ID carries **no** ``m_`` prefix — ``essay_fold`` builds
    ``known_ids`` straight from ``SELECT id FROM memories`` and compares it
    against ``LOOSE_TAG_RE`` group 1, which is the bare hex run.  ``m_`` is a
    citation *surface* form that only exists inside ``[m_<id>]`` in a body; use
    :func:`_cite` to build one.  Both the citation regex
    (``[0-9a-fA-F]{4,40}``) and the near-miss ``LOOSE_TAG_RE`` guard depend on
    that shape, so tests must not use readable non-hex IDs.
    """
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()[:12]


def _cite(label: str) -> str:
    """The ``[m_<id>]`` citation surface for the memory ``_mid(label)``."""
    return f"[m_{_mid(label)}]"


def _entry(
    memory_id: str, summary: str | None = None, *, project: str = "",
) -> MemoryEntry:
    return MemoryEntry(
        id=memory_id,
        created_at=datetime.now(timezone.utc),
        summary=summary or f"Summary for {memory_id}",
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


@pytest.fixture
def store(tmp_path):
    result = MemoryStore("curation", db_dir=str(tmp_path))
    yield result
    result.close()


@pytest.fixture
def service(store):
    return EntityService(
        store._conn,
        actor_node="agent:test:curator",
        activation_window_threshold=3,
        mutations_enabled=True,
    )


def _curation_context(mode: str = "write", **kwargs) -> CurationExecutionContext:
    return CurationExecutionContext(
        mode=mode,
        trigger_id="curation-turn-1",
        actor_node="agent:test:curator",
        batch=CurationBatch(reason="time-based", memory_ids=("m1",)),
        **kwargs,
    )


def _exec_context() -> EntityExecutionContext:
    return EntityExecutionContext(
        actor_node="agent:test:curator",
        source_message_id="curation-turn-1",
        source_author="agent:test:curator",
        source_content="synthetic curation batch summary",
    )


def _activate(service: EntityService, display_name: str, entity_type="person") -> str:
    """Create and activate an entity through the self-curation origin path."""
    created = service.create_pending_entity(
        entity_type,
        display_name,
        f"{display_name} identity note",
        origin="self-curation",
        context=_exec_context(),
        reason="test setup",
    )
    key = created["entity_key"]
    service.activate_if_eligible(key, context=_exec_context(), reason="test setup")
    row = service.get_entity(key)
    if row["status"] != "active":
        service.connection.execute(
            "UPDATE entities SET status='active' WHERE entity_key = ?", (key,)
        )
        service.connection.commit()
    return key


def _force_active(store, entity_key: str) -> None:
    """Activate a row directly, then COMMIT.

    An uncommitted raw UPDATE leaves sqlite's implicit transaction open, and the
    next ``BEGIN IMMEDIATE`` inside EntityService raises "cannot start a
    transaction within a transaction".
    """
    store._conn.execute(
        "UPDATE entities SET status='active' WHERE entity_key = ?", (entity_key,)
    )
    store._conn.commit()


DIGEST_SECTIONS = (
    "# Standing digest\n\n"
    "## Timeline\n- 2026-07-30: something happened.\n\n"
    "## Narrative\nAn arc that is load-bearing.\n\n"
    "## Projects\nmesh — the platform.\n\n"
    "## People\nProject Owner — primary user.\n\n"
    "## Standing decisions & conventions\n- Curate after every batch.\n\n"
    "## Open threads / where-we-are\n- Self-curation lands.\n\n"
    "## Agent narrative\nI maintain my own state now.\n"
)


@pytest.fixture
def digest_file(tmp_path) -> Path:
    path = tmp_path / "agent-curator.md"
    path.write_text(DIGEST_SECTIONS)
    return path


def _memory_system(store):
    """A MemorySystemV2 carrying only what the curation paths actually touch.

    Built with ``__new__`` so the *real* ``correct_entity_link`` and
    ``_emit_curation_batch`` implementations run — the alternative, a
    hand-written stub, would test the stub instead of the code.
    """
    from mesh.memory.system_v2 import MemorySystemV2

    system = MemorySystemV2.__new__(MemorySystemV2)
    system._store = store
    system._entity_activation_window_threshold = 3
    system._entity_resolution_enabled = True
    system._embedder = None
    system._pool = []
    system._curation_batch_cb = None
    # Enough for _build_router_context_blocks() when a test drives a real
    # _call_router_full() against this stub instead of stopping short of it.
    system._personality_cache = ""
    return system


def _node(tmp_path, store, *, mode="write", digest: Path | None = None, **overrides):
    """An AgentNode wired to a real store, with a live curation scope available."""
    config_kwargs = dict(
        id="agent:test:curator",
        tools=[],
        entity_resolution_mode=mode if mode != "off" else "off",
        entity_self_curation_enabled=True,
        essay_token_budget=4000,
        standing_digest_budget_tokens=32000,
    )
    if digest is not None:
        config_kwargs["standing_digest_enabled"] = True
        config_kwargs["standing_digest_path"] = str(digest)
    config_kwargs.update(overrides)
    node = AgentNode(NodeConfig(**config_kwargs), tool_registry=get_registry())
    node._memory_system = _memory_system(store)
    return node


def _trigger():
    return SimpleNamespace(
        id="curation-turn-1",
        from_node="agent:test:curator",
        content="synthetic curation batch summary",
    )


async def _call(node, name: str, arguments: dict, context) -> str:
    token = CURRENT_CURATION_CONTEXT.set(context)
    try:
        return await node._execute_entity_tool(name, arguments, _trigger())
    finally:
        CURRENT_CURATION_CONTEXT.reset(token)


async def _router(**kwargs) -> RouterV2:
    async def worker_fn(*_args, **_kw):
        return None

    async def send_fn(*_args, **_kw):
        return None

    return RouterV2(worker_fn, send_fn, node_id="agent:test:curator", **kwargs)


# ─────────────────────────────────────────────────────────────────────
# Claims 1-4, 39-40 — post-commit batch emission
# ─────────────────────────────────────────────────────────────────────


class TestBatchEmission:
    """The formation → curation handoff (§4.1)."""

    def _system(self, store):
        return _memory_system(store)

    def test_claim01_one_batch_per_nonempty_formation(self, store):
        seen: list[CurationBatch] = []
        system = self._system(store)
        system.set_curation_batch_callback(seen.append)
        system._emit_curation_batch(
            reason="token-pressure",
            new_entries=[_entry(_mid("m1")), _entry(_mid("m2"))],
            entity_mutations=[{"op": "link", "entity_key": "person:owner"}],
            formed_at="2026-07-30T00:00:00+00:00",
        )
        assert len(seen) == 1
        assert seen[0].memory_ids == (_mid("m1"), _mid("m2"))
        assert seen[0].reason == "token-pressure"
        assert seen[0].entity_keys == ("person:owner",)

    def test_claim02_empty_formation_emits_no_batch(self, store):
        seen: list[CurationBatch] = []
        system = self._system(store)
        system.set_curation_batch_callback(seen.append)
        system._emit_curation_batch(
            reason="time-based",
            new_entries=[],
            entity_mutations=[],
            formed_at="2026-07-30T00:00:00+00:00",
        )
        assert seen == []

    def test_claim03_raising_callback_does_not_break_formation(self, store):
        def explode(_batch):
            raise RuntimeError("curation pipeline is broken")

        system = self._system(store)
        system.set_curation_batch_callback(explode)
        # Must not propagate: formation has already committed, and a curation
        # failure must not feed the three-strike parse-failure fallback.
        system._emit_curation_batch(
            reason="startup",
            new_entries=[_entry(_mid("m1"))],
            entity_mutations=[],
            formed_at="2026-07-30T00:00:00+00:00",
        )

    def test_claim04_callback_is_synchronous_and_never_awaited(self, store):
        """The callback must enqueue and return; awaiting would hold the lock."""
        import inspect

        from mesh.memory.system_v2 import MemorySystemV2

        assert not inspect.iscoroutinefunction(
            MemorySystemV2._emit_curation_batch
        ), "_emit_curation_batch must stay synchronous"
        assert not inspect.iscoroutinefunction(RouterV2.enqueue_curation_batch)

        # And the enqueue path must complete with no running event loop at all,
        # which is only possible if nothing inside it awaits.
        called: list[str] = []
        system = self._system(store)
        system.set_curation_batch_callback(lambda b: called.append(b.reason))
        system._emit_curation_batch(
            reason="shutdown",
            new_entries=[_entry(_mid("m1"))],
            entity_mutations=[],
            formed_at="2026-07-30T00:00:00+00:00",
        )
        assert called == ["shutdown"]

    @pytest.mark.parametrize(
        "reason",
        ["startup", "time-based", "token-pressure", "shutdown", "window-drop-future"],
    )
    def test_claim39_all_reasons_forwarded_unfiltered(self, store, reason):
        seen: list[CurationBatch] = []
        system = self._system(store)
        system.set_curation_batch_callback(seen.append)
        system._emit_curation_batch(
            reason=reason,
            new_entries=[_entry(_mid("m1"))],
            entity_mutations=[],
            formed_at="2026-07-30T00:00:00+00:00",
        )
        assert [b.reason for b in seen] == [reason]

    def test_claim40_parse_failure_path_emits_no_batch(self, store):
        """A fallback record is not successful curation input."""
        import inspect

        from mesh.memory.system_v2 import MemorySystemV2

        source = inspect.getsource(MemorySystemV2._write_parse_failure_fallback)
        assert "_emit_curation_batch" not in source
        assert "_curation_batch_cb" not in source
        # Emission is reached only from the normal persist path.
        persist = inspect.getsource(MemorySystemV2._persist_v3_entries_atomic)
        assert "_emit_curation_batch" in persist


# ─────────────────────────────────────────────────────────────────────
# Claims 5-6, 26, 31-33 — queue, ordering, lifecycle, history exclusion
# ─────────────────────────────────────────────────────────────────────


class TestQueueAndLifecycle:
    """Delivery semantics (§4.3, §4.4)."""

    @pytest.mark.asyncio
    async def test_claim05_batches_run_independently_of_the_router_turn_lock(self):
        """Curation drains while a message turn holds the lock.

        Replaces the original "curation waits for the lock" invariant.  Holding
        `_router_turn_lock` across a 1-4 minute curation LLM call starved
        message processing, so per-call state moved onto a contextvar and
        curation no longer takes the lock at all.  FIFO order between curation
        batches is still guaranteed — by the single drain task, not the lock.
        """
        router = await _router()
        ran: list[str] = []
        both_done = asyncio.Event()

        async def fake_turn(batch):
            ran.append(batch.memory_ids[0])
            if len(ran) == 2:
                both_done.set()

        router._run_curation_turn = fake_turn

        await router._router_turn_lock.acquire()
        try:
            router.enqueue_curation_batch(CurationBatch("time-based", ("m1",)))
            router.enqueue_curation_batch(CurationBatch("time-based", ("m2",)))
            await asyncio.wait_for(both_done.wait(), timeout=5.0)
            assert ran == ["m1", "m2"], (
                "curation must drain in FIFO order without waiting for the "
                "message turn to finish"
            )
        finally:
            router._router_turn_lock.release()

        assert await router.wait_for_curation_idle(timeout=5.0)
        await router.shutdown_curation(timeout=5.0)

    @pytest.mark.asyncio
    async def test_claim05b_message_turns_remain_serialized_with_each_other(self):
        """Removing the lock from curation must not unserialise messages."""
        router = await _router()
        acquire = "async with self._router_turn_lock"
        assert acquire in inspect.getsource(RouterV2.on_message), (
            "message turns must still be serialised behind _router_turn_lock"
        )
        # Look at the code only — the docstring names the lock to explain why
        # it is absent.
        drain = inspect.getsource(RouterV2._curation_drain_loop)
        drain_code = drain.replace(RouterV2._curation_drain_loop.__doc__ or "", "")
        assert acquire not in drain_code, (
            "curation must not acquire the router turn lock"
        )
        await router.shutdown_curation(timeout=5.0)

    @pytest.mark.asyncio
    async def test_claim06_every_batch_gets_one_turn_no_coalescing(self):
        router = await _router()
        ran: list[str] = []

        async def fake_turn(batch):
            ran.append(batch.memory_ids[0])
            await asyncio.sleep(0)

        router._run_curation_turn = fake_turn
        expected = [f"m{i}" for i in range(8)]
        for mid in expected:
            router.enqueue_curation_batch(CurationBatch("time-based", (mid,)))

        assert await router.wait_for_curation_idle(timeout=10.0)
        assert ran == expected, "no batch coalesced, dropped, or reordered"
        assert router.curation_status()["curation_batches_seen"] == 8
        await router.shutdown_curation(timeout=5.0)

    @pytest.mark.asyncio
    async def test_claim26_curation_turn_bypasses_all_history_paths(self):
        """internal_turn=True must reach the tool loop, and no history write."""
        import inspect

        captured: dict = {}

        async def router_process_fn(**kwargs):
            captured.update(kwargs)
            return "nothing to do"

        router = await _router(router_process_fn=router_process_fn)
        router._curation_entity_service = lambda: None
        router._render_curation_instruction = lambda batch: ("INSTRUCTION", "BATCH")

        await router._run_curation_turn(CurationBatch("time-based", ("m1",)))

        assert captured.get("internal_turn") is True
        assert captured.get("execution_scope_kind") == "curation"

        # The three persistent paths are gated on internal_turn in agent_node.
        source = inspect.getsource(AgentNode._router_process_with_llm)
        assert "internal_turn" in source, (
            "_router_process_with_llm must honor internal_turn to skip "
            "router_history, agent history, and persisted history"
        )

    @pytest.mark.asyncio
    async def test_claim31_toolless_response_does_not_synthesize_send_message(self):
        import inspect

        source = inspect.getsource(AgentNode._router_process_with_llm)
        # Locate the natural-text -> send_message synthesis and prove it is
        # guarded by internal_turn.
        assert "internal_turn" in source
        assert "send_message" in source
        assert "send_message" not in curation_tool_names(groups_enabled=True), (
            "send_message must never be in the curation allowlist"
        )
        assert "send_message" not in curation_tool_names()

    @pytest.mark.asyncio
    async def test_claim32_startup_formation_cannot_emit_before_registration(self):
        """Registration happens after _init_router_v2(), not at memory build."""
        import inspect

        connect = inspect.getsource(AgentNode.connect)
        assert "_register_curation_callback" in connect
        init_pos = connect.index("_init_router_v2")
        reg_pos = connect.index("_register_curation_callback")
        assert reg_pos > init_pos, (
            "the curation callback must be registered after router construction "
            "or startup formation races a missing router"
        )

    @pytest.mark.asyncio
    async def test_claim33_shutdown_drains_before_closing_dependencies(self):
        import inspect

        source = inspect.getsource(AgentNode.disconnect)
        assert "shutdown_curation" in source or "wait_for_curation_idle" in source
        drain = max(
            source.find("shutdown_curation"), source.find("wait_for_curation_idle")
        )
        assert drain != -1
        for closer in ("_memory_system.close", "llm_client.close"):
            pos = source.find(closer)
            if pos != -1:
                assert drain < pos, (
                    f"curation must drain before {closer}; shutdown formation's "
                    "batch would otherwise lose its dependencies"
                )

    @pytest.mark.asyncio
    async def test_claim_failure_is_recorded_and_fifo_survives_poison_batch(self):
        router = await _router()
        ran: list[str] = []

        async def flaky(batch):
            ran.append(batch.memory_ids[0])
            if batch.memory_ids[0] == "m1":
                raise RuntimeError("boom")

        router._run_curation_turn = flaky
        router._curation_entity_service = lambda: None
        router.enqueue_curation_batch(CurationBatch("time-based", ("m1",)))
        router.enqueue_curation_batch(CurationBatch("time-based", ("m2",)))
        assert await router.wait_for_curation_idle(timeout=5.0)

        assert ran == ["m1", "m2"], "a poison batch must not block the FIFO"
        status = router.curation_status()
        assert status["last_failed_curation_memory_ids"] == ["m1"]
        # The following successful turn receives all pending recovery IDs in
        # its evidence block and therefore clears them (§10.2).
        assert status["pending_curation_recovery_ids"] == []
        await router.shutdown_curation(timeout=5.0)


# ─────────────────────────────────────────────────────────────────────
# Claims 7-10, 27-28, 35-38 — authority, gating, fail-closed
# ─────────────────────────────────────────────────────────────────────


class TestAuthorityAndGating:
    """The capability boundary (§3.6)."""

    @pytest.mark.asyncio
    async def test_claim07_no_capability_fails_closed(self, tmp_path, store):
        node = _node(tmp_path, store)
        before = store._conn.total_changes
        # No CURRENT_CURATION_CONTEXT set at all.
        result = await node._execute_entity_tool(
            "entity_create",
            {"entity_type": "person", "display_name": "Lily", "reason": "x"},
            _trigger(),
        )
        assert "requires a live self-curation execution scope" in result
        assert store._conn.total_changes == before
        assert (
            store._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0
        )

    @pytest.mark.asyncio
    async def test_claim07b_registry_stub_is_fail_closed(self, store):
        """The global registry implementation can never mutate."""
        import mesh.tool_implementations as tools

        before = store._conn.total_changes
        for name, kwargs in [
            ("entity_create", {"entity_type": "person", "display_name": "L",
                               "reason": "r"}),
            ("entity_merge", {"loser_key": "a", "winner_key": "b", "reason": "r"}),
            ("entity_edit", {"entity_key": "a", "operation": "retire",
                             "reason": "r"}),
            ("entity_group_create", {"display_name": "G", "purpose": "p",
                                     "reason": "r"}),
            ("entity_group_member_add", {"group_key": "g", "member_key": "m",
                                         "reason": "r"}),
            ("entity_group_member_remove", {"group_key": "g", "member_key": "m",
                                            "reason": "r"}),
        ]:
            result = await getattr(tools, name)(**kwargs)
            assert "requires a live self-curation execution scope" in result
        assert store._conn.total_changes == before

    @pytest.mark.asyncio
    async def test_claim08_out_of_allowlist_call_is_rejected(self, tmp_path, store):
        """A salvaged/native call outside the phase allowlist never executes."""
        node = _node(tmp_path, store)
        # Phase 1 context: groups not enabled.
        context = _curation_context(groups_enabled=False)
        before = store._conn.total_changes
        result = await _call(
            node,
            "entity_group_create",
            {"display_name": "Fishing crew", "purpose": "rec", "reason": "r"},
            context,
        )
        assert "entity_self_curation_groups_enabled" in result
        assert store._conn.total_changes == before

        # An unknown name is refused before any dispatch.
        assert "unknown special tool" in await _call(
            node, "entity_obliterate", {"reason": "r"}, context
        )

    @pytest.mark.asyncio
    async def test_claim09_shadow_create_logs_event_and_commits_nothing(
        self, tmp_path, store
    ):
        node = _node(tmp_path, store, mode="shadow")
        context = _curation_context(mode="shadow")
        result = await _call(
            node,
            "entity_create",
            {"entity_type": "person", "display_name": "Lily", "reason": "new person"},
            context,
        )
        payload = json.loads(result)
        assert payload["would_apply"] is True and payload["applied"] is False
        # No registry row.
        assert (
            store._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0
        )
        # But an audit row suffixed _shadow.
        events = [
            row[0]
            for row in store._conn.execute(
                "SELECT event_type FROM entity_events"
            ).fetchall()
        ]
        assert any(name.endswith("_shadow") for name in events), events

    @pytest.mark.asyncio
    async def test_claim10_retired_alias_recreation_is_rejected(
        self, tmp_path, store, service
    ):
        """A self-curating agent must not silently recreate what it merged away."""
        key = _activate(service, "Lily")
        service.retire_entity(
            key, context=_exec_context(), reason="merged away last week"
        )

        node = _node(tmp_path, store)
        result = await _call(
            node,
            "entity_create",
            {"entity_type": "person", "display_name": "Lily", "reason": "recreate"},
            _curation_context(),
        )
        assert "Error" in result, result
        assert "retired" in result.lower(), result

    def test_claim10b_self_curation_origin_is_in_the_reject_tuple(self):
        import inspect

        source = inspect.getsource(EntityService.create_pending_entity)
        assert "self-curation" in source, (
            'origin.startswith(("formation", "historical", "self-curation")) '
            "is the retired-alias gate for this path"
        )

    def test_claim27_master_flag_off_registers_no_callback(self):
        config = NodeConfig(
            id="agent:test:off",
            entity_resolution_mode="write",
            entity_self_curation_enabled=False,
        )
        assert resolve_self_curation_mode(config) == "off"

    def test_claim28_resolution_mode_off_always_wins(self):
        config = NodeConfig(
            id="agent:test:off2",
            entity_resolution_mode="off",
            entity_self_curation_enabled=True,
            entity_self_curation_mode="write",
        )
        assert resolve_self_curation_mode(config) == "off"
        # Fail-closed schemas still exist globally...
        assert get_registry().get("entity_create") is not None
        # ...but the phase allowlist is what the turn offers.
        assert "entity_create" in curation_tool_names()

    def test_claim28b_shadow_cannot_escalate_to_write(self):
        config = NodeConfig(
            id="agent:test:shadow",
            entity_resolution_mode="shadow",
            entity_self_curation_enabled=True,
            entity_self_curation_mode="write",
        )
        assert resolve_self_curation_mode(config) == "shadow"

    def test_claim35_worker_scopes_never_carry_curation_mutations(self):
        import inspect

        source = inspect.getsource(AgentNode)
        assert "_strip_curation_tools" in source, (
            "worker ExecutionCapabilityScope construction must strip "
            "SELF_CURATION_MUTATION_TOOLS even if YAML requests them"
        )
        stripped = AgentNode._strip_curation_tools(
            frozenset(SELF_CURATION_MUTATION_TOOLS | {"memory_get", "token_count"})
        )
        for name in SELF_CURATION_MUTATION_TOOLS:
            assert name not in stripped, name
        # Read-only tools remain governed by ordinary worker configuration.
        assert {"memory_get", "token_count"} <= set(stripped)

    def test_claim36_invalid_enrollment_fails_validation(self, digest_file):
        def base(**over):
            kwargs = dict(
                id="agent:test:enroll",
                entity_resolution_mode="write",
                entity_self_curation_enabled=True,
                memory_formation_v3_enabled=True,
                context_mode="rolling-window",
                router_mode="full",
                project_maps_enabled=False,
                standing_digest_enabled=True,
                standing_digest_path=str(digest_file),
            )
            kwargs.update(over)
            return NodeConfig(**kwargs)

        assert validate_self_curation_enrollment(base()) == []

        def errors(**over):
            return " ".join(validate_self_curation_enrollment(base(**over)))

        assert "memory_formation_v3_enabled" in errors(
            memory_formation_v3_enabled=False
        )
        assert "rolling-window" in errors(context_mode="cc-session")
        assert "router_mode=full" in errors(router_mode="classifier")
        assert "project_maps_enabled" in errors(project_maps_enabled=True)
        assert "standing_digest" in errors(standing_digest_enabled=False)
        assert "not found" in errors(standing_digest_path="/nonexistent/digest.md")

    def test_claim36b_digest_missing_a_section_fails_validation(self, tmp_path):
        broken = tmp_path / "broken.md"
        broken.write_text("## Timeline\n- only one section\n")
        config = NodeConfig(
            id="agent:test:enroll2",
            entity_resolution_mode="write",
            entity_self_curation_enabled=True,
            memory_formation_v3_enabled=True,
            context_mode="rolling-window",
            router_mode="full",
            project_maps_enabled=False,
            standing_digest_enabled=True,
            standing_digest_path=str(broken),
        )
        joined = " ".join(validate_self_curation_enrollment(config))
        assert "standing digest" in joined and "section" in joined.lower()

    @pytest.mark.asyncio
    async def test_claim37_curation_narrows_entity_link_correct(
        self, tmp_path, store, service
    ):
        node = _node(tmp_path, store)
        context = _curation_context()
        store.insert(_entry(_mid("m1")))

        # Creation through the reused correction tool is rejected in curation.
        result = await _call(
            node,
            "entity_link_correct",
            {
                "memory_id": _mid("m1"),
                "reason": "r",
                "new_entity_type": "person",
                "new_display_name": "Lily",
                "naming_surface": "Lily",
            },
            context,
        )
        assert "not accepted in a self-curation turn" in result
        assert "entity_create" in result

        # entity_create then link by returned key succeeds.
        created = json.loads(
            await _call(
                node,
                "entity_create",
                {"entity_type": "person", "display_name": "Lily", "reason": "r"},
                context,
            )
        )
        key = created["entity_key"]
        linked = await _call(
            node,
            "entity_link_correct",
            {"memory_id": _mid("m1"), "reason": "link it", "add_entity_key": key},
            context,
        )
        assert "Error" not in linked, linked
        assert key in {
            row["entity_key"] for row in service.links_for_memory(_mid("m1"))
        }

    @pytest.mark.asyncio
    async def test_claim38_socket_path_requires_a_live_curation_scope(
        self, tmp_path, store, monkeypatch
    ):
        """Missing/forged/wrong-kind capability all fail closed at the socket."""
        import aiohttp

        import mesh.paths

        monkeypatch.setattr(mesh.paths, "real_home", lambda: tmp_path)
        node = _node(tmp_path, store)
        socket_path = await node._start_tool_socket()
        try:
            connector = aiohttp.UnixConnector(path=socket_path)
            async with aiohttp.ClientSession(connector=connector) as session:
                for capability in ["", "forged-token-1234"]:
                    async with session.post(
                        "http://localhost/tool",
                        json={
                            "name": "entity_create",
                            "arguments": {
                                "entity_type": "person",
                                "display_name": "Lily",
                                "reason": "socket attempt",
                            },
                            "capability": capability,
                        },
                    ) as response:
                        assert response.status == 403, capability
                        payload = await response.json()
                    assert "self-curation" in payload["error"], payload
        finally:
            await node._stop_tool_socket()
        assert (
            store._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0
        )

    def test_claim38b_curation_mutations_route_through_the_agent_socket(self):
        from mesh.mcp_server import AGENT_LOCAL_TOOLS

        for name in SELF_CURATION_MUTATION_TOOLS:
            assert name in AGENT_LOCAL_TOOLS, (
                f"{name} must reach AgentNode's scoped dispatcher instead of a "
                "process-local registry implementation"
            )


# ─────────────────────────────────────────────────────────────────────
# Claims 11-12 — merge
# ─────────────────────────────────────────────────────────────────────


class TestMerge:
    """`retired` + `replacement_key`, one transaction (§3.4)."""

    def test_claim11_merge_moves_links_aliases_and_retires_loser(
        self, store, service
    ):
        winner = _activate(service, "Project Owner")
        loser = _activate(service, "A. Owner")
        store.insert(_entry(_mid("m1")))
        store.insert(_entry(_mid("m2")))
        service.link_memory(_mid("m1"), loser, window_key="w1", activate=True)
        service.link_memory(_mid("m2"), winner, window_key="w1", activate=True)
        service.add_alias(
            loser, "AK", source="test", context=_exec_context(), reason="alias"
        )

        result = service.merge_entities(
            loser, winner, context=_exec_context(), reason="same person"
        )

        # Links moved, memories preserved.
        assert {r["entity_key"] for r in service.links_for_memory(_mid("m1"))} == {winner}
        assert store._conn.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 2
        # Loser retired with replacement_key.
        row = service.get_entity(loser)
        assert row["status"] == "retired"
        assert row["replacement_key"] == winner
        # Aliases copied.
        aliases = {
            r[0]
            for r in store._conn.execute(
                "SELECT display_alias FROM entity_aliases WHERE entity_key = ?", (winner,)
            ).fetchall()
        }
        assert "ak" in {a.lower() for a in aliases}
        # Audit row exists.
        events = [
            r[0]
            for r in store._conn.execute(
                "SELECT event_type FROM entity_events"
            ).fetchall()
        ]
        assert "entity_merged" in events
        assert result["winner"]["entity_key"] == winner
        assert result["loser"]["entity_key"] == loser
        assert result["merged"] is True

    def test_claim11b_loser_dossier_is_retained_as_historical_source(
        self, store, service
    ):
        winner = _activate(service, "Project Owner")
        loser = _activate(service, "A. Owner")
        store.insert(_entry(_mid("m1")))
        service.link_memory(_mid("m1"), loser, window_key="w1", activate=True)
        service.publish_dossier(
            loser,
            body="A. Owner is the primary user. " + _cite("m1"),
            title="A. Owner",
            token_budget=4000,
            measure=estimate_tokens,
            context=_exec_context(),
            reason="seed",
        )
        result = service.merge_entities(
            loser, winner, context=_exec_context(), reason="same person"
        )
        # The only copy of the loser's prose must survive the merge so an
        # interrupted essay_edit cannot destroy it.
        assert store.get_essay(loser) is not None
        assert result.get("loser_dossier")

    def test_claim12_merge_does_not_violate_self_membership_constraint(
        self, store, service
    ):
        group = service.create_pending_entity(
            "group",
            "Fishing crew",
            "rec-fishing regulars",
            origin="self-curation",
            context=_exec_context(),
            reason="setup",
        )["entity_key"]
        _force_active(store, group)
        winner = _activate(service, "Project Owner")
        loser = _activate(service, "Al")
        service.add_group_member(
            group, winner, context=_exec_context(), reason="member"
        )
        service.add_group_member(
            group, loser, context=_exec_context(), reason="member"
        )

        # Both are members; merging loser into winner must drop the duplicate
        # rather than create (group, group) or a duplicate membership row.
        service.merge_entities(
            loser, winner, context=_exec_context(), reason="same person"
        )
        rows = store._conn.execute(
            "SELECT group_key, member_key FROM entity_group_members"
        ).fetchall()
        assert rows == [(group, winner)], rows
        for group_key, member_key in rows:
            assert group_key != member_key

    def test_claim12b_group_merges_only_into_a_group(self, store, service):
        person = _activate(service, "Project Owner")
        group = service.create_pending_entity(
            "group", "Crew", "purpose",
            origin="self-curation", context=_exec_context(), reason="s",
        )["entity_key"]
        _force_active(store, group)
        with pytest.raises(EntityError):
            service.merge_entities(
                group, person, context=_exec_context(), reason="type mismatch"
            )


# ─────────────────────────────────────────────────────────────────────
# Claims 13-16, 21, 24-25, 29-30, 34 — dossier/digest validation
# ─────────────────────────────────────────────────────────────────────


class TestDossierAndDigestValidation:
    """Pre-commit refusal, never truncation (§6)."""

    def _linked(self, store, service, name="Project Owner"):
        key = _activate(service, name)
        store.insert(_entry(_mid("m1")))
        service.link_memory(_mid("m1"), key, window_key="w1", activate=True)
        return key

    def test_claim13_over_budget_dossier_is_refused_not_truncated(
        self, store, service
    ):
        key = self._linked(store, service)
        body = ("word " * 5000) + _cite("m1")
        with pytest.raises(EntityError, match="budget|ceiling|token"):
            service.publish_dossier(
                key,
                body=body,
                title="Project Owner",
                token_budget=100,
                measure=estimate_tokens,
                context=_exec_context(),
                reason="oversize",
            )
        # Nothing written, and certainly nothing truncated.
        assert store.get_essay(key) is None

    @pytest.mark.asyncio
    async def test_claim14_over_budget_digest_edit_is_refused(
        self, tmp_path, store, digest_file
    ):
        # A small explicit ceiling so the refusal is exercised without writing a
        # 32k-token fixture; the gate reads standing_digest_budget_tokens.
        node = _node(
            tmp_path, store, digest=digest_file, standing_digest_budget_tokens=200
        )
        original = digest_file.read_text()
        result = await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": "- " + ("padding " * 4000),
                "reason": "blow the ceiling",
            },
            _curation_context(),
        )
        assert "Error" in result, result
        assert digest_file.read_text() == original, "digest bytes must be unchanged"

    @pytest.mark.asyncio
    async def test_digest_edit_accepts_a_citation_to_a_real_memory(
        self, tmp_path, store, digest_file
    ):
        """A citation to a minted memory must resolve, not be refused.

        Regression for the citation-prefix mismatch: ``extract_citations``
        yields the ``m_<id>`` surface form while ``memories.id`` stores the
        bare hex, so the validator looked up ``m_<id>`` and found nothing.
        Every cited digest write was refused with "unresolvable citation" —
        and since the constitution requires citations, self-curation could not
        write a cited claim at all.
        """
        store.insert(_entry(_mid("m1")))
        node = _node(tmp_path, store, digest=digest_file)
        result = await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": "- 2026-07-30: something happened. " + _cite("m1"),
                "reason": "cite the source memory",
            },
            _curation_context(),
        )
        assert not result.startswith("Error"), result
        assert "unresolvable citation" not in result, result

    @pytest.mark.asyncio
    async def test_digest_edit_still_refuses_a_citation_to_no_memory(
        self, tmp_path, store, digest_file
    ):
        """The prefix fix must not weaken the gate into accepting anything."""
        node = _node(tmp_path, store, digest=digest_file)
        original = digest_file.read_text()
        result = await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": (
                    "- 2026-07-30: something happened. [m_ffffffffffff]"
                ),
                "reason": "cite a memory that does not exist",
            },
            _curation_context(),
        )
        assert "unresolvable citation" in result, result
        assert digest_file.read_text() == original, "digest bytes must be unchanged"

    def test_claim15_unresolvable_citation_aborts_commit(self, store, service):
        key = self._linked(store, service)
        with pytest.raises(EntityError, match="unresolvable citation"):
            service.publish_dossier(
                key,
                body="Project Owner did a thing. [m_deadbeef]",
                title="Project Owner",
                token_budget=4000,
                measure=estimate_tokens,
                context=_exec_context(),
                reason="bad citation",
            )
        assert store.get_essay(key) is None

    def test_claim16_linked_cross_project_citation_is_accepted(self, store, service):
        key = _activate(service, "Mesh Autopilot", entity_type="project")
        foreign = _mid("foreign")
        store.insert(_entry(foreign, project="other-project"))
        service.link_memory(foreign, key, window_key="w1", activate=True)

        body = "Autopilot evidence is cross-project. " + _cite("foreign")
        service.publish_dossier(
            key,
            body=body,
            title="Mesh Autopilot",
            token_budget=4000,
            measure=estimate_tokens,
            context=_exec_context(),
            reason="linked cross-project citation",
        )

        assert store.get_essay(key)["body"] == body

    def test_linked_cross_project_citation_survives_dossier_update(
        self, store, service
    ):
        key = _activate(service, "Mesh Autopilot", entity_type="project")
        foreign = _mid("foreign")
        store.insert(_entry(foreign, project="other-project"))
        service.link_memory(foreign, key, window_key="w1", activate=True)

        original = "Autopilot evidence is cross-project. " + _cite("foreign")
        service.publish_dossier(
            key,
            body=original,
            title="Mesh Autopilot",
            token_budget=4000,
            measure=estimate_tokens,
            context=_exec_context(),
            reason="seed linked cross-project citation",
        )
        updated = "Autopilot evidence remains cross-project. " + _cite("foreign")
        service.publish_dossier(
            key,
            body=updated,
            title="Mesh Autopilot",
            token_budget=4000,
            measure=estimate_tokens,
            context=_exec_context(),
            reason="update with unchanged cross-project citation",
        )

        assert store.get_essay(key)["body"] == updated

    def test_claim16_unlinked_citation_aborts_create_and_update(self, store, service):
        key = self._linked(store, service)
        # m2 exists but is not linked to this entity.
        store.insert(_entry(_mid("m2")))
        with pytest.raises(EntityError, match="out-of-scope citation"):
            service.publish_dossier(
                key,
                body="Project Owner did a thing. " + _cite("m2"),
                title="Project Owner",
                token_budget=4000,
                measure=estimate_tokens,
                context=_exec_context(),
                reason="unlinked citation",
            )
        assert store.get_essay(key) is None

        original = "Project Owner did a thing. " + _cite("m1")
        service.publish_dossier(
            key,
            body=original,
            title="Project Owner",
            token_budget=4000,
            measure=estimate_tokens,
            context=_exec_context(),
            reason="seed linked citation",
        )
        with pytest.raises(EntityError, match="out-of-scope citation"):
            service.publish_dossier(
                key,
                body=original + " Another claim. " + _cite("m2"),
                title="Project Owner",
                token_budget=4000,
                measure=estimate_tokens,
                context=_exec_context(),
                reason="unlinked update citation",
            )
        assert store.get_essay(key)["body"] == original

    def test_claim16b_group_citation_must_be_bridge_evidence(self, store, service):
        group, owner, bob = _group_with_bridge(store, service, windows=3)
        service.activate_group_if_eligible(
            group, context=_exec_context(), reason="gate"
        )
        # A memory linked to only one member is not bridge evidence.
        store.insert(_entry(_mid("solo")))
        service.link_memory(_mid("solo"), owner, window_key="w-solo", activate=True)
        body = (
            "The crew works together. " + _cite("solo") + "\n\n"
            f"{render_roster_block(service.group_members(group))}\n"
        )
        with pytest.raises(EntityError, match="out-of-scope citation"):
            service.publish_dossier(
                group,
                body=body,
                title="Crew",
                token_budget=4000,
                measure=estimate_tokens,
                context=_exec_context(),
                reason="non-bridge citation",
            )

    def test_claim24_verified_hash_tracks_the_body(self, store, service):
        key = self._linked(store, service)
        first = service.publish_dossier(
            key,
            body="Project Owner is the primary user. " + _cite("m1"),
            title="Project Owner",
            token_budget=4000,
            measure=estimate_tokens,
            context=_exec_context(),
            reason="v1",
        )
        entity = service.get_entity(key)
        same = service.publish_dossier(
            key,
            body="Project Owner is the primary user. " + _cite("m1"),
            title="Project Owner",
            token_budget=4000,
            measure=estimate_tokens,
            expected_evidence_version=int(entity["evidence_version"]),
            context=_exec_context(),
            reason="v1 again",
        )
        assert same["verified_hash"] == first["verified_hash"], "stable for same body"
        entity = service.get_entity(key)
        changed = service.publish_dossier(
            key,
            body="Project Owner is the primary user and the project owner. " + _cite("m1"),
            title="Project Owner",
            token_budget=4000,
            measure=estimate_tokens,
            expected_evidence_version=int(entity["evidence_version"]),
            context=_exec_context(),
            reason="v2",
        )
        assert changed["verified_hash"] != first["verified_hash"]

    def test_claim25_bracket_token_aborts_dossier(self, store, service):
        key = self._linked(store, service)
        with pytest.raises(EntityError, match="placeholder|token"):
            service.publish_dossier(
                key,
                body="Project Owner did a thing. [[m_handle_3]]",
                title="Project Owner",
                token_budget=4000,
                measure=estimate_tokens,
                context=_exec_context(),
                reason="unexpanded handle",
            )
        assert store.get_essay(key) is None

    @pytest.mark.asyncio
    async def test_claim25b_digest_edit_removing_a_section_aborts(
        self, tmp_path, store, digest_file
    ):
        node = _node(tmp_path, store, digest=digest_file)
        original = digest_file.read_text()
        result = await _call(
            node,
            "digest_edit",
            {
                "old_text": "## Agent narrative\nI maintain my own state now.\n",
                "new_text": "",
                "reason": "drop a must-keep section",
            },
            _curation_context(),
        )
        assert "Error" in result, result
        assert digest_file.read_text() == original

    @pytest.mark.asyncio
    async def test_claim25c_digest_edit_with_bracket_token_aborts(
        self, tmp_path, store, digest_file
    ):
        node = _node(tmp_path, store, digest=digest_file)
        original = digest_file.read_text()
        result = await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": "- 2026-07-30: something happened. [[m_handle_1]]",
                "reason": "unexpanded handle",
            },
            _curation_context(),
        )
        assert "Error" in result, result
        assert digest_file.read_text() == original

    def test_claim29_concurrent_evidence_version_bump_aborts(self, store, service):
        key = self._linked(store, service)
        entity = service.get_entity(key)
        stale_version = int(entity["evidence_version"])
        # Something else bumps evidence between compose and commit.
        store.insert(_entry(_mid("m2")))
        service.link_memory(_mid("m2"), key, window_key="w2", activate=True)

        with pytest.raises(EntityError, match="evidence_version changed"):
            service.publish_dossier(
                key,
                body="Project Owner is the primary user. " + _cite("m1"),
                title="Project Owner",
                token_budget=4000,
                measure=estimate_tokens,
                expected_evidence_version=stale_version,
                context=_exec_context(),
                reason="stale compose",
            )
        assert store.get_essay(key) is None

    def test_claim30_estimate_tokens_is_the_ruler_no_count_words(self):
        import inspect

        from mesh import agent_node as agent_node_mod
        from mesh.memory import entities as entities_mod

        publish = inspect.getsource(EntityService.publish_dossier)
        assert "count_words" not in publish
        for name in (
            "_execute_curation_essay_edit",
            "_execute_curation_digest_edit",
        ):
            source = inspect.getsource(getattr(AgentNode, name))
            assert "count_words" not in source, name
            assert "estimate_tokens" in source, name

    @pytest.mark.asyncio
    async def test_claim34_shadow_artifact_edits_validate_but_do_not_write(
        self, tmp_path, store, service, digest_file
    ):
        key = self._linked(store, service)
        service.publish_dossier(
            key,
            body="Project Owner is the primary user. " + _cite("m1"),
            title="Project Owner",
            token_budget=4000,
            measure=estimate_tokens,
            context=_exec_context(),
            reason="seed",
        )
        before_essay = store.get_essay(key)["body"]
        before_digest = digest_file.read_text()

        node = _node(tmp_path, store, mode="shadow", digest=digest_file)
        context = _curation_context(mode="shadow")

        essay_result = await _call(
            node,
            "essay_edit",
            {
                "key": key,
                "old_text": "primary user",
                "new_text": "primary user and project owner",
                "reason": "shadow dossier update",
            },
            context,
        )
        payload = json.loads(essay_result)
        assert payload["would_apply"] is True and payload["applied"] is False

        digest_result = await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": "- 2026-07-30: something specific happened.",
                "reason": "shadow digest update",
            },
            context,
        )
        assert "Error" not in digest_result, digest_result

        assert store.get_essay(key)["body"] == before_essay
        assert digest_file.read_text() == before_digest
        events = [
            r[0]
            for r in store._conn.execute(
                "SELECT event_type FROM entity_events"
            ).fetchall()
        ]
        assert any(e.endswith("_shadow") for e in events), events

    @pytest.mark.asyncio
    async def test_claim41_shadow_overlay_composes_within_one_turn(
        self, tmp_path, store, service, digest_file
    ):
        """create -> link, then two successive exact-string edits, all shadow."""
        store.insert(_entry(_mid("m1")))
        node = _node(tmp_path, store, mode="shadow", digest=digest_file)
        context = _curation_context(mode="shadow")
        before_digest = digest_file.read_text()

        created = json.loads(
            await _call(
                node,
                "entity_create",
                {"entity_type": "person", "display_name": "Lily", "reason": "r"},
                context,
            )
        )
        provisional = created["entity_key"]
        assert "shadow" in provisional, provisional
        assert provisional in context.entities

        # A following shadow link accepts the provisional key.
        linked = await _call(
            node,
            "entity_link_correct",
            {"memory_id": _mid("m1"), "reason": "link", "add_entity_key": provisional},
            context,
        )
        assert "Error" not in linked, linked

        # Two successive digest edits compose against the prior candidate.
        first = await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: something happened.",
                "new_text": "- 2026-07-30: first edit.",
                "reason": "edit one",
            },
            context,
        )
        assert "Error" not in first, first
        second = await _call(
            node,
            "digest_edit",
            {
                "old_text": "- 2026-07-30: first edit.",
                "new_text": "- 2026-07-30: second edit.",
                "reason": "edit two",
            },
            context,
        )
        assert "Error" not in second, second
        assert context.digest_text is not None
        assert "second edit" in context.digest_text

        # Authoritative state untouched after the turn.
        assert digest_file.read_text() == before_digest
        assert (
            store._conn.execute(
                "SELECT COUNT(*) FROM entities WHERE status <> 'retired'"
            ).fetchone()[0]
            == 0
        )
        assert len(context.intents) >= 2


# ─────────────────────────────────────────────────────────────────────
# Claims 17-23 — groups
# ─────────────────────────────────────────────────────────────────────


def _group_with_bridge(store, service, *, windows: int, members: int = 2):
    """Build an active-eligible group with `windows` distinct bridge windows."""
    group = service.create_pending_entity(
        "group",
        "Fishing crew",
        "rec-fishing regulars",
        origin="self-curation",
        context=_exec_context(),
        reason="setup",
    )["entity_key"]
    owner = _activate(service, "Project Owner")
    bob = _activate(service, "Bob")
    keys = [owner, bob][:members]
    for key in keys:
        service.add_group_member(
            group, key, context=_exec_context(), reason="member"
        )
    for index in range(windows):
        memory_id = _mid(f"bridge{index}")
        store.insert(_entry(memory_id))
        for key in keys:
            service.link_memory(
                memory_id, key, window_key=f"w{index}", activate=True
            )
    return group, owner, bob


class TestGroups:
    """Deterministic activation and the protected roster (§5)."""

    def test_claim17_activation_requires_two_active_members(self, store, service):
        group, owner, _bob = _group_with_bridge(
            store, service, windows=3, members=1
        )
        report = service.activate_group_if_eligible(
            group, context=_exec_context(), reason="gate"
        )
        assert report["activated"] is False
        assert service.get_entity(group)["status"] != "active"
        assert service.active_group_member_count(group) == 1

    def test_claim18_activation_requires_the_window_threshold(
        self, store, service
    ):
        group, _alan, _bob = _group_with_bridge(store, service, windows=2)
        report = service.activate_group_if_eligible(
            group, context=_exec_context(), reason="gate"
        )
        assert report["activated"] is False, report
        assert len(service.group_bridge_windows(group)) == 2

        # A third distinct window tips it over.
        store.insert(_entry(_mid("bridge2")))
        for key in (_alan, _bob):
            service.link_memory(_mid("bridge2"), key, window_key="w2", activate=True)
        report = service.activate_group_if_eligible(
            group, context=_exec_context(), reason="gate"
        )
        assert report["activated"] is True, report
        assert service.get_entity(group)["status"] == "active"

    def test_claim19_single_member_memory_is_not_bridge_evidence(
        self, store, service
    ):
        group, owner, _bob = _group_with_bridge(store, service, windows=1)
        store.insert(_entry(_mid("solo")))
        service.link_memory(_mid("solo"), owner, window_key="w-solo", activate=True)
        windows = service.group_bridge_windows(group)
        assert "w-solo" not in windows, windows
        assert "solo" not in service.group_bridge_memory_ids(group)

    @pytest.mark.asyncio
    async def test_claim20_essay_edit_touching_the_roster_is_rejected(
        self, tmp_path, store, service
    ):
        group, _alan, _bob = _group_with_bridge(store, service, windows=3)
        service.activate_group_if_eligible(
            group, context=_exec_context(), reason="gate"
        )
        roster = render_roster_block(service.group_members(group))
        body = (
            f'The crew fishes together. {_cite("bridge0")}\n\n{roster}\n'
        )
        service.publish_dossier(
            group,
            body=body,
            title="Fishing crew",
            token_budget=4000,
            measure=estimate_tokens,
            context=_exec_context(),
            reason="seed",
        )

        node = _node(
            tmp_path, store, entity_self_curation_groups_enabled=True
        )
        context = _curation_context(groups_enabled=True)
        first_member = service.group_members(group)[0]["member_key"]
        result = await _call(
            node,
            "essay_edit",
            {
                "key": group,
                "old_text": f"- {first_member}",
                "new_text": "- person:impostor  Impostor  — ",
                "reason": "prose membership change",
            },
            context,
        )
        assert "roster" in result.lower() and "Error" in result, result
        assert store.get_essay(group)["body"] == body

    def test_claim21_roster_mismatch_at_commit_aborts(self, store, service):
        group, _alan, _bob = _group_with_bridge(store, service, windows=3)
        service.activate_group_if_eligible(
            group, context=_exec_context(), reason="gate"
        )
        stale_roster = render_roster_block(
            [{"member_key": "person:ghost", "display_name": "Ghost", "role": ""}]
        )
        body = (
            f'The crew fishes together. {_cite("bridge0")}\n\n{stale_roster}\n'
        )
        with pytest.raises(EntityError, match="roster"):
            service.publish_dossier(
                group,
                body=body,
                title="Fishing crew",
                token_budget=4000,
                measure=estimate_tokens,
                context=_exec_context(),
                reason="stale roster",
            )
        assert store.get_essay(group) is None

    def test_claim22_retired_member_row_is_rewritten_to_replacement(
        self, store, service
    ):
        group, owner, bob = _group_with_bridge(store, service, windows=3)
        service.activate_group_if_eligible(
            group, context=_exec_context(), reason="gate"
        )
        winner = _activate(service, "Project Owner")
        service.merge_entities(
            owner, winner, context=_exec_context(), reason="same person"
        )
        service.reconcile_group_membership(
            group, context=_exec_context(), reason="post-merge"
        )
        members = {m["member_key"] for m in service.group_members(group)}
        assert owner not in members
        assert winner in members and bob in members

    def test_claim23_dropping_to_one_member_reports_degraded_not_retired(
        self, store, service
    ):
        group, owner, bob = _group_with_bridge(store, service, windows=3)
        service.activate_group_if_eligible(
            group, context=_exec_context(), reason="gate"
        )
        service.remove_group_member(
            group, bob, context=_exec_context(), reason="left the crew"
        )
        report = service.reconcile_group_membership(
            group, context=_exec_context(), reason="reconcile"
        )
        assert report["degraded"] is True, report
        # Silent deletion is exactly what the 2026-07-06 audit fixed.
        assert service.get_entity(group)["status"] == "active"


# ─────────────────────────────────────────────────────────────────────
# Supporting contracts: prompt rendering and tool scoping
# ─────────────────────────────────────────────────────────────────────


class TestInstructionAndScopes:
    def test_phase1_allowlist_omits_group_tools(self):
        phase1 = curation_tool_names(groups_enabled=False)
        for name in SELF_CURATION_GROUP_TOOLS:
            assert name not in phase1
        phase2 = curation_tool_names(groups_enabled=True)
        assert SELF_CURATION_GROUP_TOOLS <= phase2
        assert SELF_CURATION_READ_TOOLS <= phase1

    def test_update_instruction_leaves_no_unfilled_anchor(self):
        from mesh.memory.curation import (
            load_update_template,
            render_update_instruction,
        )

        rendered = render_update_instruction(
            load_update_template(),
            batch_block="m_m1  retrieval key  [entities: person:owner]",
            registry_block="person:owner  person  Project Owner  — primary user",
            budgets_block="Standing digest: 100 / 32000 tokens",
            groups_block="",
        )
        assert find_bracket_tokens(rendered) == [], (
            "an unreplaced [[ANCHOR]] would reach the model verbatim"
        )
        assert "Do not call" in rendered and "send_message" in rendered
        # The constitution's seven sections must be present verbatim.
        for section in (
            "Timeline",
            "Narrative",
            "Projects",
            "People",
            "Standing",
            "Open threads",
            "Agent narrative",
        ):
            assert section in rendered, section
        assert "GRADUATE" in rendered

    def test_digest_section_errors_detects_missing_and_empty(self):
        assert digest_section_errors(DIGEST_SECTIONS) == []
        missing = digest_section_errors(
            DIGEST_SECTIONS.replace(
                "## Agent narrative\nI maintain my own state now.\n", ""
            )
        )
        assert missing, "a removed must-keep section must be reported"

    def test_roster_block_round_trips(self):
        members = [
            {"member_key": "person:owner", "display_name": "Project Owner", "role": "lead"},
            {"member_key": "person:bob", "display_name": "Bob", "role": ""},
        ]
        block = render_roster_block(members)
        assert roster_block_of(f"prose\n\n{block}\n") == block
        # One canonical renderer: order is by member_key, not insertion order.
        assert render_roster_block(list(reversed(members))) == block

    def test_citation_extraction_is_first_seen_order(self):
        body = "a [m_beef] b [m_cafe] c [m_beef]"
        assert extract_citations(body) == ["beef", "cafe"] or extract_citations(
            body
        ) == ["m_beef", "m_cafe"]


class TestVerifierRepairs:
    """Regression coverage for concrete gaps found by independent verification."""

    def test_agent_config_reaches_router_curation_runtime(self, tmp_path, store):
        node = _node(
            tmp_path,
            store,
            entity_self_curation_groups_enabled=True,
            standing_digest_budget_tokens=12345,
            essay_token_budget=2345,
            entity_activation_window_threshold=4,
            entity_registry_injection_cap=77,
            curation_stale_group_batches=9,
            curation_failure_alert_threshold=2,
        )
        config = node._router_v2_config
        assert config.entity_self_curation_groups_enabled is True
        assert config.entity_self_curation_mode == "write"
        assert config.standing_digest_budget_tokens == 12345
        assert config.essay_token_budget == 2345
        assert config.entity_activation_window_threshold == 4
        assert config.entity_registry_injection_cap == 77
        assert config.curation_stale_group_batches == 9
        assert config.curation_failure_alert_threshold == 2

    def test_invalid_enrollment_aborts_instead_of_silently_disabling(
        self, tmp_path, store
    ):
        node = _node(tmp_path, store)
        with pytest.raises(ValueError, match="invalid entity self-curation"):
            node._register_curation_callback()

    @pytest.mark.asyncio
    async def test_lifecycle_rows_use_first_class_event_types(self, store):
        router = await _router(memory_system=_memory_system(store))
        batch = CurationBatch(
            "time-based", (_mid("lifecycle"),), formed_at="now"
        )
        router._record_curation_failure(batch, RuntimeError("boom"))
        router._record_curation_turn(batch, batch.turn_id(router._node_id), "")
        events = [
            row[0]
            for row in store._conn.execute(
                "SELECT event_type FROM entity_events ORDER BY sequence"
            ).fetchall()
        ]
        assert events == ["curation_turn_failed", "curation_turn"]

    def test_group_reconciliation_republishes_protected_roster(
        self, store, service
    ):
        group, owner, _bob = _group_with_bridge(store, service, windows=3)
        service.activate_group_if_eligible(
            group, context=_exec_context(), reason="gate"
        )
        roster = render_roster_block(service.group_members(group))
        body = f'The crew fishes together. {_cite("bridge0")}\n\n{roster}\n'
        service.publish_dossier(
            group,
            body=body,
            title="Fishing crew",
            token_budget=4000,
            measure=estimate_tokens,
            context=_exec_context(),
            reason="seed",
        )

        service.add_group_member(
            group,
            owner,
            role="captain",
            source="self-curation",
            context=_exec_context(),
            reason="role correction",
        )
        assert store.get_essay(group)["verified_hash"] == ""
        report = service.reconcile_group_membership(
            group,
            context=_exec_context(),
            reason="reconcile",
            token_budget=4000,
            measure=estimate_tokens,
        )
        assert report["roster_reconciled"] is True, report
        essay = store.get_essay(group)
        assert roster_block_of(essay["body"]) == render_roster_block(
            service.group_members(group)
        )
        assert essay["verified_hash"]

    @pytest.mark.asyncio
    async def test_shadow_group_overlay_composes_after_provisional_create(
        self, tmp_path, store, service
    ):
        member = _activate(service, "Project Owner")
        node = _node(
            tmp_path,
            store,
            mode="shadow",
            entity_self_curation_groups_enabled=True,
        )
        context = _curation_context(
            mode="shadow", groups_enabled=True
        )
        created = json.loads(await _call(
            node,
            "entity_group_create",
            {
                "display_name": "Fishing crew",
                "purpose": "People who fish together",
                "reason": "relationship evidence",
            },
            context,
        ))
        group = created["entity_key"]
        result = await _call(
            node,
            "entity_group_member_add",
            {
                "group_key": group,
                "member_key": member,
                "reason": "member evidence",
            },
            context,
        )
        assert "Error" not in result, result
        assert [row["member_key"] for row in context.group_members[group]] == [
            member
        ]
        assert service.get_entity(group) is None

    @pytest.mark.asyncio
    async def test_shadow_entity_edit_runs_write_path_validation(
        self, tmp_path, store, service
    ):
        entity = _activate(service, "Project Owner")
        node = _node(tmp_path, store, mode="shadow")
        context = _curation_context(mode="shadow")
        result = await _call(
            node,
            "entity_edit",
            {
                "entity_key": entity,
                "operation": "update_details",
                "display_name": "!!!",
                "reason": "invalid candidate",
            },
            context,
        )
        assert "Error" in result and "slug" in result
        assert service.get_entity(entity)["display_name"] == "Project Owner"

    def test_pending_group_with_old_nonzero_evidence_becomes_stale(
        self, store, service
    ):
        group, _alan, _bob = _group_with_bridge(store, service, windows=1)
        router_config = RouterV2Config(
            curation_stale_group_batches=2,
            entity_activation_window_threshold=3,
        )

        async def worker_fn(*_args, **_kwargs):
            return None

        async def send_fn(*_args, **_kwargs):
            return None

        router = RouterV2(
            worker_fn,
            send_fn,
            config=router_config,
            node_id="agent:test:curator",
            memory_system=_memory_system(store),
        )
        router._curation_turn_sequence = 1
        assert group not in router._curation_group_reports(service)[1]
        router._curation_turn_sequence = 2
        assert group not in router._curation_group_reports(service)[1]
        router._curation_turn_sequence = 3
        assert group in router._curation_group_reports(service)[1]

    def test_shadow_pre_turn_group_checks_do_not_activate_live_rows(
        self, store, service
    ):
        group, _alan, _bob = _group_with_bridge(store, service, windows=3)
        router = RouterV2(
            lambda *_args, **_kwargs: None,
            lambda *_args, **_kwargs: None,
            config=RouterV2Config(
                entity_self_curation_mode="shadow",
                entity_self_curation_groups_enabled=True,
                entity_activation_window_threshold=3,
            ),
            node_id="agent:test:curator",
            memory_system=_memory_system(store),
        )
        router._curation_turn_sequence = 1
        router._curation_group_reports(service)
        assert service.get_entity(group)["status"] == "pending"


# ─────────────────────────────────────────────────────────────────────
# Phase 3 — agent-driven backfill (§9, "Phase 3")
# ─────────────────────────────────────────────────────────────────────


def _backfill_entry(
    memory_id: str,
    created_at: str,
    *,
    formation_source: str = "live-extraction",
) -> MemoryEntry:
    """A persistable memory row with an explicit oldest-first timestamp."""
    entry = _entry(memory_id)
    entry.created_at = datetime.fromisoformat(created_at)
    entry.formation_source = formation_source
    return entry


def _seed_memories(store, count: int, *, start_day: int = 1, **kwargs) -> list[str]:
    ids: list[str] = []
    for index in range(count):
        memory_id = _mid(f"backfill-{index}")
        store.insert(
            _backfill_entry(
                memory_id,
                f"2026-06-{start_day + index:02d}T00:00:00+00:00",
                **kwargs,
            )
        )
        ids.append(memory_id)
    store._conn.commit()
    return ids


def _mark_curated(
    store,
    memory_ids,
    *,
    event_type: str = "curation_turn",
    mode: str | None = None,
) -> None:
    """Write the audit row a completed curation turn leaves behind."""
    service = EntityService(
        store._conn,
        actor_node="agent:test:curator",
        mutations_enabled=True,
    )
    service.record_curation_event(
        event_type,
        reason="test marker",
        run_key=f"curation-test-{event_type}-{memory_ids[0]}",
        details={
            "memory_ids": list(memory_ids),
            **({"mode": mode} if mode is not None else {}),
        },
    )


async def _backfill_router(store, **overrides) -> RouterV2:
    router_kwargs = {
        key: overrides.pop(key)
        for key in list(overrides)
        if key not in RouterV2Config.__dataclass_fields__
    }
    config_kwargs = dict(
        entity_self_curation_mode="write",
        entity_self_curation_backfill_slice_size=3,
        entity_self_curation_backfill_max_batches=50,
    )
    config_kwargs.update(overrides)

    async def worker_fn(*_args, **_kwargs):
        return None

    async def send_fn(*_args, **_kwargs):
        return None

    return RouterV2(
        worker_fn,
        send_fn,
        config=RouterV2Config(**config_kwargs),
        node_id="agent:test:curator",
        memory_system=_memory_system(store),
        **router_kwargs,
    )


class TestPhase3Backfill:
    """The bounded oldest-first slicer and its trigger (§9, Phase 3)."""

    # ── The pure slicer ──────────────────────────────────────────────

    def test_slicer_chunks_oldest_first_and_stamps_the_backfill_reason(self):
        from mesh.memory.curation import (
            CURATION_BACKFILL_REASON,
            slice_backfill_batches,
        )

        rows = [(f"m{i}", f"2026-06-{i + 1:02d}") for i in range(7)]
        batches = slice_backfill_batches(
            rows, curated_ids=(), slice_size=3, max_batches=50
        )
        assert [b.memory_ids for b in batches] == [
            ("m0", "m1", "m2"), ("m3", "m4", "m5"), ("m6",),
        ]
        assert {b.reason for b in batches} == {CURATION_BACKFILL_REASON}
        # formed_at is the slice's oldest row, so turn_id is stable + distinct.
        assert [b.formed_at for b in batches] == [
            "2026-06-01", "2026-06-04", "2026-06-07",
        ]
        assert len({b.turn_id("agent:test:curator") for b in batches}) == 3

    def test_slicer_skips_an_already_backfilled_prefix_so_runs_resume(self):
        """A curated *leading* run is skipped, not a stop — else run 2 is a no-op."""
        from mesh.memory.curation import slice_backfill_batches

        rows = [(f"m{i}", "2026-06-01") for i in range(6)]
        batches = slice_backfill_batches(
            rows, curated_ids={"m0", "m1", "m2"}, slice_size=2, max_batches=50
        )
        assert [b.memory_ids for b in batches] == [("m3", "m4"), ("m5",)]

    def test_slicer_stops_at_the_curated_frontier(self):
        from mesh.memory.curation import slice_backfill_batches

        rows = [(f"m{i}", "2026-06-01") for i in range(6)]
        batches = slice_backfill_batches(
            rows, curated_ids={"m3"}, slice_size=10, max_batches=50
        )
        assert [b.memory_ids for b in batches] == [("m0", "m1", "m2")], (
            "the walk must stop at the frontier, never leapfrog it"
        )

    def test_slicer_respects_max_batches(self):
        from mesh.memory.curation import slice_backfill_batches

        rows = [(f"m{i}", "2026-06-01") for i in range(100)]
        batches = slice_backfill_batches(
            rows, curated_ids=(), slice_size=4, max_batches=3
        )
        assert len(batches) == 3
        assert sum(len(b.memory_ids) for b in batches) == 12, (
            "the remainder is left for the next invocation, not curated now"
        )

    def test_slicer_empty_walk_is_a_no_op(self):
        from mesh.memory.curation import slice_backfill_batches

        assert slice_backfill_batches([], curated_ids=(), slice_size=3,
                                      max_batches=5) == []
        assert slice_backfill_batches(
            [("m0", "x")], curated_ids={"m0"}, slice_size=3, max_batches=5
        ) == []

    # ── Router-level planning against a real store ───────────────────

    @pytest.mark.asyncio
    async def test_backfill_finds_uncurated_batches(self, store):
        ids = _seed_memories(store, 7)
        router = await _backfill_router(store)
        batches = router.plan_curation_backfill()
        assert [list(b.memory_ids) for b in batches] == [
            ids[0:3], ids[3:6], ids[6:7],
        ]

    @pytest.mark.asyncio
    async def test_backfill_reads_curated_ids_from_curation_turn_events(self, store):
        ids = _seed_memories(store, 6)
        _mark_curated(store, ids[4:])
        router = await _backfill_router(store)
        assert router._curated_memory_ids() == set(ids[4:])
        assert [list(b.memory_ids) for b in router.plan_curation_backfill()] == [
            ids[0:3], ids[3:4],
        ]

    @pytest.mark.asyncio
    async def test_backfill_treats_a_failed_turn_as_uncurated(self, store):
        """A failed turn leaves repair work; §10.2 names backfill as the path."""
        ids = _seed_memories(store, 3)
        _mark_curated(store, ids, event_type="curation_turn_failed")
        router = await _backfill_router(store)
        assert router._curated_memory_ids() == set()
        assert [list(b.memory_ids) for b in router.plan_curation_backfill()] == [ids]

    @pytest.mark.asyncio
    async def test_write_backfill_revisits_shadow_only_batches(self, store):
        """A dry-run marker must not block later authoritative application."""
        ids = _seed_memories(store, 3)
        _mark_curated(store, ids, mode="shadow")

        shadow = await _backfill_router(
            store, entity_self_curation_mode="shadow",
        )
        assert shadow.plan_curation_backfill() == []

        write = await _backfill_router(
            store, entity_self_curation_mode="write",
        )
        assert [list(batch.memory_ids) for batch in write.plan_curation_backfill()] == [
            ids
        ]

    @pytest.mark.asyncio
    async def test_curation_turn_records_the_mode_used_for_backfill_markers(self, store):
        ids = _seed_memories(store, 2)
        router = await _backfill_router(
            store, entity_self_curation_mode="shadow",
        )
        batch = CurationBatch(reason="backfill", memory_ids=tuple(ids))
        router._record_curation_turn(batch, "curation-shadow-test", "")
        payload = store._conn.execute(
            "SELECT details_json FROM entity_events "
            "WHERE run_key = 'curation-shadow-test'"
        ).fetchone()[0]
        assert json.loads(payload)["mode"] == "shadow"

    @pytest.mark.asyncio
    async def test_backfill_excludes_parse_failure_placeholders(self, store):
        good = _seed_memories(store, 2)
        store.insert(
            _backfill_entry(
                _mid("fallback"),
                "2026-06-09T00:00:00+00:00",
                formation_source="live-extraction-fallback",
            )
        )
        store._conn.commit()
        router = await _backfill_router(store)
        rows = [row[0] for row in router._backfill_candidate_rows()]
        assert rows == good, "claim 40: a fallback record is not curation input"

    @pytest.mark.asyncio
    async def test_backfill_respects_max_batches_limit(self, store):
        ids = _seed_memories(store, 12)
        router = await _backfill_router(store)
        assert len(router.plan_curation_backfill(2)) == 2
        # The configured ceiling clamps a larger explicit request.
        router._config.entity_self_curation_backfill_max_batches = 2
        assert len(router.plan_curation_backfill(99)) == 2
        assert sum(
            len(b.memory_ids) for b in router.plan_curation_backfill(99)
        ) == 6 < len(ids)

    @pytest.mark.asyncio
    async def test_backfill_stops_at_already_curated_marker(self, store):
        ids = _seed_memories(store, 9)
        _mark_curated(store, [ids[4]])
        router = await _backfill_router(store)
        queued = [
            mid
            for batch in router.plan_curation_backfill()
            for mid in batch.memory_ids
        ]
        assert queued == ids[0:4]
        assert ids[5] not in queued

    # ── Enqueue semantics: the Phase 1 FIFO, unchanged ───────────────

    @pytest.mark.asyncio
    async def test_backfill_enqueues_onto_the_existing_curation_queue(self, store):
        ids = _seed_memories(store, 5)
        router = await _backfill_router(store)
        ran: list[tuple[str, ...]] = []

        async def fake_turn(batch):
            ran.append(batch.memory_ids)

        router._run_curation_turn = fake_turn
        result = router.enqueue_curation_backfill()
        assert result["status"] == "queued"
        assert result["queued"] == 2 and result["memory_ids"] == 5
        assert len(result["turn_ids"]) == 2

        assert await router.wait_for_curation_idle(timeout=5.0)
        assert ran == [tuple(ids[0:3]), tuple(ids[3:5])], "FIFO, oldest-first"
        assert router.curation_status()["curation_backfill_slices_queued"] == 2
        await router.shutdown_curation(timeout=5.0)

    @pytest.mark.asyncio
    async def test_backfill_does_not_duplicate_a_queued_live_batch(self, store):
        """Startup backfill must not requeue the startup formation batch."""
        ids = _seed_memories(store, 5)
        router = await _backfill_router(store)
        await router._router_turn_lock.acquire()
        try:
            assert router.enqueue_curation_batch(
                CurationBatch(
                    reason="startup",
                    memory_ids=tuple(ids[3:]),
                    formed_at="2026-06-05T00:00:00+00:00",
                )
            )
            result = router.enqueue_curation_backfill()
            assert result["memory_ids"] == 3
            queued = list(router._curation_queue._queue)
            assert [tuple(batch.memory_ids) for batch in queued] == [
                tuple(ids[3:]),
                tuple(ids[:3]),
            ]
        finally:
            router._router_turn_lock.release()
        assert await router.wait_for_curation_idle(timeout=5.0)
        await router.shutdown_curation(timeout=5.0)

    @pytest.mark.asyncio
    async def test_repeated_trigger_does_not_duplicate_pending_backfill(self, store):
        ids = _seed_memories(store, 5)
        router = await _backfill_router(store)
        await router._router_turn_lock.acquire()
        try:
            first = router.enqueue_curation_backfill()
            second = router.enqueue_curation_backfill()
            assert first["memory_ids"] == len(ids)
            assert second["status"] == "empty"
            assert router._curation_queue.qsize() == 2
        finally:
            router._router_turn_lock.release()
        assert await router.wait_for_curation_idle(timeout=5.0)
        await router.shutdown_curation(timeout=5.0)

    @pytest.mark.asyncio
    async def test_backfill_runs_alongside_a_held_router_turn(self, store):
        """Backfill is a batch of internal curation turns — now independent.

        Previously asserted the inverse ("the lock still rules").  Backfill can
        be minutes of LLM work, which is exactly the load that must not block
        an incoming message.
        """
        _seed_memories(store, 3)
        router = await _backfill_router(store)
        ran: list[str] = []
        done = asyncio.Event()

        async def fake_turn(batch):
            ran.append(batch.reason)
            done.set()

        router._run_curation_turn = fake_turn
        await router._router_turn_lock.acquire()
        try:
            # Synchronous, non-awaiting: safe to call from inside a router turn.
            assert router.enqueue_curation_backfill()["queued"] == 1
            await asyncio.wait_for(done.wait(), timeout=5.0)
            assert ran == ["backfill"], (
                "backfill must drain without waiting for the message turn"
            )
        finally:
            router._router_turn_lock.release()
        assert await router.wait_for_curation_idle(timeout=5.0)
        await router.shutdown_curation(timeout=5.0)

    @pytest.mark.asyncio
    async def test_backfill_empty_queue_is_a_no_op(self, store):
        router = await _backfill_router(store)
        result = router.enqueue_curation_backfill()
        assert result["status"] == "empty"
        assert result["queued"] == 0
        assert router._curation_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_backfill_is_disabled_when_curation_is_off(self, store):
        _seed_memories(store, 5)
        router = await _backfill_router(store, entity_self_curation_mode="off")
        result = router.enqueue_curation_backfill()
        assert result["status"] == "disabled"
        assert result["queued"] == 0 and router._curation_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_backfill_in_shadow_mode_stays_shadow(self, store):
        """Mode is carried by the shared pipeline, not re-decided per batch."""
        ids = _seed_memories(store, 3)
        planner = await _backfill_router(
            store, entity_self_curation_mode="shadow",
        )
        result = planner.enqueue_curation_backfill()
        assert result["mode"] == "shadow" and result["queued"] == 1

        # The queued slice runs through the ordinary curation turn, so it
        # inherits the turn's shadow authority instead of re-deciding it.
        batch = planner._curation_queue.get_nowait()
        assert batch.reason == "backfill" and list(batch.memory_ids) == ids

        captured: dict = {}

        async def router_process_fn(**kwargs):
            captured.update(kwargs)
            return "nothing to do"

        runner = await _router(router_process_fn=router_process_fn)
        runner._config.entity_self_curation_mode = "shadow"
        runner._curation_entity_service = lambda: None
        runner._render_curation_instruction = lambda b: ("INSTRUCTION", "BATCH")
        await runner._run_curation_turn(batch)

        assert captured.get("execution_scope_kind") == "curation"
        assert captured.get("internal_turn") is True
        assert captured["trigger_msg"].metadata["curation_reason"] == "backfill"
        assert "send_message" not in captured["tool_names"]

    @pytest.mark.asyncio
    async def test_shadow_backfill_reports_would_apply_intents(self, tmp_path, store):
        """A shadow slice's mutations validate and log, changing nothing."""
        node = _node(tmp_path, store, mode="shadow")
        context = _curation_context(mode="shadow")
        context.batch = CurationBatch(
            reason="backfill", memory_ids=tuple(_seed_memories(store, 2)),
        )
        result = await _call(
            node,
            "entity_create",
            {
                "entity_type": "person",
                "display_name": "Backfilled Person",
                "reason": "old history mentions them repeatedly",
            },
            context,
        )
        assert "would" in result.lower() or "shadow" in result.lower()
        assert store._conn.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()[0] == 0, "shadow backfill commits nothing"
        assert context.intents and context.entities
        shadow_events = store._conn.execute(
            "SELECT COUNT(*) FROM entity_events WHERE event_type LIKE '%_shadow'"
        ).fetchone()[0]
        assert shadow_events >= 1

    # ── The entity_backfill tool ─────────────────────────────────────

    def test_entity_backfill_tool_is_registered_with_optional_max_batches(self):
        import mesh.tool_implementations  # noqa: F401  (registers the tool)

        schema = get_registry().get("entity_backfill")
        assert schema is not None
        params = {p.name: p for p in schema.parameters}
        assert set(params) == {"max_batches"}
        assert params["max_batches"].required is False

    def test_entity_backfill_is_never_offered_inside_a_curation_turn(self):
        assert "entity_backfill" not in curation_tool_names()
        assert "entity_backfill" not in curation_tool_names(groups_enabled=True)
        assert "entity_backfill" not in SELF_CURATION_MUTATION_TOOLS

    def test_entity_backfill_is_enabled_only_for_an_enrolled_agent(self, tmp_path, store):
        enrolled = _node(tmp_path, store, mode="write")
        assert "entity_backfill" in enrolled.enabled_tools
        plain = _node(
            tmp_path, store, mode="write", entity_self_curation_enabled=False,
        )
        assert "entity_backfill" not in plain.enabled_tools

    @pytest.mark.asyncio
    async def test_entity_backfill_tool_returns_the_expected_shape(
        self, tmp_path, store
    ):
        ids = _seed_memories(store, 5)
        node = _node(tmp_path, store, mode="write")
        node._router_v2 = await _backfill_router(store)
        node._router_v2._run_curation_turn = lambda batch: asyncio.sleep(0)

        raw = await node._execute_entity_tool("entity_backfill", {}, _trigger())
        payload = json.loads(raw)
        assert payload["status"] == "queued"
        assert payload["mode"] == "write"
        assert payload["queued"] == 2
        assert payload["memory_ids"] == len(ids)
        assert payload["slice_size"] == 3
        assert len(payload["turn_ids"]) == 2
        assert await node._router_v2.wait_for_curation_idle(timeout=5.0)
        await node._router_v2.shutdown_curation(timeout=5.0)

    @pytest.mark.asyncio
    async def test_entity_backfill_tool_honors_max_batches(self, tmp_path, store):
        _seed_memories(store, 12)
        node = _node(tmp_path, store, mode="write")
        node._router_v2 = await _backfill_router(store)
        node._router_v2._run_curation_turn = lambda batch: asyncio.sleep(0)

        payload = json.loads(
            await node._execute_entity_tool(
                "entity_backfill", {"max_batches": 2}, _trigger()
            )
        )
        assert payload["queued"] == 2 and payload["memory_ids"] == 6
        for bad, expected in ((0, "at least 1"), ("many", "must be an integer")):
            result = await node._execute_entity_tool(
                "entity_backfill", {"max_batches": bad}, _trigger()
            )
            assert result.startswith("Error") and expected in result
        assert await node._router_v2.wait_for_curation_idle(timeout=5.0)
        await node._router_v2.shutdown_curation(timeout=5.0)

    @pytest.mark.asyncio
    async def test_entity_backfill_is_refused_inside_a_curation_turn(
        self, tmp_path, store
    ):
        _seed_memories(store, 3)
        node = _node(tmp_path, store, mode="write")
        node._router_v2 = await _backfill_router(store)
        result = await _call(node, "entity_backfill", {}, _curation_context())
        assert result.startswith("Error") and "self-curation turn" in result
        assert node._router_v2._curation_queue.qsize() == 0, (
            "a curation turn must never be able to schedule further turns"
        )

    @pytest.mark.asyncio
    async def test_entity_backfill_reports_disabled_and_unavailable(
        self, tmp_path, store
    ):
        _seed_memories(store, 3)
        off = _node(tmp_path, store, mode="off", entity_self_curation_mode="off")
        payload = json.loads(
            await off._execute_entity_tool("entity_backfill", {}, _trigger())
        )
        assert payload["status"] == "disabled" and payload["queued"] == 0

        no_router = _node(tmp_path, store, mode="write")
        no_router._router_v2 = None
        payload = json.loads(
            await no_router._execute_entity_tool("entity_backfill", {}, _trigger())
        )
        assert payload["status"] == "unavailable" and payload["queued"] == 0

    def test_entity_backfill_registry_stub_fails_closed(self):
        from mesh.tool_implementations import entity_backfill

        result = asyncio.run(entity_backfill())
        assert result.startswith("Error") and "in-process" in result

    def test_entity_backfill_routes_through_the_agent_socket(self):
        from mesh.mcp_server import AGENT_LOCAL_TOOLS

        assert "entity_backfill" in AGENT_LOCAL_TOOLS

    # ── Startup trigger ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_startup_backfill_fires_after_the_formation_chain(
        self, tmp_path, store
    ):
        import inspect

        source = inspect.getsource(AgentNode._v3_startup_formation_chain)
        assert "_maybe_backfill_on_startup" in source
        assert source.index("form_un_formed") < source.index(
            "_maybe_backfill_on_startup"
        ), "backfill must run after the startup formation batch is enqueued"

        _seed_memories(store, 4)
        node = _node(tmp_path, store, mode="write")
        node._router_v2 = await _backfill_router(store)
        node._router_v2._run_curation_turn = lambda batch: asyncio.sleep(0)
        await node._maybe_backfill_on_startup()
        assert node._router_v2.curation_status()["curation_backfill_runs"] == 1
        assert await node._router_v2.wait_for_curation_idle(timeout=5.0)
        await node._router_v2.shutdown_curation(timeout=5.0)

    @pytest.mark.asyncio
    async def test_startup_backfill_is_skippable_by_config(self, tmp_path, store):
        _seed_memories(store, 4)
        node = _node(
            tmp_path,
            store,
            mode="write",
            entity_self_curation_backfill_on_startup=False,
        )
        node._router_v2 = await _backfill_router(store)
        await node._maybe_backfill_on_startup()
        assert node._router_v2._curation_queue.qsize() == 0
        assert node._router_v2.curation_status()["curation_backfill_runs"] == 0

    # ── Config ───────────────────────────────────────────────────────

    def test_backfill_config_defaults_and_validation(self):
        config = NodeConfig(id="agent:test:curator", tools=[])
        assert config.entity_self_curation_backfill_on_startup is True
        assert config.entity_self_curation_backfill_max_batches == 50
        assert config.entity_self_curation_backfill_slice_size == 10
        for field in (
            "entity_self_curation_backfill_max_batches",
            "entity_self_curation_backfill_slice_size",
        ):
            with pytest.raises(ValueError, match="at least 1"):
                NodeConfig(id="agent:test:curator", tools=[], **{field: 0})


# ─────────────────────────────────────────────────────────────────────
# Shadow-mode live-test fixes — rejection logging, digest path,
# overlay essay_edit lookup, token accounting
# ─────────────────────────────────────────────────────────────────────


class TestShadowLiveTestFixes:
    """The four defects the 22-memory shadow-mode live test surfaced."""

    # ── Shadow essay_edit overlay lookup ─────────────────────────────

    @pytest.mark.asyncio
    async def test_shadow_create_then_essay_edit_reports_the_activation_gate(
        self, tmp_path, store
    ):
        """A provisional key must resolve through the overlay, not 404.

        Before the fix every shadow ``essay_edit`` on a just-created entity
        returned "not in the entity registry", which reads as "your create
        failed" and hides the real blocker: the entity is pending until it
        crosses the activation window threshold.
        """
        node = _node(tmp_path, store, mode="shadow")
        context = _curation_context(mode="shadow")

        created = json.loads(
            await _call(
                node,
                "entity_create",
                {
                    "entity_type": "person",
                    "display_name": "Jessica",
                    "identity_note": "graduate student",
                    "reason": "new person in this batch",
                },
                context,
            )
        )
        key = created["entity_key"]
        assert "#shadow" in key, key
        assert created["status"] == "pending"

        result = await _call(
            node,
            "essay_edit",
            {
                "key": key,
                "old_text": "",
                "new_text": "Jessica is a graduate student. " + _cite("m1"),
                "reason": "dossier for the new entity",
            },
            context,
        )
        assert result.startswith("Error: "), result
        assert "not in the entity registry" not in result, result
        assert "pending" in result, result
        assert "activates" in result or "activate" in result, result

    @pytest.mark.asyncio
    async def test_essay_edit_still_rejects_a_key_that_exists_nowhere(
        self, tmp_path, store
    ):
        node = _node(tmp_path, store, mode="shadow")
        context = _curation_context(mode="shadow")
        result = await _call(
            node,
            "essay_edit",
            {
                "key": "person:nobody",
                "old_text": "",
                "new_text": "text",
                "reason": "unknown key",
            },
            context,
        )
        assert "not in the entity registry" in result, result

    @pytest.mark.asyncio
    async def test_shadow_essay_edit_accepts_a_citation_to_a_linked_memory(
        self, tmp_path, store, service
    ):
        """The shadow ``essay_edit`` overlay must accept a real citation.

        Shadow ``essay_edit`` routes through ``publish_dossier(validate_only=
        True)``, so it ran the same ``m_``-prefixed lookup against the bare
        ``memories.id`` column and refused every cited dossier.
        """
        key = _activate(service, "Project Owner")
        store.insert(_entry(_mid("m1")))
        service.link_memory(
            _mid("m1"), key, window_key="w1", context=_exec_context(), reason="setup"
        )
        node = _node(tmp_path, store, mode="shadow")
        result = await _call(
            node,
            "essay_edit",
            {
                "key": key,
                "old_text": "",
                "new_text": "Project Owner is the primary user. " + _cite("m1"),
                "reason": "cited dossier in shadow mode",
            },
            _curation_context(mode="shadow"),
        )
        assert not result.startswith("Error"), result
        payload = json.loads(result)
        assert payload["citations"] == ["m_" + _mid("m1")], payload
        assert payload["applied"] is False, "shadow mode must not write"

    # ── Rejection logging ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_refused_curation_calls_land_on_the_turn_context(
        self, tmp_path, store
    ):
        from mesh.tools import ToolCall

        node = _node(tmp_path, store, mode="shadow")
        context = _curation_context(mode="shadow")
        token = CURRENT_CURATION_CONTEXT.set(context)
        try:
            await node._execute_all_tools(
                [
                    ToolCall(
                        name="essay_edit",
                        arguments={
                            "key": "person:nobody",
                            "old_text": "",
                            "new_text": "x",
                            "reason": "unknown key",
                        },
                        raw_xml="",
                        call_id="c1",
                    ),
                ],
                _trigger(),
            )
        finally:
            CURRENT_CURATION_CONTEXT.reset(token)

        assert len(context.rejections) == 1, context.rejections
        assert context.rejections[0]["tool"] == "essay_edit"
        assert "not in the entity registry" in context.rejections[0]["error"]
        assert context.summary()["rejections"] == 1

    def test_rejection_log_is_capped(self):
        from mesh.memory.curation import REJECTION_LOG_CAP, REJECTION_TEXT_CHARS

        context = _curation_context(mode="shadow")
        for _ in range(REJECTION_LOG_CAP + 10):
            context.record_rejection("essay_edit", "Error: " + "x" * 2000)
        assert len(context.rejections) == REJECTION_LOG_CAP
        assert len(context.rejections[0]["error"]) == REJECTION_TEXT_CHARS

    @pytest.mark.asyncio
    async def test_curation_turn_row_carries_rejections_and_tokens(self, store):
        router = await _backfill_router(store)
        router._last_router_call_tools = [("essay_edit", ""), ("digest_get", "")]
        router._last_curation_rejections = [
            {"tool": "essay_edit", "error": "Error: 'person:x' is pending"},
        ]
        router._last_router_call_usage = {
            "input_tokens": 1234, "output_tokens": 567, "llm_calls": 4,
        }
        batch = CurationBatch(reason="time-based", memory_ids=(_mid("m1"),))
        router._record_curation_turn(batch, "turn-1", "done")

        row = store._conn.execute(
            "SELECT details_json FROM entity_events "
            "WHERE event_type = 'curation_turn' ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        details = json.loads(row[0])
        assert details["rejections"] == [
            {"tool": "essay_edit", "error": "Error: 'person:x' is pending"},
        ]
        assert details["tokens_in"] == 1234
        assert details["tokens_out"] == 567
        assert details["llm_calls"] == 4

        status = router.curation_status()
        assert status["last_rejections"][0]["tool"] == "essay_edit"
        assert status["last_tokens_in"] == 1234
        assert status["last_tokens_out"] == 567

    # ── digest_get for MemorySystemV2 agents ─────────────────────────

    def test_digest_path_resolves_for_a_memory_system_v2_agent(
        self, tmp_path, store, digest_file
    ):
        import mesh.tool_implementations as ti

        node = _node(tmp_path, store, mode="shadow", digest=digest_file)
        previous_system = ti._memory_system
        previous_path = ti._standing_digest_path
        try:
            ti._memory_system = node._memory_system
            # MemorySystemV2 has neither ``_config`` nor ``config``.
            assert getattr(ti._memory_system, "_config", None) is None
            assert getattr(ti._memory_system, "config", None) is None
            ti._standing_digest_path = ""
            assert ti._resolve_digest_path() is None
            assert ti.digest_get().startswith("Error: no standing_digest_path")

            ti._standing_digest_path = node.config.standing_digest_path
            assert ti._resolve_digest_path() == str(digest_file)
            assert "2026-07-30" in ti.digest_get()
        finally:
            ti._memory_system = previous_system
            ti._standing_digest_path = previous_path


# ─────────────────────────────────────────────────────────────────────
# Async curation isolation — per-router-call state in contextvars
#
# These cover the invariant that makes `_router_turn_lock` removable from
# `_curation_drain_loop`: every mutable per-call field must live in a
# task-local `RouterCallState`, not on the shared RouterV2 instance.
# ─────────────────────────────────────────────────────────────────────


class TestAsyncCurationIsolation:
    """Curation runs concurrently with message processing (§ async plan)."""

    @pytest.mark.asyncio
    async def test_async01_concurrent_router_calls_keep_separate_call_state(self):
        """Two overlapping `_call_router_full` calls must not share per-call state.

        Each task writes a distinct marker into the per-call fields, then yields
        so the other task's prologue runs.  On the shared-instance design the
        second call's reset wipes the first call's ledger; with contextvars each
        task reads back exactly what it wrote.
        """
        router = await _router()

        started = asyncio.Event()
        both_in_flight = asyncio.Event()
        snapshots: dict[str, dict] = {}

        async def fake_process(**kwargs):
            tag = kwargs["trigger_msg"].content
            # Write this call's markers into the per-call ledger.
            router._last_router_call_tools.append((f"tool-{tag}", ""))
            router._router_call_worker_launches.append(f"worker-{tag}")
            router._router_call_worker_task_keys.add(f"key-{tag}")
            router._last_router_call_usage[tag] = 1
            router._last_worker_launch = {"worker_id": f"worker-{tag}"}
            router._last_router_call_sent_message = True
            router._last_router_failure_class = f"fail-{tag}"

            # Make the two calls genuinely overlap.
            if not started.is_set():
                started.set()
                await both_in_flight.wait()
            else:
                both_in_flight.set()
            await asyncio.sleep(0)

            # Read back — must still be this call's own values.
            snapshots[tag] = {
                "tools": list(router._last_router_call_tools),
                "launches": list(router._router_call_worker_launches),
                "task_keys": set(router._router_call_worker_task_keys),
                "usage": dict(router._last_router_call_usage),
                "last_launch": router._last_worker_launch,
                "sent_message": router._last_router_call_sent_message,
                "failure_class": router._last_router_failure_class,
                "trigger": router._trigger_nodes(),
            }
            return ""

        router._router_process_fn = fake_process

        def _msg(tag: str) -> Message:
            return Message(
                type=MessageType.MESSAGE,
                from_node=f"user:{tag}",
                to_node="agent:test:curator",
                content=tag,
            )

        await asyncio.gather(
            router._call_router_full(_msg("a")),
            router._call_router_full(_msg("b")),
        )

        assert set(snapshots) == {"a", "b"}
        for tag in ("a", "b"):
            snap = snapshots[tag]
            assert snap["tools"] == [(f"tool-{tag}", "")], (
                f"call {tag} lost its tool ledger to the concurrent call: "
                f"{snap['tools']}"
            )
            assert snap["launches"] == [f"worker-{tag}"], snap["launches"]
            assert snap["task_keys"] == {f"key-{tag}"}, snap["task_keys"]
            assert snap["usage"] == {tag: 1}, snap["usage"]
            assert snap["last_launch"] == {"worker_id": f"worker-{tag}"}
            assert snap["sent_message"] is True
            assert snap["failure_class"] == f"fail-{tag}"
            assert snap["trigger"] == (f"user:{tag}", "agent:test:curator")

    @pytest.mark.asyncio
    async def test_async02_curation_starts_while_a_message_turn_is_running(self):
        """Curation must not wait behind an in-flight message turn.

        This is the whole point of the change: `_router_turn_lock` is held for
        the length of a 1-4 minute LLM call, so serialising curation behind it
        starves message processing.
        """
        router = await _router()
        ran: list[str] = []
        started = asyncio.Event()

        async def fake_turn(batch):
            ran.append(batch.memory_ids[0])
            started.set()

        router._run_curation_turn = fake_turn

        await router._router_turn_lock.acquire()
        try:
            router.enqueue_curation_batch(CurationBatch("time-based", ("m1",)))
            await asyncio.wait_for(started.wait(), timeout=5.0)
            assert ran == ["m1"], (
                "curation must start while a message turn holds the lock"
            )
        finally:
            router._router_turn_lock.release()

        assert await router.wait_for_curation_idle(timeout=5.0)
        await router.shutdown_curation(timeout=5.0)

    @pytest.mark.asyncio
    async def test_async03_curation_batches_still_run_fifo_and_serially(self):
        """Curation turns stay ordered and non-overlapping with each other."""
        router = await _router()
        ran: list[str] = []
        overlap: list[str] = []
        in_turn = False

        async def fake_turn(batch):
            nonlocal in_turn
            if in_turn:
                overlap.append(batch.memory_ids[0])
            in_turn = True
            ran.append(batch.memory_ids[0])
            await asyncio.sleep(0)
            in_turn = False

        router._run_curation_turn = fake_turn
        router.enqueue_curation_batch(CurationBatch("time-based", ("m1",)))
        router.enqueue_curation_batch(CurationBatch("time-based", ("m2",)))
        router.enqueue_curation_batch(CurationBatch("time-based", ("m3",)))

        assert await router.wait_for_curation_idle(timeout=5.0)
        assert ran == ["m1", "m2", "m3"], "FIFO order preserved"
        assert overlap == [], "curation turns must not overlap each other"
        await router.shutdown_curation(timeout=5.0)


class TestConcurrentDigestEdit:
    """`digest_edit` must read-modify-write inside one exclusive lock."""

    @pytest.mark.asyncio
    async def test_async04_digest_edit_rereads_under_the_exclusive_lock(
        self, tmp_path
    ):
        """A competing edit that lands while `digest_edit` waits must survive.

        Deterministic interleaving without test hooks: the main task holds an
        exclusive flock on the digest, so a `digest_edit` running in a worker
        thread cannot write.  The broken read-then-lock ordering snapshots the
        file *before* blocking, so the edit the main task commits while the
        lock is held is silently overwritten.
        """
        import fcntl
        import threading

        import mesh.tool_implementations as ti

        path = tmp_path / "digest.md"
        path.write_text("AAA\nBBB\n")

        previous_path = ti._standing_digest_path
        previous_system = ti._memory_system
        ti._standing_digest_path = str(path)
        ti._memory_system = None
        try:
            result: dict[str, str] = {}
            release = threading.Event()

            def _competing_edit():
                result["out"] = ti.digest_edit("BBB", "B1")

            with open(path, "r+") as holder:
                fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
                thread = threading.Thread(target=_competing_edit)
                thread.start()
                # Give the thread time to reach its read and/or block on flock.
                await asyncio.sleep(0.3)
                # Commit a different edit while we still hold the lock.
                holder.seek(0)
                holder.write("A1\nBBB\n")
                holder.truncate()
                holder.flush()
                os.fsync(holder.fileno())
                fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
                release.set()

            thread.join(timeout=10)
            assert not thread.is_alive(), "digest_edit deadlocked"

            final = path.read_text()
            assert final == "A1\nB1\n", (
                "digest_edit clobbered a concurrent edit — it must re-read the "
                f"file inside the exclusive lock. Got: {final!r} "
                f"(tool said: {result.get('out')!r})"
            )
        finally:
            ti._standing_digest_path = previous_path
            ti._memory_system = previous_system


class TestConcurrentEssayEdit:
    """`essay_edit` must not silently drop a concurrent writer's patch."""

    def test_async05_essay_cas_rejects_a_stale_revision(self, store):
        """A write gated on a superseded revision is refused, not applied."""
        store.create_essay("person:owner", body="ORIGINAL", title="Project Owner")
        first = store.get_essay("person:owner")
        stale_revision = int(first["patch_count"])

        # Another writer lands in the read → write window.
        ok, _ = store.update_essay_if_revision(
            "person:owner", stale_revision, body="WRITER-A"
        )
        assert ok, "the first writer at the observed revision must succeed"

        # Our snapshot is now stale — the CAS must refuse.
        ok, message = store.update_essay_if_revision(
            "person:owner", stale_revision, body="WRITER-B"
        )
        assert not ok, "a stale-revision write must be refused"
        assert "modified concurrently" in message
        assert store.get_essay("person:owner")["body"] == "WRITER-A", (
            "the refused write must not have touched the row"
        )

    def test_async06_essay_edit_surfaces_a_concurrent_modification(
        self, tmp_path, store
    ):
        """`essay_edit` reports a conflict instead of clobbering.

        The competing write is injected at the real seam: `get_essay` is what
        `essay_edit` reads its snapshot from, so landing another patch as that
        read returns reproduces exactly the interleaving the CAS defends
        against.
        """
        import mesh.tool_implementations as ti

        store.create_essay("person:owner", body="AAA BBB", title="Project Owner")

        real_get_essay = store.get_essay
        fired: list[bool] = []

        def racing_get_essay(entity_key: str):
            snapshot = real_get_essay(entity_key)
            if not fired:
                fired.append(True)
                # A concurrent writer commits between our read and our write.
                real_update = store.update_essay
                real_update(entity_key, body="AAA ZZZ")
            return snapshot

        previous_resolver = ti._get_essay_store
        try:
            ti._get_essay_store = lambda: store
            store.get_essay = racing_get_essay
            result = ti.essay_edit("person:owner", old_text="BBB", new_text="B1")
        finally:
            store.get_essay = real_get_essay
            ti._get_essay_store = previous_resolver

        assert "modified concurrently" in result, (
            f"essay_edit must surface the conflict, got: {result!r}"
        )
        assert store.get_essay("person:owner")["body"] == "AAA ZZZ", (
            "the concurrent writer's patch must survive"
        )

    @pytest.mark.asyncio
    async def test_async10_curation_essay_edit_passes_observed_revision(
        self, tmp_path, store, service, monkeypatch
    ):
        """The curation-specific dossier writer must use the same CAS guard."""
        key = _activate(service, "Project Owner")
        store.insert(_entry(_mid("m1")))
        service.link_memory(
            _mid("m1"), key, window_key="w1", context=_exec_context(),
            reason="setup",
        )
        citation = _cite("m1")
        service.publish_dossier(
            key,
            body=f"Project Owner has the original summary. {citation}",
            title="Project Owner",
            token_budget=4000,
            measure=estimate_tokens,
            context=_exec_context(),
            reason="seed",
        )

        real_publish = EntityService.publish_dossier
        raced: list[bool] = []

        def racing_publish(entity_service, entity_key, **kwargs):
            if not raced:
                raced.append(True)
                store.update_essay(
                    entity_key,
                    body=f"Project Owner has the concurrent summary. {citation}",
                )
            return real_publish(entity_service, entity_key, **kwargs)

        monkeypatch.setattr(EntityService, "publish_dossier", racing_publish)
        node = _node(tmp_path, store)
        result = await _call(
            node,
            "essay_edit",
            {
                "key": key,
                "old_text": "original",
                "new_text": "curation",
                "reason": "exercise curation CAS",
            },
            _curation_context(),
        )

        assert "modified concurrently" in result, result
        assert store.get_essay(key)["body"] == (
            f"Project Owner has the concurrent summary. {citation}"
        )


class TestCurationDigestSnapshotIsolation:
    @pytest.mark.asyncio
    async def test_async11_second_curation_edit_rereads_locked_file(
        self, tmp_path, store, digest_file
    ):
        """A later curation edit must retain a message-turn edit in between."""
        import mesh.tool_implementations as ti

        node = _node(tmp_path, store, digest=digest_file)
        context = _curation_context()
        first = await _call(
            node,
            "digest_edit",
            {
                "old_text": "something happened",
                "new_text": "the first curation edit landed",
            },
            context,
        )
        assert not first.startswith("Error"), first

        previous_path = ti._standing_digest_path
        previous_system = ti._memory_system
        ti._standing_digest_path = str(digest_file)
        ti._memory_system = None
        try:
            external = ti.digest_edit(
                "mesh — the platform.", "mesh — externally edited."
            )
            assert not external.startswith("Error"), external
        finally:
            ti._standing_digest_path = previous_path
            ti._memory_system = previous_system

        second = await _call(
            node,
            "digest_edit",
            {"old_text": "Project Owner — primary user.", "new_text": "Project Owner — curated."},
            context,
        )
        assert not second.startswith("Error"), second
        final = digest_file.read_text()
        assert "the first curation edit landed" in final
        assert "mesh — externally edited." in final
        assert "Project Owner — curated." in final


class TestSocketBoundaryStatePropagation:
    """Per-call router state must survive the tool-socket task boundary."""

    def test_async07_execution_scope_carries_the_router_call_state(self):
        """The scope is the only channel across the aiohttp task boundary."""
        import dataclasses

        from mesh.agent_node import ExecutionCapabilityScope

        fields = {f.name for f in dataclasses.fields(ExecutionCapabilityScope)}
        assert "router_call_state" in fields, (
            "ExecutionCapabilityScope must carry the originating router call's "
            "state — contextvars do not cross into the socket request task"
        )

    def test_async08_socket_handler_rebinds_the_call_state(self):
        """`handle_tool_call` must bind and then reset the contextvar."""
        from mesh.agent_node import AgentNode

        source = inspect.getsource(AgentNode._start_tool_socket)
        assert "_CTX_ROUTER_CALL_STATE.set(" in source, (
            "socket handler must rebind the router call state before "
            "executing tools"
        )
        assert "_CTX_ROUTER_CALL_STATE.reset(" in source, (
            "socket handler must reset the contextvar in its finally block"
        )
        assert "_CC_TRIGGER_CTX.set(" in source, (
            "socket handler must rebind the trigger context too"
        )

    @pytest.mark.asyncio
    async def test_async09_rebound_state_is_visible_to_a_separate_task(self):
        """A tool running on another task writes into the originating call.

        This is the exact shape of a subprocess-backed router (harness, Codex,
        Claude Code) calling back over the socket: the tool executes on an
        aiohttp request task, but its tool-ledger and send_message writes must
        land in the router call that dispatched it.
        """
        from mesh.router_v2 import _CTX_ROUTER_CALL_STATE, RouterCallState

        router = await _router()
        captured: dict[str, RouterCallState] = {}
        ready = asyncio.Event()
        proceed = asyncio.Event()

        async def owning_call():
            router._init_call_state(
                Message(
                    type=MessageType.MESSAGE,
                    from_node="user:testuser",
                    to_node="agent:test:curator",
                    content="hi",
                )
            )
            captured["state"] = _CTX_ROUTER_CALL_STATE.get()
            ready.set()
            await proceed.wait()
            # Back on the owning task: the socket tool's writes are visible.
            assert router._last_router_call_tools == [("essay_edit", "")]
            assert router._last_router_call_sent_message is True

        # The owning task is started from *this* context, so this scope never
        # holds the call state — which is exactly the aiohttp situation: request
        # tasks descend from the server's context, not from the router call's.
        owner = asyncio.create_task(owning_call())
        await ready.wait()
        scope_state = captured["state"]

        async def socket_task():
            assert _CTX_ROUTER_CALL_STATE.get() is None, (
                "an aiohttp-style request task does not inherit the call state"
            )
            # This is what handle_tool_call() does with scope.router_call_state.
            token = _CTX_ROUTER_CALL_STATE.set(scope_state)
            try:
                router._last_router_call_tools.append(("essay_edit", ""))
                router._last_router_call_sent_message = True
            finally:
                _CTX_ROUTER_CALL_STATE.reset(token)
            assert _CTX_ROUTER_CALL_STATE.get() is None, "must reset on exit"

        await asyncio.create_task(socket_task())
        proceed.set()
        await owner
        assert scope_state.tools == [("essay_edit", "")]

    @pytest.mark.asyncio
    async def test_async12_status_aggregates_only_active_router_scopes(self):
        """A completed call's task-local CC events are not live activity."""
        router = await _router()
        node = AgentNode.__new__(AgentNode)
        node._router_v2 = router
        node._execution_scopes = {}
        node._router_cc_events_fallback = []

        state = router._init_call_state(
            Message(
                type=MessageType.MESSAGE,
                from_node="user:testuser",
                to_node="agent:test:curator",
                content="status isolation",
            )
        )
        event = SimpleNamespace(message="tool activity")
        state.router_cc_events.append(event)

        # No registered scope means this is a completed call, even though its
        # ledger remains readable by the caller on this task.
        assert node._all_router_cc_events() == []

        node._execution_scopes["live"] = SimpleNamespace(router_call_state=state)
        assert node._all_router_cc_events() == [event]

    @pytest.mark.asyncio
    async def test_async13_call_state_is_bound_to_its_router_instance(self):
        """The module ContextVar must not bridge two RouterV2 objects."""
        first = await _router()
        second = await _router()
        first_state = first._init_call_state()
        first._last_router_call_tools.append(("first", ""))

        second._last_router_call_tools.append(("second", ""))

        assert first_state.tools == [("first", "")]
        assert second._last_router_call_tools == [("second", "")]

    @pytest.mark.asyncio
    async def test_async14_curation_drain_discards_creator_call_context(self):
        """create_task inheritance must not share the message call's state."""
        router = await _router()
        parent_state = router._init_call_state()
        parent_state.curation_rejections.append({"parent": True})
        observed: list[object] = []

        async def fake_turn(_batch):
            observed.append(router._get_call_state())
            router._last_curation_rejections = [{"curation": True}]

        router._run_curation_turn = fake_turn
        router.enqueue_curation_batch(CurationBatch("time-based", ("m1",)))
        assert await router.wait_for_curation_idle(timeout=5.0)
        await router.shutdown_curation(timeout=5.0)

        assert observed and observed[0] is not parent_state
        assert parent_state.curation_rejections == [{"parent": True}]


# ── Citation surface and curation evidence windows ────────────────
#
# Two defects found in coder1's 2026-08-01 digest audit, both structural:
#
#   F1  The batch listing printed bare hex memory IDs, so the model had to
#       invent a citation delimiter.  Agents that guessed ``[m:<id>]`` wrote
#       digests whose every citation was invisible to ``CITATION_RE``, which
#       left the ghost-citation gate passing vacuously.
#
#   F2  The self-curation path linked memories with ``window_key = NULL`` and
#       never evaluated activation, so entities it built could never reach the
#       distinct-window threshold and therefore could never receive an essay.


class TestCanonicalCitationSurface:
    def test_batch_block_prints_memories_as_canonical_citations(self):
        """The listing is the only citation surface the model sees first."""
        from mesh.memory.curation import render_batch_block

        mid = _mid("m1")
        block = render_batch_block(
            CurationBatch(reason="time-based", memory_ids=(mid,)),
            [{"id": mid, "retrieval_key": "k", "digest_candidate": True,
              "entities": ["person:owner"]}],
        )
        assert f"[m_{mid}]" in block
        assert f"[m:{mid}]" not in block
        # And the bare ID never appears unbracketed, which is what let an
        # agent copy a delimiter-free surface and invent its own.
        assert extract_citations(block) == [f"m_{mid}"]

    def test_legacy_colon_citations_are_ignored_by_default(self):
        """Canonical-only is the default: the dossier gate depends on it."""
        mid = _mid("m1")
        assert extract_citations(f"claim [m:{mid}] here") == []
        assert extract_citations(f"claim [m_{mid}] here") == [f"m_{mid}"]

    def test_legacy_colon_citations_are_readable_on_request(self):
        """The digest gate opts in, so legacy digests stop passing vacuously."""
        mid = _mid("m1")
        other = _mid("m2")
        body = f"one [m:{mid}] two [m_{other}]"
        assert extract_citations(body, include_legacy=True) == [
            f"m_{mid}", f"m_{other}",
        ]

    def test_legacy_surface_is_not_reported_as_a_malformed_reference(self):
        """Flagging it would refuse every edit to a legacy digest."""
        from mesh.memory.curation import find_loose_citations

        assert find_loose_citations(f"claim [m:{_mid('m1')}]") == []

    def test_constitutions_state_the_canonical_form_and_reject_the_colon(self):
        from mesh.memory.curation import DIGEST_CONSTITUTION, ESSAY_CONSTITUTION

        for text in (DIGEST_CONSTITUTION, ESSAY_CONSTITUTION):
            assert "[m_3f723ef2c694]" in text
            assert "[m:3f723ef2c694]" in text  # named as the wrong form
            assert "m_3f723ef2c694     (no brackets)" in text


class TestCurationEvidenceWindows:
    def test_curation_window_key_is_derived_and_stable(self):
        from mesh.memory.curation import curation_window_key

        batch = CurationBatch(
            reason="time-based", memory_ids=("m1",), formed_at="2026-08-01T00:00:00Z"
        )
        turn_id = batch.turn_id("agent:test:curator")
        first = curation_window_key(turn_id)
        assert first == curation_window_key(turn_id)  # same batch, same key
        assert len(first) == 16 and int(first, 16) >= 0  # formation key shape
        assert first != curation_window_key(turn_id + "x")  # different batch

    def test_curation_window_key_cannot_collide_with_a_formation_key(self):
        """Both live in one column, so the namespaces must be disjoint."""
        from mesh.memory.curation import curation_window_key
        from mesh.memory.formation_v3 import formation_window_key

        turn_id = "curation-agent:test:curator-2026-08-01T00:00:00Z-m1"
        assert curation_window_key(turn_id) != formation_window_key(
            0, 60, None, None
        )

    def test_curation_link_records_its_evidence_window(self, store, service):
        """The forward fix for F2: the curation path stops writing NULL."""
        from mesh.memory.curation import curation_window_key

        store.insert(_entry(_mid("m1")))
        key = _activate(service, "Windowed Person")
        turn_id = "curation-agent:test:curator-2026-08-01T00:00:00Z-m1"
        service.correct_link_transactional(
            _mid("m1"),
            reason="link it",
            context=EntityExecutionContext(
                actor_node="agent:test:curator",
                source_message_id=turn_id,
                source_author="agent:test:curator",
                curation_turn_id=turn_id,
            ),
            prepared_snapshot=service._memory_snapshot(_mid("m1")),
            add_entity_key=key,
        )
        row = store._conn.execute(
            "SELECT window_key, assignment_source FROM memory_entities "
            "WHERE memory_id = ? AND entity_key = ?",
            (_mid("m1"), key),
        ).fetchone()
        assert row[0] == curation_window_key(turn_id)
        assert row[1] == "interactive-correction"

    def test_interactive_correction_outside_curation_stays_unwindowed(
        self, store, service
    ):
        """A single human act carries no window provenance to record."""
        store.insert(_entry(_mid("m1")))
        key = _activate(service, "Unwindowed Person")
        service.correct_link_transactional(
            _mid("m1"),
            reason="link it",
            context=_exec_context(),  # no curation_turn_id
            prepared_snapshot=service._memory_snapshot(_mid("m1")),
            add_entity_key=key,
        )
        window_key = store._conn.execute(
            "SELECT window_key FROM memory_entities "
            "WHERE memory_id = ? AND entity_key = ?",
            (_mid("m1"), key),
        ).fetchone()[0]
        assert window_key is None

    def test_entity_activates_after_three_distinct_curation_batches(
        self, store, service
    ):
        """The other half of F2: the curation path must evaluate admission.

        Populating ``window_key`` alone is not enough — before this change
        ``correct_link_transactional`` never called the activation rule, so an
        entity could accumulate evidence across any number of batches and stay
        pending forever, which is what kept 27 of coder1's 29 entities from
        ever receiving an essay.
        """
        created = service.create_pending_entity(
            "person",
            "Recurring Person",
            "note",
            origin="self-curation",
            context=_exec_context(),
            reason="test setup",
        )
        key = created["entity_key"]
        assert service.get_entity(key)["status"] == "pending"

        for index in range(3):
            memory_id = _mid(f"batch{index}")
            store.insert(_entry(memory_id))
            turn_id = f"curation-agent:test:curator-2026-08-0{index + 1}T00:00:00Z-m"
            service.correct_link_transactional(
                memory_id,
                reason="link it",
                context=EntityExecutionContext(
                    actor_node="agent:test:curator",
                    source_message_id=turn_id,
                    source_author="agent:test:curator",
                    curation_turn_id=turn_id,
                ),
                prepared_snapshot=service._memory_snapshot(memory_id),
                add_entity_key=key,
            )
            expected = "active" if index >= 2 else "pending"
            assert service.get_entity(key)["status"] == expected, (
                f"after {index + 1} distinct batches"
            )

    def test_repeated_links_from_one_batch_do_not_activate(self, store, service):
        """Activation means recurrence, not volume inside a single batch."""
        created = service.create_pending_entity(
            "person",
            "Bursty Person",
            "note",
            origin="self-curation",
            context=_exec_context(),
            reason="test setup",
        )
        key = created["entity_key"]
        turn_id = "curation-agent:test:curator-2026-08-01T00:00:00Z-m1"
        for index in range(5):
            memory_id = _mid(f"burst{index}")
            store.insert(_entry(memory_id))
            service.correct_link_transactional(
                memory_id,
                reason="link it",
                context=EntityExecutionContext(
                    actor_node="agent:test:curator",
                    source_message_id=turn_id,
                    source_author="agent:test:curator",
                    curation_turn_id=turn_id,
                ),
                prepared_snapshot=service._memory_snapshot(memory_id),
                add_entity_key=key,
            )
        assert service.get_entity(key)["status"] == "pending"


class TestCurationWindowKeyThroughTheToolPath:
    """The service-level fix only matters if the real handler carries it.

    ``_execute_entity_link_correct`` builds its own ``EntityExecutionContext``
    (it serves both the interactive correction and the curation turn), so a fix
    applied only to ``_curation_execution_context`` would never reach
    production.  These exercise the tool handler end to end.
    """

    @staticmethod
    def _curation_trigger(turn_id: str = "curation-turn-1"):
        return SimpleNamespace(
            id=turn_id,
            from_node="agent:test:curator",
            content="synthetic curation batch summary",
            metadata={"internal_curation": True},
        )

    def test_curation_turn_id_is_read_from_the_trigger_metadata(self, tmp_path, store):
        node = _node(tmp_path, store)
        assert node._curation_turn_id_of(self._curation_trigger()) == "curation-turn-1"
        # An ordinary user message is not a curation batch.
        assert node._curation_turn_id_of(_trigger()) is None
        assert node._curation_turn_id_of(
            SimpleNamespace(id="m-1", metadata={"internal_curation": False})
        ) is None
        assert node._curation_turn_id_of(
            SimpleNamespace(id=None, metadata={"internal_curation": True})
        ) is None

    @pytest.mark.asyncio
    async def test_link_through_the_tool_handler_records_the_window(
        self, tmp_path, store, service
    ):
        from mesh.memory.curation import curation_window_key

        node = _node(tmp_path, store)
        context = _curation_context()
        store.insert(_entry(_mid("m1")))
        created = json.loads(
            await _call(
                node,
                "entity_create",
                {"entity_type": "person", "display_name": "Handler", "reason": "r"},
                context,
            )
        )
        key = created["entity_key"]

        trigger = self._curation_trigger()
        token = CURRENT_CURATION_CONTEXT.set(context)
        try:
            result = await node._execute_entity_tool(
                "entity_link_correct",
                {"memory_id": _mid("m1"), "reason": "link", "add_entity_key": key},
                trigger,
            )
        finally:
            CURRENT_CURATION_CONTEXT.reset(token)
        assert "Error" not in result, result

        window_key = store._conn.execute(
            "SELECT window_key FROM memory_entities "
            "WHERE memory_id = ? AND entity_key = ?",
            (_mid("m1"), key),
        ).fetchone()[0]
        assert window_key == curation_window_key(trigger.id)


# ─────────────────────────────────────────────────────────────────────
# Essay generation folded into the curation turn
# (mesh/memory/entity_essays.py + RouterV2._generate_curation_essays)
# ─────────────────────────────────────────────────────────────────────


class _StubLLM:
    """An LLM client whose completions are scripted, in order."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def complete(self, prompt: str, **_kwargs) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("stub LLM exhausted")
        return self.replies.pop(0)


def _entity_with_evidence(store, service, label: str, *, windows: int = 3):
    """An active entity with ``windows`` linked memories, one per window."""
    key = service.create_pending_entity(
        "person", label, f"{label} note",
        origin="self-curation", context=_exec_context(), reason="setup",
    )["entity_key"]
    ids = []
    for index in range(windows):
        memory_id = _mid(f"{label}-{index}")
        store.insert(_entry(memory_id))
        store._conn.commit()
        service.link_memory(
            memory_id, key, window_key=f"w-{label}-{index}", activate=True,
        )
        ids.append(memory_id)
    return key, ids


class TestEssayGenerationInCuration:
    def test_active_entities_missing_essays_lists_only_undossiered(
        self, store, service,
    ):
        from mesh.memory.entity_essays import active_entities_missing_essays

        key, ids = _entity_with_evidence(store, service, "Nora")
        assert service.get_entity(key)["status"] == "active"
        assert active_entities_missing_essays(store._conn) == [key]

        service.publish_dossier(
            key,
            body=f"## Identity\n\nNora appears in [m_{ids[0]}].\n",
            title="Nora",
            token_budget=4000,
            measure=estimate_tokens,
            context=_exec_context(),
            reason="test",
        )
        assert active_entities_missing_essays(store._conn) == []

    def test_limit_bounds_the_drain(self, store, service):
        from mesh.memory.entity_essays import active_entities_missing_essays

        _entity_with_evidence(store, service, "Ann")
        _entity_with_evidence(store, service, "Bea")
        assert len(active_entities_missing_essays(store._conn)) == 2
        assert len(active_entities_missing_essays(store._conn, limit=1)) == 1

    def test_generate_writes_essay_through_publish_dossier(self, store, service):
        from mesh.memory.entity_essays import generate_essay_for_entity

        key, ids = _entity_with_evidence(store, service, "Cleo")
        body = f"## Identity\n\nCleo is cited at [m_{ids[0]}] and [m_{ids[1]}].\n"
        client = _StubLLM(body)

        result = asyncio.run(generate_essay_for_entity(
            client, service, store._conn, key,
            node_id="agent:test:curator",
            context=_exec_context(),
            measure=estimate_tokens,
        ))

        assert result["status"] == "written", result
        assert result["citations"] == 2
        assert result["repaired"] is False
        row = store._conn.execute(
            "SELECT body FROM essays WHERE entity_key = ?", (key,)
        ).fetchone()
        assert row is not None and "Cleo is cited" in row[0]
        # The constitution and the evidence both reached the model.
        assert "ESSAY CONSTITUTION" in client.prompts[0]
        assert f"[m_{ids[0]}]" in client.prompts[0]

    def test_ghost_citation_is_refused_then_repaired(self, store, service):
        """A hallucinated handle gets exactly one bounded repair attempt."""
        from mesh.memory.entity_essays import generate_essay_for_entity

        key, ids = _entity_with_evidence(store, service, "Dot")
        ghost = _mid("never-stored")
        client = _StubLLM(
            f"## Identity\n\nDot is cited at [m_{ghost}].\n",
            f"## Identity\n\nDot is cited at [m_{ids[0]}].\n",
        )

        result = asyncio.run(generate_essay_for_entity(
            client, service, store._conn, key,
            node_id="agent:test:curator",
            context=_exec_context(),
            measure=estimate_tokens,
        ))

        assert result["status"] == "written", result
        assert result["repaired"] is True
        assert len(client.prompts) == 2
        assert "PREVIOUS ATTEMPT REFUSED" in client.prompts[1]
        body = store._conn.execute(
            "SELECT body FROM essays WHERE entity_key = ?", (key,)
        ).fetchone()[0]
        assert ghost not in body

    def test_second_refusal_is_reported_not_retried(self, store, service):
        from mesh.memory.entity_essays import generate_essay_for_entity

        key, _ids = _entity_with_evidence(store, service, "Eli")
        ghost = _mid("never-stored-2")
        client = _StubLLM(
            f"## Identity\n\nEli [m_{ghost}].\n",
            f"## Identity\n\nEli again [m_{ghost}].\n",
        )

        result = asyncio.run(generate_essay_for_entity(
            client, service, store._conn, key,
            node_id="agent:test:curator",
            context=_exec_context(),
            measure=estimate_tokens,
        ))

        assert result["status"] == "failed"
        assert "refused" in result["error"]
        # Never a loop: two attempts, then report.
        assert len(client.prompts) == 2
        assert store._conn.execute(
            "SELECT COUNT(*) FROM essays WHERE entity_key = ?", (key,)
        ).fetchone()[0] == 0

    def test_validate_only_runs_gates_but_writes_nothing(self, store, service):
        from mesh.memory.entity_essays import generate_essay_for_entity

        key, ids = _entity_with_evidence(store, service, "Fay")
        client = _StubLLM(f"## Identity\n\nFay [m_{ids[0]}].\n")

        result = asyncio.run(generate_essay_for_entity(
            client, service, store._conn, key,
            node_id="agent:test:curator",
            context=_exec_context(),
            measure=estimate_tokens,
            validate_only=True,
        ))

        assert result["status"] == "written"
        assert result["validate_only"] is True
        assert store._conn.execute(
            "SELECT COUNT(*) FROM essays WHERE entity_key = ?", (key,)
        ).fetchone()[0] == 0

    def test_router_hook_is_off_by_default(self, store, service):
        key, ids = _entity_with_evidence(store, service, "Gus")
        router = asyncio.run(_router(config=RouterV2Config(
            entity_self_curation_mode="write",
        )))
        router._memory = _memory_system(store)
        router._memory._llm_client = _StubLLM(
            f"## Identity\n\nGus [m_{ids[0]}].\n"
        )

        result = asyncio.run(router._generate_curation_essays())

        assert result == {"skipped": "disabled"}
        assert store._conn.execute(
            "SELECT COUNT(*) FROM essays WHERE entity_key = ?", (key,)
        ).fetchone()[0] == 0

    def test_router_hook_writes_when_enabled(self, store, service):
        key, ids = _entity_with_evidence(store, service, "Hal")
        router = asyncio.run(_router(config=RouterV2Config(
            entity_self_curation_mode="write",
            entity_self_curation_essays_enabled=True,
        )))
        router._memory = _memory_system(store)
        router._memory._llm_client = _StubLLM(
            f"## Identity\n\nHal is cited at [m_{ids[0]}].\n"
        )

        summary = asyncio.run(router._generate_curation_essays())

        assert summary["written"] == [key], summary
        assert store._conn.execute(
            "SELECT COUNT(*) FROM essays WHERE entity_key = ?", (key,)
        ).fetchone()[0] == 1

    def test_router_hook_respects_per_turn_cap(self, store, service):
        _entity_with_evidence(store, service, "Ivy")
        _entity_with_evidence(store, service, "Jon")
        router = asyncio.run(_router(config=RouterV2Config(
            entity_self_curation_mode="write",
            entity_self_curation_essays_enabled=True,
            entity_self_curation_essays_max_per_turn=1,
        )))
        router._memory = _memory_system(store)
        router._memory._llm_client = _StubLLM(
            "## Identity\n\nNo citations here.\n"
        )

        summary = asyncio.run(router._generate_curation_essays())

        assert len(summary["considered"]) == 1
        assert store._conn.execute(
            "SELECT COUNT(*) FROM essays"
        ).fetchone()[0] == 1

    def test_router_hook_shadow_mode_writes_nothing(self, store, service):
        key, ids = _entity_with_evidence(store, service, "Kim")
        router = asyncio.run(_router(config=RouterV2Config(
            entity_self_curation_mode="shadow",
            entity_self_curation_essays_enabled=True,
        )))
        router._memory = _memory_system(store)
        router._memory._llm_client = _StubLLM(
            f"## Identity\n\nKim [m_{ids[0]}].\n"
        )

        summary = asyncio.run(router._generate_curation_essays())

        assert summary["written"] == [key]
        assert store._conn.execute(
            "SELECT COUNT(*) FROM essays WHERE entity_key = ?", (key,)
        ).fetchone()[0] == 0

    def test_router_hook_curation_off_is_a_no_op(self, store, service):
        _entity_with_evidence(store, service, "Lee")
        router = asyncio.run(_router(config=RouterV2Config(
            entity_self_curation_mode="off",
            entity_self_curation_essays_enabled=True,
        )))
        router._memory = _memory_system(store)

        result = asyncio.run(router._generate_curation_essays())

        assert result == {"skipped": "curation off"}
        assert store._conn.execute(
            "SELECT COUNT(*) FROM essays"
        ).fetchone()[0] == 0


# ─────────────────────────────────────────────────────────────────────
# Memory-ID normalization at the tool-argument boundary
# (mesh/memory/ids.py + correct_entity_link + the shadow path + memory_get)
#
# Curation renders memories as ``[m_<id>]`` and tells the model to copy the
# handle exactly, but ``memories.id`` stores the bare hex.  The correction
# path used to do an exact lookup with no normalization, which rejected
# 1,963/3,292 (59.6%) of all curation rejections as "unknown memory ID".
# ─────────────────────────────────────────────────────────────────────


#: Bare 12-hex IDs that are *well-formed* but name no row — the six observed
#: transcription typos (Hamming distance 1-2 from a real batch ID).  These
#: must stay rejected: normalization is shape-only and never repairs digits.
def _typo_shapes(real: str) -> list[str]:
    def flip(index: int, digit: str) -> str:
        return real[:index] + digit + real[index + 1:]

    swap = "0" if real[0] != "0" else "1"
    alt = "9" if real[-1] != "9" else "8"
    return [
        flip(0, swap),                                  # distance 1, first
        flip(11, alt),                                  # distance 1, last
        flip(5, "0" if real[5] != "0" else "2"),        # distance 1, middle
        flip(0, swap)[:11] + alt,                       # distance 2
        real[1:] + real[0],                             # rotated
        real[:10] + real[11] + real[10],                # transposed tail
    ]


class TestMemoryIdNormalizer:
    """The shared helper: a closed accept-list, not a permissive strip."""

    def test_accepts_the_three_canonical_forms(self):
        from mesh.memory.ids import normalize_memory_id

        bare = _mid("m1")
        assert normalize_memory_id(bare) == bare
        assert normalize_memory_id(f"m_{bare}") == bare
        assert normalize_memory_id(f"[m_{bare}]") == bare
        # Surrounding whitespace cannot change which memory is named.
        assert normalize_memory_id(f"  [m_{bare}]  ") == bare

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "m_",
            "0123456789a",            # 11 hex
            "0123456789abc",          # 13 hex
            "0123456789AB",           # uppercase
            "m_0123456789AB",         # uppercase behind the handle
            "0123456789ag",           # non-hex
            "mem-0123456789ab",       # other prefix
            "m-0123456789ab",         # wrong separator
            "[m:0123456789ab]",       # colon surface
            "[0123456789ab]",         # bracketed but no m_
            "[m_0123456789ab",        # half bracket
            "m_0123456789ab]",        # half bracket
            "m_m_0123456789ab",       # doubled prefix
            "see [m_0123456789ab]",   # embedded in prose
            None,
            12,
        ],
    )
    def test_rejects_every_other_shape(self, bad):
        from mesh.memory.ids import (
            MemoryIdError,
            normalize_memory_id,
            try_normalize_memory_id,
        )

        assert try_normalize_memory_id(bad) is None
        with pytest.raises(MemoryIdError):
            normalize_memory_id(bad)

    def test_normalization_never_repairs_a_typo(self):
        """A well-formed typo normalizes to itself — it is not snapped to a real ID."""
        from mesh.memory.ids import normalize_memory_id

        real = _mid("m1")
        for typo in _typo_shapes(real):
            assert normalize_memory_id(typo) == typo
            assert typo != real


class TestCorrectEntityLinkAcceptsCitationHandles:
    """The write path: MemorySystemV2.correct_entity_link."""

    async def _link(self, node, context, key, memory_ref):
        return await _call(
            node,
            "entity_link_correct",
            {"memory_id": memory_ref, "reason": "link", "add_entity_key": key},
            context,
        )

    async def _setup(self, tmp_path, store, label="Handle"):
        node = _node(tmp_path, store)
        context = _curation_context()
        created = json.loads(
            await _call(
                node,
                "entity_create",
                {"entity_type": "person", "display_name": label, "reason": "r"},
                context,
            )
        )
        return node, context, created["entity_key"]

    def _links(self, store, memory_id):
        return [
            row[0]
            for row in store._conn.execute(
                "SELECT entity_key FROM memory_entities WHERE memory_id = ?",
                (memory_id,),
            ).fetchall()
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("surface", ["bare", "handle", "bracketed"])
    async def test_every_canonical_surface_links_the_same_row(
        self, tmp_path, store, service, surface
    ):
        bare = _mid("m1")
        store.insert(_entry(bare))
        node, context, key = await self._setup(tmp_path, store)
        ref = {
            "bare": bare,
            "handle": f"m_{bare}",
            "bracketed": f"[m_{bare}]",
        }[surface]

        result = await self._link(node, context, key, ref)

        assert "Error" not in result, result
        # The link lands under the BARE id regardless of the surface used.
        assert self._links(store, bare) == [key]
        assert self._links(store, ref) == [] if ref != bare else True

    @pytest.mark.asyncio
    async def test_handle_form_used_to_be_rejected_and_now_resolves(
        self, tmp_path, store, service
    ):
        """The exact regression: the m_ handle the model is told to copy."""
        bare = _mid("m1")
        store.insert(_entry(bare))
        node, context, key = await self._setup(tmp_path, store)

        result = await self._link(node, context, key, f"m_{bare}")

        assert "unknown memory ID" not in result, result
        assert "Error" not in result, result

    @pytest.mark.asyncio
    async def test_the_six_typo_shapes_stay_rejected(
        self, tmp_path, store, service
    ):
        bare = _mid("m1")
        store.insert(_entry(bare))
        node, context, key = await self._setup(tmp_path, store)

        for typo in _typo_shapes(bare):
            for ref in (typo, f"m_{typo}", f"[m_{typo}]"):
                result = await self._link(node, context, key, ref)
                assert "unknown memory ID" in result, (ref, result)
                assert self._links(store, typo) == []
        # The real memory was never touched by any of the near misses.
        assert self._links(store, bare) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad", ["0123456789AB", "mem-0123456789ab", "[m_0123456789ab", "0123456789a"]
    )
    async def test_malformed_shapes_are_rejected_as_malformed(
        self, tmp_path, store, service, bad
    ):
        store.insert(_entry(_mid("m1")))
        node, context, key = await self._setup(tmp_path, store)

        result = await self._link(node, context, key, bad)

        assert "Error" in result, result
        assert "malformed memory_id" in result, result
        assert store._conn.execute(
            "SELECT COUNT(*) FROM memory_entities"
        ).fetchone()[0] == 0

    @pytest.mark.asyncio
    async def test_no_cross_db_fallback(self, tmp_path, store, service):
        """A memory that exists only in another agent's DB stays unknown.

        Normalization must widen the accepted *surface*, never the search
        scope: resolution stays bound to this store's connection.
        """
        other_dir = tmp_path / "other-agent"
        other_dir.mkdir()
        other = MemoryStore("other", db_dir=str(other_dir))
        try:
            foreign = _mid("lives-elsewhere")
            other.insert(_entry(foreign))
            other._conn.commit()
            assert other.get(foreign) is not None
            assert store.get(foreign) is None

            node, context, key = await self._setup(tmp_path, store)
            for ref in (foreign, f"m_{foreign}", f"[m_{foreign}]"):
                result = await self._link(node, context, key, ref)
                assert "unknown memory ID" in result, (ref, result)
            assert store._conn.execute(
                "SELECT COUNT(*) FROM memory_entities"
            ).fetchone()[0] == 0
        finally:
            other.close()


class TestShadowEntityLinkCorrectNormalizes:
    """The shadow path bypasses MemorySystemV2 and had the identical drift."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("surface", ["bare", "handle", "bracketed"])
    async def test_shadow_accepts_every_canonical_surface(
        self, tmp_path, store, digest_file, surface
    ):
        bare = _mid("m1")
        store.insert(_entry(bare))
        node = _node(tmp_path, store, mode="shadow", digest=digest_file)
        context = _curation_context(mode="shadow")
        created = json.loads(
            await _call(
                node,
                "entity_create",
                {"entity_type": "person", "display_name": "Shadow", "reason": "r"},
                context,
            )
        )
        key = created["entity_key"]
        ref = {"bare": bare, "handle": f"m_{bare}", "bracketed": f"[m_{bare}]"}[surface]

        result = await _call(
            node,
            "entity_link_correct",
            {"memory_id": ref, "reason": "link", "add_entity_key": key},
            context,
        )

        assert "Error" not in result, result
        # The overlay is keyed on the bare id whichever surface came in, so a
        # following shadow call composes against it.
        assert key in context.links[bare]
        assert ref not in context.links if ref != bare else True

    @pytest.mark.asyncio
    async def test_shadow_rejects_malformed_and_typos(
        self, tmp_path, store, digest_file
    ):
        bare = _mid("m1")
        store.insert(_entry(bare))
        node = _node(tmp_path, store, mode="shadow", digest=digest_file)
        context = _curation_context(mode="shadow")
        created = json.loads(
            await _call(
                node,
                "entity_create",
                {"entity_type": "person", "display_name": "Shadow", "reason": "r"},
                context,
            )
        )
        key = created["entity_key"]

        malformed = await _call(
            node,
            "entity_link_correct",
            {"memory_id": "mem-0123456789ab", "reason": "r", "add_entity_key": key},
            context,
        )
        assert "malformed memory_id" in malformed, malformed

        for typo in _typo_shapes(bare):
            result = await _call(
                node,
                "entity_link_correct",
                {"memory_id": f"m_{typo}", "reason": "r", "add_entity_key": key},
                context,
            )
            assert "unknown memory ID" in result, (typo, result)
        assert bare not in context.links


class TestMemoryGetAcceptsCitationHandles:
    """Digests and essays hand the agent [m_xxxx]; memory_get keys on bare hex."""

    def _install(self, monkeypatch, entry):
        from mesh import tool_implementations

        class _Stub:
            _payload_max_chars = 6000

            def get_entry(self, entry_id):
                return entry if entry_id == entry.id else None

        monkeypatch.setattr(tool_implementations, "_memory_system", _Stub())
        return tool_implementations

    @pytest.mark.parametrize("surface", ["bare", "handle", "bracketed"])
    def test_every_canonical_surface_resolves(self, monkeypatch, surface):
        bare = _mid("m1")
        entry = _entry(bare)
        tools = self._install(monkeypatch, entry)
        ref = {"bare": bare, "handle": f"m_{bare}", "bracketed": f"[m_{bare}]"}[surface]

        out = tools.memory_get(ref)

        assert "No memory entry found" not in out, out
        assert f"**ID**: {bare}" in out

    def test_unknown_and_malformed_still_report_not_found(self, monkeypatch):
        bare = _mid("m1")
        tools = self._install(monkeypatch, _entry(bare))

        for ref in (_mid("nope"), f"m_{_mid('nope')}", "mem-0123456789ab", "garbage"):
            out = tools.memory_get(ref)
            assert "No memory entry found" in out, (ref, out)
