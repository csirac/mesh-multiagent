"""Regressions for the durable entity registry and correction transactions."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from mesh.memory.entities import (
    EntityError,
    EntityExecutionContext,
    EntityService,
    make_entity_slug,
    normalize_alias,
    serialize_registry_for_injection,
)
from mesh.llm import estimate_tokens
from mesh.memory.essay_fold import _ensure_essays_table, _exec_essay_edit
from mesh.memory.store import MemoryEntry, MemoryStore
from scripts.backfill_digest_entities import (
    _digest_map,
    backfill_database,
    extract_digest_entities,
)


def _entry(memory_id: str, summary: str | None = None) -> MemoryEntry:
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
    )


@pytest.fixture
def store(tmp_path):
    result = MemoryStore("entities", db_dir=str(tmp_path))
    yield result
    result.close()


@pytest.fixture
def service(store):
    return EntityService(
        store._conn,
        actor_node="agent:test",
        activation_window_threshold=3,
        mutations_enabled=True,
    )


def _user_context(content: str = "Lily is the person I mean."):
    return EntityExecutionContext(
        actor_node="agent:test",
        source_message_id="msg-user-1",
        source_author="user:testuser",
        source_content=content,
    )


def _active(
    service: EntityService,
    display_name: str,
    entity_type: str = "person",
    *,
    identity_note: str = "",
    content: str | None = None,
) -> str:
    context = _user_context(content or f"Please name {display_name} now.")
    result = service.create_user_named_entity(
        entity_type,
        display_name,
        naming_surface=display_name,
        identity_note=identity_note,
        context=context,
        reason="test user naming",
    )
    assert result["created"] is True
    return result["entity"]["entity_key"]


class TestNormalizationAndKeys:
    @pytest.mark.parametrize(
        ("surface", "alias", "slug"),
        [
            ("Mesh-Infra", "mesh infra", "mesh-infra"),
            ("MESH_infra", "mesh infra", "mesh-infra"),
            ("mesh infra", "mesh infra", "mesh-infra"),
            ("Paper–Review Flow", "paper review flow", "paper-review-flow"),
            ("Ｃafé", "café", "café"),
            ("q\u0301", "q\u0301", "q\u0301"),
            ("नमस्ते", "नमस्ते", "नमस्ते"),
        ],
    )
    def test_nfkc_casefold_and_separator_normalization(
        self, surface, alias, slug
    ):
        assert normalize_alias(surface) == alias
        assert make_entity_slug(surface) == slug

    def test_empty_slug_is_rejected(self, service):
        with pytest.raises(EntityError, match="non-empty entity slug"):
            service.create_pending_entity("project", "— 🐟 !!!")

    def test_service_generates_keys_and_ordinal_collisions(self, service):
        first = service.create_pending_entity("person", "Lily")
        second = service.create_pending_entity("person", "Lily")
        assert first["entity_key"] == "person:lily"
        assert second["entity_key"] == "person:lily-2"
        assert "entity_key" not in inspect.signature(
            service.create_pending_entity
        ).parameters


class TestFormationRegistrySerialization:
    def test_active_then_newest_pending_with_field_limits(self, service, store):
        active_key = _active(
            service,
            "Project Owner",
            identity_note="A" * 200,
        )
        service.add_alias(
            active_key,
            "an alias that is intentionally much longer than sixty-four "
            "characters for serialization",
            source="test",
        )
        older = service.create_pending_entity("project", "Older Pending")
        newer = service.create_pending_entity("project", "Newer Pending")
        store._conn.execute(
            "UPDATE entities SET created_at = ? WHERE entity_key = ?",
            ("2026-01-01T00:00:00+00:00", older["entity_key"]),
        )
        store._conn.execute(
            "UPDATE entities SET created_at = ? WHERE entity_key = ?",
            ("2026-01-02T00:00:00+00:00", newer["entity_key"]),
        )
        store._conn.commit()

        serialized = serialize_registry_for_injection(service, injection_cap=2)

        assert serialized.entity_keys == (active_key, newer["entity_key"])
        assert older["entity_key"] not in serialized.payload
        rows = [
            json.loads(line)
            for line in serialized.payload.splitlines()
            if line.startswith("{")
        ]
        assert rows[0]["identity_note"] == "A" * 128
        assert len(rows[0]["aliases"]) <= 64
        assert serialized.candidates_injected == 2
        assert serialized.serialized_token_count == estimate_tokens(
            serialized.payload
        )

    def test_active_admission_cap_does_not_evict(self, store):
        capped = EntityService(
            store._conn,
            actor_node="agent:test",
            mutations_enabled=True,
            active_entity_cap=1,
        )
        _active(capped, "First")
        with pytest.raises(EntityError, match="active entity cap"):
            _active(capped, "Second")
        assert [
            row["display_name"]
            for row in capped.list_registry(statuses=("active",))
        ] == ["First"]


class TestDigestEntityBackfill:
    def test_discovers_deployed_and_legacy_digest_names(self, tmp_path):
        deployed = tmp_path / "coder1_digest.md"
        legacy = tmp_path / "agent-alice.md"
        deployed.write_text("## Projects\n", encoding="utf-8")
        legacy.write_text("## People\n", encoding="utf-8")

        assert _digest_map(tmp_path) == {
            "alice": legacy,
            "coder1": deployed,
        }

        duplicate = tmp_path / "agent-coder1.md"
        duplicate.write_text("## Projects\n", encoding="utf-8")
        with pytest.raises(ValueError, match="multiple standing digests"):
            _digest_map(tmp_path)

    def test_people_parser_keeps_roles_out_of_aliases(self):
        digest = """\
## People

- **Ada** (`agent:researcher:ada`, self): Researcher agent.
- **Martin Hirzel** (IBM Research, White Plains): IBM manager.
- **Mandana Vaziri** (IBM Research, White Plains): IBM manager.
- **Project Owner:** Primary user.
- **Dylan Shell** (mentor@example.edu): Mentor.
"""
        candidates = extract_digest_entities(digest)
        by_name = {item.display_name: item for item in candidates}

        assert set(by_name) == {
            "Ada",
            "Project Owner",
            "Dylan Shell",
            "Mandana Vaziri",
            "Martin Hirzel",
        }
        assert {
            "Ada",
            "agent:researcher:ada",
            "ada",
        } <= by_name["Ada"].aliases
        assert "agent:researcher:ada`, self" not in by_name["Ada"].aliases
        assert by_name["Ada"].identity_note == "self. Researcher agent."
        assert "IBM Research, White Plains" not in by_name[
            "Martin Hirzel"
        ].aliases
        assert "IBM Research, White Plains" not in by_name[
            "Mandana Vaziri"
        ].aliases
        assert by_name["Project Owner"].display_name == "Project Owner"
        assert "mentor@example.edu" in by_name["Dylan Shell"].aliases

    def test_copied_database_backfill_is_active_and_idempotent(
        self, tmp_path
    ):
        digest = """\
## Projects

### hello-world (mesh agent platform)
- **Path:** `/tmp/hello-world`
- **Role:** The active repository.

## People

- **Bob** (`agent:sysadmin:bob`): Sysadmin agent.
"""
        candidates = extract_digest_entities(digest)
        assert [
            (item.entity_type, item.display_name)
            for item in candidates
        ] == [("person", "Bob"), ("project", "hello-world")]
        bob = candidates[0]
        assert {"Bob", "agent:sysadmin:bob", "bob"} <= bob.aliases

        store = MemoryStore("source", db_dir=str(tmp_path))
        source_path = Path(store._db_path)
        store.close()
        copy_dir = tmp_path / "copies"
        copy_dir.mkdir()
        database_path = copy_dir / "copy.db"
        shutil.copy2(source_path, database_path)
        first = backfill_database(database_path, candidates)
        second = backfill_database(database_path, candidates)
        assert first == {"created": 2, "existing": 0, "aliases_added": 0}
        assert second["created"] == 0
        assert second["existing"] == 2

        connection = sqlite3.connect(database_path)
        assert connection.execute(
            "SELECT display_name, status FROM entities ORDER BY display_name"
        ).fetchall() == [("Bob", "active"), ("hello-world", "active")]
        service = EntityService(connection)
        assert [
            row["entity_key"]
            for row in service.resolve_alias("agent:sysadmin:bob")
        ] == ["person:bob"]
        connection.close()

    def test_retired_tombstone_still_forces_ordinal(self, service):
        first = service.create_pending_entity("person", "Lily")
        service.retire_entity(first["entity_key"], reason="wrong Lily")
        second = service.create_pending_entity(
            "person", "Lily", origin="user-correction"
        )
        assert second["entity_key"] == "person:lily-2"

    def test_legitimate_slug_ending_in_ordinal_gets_collision_suffix(self, service):
        first = service.create_pending_entity("project", "phase 2")
        second = service.create_pending_entity("project", "phase 2")
        assert first["entity_key"] == "project:phase-2"
        assert second["entity_key"] == "project:phase-2-2"


class TestAliasesAndActivation:
    def test_two_lilies_share_alias_and_resolution_returns_both(self, service):
        lily_a = _active(
            service, "Lily", identity_note="Austin collaborator"
        )
        lily_b = _active(
            service, "Lily", identity_note="Work collaborator"
        )
        matches = service.resolve_alias("LILY")
        assert [item["entity_key"] for item in matches] == [lily_a, lily_b]
        assert {item["identity_note"] for item in matches} == {
            "Austin collaborator",
            "Work collaborator",
        }

    def test_ambiguous_formation_alias_links_nothing_and_audits(
        self, store, service
    ):
        _active(service, "Lily", identity_note="Austin")
        _active(service, "Lily", identity_note="Work")
        store.insert(_entry("ambiguous-memory"))
        store._conn.execute("BEGIN IMMEDIATE")
        results = service.apply_formation_mutations_in_transaction(
            [{
                "op": "resolve_and_link",
                "memory_id": "ambiguous-memory",
                "entity_type": "person",
                "display_name": "Lily",
                "window_key": "window-1",
            }]
        )
        store._conn.commit()
        assert results == [{"status": "ambiguous"}]
        assert service.links_for_memory("ambiguous-memory") == []
        event = service.events_for_memory("ambiguous-memory")[-1]
        assert event["event_type"] == "entity_unresolved"
        assert len(event["details"]["candidates"]) == 2

    def test_retired_alias_cannot_be_recreated_from_formation(
        self, store, service
    ):
        lily = _active(service, "Lily")
        service.retire_entity(lily, reason="retired identity")
        store.insert(_entry("historical-memory"))
        store._conn.execute("BEGIN IMMEDIATE")
        results = service.apply_formation_mutations_in_transaction(
            [{
                "memory_id": "historical-memory",
                "entity_type": "person",
                "display_name": "Lily",
                "window_key": "old-window",
            }]
        )
        store._conn.commit()
        assert results == [{"status": "retired-alias-rejected"}]
        assert service.links_for_memory("historical-memory") == []
        assert len(service.list_registry(statuses=())) == 1

    def test_retired_secondary_alias_cannot_be_recreated_from_formation(
        self, service
    ):
        lily = _active(service, "Lily")
        service.retire_entity(lily, reason="retired identity")
        with pytest.raises(
            EntityError,
            match="retired entity alias cannot be recreated",
        ):
            service.create_pending_entity(
                "person",
                "Lily Smith",
                aliases=["Lily"],
                origin="formation",
            )
        assert len(service.list_registry(statuses=())) == 1

    def test_recurrence_uses_distinct_windows(self, store, service):
        entity = service.create_pending_entity("project", "Mesh")
        key = entity["entity_key"]
        for memory_id in ("m1", "m2", "m3", "m4"):
            store.insert(_entry(memory_id))

        service.link_memory("m1", key, window_key="w1", activate=True)
        assert service.get_entity(key)["status"] == "pending"
        service.link_memory("m2", key, window_key="w1", activate=True)
        assert service.get_entity(key)["status"] == "pending"
        service.link_memory("m3", key, window_key="w2", activate=True)
        assert service.get_entity(key)["status"] == "pending"
        service.link_memory("m4", key, window_key="w3", activate=True)
        assert service.get_entity(key)["status"] == "active"

    def test_duplicate_retry_same_memory_is_idempotent(self, store, service):
        store.insert(_entry("m1"))
        key = service.create_pending_entity("event", "Launch")["entity_key"]
        assert service.link_memory("m1", key, window_key="w1") is True
        version = service.get_entity(key)["evidence_version"]
        assert service.link_memory("m1", key, window_key="w1") is False
        assert service.get_entity(key)["evidence_version"] == version
        linked_events = [
            event
            for event in service.events_for_memory("m1")
            if event["event_type"] == "memory_entity_linked"
        ]
        assert len(linked_events) == 1

    def test_collision_scan_keeps_formation_candidate_pending(
        self, store, service
    ):
        _active(service, "Lily", identity_note="Known Lily")
        candidate = service.create_pending_entity(
            "person",
            "Lily",
            identity_note="Possibly another Lily",
            origin="manual-proposal",
        )
        for index in range(3):
            memory_id = f"collision-{index}"
            store.insert(_entry(memory_id))
            service.link_memory(
                memory_id,
                candidate["entity_key"],
                window_key=f"window-{index}",
                activate=True,
            )
        assert service.get_entity(candidate["entity_key"])["status"] == "pending"
        assert any(
            event["event_type"] == "entity_activation_collision"
            for event in service.events_for_entity(candidate["entity_key"])
        )

    def test_immediate_user_naming_requires_verbatim_surface(self, service):
        success = service.create_user_named_entity(
            "person",
            "Kaylee",
            naming_surface="Kaylee",
            context=_user_context("Kaylee is my friend."),
        )
        assert success["created"] is True
        assert success["entity"]["status"] == "active"

        failure = service.create_user_named_entity(
            "person",
            "Lee Ann",
            naming_surface="Lee Ann",
            context=_user_context("She is my friend."),
        )
        assert failure["created"] is False
        assert "verbatim" in failure["error"]
        assert service.resolve_alias("Lee Ann") == []
        rejected = service.events_for_entity("person:lee-ann")
        assert rejected == []
        row = store_event(service, "entity_creation_rejected")
        assert row["source_message_id"] == "msg-user-1"

    def test_non_user_source_cannot_immediately_activate(self, service):
        failure = service.create_user_named_entity(
            "person",
            "Kaylee",
            naming_surface="Kaylee",
            context=EntityExecutionContext(
                actor_node="agent:test",
                source_message_id="agent-msg",
                source_author="agent:researcher",
                source_content="Kaylee",
            ),
        )
        assert failure["created"] is False
        assert "user-authored" in failure["error"]


def store_event(service: EntityService, event_type: str) -> dict:
    row = service.connection.execute(
        "SELECT sequence, event_type, source_message_id, source_author, reason "
        "FROM entity_events WHERE event_type = ? ORDER BY sequence DESC LIMIT 1",
        (event_type,),
    ).fetchone()
    assert row is not None
    return {
        "sequence": row[0],
        "event_type": row[1],
        "source_message_id": row[2],
        "source_author": row[3],
        "reason": row[4],
    }


class TestCurationEvents:
    def test_curation_essay_event_is_recorded_and_unknown_type_is_rejected(
        self, service
    ):
        sequence = service.record_curation_event(
            "curation_essays",
            run_key="curation-essay-test",
            reason="essay generation during curation turn",
            details={"written": ["person:test"]},
        )

        row = store_event(service, "curation_essays")
        assert row["sequence"] == sequence
        assert row["reason"] == "essay generation during curation turn"

        with pytest.raises(EntityError, match="unsupported curation event type"):
            service.record_curation_event("unsupported_curation_event")


class TestLinksGroupsAndRegistry:
    def test_remove_then_readd_preserves_ordered_history(self, store, service):
        store.insert(_entry("m1"))
        key = _active(service, "Mesh", entity_type="project")
        service.link_memory("m1", key)
        service.unlink_memory("m1", key)
        service.link_memory("m1", key)
        assert [item["entity_key"] for item in service.links_for_memory("m1")] == [key]
        history = [
            event["event_type"]
            for event in service.events_for_memory("m1")
            if event["event_type"].startswith("memory_entity_")
        ]
        assert history == [
            "memory_entity_linked",
            "memory_entity_unlinked",
            "memory_entity_linked",
        ]

    def test_group_membership_only_bumps_group(self, service):
        group = _active(service, "Research Team", entity_type="group")
        member = _active(service, "Lily")
        group_start = service.get_entity(group)["evidence_version"]
        member_start = service.get_entity(member)["evidence_version"]

        assert service.add_group_member(group, member, role="lead") is True
        assert service.add_group_member(group, member, role="lead") is False
        assert service.add_group_member(group, member, role="reviewer") is True
        assert service.remove_group_member(group, member) is True
        assert service.get_entity(group)["evidence_version"] == group_start + 3
        assert service.get_entity(member)["evidence_version"] == member_start

    def test_list_registry_has_no_cap(self, service):
        for index in range(125):
            service.create_pending_entity("event", f"Event {index:03d}")
        registry = service.list_registry(statuses=("pending",))
        assert len(registry) == 125
        assert registry == sorted(
            registry,
            key=lambda item: (
                item["entity_type"],
                item["display_name"].casefold(),
                item["entity_key"],
            ),
        )

    def test_read_apis_and_dossier_synchronization(self, store, service):
        store.insert(_entry("m1"))
        group = _active(service, "Team", entity_type="group")
        member = _active(service, "Lily")
        service.link_memory("m1", member)
        service.add_group_member(group, member, role="lead")
        assert service.get_entity(member)["display_name"] == "Lily"
        assert service.memory_ids_for_entity(member) == ["m1"]
        assert service.group_members(group)[0]["role"] == "lead"
        assert service.events_for_entity(member)
        assert service.events_for_memory("m1")
        assert service.dossier_needs_work(member) is True

        store.create_essay(
            member,
            body="Lily dossier.",
            title="Lily",
            citations=["m1"],
            cross_refs=[],
        )
        from mesh.memory.entities import canonical_dossier_hash

        entity = service.get_entity(member)
        digest_hash = canonical_dossier_hash(
            "Lily", "Lily dossier.", ["m1"], []
        )
        store._conn.execute(
            "UPDATE essays SET curated_version = ?, verified_hash = ?, "
            "verified_at = ? WHERE entity_key = ?",
            (
                entity["evidence_version"],
                digest_hash,
                "2026-07-28",
                member,
            ),
        )
        store._conn.commit()
        assert service.dossier_needs_work(member) is False


class TestTransactionalCorrection:
    def test_source_edit_and_link_swap_commit_together(self, store, service):
        store.insert(_entry("m1", "Old source"))
        old_key = _active(service, "Old Project", entity_type="project")
        new_key = _active(service, "New Project", entity_type="project")
        service.link_memory("m1", old_key)
        snapshot = service._memory_snapshot("m1")

        result = service.correct_link_transactional(
            "m1",
            reason="correct attribution",
            context=_user_context("Use New Project."),
            prepared_snapshot=snapshot,
            memory_patch={"summary": "Corrected source"},
            remove_entity_key=old_key,
            add_entity_key=new_key,
            reflection_embedding=np.full(4, 2, dtype=np.float32),
            retrieval_key_embedding=np.full(4, 3, dtype=np.float32),
            embeddings_prepared=True,
        )
        assert result["changed"] is True
        assert store.get("m1").summary == "Corrected source"
        assert [link["entity_key"] for link in service.links_for_memory("m1")] == [
            new_key
        ]
        assert store.search_fts("Corrected source")[0][0] == "m1"

    def test_forced_event_failure_rolls_back_every_surface(self, store, service):
        store.insert(_entry("m1", "Old source"))
        old_key = _active(service, "Old Project", entity_type="project")
        new_key = _active(service, "New Project", entity_type="project")
        service.link_memory("m1", old_key)
        snapshot = service._memory_snapshot("m1")
        versions = {
            key: service.get_entity(key)["evidence_version"]
            for key in (old_key, new_key)
        }
        event_count = store._conn.execute(
            "SELECT COUNT(*) FROM entity_events"
        ).fetchone()[0]
        store._conn.execute(
            """
            CREATE TRIGGER fail_correction_events
            BEFORE INSERT ON entity_events
            BEGIN
                SELECT RAISE(ABORT, 'forced event failure');
            END
            """
        )
        store._conn.commit()

        with pytest.raises(sqlite3.IntegrityError, match="forced event failure"):
            service.correct_link_transactional(
                "m1",
                reason="must roll back",
                context=_user_context("Use New Project."),
                prepared_snapshot=snapshot,
                memory_patch={"summary": "Must not persist"},
                remove_entity_key=old_key,
                add_entity_key=new_key,
                reflection_embedding=np.ones(4, dtype=np.float32),
                retrieval_key_embedding=np.ones(4, dtype=np.float32),
                embeddings_prepared=True,
            )

        assert store.get("m1").summary == "Old source"
        assert [link["entity_key"] for link in service.links_for_memory("m1")] == [
            old_key
        ]
        assert store.search_fts("Must not persist") == []
        assert {
            key: service.get_entity(key)["evidence_version"]
            for key in (old_key, new_key)
        } == versions
        assert store._conn.execute(
            "SELECT COUNT(*) FROM entity_events"
        ).fetchone()[0] == event_count

    def test_edit_linked_memory_bumps_all_and_only_linked_entities(
        self, store, service
    ):
        store.insert(_entry("m1"))
        linked_a = _active(service, "Alice")
        linked_b = _active(service, "Bob")
        unlinked = _active(service, "Carol")
        service.link_memory("m1", linked_a)
        service.link_memory("m1", linked_b)
        before = {
            key: service.get_entity(key)["evidence_version"]
            for key in (linked_a, linked_b, unlinked)
        }
        result = service.edit_memory_transactional(
            "m1",
            {"summary": "New summary"},
            prepared_snapshot=service._memory_snapshot("m1"),
            reflection_embedding=np.ones(4, dtype=np.float32),
            retrieval_key_embedding=np.ones(4, dtype=np.float32),
            embeddings_prepared=True,
        )
        assert result["affected_entity_keys"] == sorted([linked_a, linked_b])
        assert service.get_entity(linked_a)["evidence_version"] == before[linked_a] + 1
        assert service.get_entity(linked_b)["evidence_version"] == before[linked_b] + 1
        assert service.get_entity(unlinked)["evidence_version"] == before[unlinked]

    def test_combined_source_and_swap_bumps_each_entity_once(
        self, store, service
    ):
        store.insert(_entry("m1"))
        old_key = _active(service, "Old")
        new_key = _active(service, "New")
        service.link_memory("m1", old_key)
        before = {
            key: service.get_entity(key)["evidence_version"]
            for key in (old_key, new_key)
        }
        service.correct_link_transactional(
            "m1",
            reason="combined",
            context=_user_context("New"),
            prepared_snapshot=service._memory_snapshot("m1"),
            memory_patch={"summary": "Changed"},
            remove_entity_key=old_key,
            add_entity_key=new_key,
            reflection_embedding=np.ones(4, dtype=np.float32),
            retrieval_key_embedding=np.ones(4, dtype=np.float32),
            embeddings_prepared=True,
        )
        assert service.get_entity(old_key)["evidence_version"] == before[old_key] + 1
        assert service.get_entity(new_key)["evidence_version"] == before[new_key] + 1

    def test_noop_edit_has_no_event_or_version_bump(self, store, service):
        store.insert(_entry("m1"))
        key = _active(service, "Lily")
        service.link_memory("m1", key)
        before_version = service.get_entity(key)["evidence_version"]
        before_events = len(service.events_for_memory("m1"))
        snapshot = service._memory_snapshot("m1")
        result = service.edit_memory_transactional(
            "m1",
            {"summary": snapshot["summary"]},
            prepared_snapshot=snapshot,
        )
        assert result["changed"] is False
        assert service.get_entity(key)["evidence_version"] == before_version
        assert len(service.events_for_memory("m1")) == before_events

    def test_lost_concurrent_edit_is_detected(self, store, service):
        store.insert(_entry("m1", "Prepared source"))
        snapshot = service._memory_snapshot("m1")
        store.update_entry("m1", summary="Concurrent source")
        with pytest.raises(
            Exception, match="changed while correction embeddings were prepared"
        ):
            service.edit_memory_transactional(
                "m1",
                {"summary": "Prepared replacement"},
                prepared_snapshot=snapshot,
                reflection_embedding=np.ones(4, dtype=np.float32),
                retrieval_key_embedding=np.ones(4, dtype=np.float32),
                embeddings_prepared=True,
            )
        assert store.get("m1").summary == "Concurrent source"

    @pytest.mark.asyncio
    async def test_embedding_failure_clears_superseded_vectors(
        self, store
    ):
        from mesh.memory.system_v2 import MemorySystemV2

        store.insert(_entry("m1", "Old source"))
        system = MemorySystemV2(nickname="test", llm_client=None)
        system._store = store
        system._pool = [store.get("m1")]

        class FailingEmbedder:
            async def embed_batch_to_arrays(self, _texts):
                raise RuntimeError("embedding unavailable")

        system._embedder = FailingEmbedder()
        result = await system.edit_entry("m1", summary="New source")
        assert "changed=true" in result
        updated = store.get("m1")
        assert updated.reflection_embedding is None
        assert updated.retrieval_key_embedding is None
        assert store.search_fts("New source")[0][0] == "m1"

    def test_delete_linked_memory_cascades_and_audits(self, store, service):
        store.insert(_entry("m1"))
        key = _active(service, "Lily")
        service.link_memory("m1", key)
        before = service.get_entity(key)["evidence_version"]
        assert store.delete("m1", actor_node="agent:test") is True
        assert store.get("m1") is None
        assert service.links_for_memory("m1") == []
        assert service.get_entity(key)["evidence_version"] == before + 1
        assert [
            event["event_type"] for event in service.events_for_memory("m1")
        ][-2:] == ["memory_link_removed_for_delete", "memory_deleted"]

    def test_prune_failure_leaves_pool_and_database_aligned(
        self, store, service
    ):
        from mesh.memory.system_v2 import MemorySystemV2

        older = _entry("m1")
        newer = _entry("m2")
        newer.created_at = datetime.now(timezone.utc)
        store.insert(older)
        store.insert(newer)
        key = _active(service, "Lily")
        service.link_memory("m1", key)
        system = MemorySystemV2(
            nickname="test", llm_client=None, pool_max_entries=1
        )
        system._store = store
        system._pool = [store.get("m1"), store.get("m2")]
        system._active_ids = set()
        store._conn.execute(
            """
            CREATE TRIGGER fail_prune_audit
            BEFORE INSERT ON entity_events
            BEGIN
                SELECT RAISE(ABORT, 'forced prune audit failure');
            END
            """
        )
        store._conn.commit()

        with pytest.raises(
            sqlite3.IntegrityError, match="forced prune audit failure"
        ):
            system._prune_pool()
        assert [entry.id for entry in system._pool] == ["m1", "m2"]
        assert store.get("m1") is not None
        assert store.get("m2") is not None
        assert service.links_for_memory("m1")[0]["entity_key"] == key


class TestEssayVerificationInvalidation:
    def _seed_verified(self, store: MemoryStore, key: str = "person:lily"):
        if store.get_essay(key) is None:
            store.create_essay(
                key, body="Original body.", title="Lily", citations=["m1"]
            )
        store._conn.execute(
            "UPDATE essays SET curated_version = 7, "
            "verified_hash = 'verified', verified_at = '2026-07-28' "
            "WHERE entity_key = ?",
            (key,),
        )
        store._conn.commit()

    def _assert_invalidated(self, store: MemoryStore, key: str = "person:lily"):
        row = store._conn.execute(
            "SELECT curated_version, verified_hash, verified_at "
            "FROM essays WHERE entity_key = ?",
            (key,),
        ).fetchone()
        assert row == (7, "", None)

    @pytest.mark.parametrize(
        "write_path",
        [
            "store_update",
            "store_edit",
            "store_citations_and_refs",
            "store_citations_only",
        ],
    )
    def test_store_write_paths_invalidate(self, store, write_path):
        self._seed_verified(store)
        if write_path == "store_update":
            store.update_essay("person:lily", "Updated body.", title="Lil")
        elif write_path == "store_edit":
            store.essay_edit("person:lily", "Original", "Changed")
        elif write_path == "store_citations_and_refs":
            store.update_essay_citations(
                "person:lily", ["m2"], cross_refs=["project:mesh"]
            )
        else:
            store.update_essay_citations("person:lily", ["m3"])
        self._assert_invalidated(store)

    def test_store_create_path_starts_unverified(self, store):
        store.create_essay("event:new", "New body")
        row = store._conn.execute(
            "SELECT curated_version, verified_hash, verified_at "
            "FROM essays WHERE entity_key = 'event:new'"
        ).fetchone()
        assert row == (0, "", None)

    def test_fold_create_path_starts_unverified(self, tmp_path):
        db = tmp_path / "fold.db"
        con = sqlite3.connect(db)
        _ensure_essays_table(con)
        result = _exec_essay_edit(
            con,
            {
                "key": "event:new",
                "old_text": "",
                "new_text": "New fold body.",
                "title": "New",
            },
        )
        assert "created" in result
        assert con.execute(
            "SELECT curated_version, verified_hash, verified_at "
            "FROM essays WHERE entity_key = 'event:new'"
        ).fetchone() == (0, "", None)
        con.close()

    def test_fold_update_path_invalidates(self, store):
        self._seed_verified(store)
        result = _exec_essay_edit(
            store._conn,
            {
                "key": "person:lily",
                "old_text": "Original",
                "new_text": "Updated",
            },
        )
        assert "updated" in result.lower()
        self._assert_invalidated(store)


class TestAttributionAndFailClosedBehavior:
    def test_service_mutations_are_disabled_by_default(self, store):
        read_only_service = EntityService(store._conn)
        assert read_only_service.list_registry(statuses=()) == []
        with pytest.raises(
            Exception, match="entity resolution mutations are disabled"
        ):
            read_only_service.create_pending_entity("person", "Lily")

    def test_public_tool_schema_has_no_authority_arguments(self):
        import mesh.tool_implementations  # noqa: F401
        from mesh.tools import get_registry

        definition = get_registry().get("entity_link_correct")
        names = {parameter.name for parameter in definition.parameters}
        assert "source_message_id" not in names
        assert "immediate" not in names

    def test_tool_visibility_is_feature_gated(self):
        from mesh.agent_node import AgentNode
        from mesh.config import NodeConfig
        from mesh.router_v2 import RouterV2
        from mesh.tools import ToolRegistry

        disabled = AgentNode(
            NodeConfig(
                id="agent:test:disabled",
                tools=["entity_link_correct"],
                entity_resolution_mode="off",
            ),
            tool_registry=ToolRegistry(),
        )
        shadow = AgentNode(
            NodeConfig(
                id="agent:test:shadow",
                tools=["entity_link_correct"],
                entity_resolution_mode="shadow",
            ),
            tool_registry=ToolRegistry(),
        )
        enabled = AgentNode(
            NodeConfig(
                id="agent:test:enabled",
                tools=[],
                entity_resolution_mode="write",
            ),
            tool_registry=ToolRegistry(),
        )
        assert "entity_link_correct" not in disabled.enabled_tools
        assert "entity_link_correct" not in shadow.enabled_tools
        assert "entity_link_correct" in enabled.enabled_tools

        async def worker_fn(*_args):
            return None

        async def send_fn(*_args):
            return None

        disabled_router = RouterV2(
            worker_fn,
            send_fn,
            entity_resolution_mode="off",
        )
        shadow_router = RouterV2(
            worker_fn,
            send_fn,
            entity_resolution_mode="shadow",
        )
        enabled_router = RouterV2(
            worker_fn,
            send_fn,
            entity_resolution_mode="write",
        )
        assert "entity_link_correct" not in disabled_router._router_tool_names
        assert "entity_link_correct" not in shadow_router._router_tool_names
        assert "entity_link_correct" in enabled_router._router_tool_names

    @pytest.mark.asyncio
    async def test_no_context_handler_fails_closed_and_mutates_nothing(
        self, store
    ):
        import mesh.tool_implementations as tools

        store.insert(_entry("m1"))
        before = store._conn.total_changes
        result = await tools.entity_link_correct(
            memory_id="m1",
            reason="socket attempt",
            new_entity_type="person",
            new_display_name="Lily",
            naming_surface="Lily",
        )
        assert "requires an in-process execution context" in result
        assert store._conn.total_changes == before
        assert store._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0

    @pytest.mark.asyncio
    async def test_model_supplied_source_message_id_is_rejected(self):
        from mesh.agent_node import AgentNode
        from mesh.config import NodeConfig
        from mesh.tools import get_registry

        node = AgentNode(
            NodeConfig(
                id="agent:test:authority",
                tools=["entity_link_correct"],
                entity_resolution_enabled=True,
            ),
            tool_registry=get_registry(),
        )
        result = await node._execute_entity_link_correct(
            {
                "memory_id": "m1",
                "reason": "forged",
                "source_message_id": "model-forged-id",
            },
            SimpleNamespace(
                id="real-message",
                from_node="user:testuser",
                content="Lily",
            ),
        )
        assert "authority metadata is context-bound" in result

    @pytest.mark.asyncio
    async def test_actual_unix_socket_path_fails_closed(
        self, tmp_path, monkeypatch
    ):
        import aiohttp
        from mesh.agent_node import AgentNode
        from mesh.config import NodeConfig
        from mesh.tools import get_registry
        import mesh.paths

        monkeypatch.setattr(mesh.paths, "real_home", lambda: tmp_path)
        node = AgentNode(
            NodeConfig(
                id="agent:test:entity-socket",
                tools=["entity_link_correct"],
                entity_resolution_enabled=True,
            ),
            tool_registry=get_registry(),
        )
        node._current_trigger_msg = SimpleNamespace(
            id="stale-global",
            from_node="user:testuser",
            content="Lily",
        )
        socket_path = await node._start_tool_socket()
        try:
            connector = aiohttp.UnixConnector(path=socket_path)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    "http://localhost/tool",
                    json={
                        "name": "entity_link_correct",
                        "arguments": {
                            "memory_id": "m1",
                            "reason": "socket attempt",
                            "new_entity_type": "person",
                            "new_display_name": "Lily",
                            "naming_surface": "Lily",
                        },
                    },
                ) as response:
                    payload = await response.json()
            assert response.status == 200
            assert "requires an in-process execution context" in payload["result"]
        finally:
            await node._stop_tool_socket()

    def test_user_naming_attribution_comes_from_context(self, service):
        result = service.create_user_named_entity(
            "person",
            "Lily",
            naming_surface="Lily",
            context=EntityExecutionContext(
                actor_node="agent:test",
                source_message_id="trusted-message",
                source_author="user:testuser",
                source_content="Lily is distinct.",
            ),
            reason="named by user",
        )
        key = result["entity"]["entity_key"]
        events = service.events_for_entity(key)
        assert {event["source_message_id"] for event in events} == {
            "trusted-message"
        }
        assert {event["source_author"] for event in events} == {"user:testuser"}


class TestSnapshotMigration:
    @pytest.mark.skip(
        reason=(
            "requires a mutable production memory snapshot whose entity tables "
            "are empty; live curation databases validly contain entity rows"
        )
    )
    def test_real_database_backup_migration_is_empty_and_idempotent(
        self, tmp_path
    ):
        memory_dir = Path("/home/testuser/.mesh/memory")
        candidates = sorted(
            path for path in memory_dir.glob("*.db") if path.is_file()
        )
        if not candidates:
            pytest.skip("no production memory database available for backup test")
        source_path = candidates[0]
        snapshot_path = tmp_path / "snapshot.db"
        source = sqlite3.connect(
            f"file:{source_path}?mode=ro", uri=True
        )
        destination = sqlite3.connect(snapshot_path)
        source.backup(destination)
        destination.close()
        source.close()

        before = sqlite3.connect(snapshot_path)
        memory_columns = [
            row[1] for row in before.execute("PRAGMA table_info(memories)")
        ]
        memory_payload = before.execute(
            f"SELECT {', '.join(memory_columns)} FROM memories ORDER BY id"
        ).fetchall()
        essay_columns = {
            row[1] for row in before.execute("PRAGMA table_info(essays)")
        }
        legacy_essay_columns = [
            name
            for name in (
                "entity_key",
                "title",
                "body",
                "citations",
                "cross_refs",
                "patch_count",
                "created_at",
                "updated_at",
            )
            if name in essay_columns
        ]
        essay_payload = before.execute(
            f"SELECT {', '.join(legacy_essay_columns)} "
            "FROM essays ORDER BY entity_key"
        ).fetchall()
        before.close()

        migrated = MemoryStore("snapshot", db_dir=str(tmp_path))
        tables = {
            row[0]
            for row in migrated._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {
            "entities",
            "entity_aliases",
            "memory_entities",
            "entity_group_members",
            "entity_events",
        } <= tables
        for table in (
            "entities",
            "entity_aliases",
            "memory_entities",
            "entity_group_members",
            "entity_events",
        ):
            assert migrated._conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0] == 0
        assert migrated._conn.execute(
            f"SELECT {', '.join(memory_columns)} FROM memories ORDER BY id"
        ).fetchall() == memory_payload
        assert migrated._conn.execute(
            f"SELECT {', '.join(legacy_essay_columns)} "
            "FROM essays ORDER BY entity_key"
        ).fetchall() == essay_payload
        assert migrated._conn.execute(
            "SELECT DISTINCT curated_version, verified_hash, verified_at "
            "FROM essays"
        ).fetchall() in ([], [(0, "", None)])
        assert EntityService(migrated._conn).list_registry(statuses=()) == []
        migrated.close()

        reopened = MemoryStore("snapshot", db_dir=str(tmp_path))
        assert EntityService(reopened._conn).list_registry(statuses=()) == []
        assert reopened._conn.execute(
            f"SELECT {', '.join(memory_columns)} FROM memories ORDER BY id"
        ).fetchall() == memory_payload
        reopened.close()
