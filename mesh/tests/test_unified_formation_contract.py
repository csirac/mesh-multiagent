"""Regression tests for the unified recall-oriented formation contract."""
from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pytest

from mesh.conversation_history import Turn
from mesh.memory.formation_v3 import LLMSegmenterV3
from mesh.memory.formation_contract import (
    FORMATION_CONTRACT_VERSION,
    FORMATION_PROMPT,
    FORMATION_PROMPT_PATH,
    FormationContractError,
    parse_formation_response,
    render_formation_prompt,
)
from mesh.memory.store import (
    MemoryEntry,
    MemoryStore,
)
from mesh.memory.system_v2 import MemorySystemV2


def _record(**overrides) -> dict:
    record = {
        "summary": "Project Owner raised $250 in a charity walk.",
        "reflection": "This is a durable personal activity and amount.",
        "trace": "I raised $250 in a charity walk.",
        "retrieval_key": "Project Owner charity walk raised $250",
        "tags": "charity,personal",
        "outcome": "",
        "topic_label": "Charity walk",
        "project": "personal",
        "event_date": "2026-07-09",
        "digest_candidate": False,
    }
    record.update(overrides)
    return record


class _LLM:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    async def complete(self, prompt: str, **_kwargs) -> str:
        self.prompts.append(prompt)
        return self.response


class _SequenceLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    async def complete(self, _prompt: str, **_kwargs) -> str:
        return self.responses.pop(0)


class _Embedder:
    async def embed_batch_to_arrays(self, texts: list[str]) -> list[np.ndarray]:
        return [np.ones(4, dtype=np.float32) for _ in texts]


def _turn(content: str) -> Turn:
    return Turn(
        role="user",
        content=content,
        timestamp=datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc),
        from_node="user:testuser",
        to_node="agent:coder:coder1",
        seq_id=1,
    )


def test_one_canonical_formation_prompt_artifact():
    assert FORMATION_CONTRACT_VERSION == 2
    assert LLMSegmenterV3.PROMPT == FORMATION_PROMPT
    assert FORMATION_PROMPT_PATH.name == "form_memories_lowbar.txt"
    assert not FORMATION_PROMPT_PATH.with_name("form_memories.txt").exists()


def test_entity_contract_is_mode_conditional():
    off = render_formation_prompt(
        agent_label="test",
        entity_resolution_mode="off",
    )
    shadow = render_formation_prompt(
        agent_label="test",
        entity_resolution_mode="shadow",
    )
    assert "ENTITY ASSIGNMENT" not in off
    assert '"existing_keys"' not in off
    assert "ENTITY ASSIGNMENT" in shadow
    assert '"existing_keys"' in shadow
    assert '"new_entities"' in shadow
    assert '"unresolved"' in shadow


def test_entity_metadata_is_required_and_preserved_when_enabled():
    record = _record(entity={
        "existing_keys": ["person:owner"],
        "new_entities": [{
            "entity_type": "project",
            "display_name": "Mesh",
            "identity_note": "The mesh platform.",
            "aliases": ["hello-world"],
        }],
        "unresolved": [{
            "surface": "the group",
            "candidates": ["group:mesh-team"],
            "reason": "not enough context",
        }],
    })
    parsed = parse_formation_response(
        json.dumps([record]),
        entity_resolution_mode="shadow",
        known_entity_statuses={
            "person:owner": "active",
            "group:mesh-team": "pending",
        },
    )
    assert parsed[0]["entity"]["existing_keys"] == ["person:owner"]
    assert parsed[0]["entity"]["new_entities"][0]["display_name"] == "Mesh"
    assert parsed[0]["entity"]["unresolved"][0]["surface"] == "the group"
    assert parsed[0]["entity_validation_failures"] == []

    with pytest.raises(FormationContractError, match="missing fields.*entity"):
        parse_formation_response(
            json.dumps([_record()]),
            entity_resolution_mode="write",
        )


def test_entity_field_validation_salvages_core_but_bad_json_does_not():
    record = _record(entity={
        "existing_keys": ["person:missing", "person:retired"],
        "new_entities": [{
            "entity_type": "topic",
            "display_name": "Not Allowed",
            "identity_note": "",
            "aliases": [],
        }],
        "unresolved": "not-a-list",
    })
    parsed = parse_formation_response(
        json.dumps([record]),
        entity_resolution_mode="shadow",
        known_entity_statuses={"person:retired": "retired"},
    )
    assert parsed[0]["summary"] == record["summary"]
    assert parsed[0]["entity"] == {
        "existing_keys": [],
        "new_entities": [],
        "unresolved": [],
    }
    failures = " ".join(parsed[0]["entity_validation_failures"])
    assert "unknown entity key" in failures
    assert "retired entity key" in failures
    assert "invalid entity_type" in failures
    assert "unresolved must be a list" in failures

    with pytest.raises(FormationContractError, match="not JSON"):
        parse_formation_response(
            "not-json",
            entity_resolution_mode="shadow",
        )


@pytest.mark.asyncio
async def test_extraction_contract_keeps_two_facts_from_one_turn():
    """The measured lowbar behavior: one turn may yield two distinct records."""
    response = json.dumps([
        _record(),
        _record(
            summary="Project Owner subscribes to Architectural Digest.",
            reflection="This is a persistent subscription preference.",
            trace="I also subscribe to Architectural Digest.",
            retrieval_key="Project Owner Architectural Digest subscription",
            tags="subscription,design,personal",
            topic_label="Architectural Digest subscription",
        ),
    ])
    llm = _LLM(response)
    extractor = LLMSegmenterV3(
        llm, window_size=10, overlap=2, defer_tail_turns=1,
    )

    records = await extractor.segment([
        _turn(
            "I raised $250 in a charity walk. "
            "By the way, I also subscribe to Architectural Digest."
        )
    ])

    assert len(records) == 2
    assert {record.metadata["topic_label"] for record in records} == {
        "Charity walk",
        "Architectural Digest subscription",
    }
    assert "two distinct\npersonal facts mentioned in the same turn = TWO records" in (
        llm.prompts[0]
    )


@pytest.mark.asyncio
async def test_any_window_parse_failure_blocks_cursor_eligible_result():
    """A partial extraction must not let the caller advance past lost facts."""
    llm = _SequenceLLM([
        json.dumps([_record(trace="turn zero durable fact")]),
        "not json",
        "still not json",
    ])
    extractor = LLMSegmenterV3(
        llm,
        window_size=3,
        overlap=1,
        defer_tail_turns=1,
    )

    with pytest.raises(ValueError, match="window 1 failed after 2 attempts"):
        await extractor.segment([
            _turn("turn zero durable fact"),
            _turn("turn one"),
            _turn("turn two"),
            _turn("turn three"),
            _turn("turn four"),
        ])


@pytest.mark.asyncio
async def test_live_formation_persists_full_contract_and_false_digest_candidate(
    tmp_path,
):
    model_trace = "I raised $250 in a charity walk."
    llm = _LLM(json.dumps([_record(trace=model_trace, project="")]))
    system = MemorySystemV2(
        nickname="unified",
        llm_client=llm,
        formation_v3_enabled=True,
        formation_v3_window_size=10,
        formation_v3_overlap=2,
        formation_v3_defer_tail=1,
    )
    system._store = MemoryStore("unified", db_dir=str(tmp_path))
    system._embedder = _Embedder()
    system._formation_lock = asyncio.Lock()
    system._integrate_entries_into_maps = lambda _entries: asyncio.sleep(0)

    assert await system.form_un_formed(
        [_turn("I raised $250 in a charity walk.")],
        "time-based",
    ) == 1

    entry = system._store.load()[0]
    assert entry.summary
    assert entry.reflection
    assert entry.trace == model_trace
    assert entry.retrieval_key
    assert entry.tags == ["charity", "personal"]
    assert entry.topic_label == "Charity walk"
    assert entry.event_date == "2026-07-09"
    assert entry.digest_candidate is False
    assert entry.formation_source == "live-extraction"
    system._store.close()


def _entry(entry_id: str, **overrides) -> MemoryEntry:
    values = {
        "id": entry_id,
        "created_at": datetime(2026, 7, 28, tzinfo=timezone.utc),
        "summary": "summary",
        "reflection": "reflection",
        "trace": "verbatim evidence",
        "trigger": "trigger",
        "retrieval_key": "retrieval",
        "topic_label": "topic",
        "project": "personal",
        "tags": ["one"],
        "outcome": "success",
        "event_date": "2026-07-09",
        "formation_source": "live-extraction",
        "digest_candidate": False,
    }
    values.update(overrides)
    return MemoryEntry(**values)


def test_event_date_and_provenance_round_trip_with_legacy_date_fallback(tmp_path):
    store = MemoryStore("roundtrip", db_dir=str(tmp_path))
    store.insert(_entry("new-row"))

    loaded = store.get("new-row")
    assert loaded.event_date == "2026-07-09"
    assert loaded.formation_source == "live-extraction"
    persisted = store._conn.execute(
        "SELECT event_date, formation_source FROM memories WHERE id='new-row'"
    ).fetchone()
    assert persisted == ("2026-07-09", "live-extraction")

    legacy = _entry(
        "legacy-fold",
        event_date="",
        formation_source="legacy-unknown",
        trigger="[fold-formed event:2026-07-10]",
    )
    store.insert(legacy)
    loaded_legacy = store.get("legacy-fold")
    assert loaded_legacy.event_date == "2026-07-10"
    assert loaded_legacy.formation_source == "legacy-nightly-fold"
    store.close()


def test_old_schema_migration_preserves_memory_and_essay_payloads(tmp_path):
    db = tmp_path / "legacy.db"
    connection = sqlite3.connect(db)
    connection.executescript(
        """
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
            reflection_embedding BLOB NOT NULL,
            retrieval_key_embedding BLOB NOT NULL,
            weight REAL NOT NULL DEFAULT 0.0,
            topic_label TEXT DEFAULT '',
            project TEXT DEFAULT '',
            digest_candidate INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE essays (
            entity_key TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            body TEXT NOT NULL DEFAULT '',
            citations TEXT NOT NULL DEFAULT '[]',
            cross_refs TEXT NOT NULL DEFAULT '[]',
            patch_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    memory_payload = (
        "old-id",
        "2026-07-01T00:00:00+00:00",
        "old summary",
        "old reflection",
        "old trace",
        "[fold-formed event:2026-06-30]",
        "old retrieval",
        '["old-tag"]',
        "success",
        b"\x00\x00\x80?",
        b"\x00\x00\x00@",
        1.25,
        "old topic",
        "old project",
        0,
    )
    essay_payload = (
        "person:owner",
        "Project Owner",
        "Exact essay body.",
        '["old-id"]',
        "[]",
        2,
        "2026-07-01",
        "2026-07-02",
    )
    connection.execute(
        "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        memory_payload,
    )
    connection.execute("INSERT INTO essays VALUES (?,?,?,?,?,?,?,?)", essay_payload)
    connection.commit()
    before_memory = connection.execute("SELECT * FROM memories").fetchone()
    before_essay = connection.execute(
        "SELECT entity_key, title, body, citations, cross_refs, patch_count, "
        "created_at, updated_at FROM essays"
    ).fetchone()
    connection.close()

    store = MemoryStore("legacy", db_dir=str(tmp_path))
    columns = {
        row[1] for row in store._conn.execute("PRAGMA table_info(memories)")
    }
    after_memory = store._conn.execute(
        "SELECT id, created_at, summary, reflection, trace, trigger_text, "
        "retrieval_key, tags, outcome, reflection_embedding, "
        "retrieval_key_embedding, weight, topic_label, project, "
        "digest_candidate FROM memories"
    ).fetchone()
    after_essay = store._conn.execute(
        "SELECT entity_key, title, body, citations, cross_refs, patch_count, "
        "created_at, updated_at FROM essays"
    ).fetchone()

    assert {"event_date", "formation_source"} <= columns
    assert after_memory == before_memory
    assert after_essay == before_essay
    assert store.get("old-id").event_date == "2026-06-30"
    store.close()


def test_insert_and_cursor_advance_roll_back_together_on_crash(tmp_path):
    store = MemoryStore("atomic", db_dir=str(tmp_path))
    store._conn.execute(
        """
        CREATE TRIGGER force_cursor_failure
        BEFORE INSERT ON formation_cursor
        BEGIN
            SELECT RAISE(ABORT, 'forced cursor failure');
        END
        """
    )
    store._conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="forced cursor failure"):
        store.insert_entry_and_advance_cursor(
            [_entry("must-rollback")],
            new_cursor=12,
            new_ts_utc="2026-07-28T00:00:00+00:00",
            entity_mutations=[{
                "op": "resolve_and_link",
                "memory_id": "must-rollback",
                "entity_type": "project",
                "display_name": "Mesh",
                "window_key": "opaque-window-12",
            }],
            entity_resolution_enabled=True,
        )

    assert store.get("must-rollback") is None
    assert store.get_formation_cursor() == (0, "")
    assert store._conn.execute(
        "SELECT value FROM meta WHERE key='memory_formation_contract'"
    ).fetchone() is None
    assert store._conn.execute(
        "SELECT memory_id FROM memory_formation_log"
    ).fetchall() == []
    assert store._conn.execute("SELECT * FROM entities").fetchall() == []
    assert store._conn.execute("SELECT * FROM memory_entities").fetchall() == []
    assert store._conn.execute("SELECT * FROM entity_events").fetchall() == []
    store.close()


def test_entity_resolver_savepoint_failure_still_commits_memory_and_cursor(
    tmp_path,
):
    store = MemoryStore("resolver-savepoint", db_dir=str(tmp_path))
    store.insert_entry_and_advance_cursor(
        [_entry("memory-survives")],
        new_cursor=22,
        new_ts_utc="2026-07-28T01:00:00+00:00",
        entity_mutations=[{
            "op": "raise",
            "message": "forced resolver failure",
        }],
        entity_resolution_enabled=True,
        entity_run_key="test-run",
    )

    assert store.get("memory-survives") is not None
    assert store.get_formation_cursor() == (
        22,
        "2026-07-28T01:00:00+00:00",
    )
    assert store._conn.execute("SELECT * FROM entities").fetchall() == []
    event = store._conn.execute(
        "SELECT event_type, reason, run_key FROM entity_events"
    ).fetchone()
    assert event == (
        "entity_resolution_failed",
        "forced resolver failure",
        "test-run",
    )
    store.close()


def test_known_projects_are_bounded_with_elision_telemetry(caplog):
    projects = [f"project-{index:03d}" for index in range(60)]
    with caplog.at_level("INFO", logger="mesh.memory.formation_v3"):
        rendered = LLMSegmenterV3._format_known_projects(projects)

    assert "project-000" in rendered
    assert "project-049" in rendered
    assert "project-050" not in rendered
    assert "shown=50 elided=10 total=60" in caplog.text
