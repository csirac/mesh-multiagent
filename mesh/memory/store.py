"""
Memory store — SQLite persistence for episodic memory entries and project maps.

Each agent has its own database at ~/.mesh/memory/{nickname}.db.
Stores three-tier memory entries (summary, reflection, trace) with
dual embeddings: reflection_embedding for diversity selection and
retrieval_key_embedding (LLM-generated task descriptor) for similarity retrieval.

v2 additions:
- project_maps table: living markdown documents per project
- project column on memories table: ties entries to project maps
"""

import json
import logging
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DTYPE = np.float32
FORMATION_CONTRACT_META_KEY = "memory_formation_contract"
AUTHORITATIVE_FORMATION_CONTRACT = "lowbar-extraction-v1"

_SENTINEL = object()  # distinguishes "not passed" from None for optional embedding args
_LEGACY_FOLD_EVENT_RE = re.compile(
    r"\[fold-formed event:(\d{4}-\d{2}-\d{2})\]"
)


@dataclass
class MemoryEntry:
    """A single episodic memory with three tiers of detail."""

    id: str
    created_at: datetime
    # Three-tier content
    summary: str       # Tier 1: one paragraph (always in context)
    reflection: str    # Tier 2: deep LLM reflection (via remember tool)
    trace: str         # Tier 3: clipped tool call trace (via remember full=true)
    trigger: str       # Original trigger message
    retrieval_key: str = ""  # LLM-generated task descriptor for retrieval
    topic_label: str = ""    # Topic label from topic segmentation classifier
    project: str = ""        # Project name (v2: ties entry to a project map)
    # Metadata
    tags: list[str] = field(default_factory=list)
    outcome: str = "success"  # success | partial | failure
    # Embeddings
    reflection_embedding: np.ndarray | None = None  # For diversity selection
    retrieval_key_embedding: np.ndarray | None = None  # For task-similarity retrieval
    weight: float = 0.0  # Cached withholding cost
    digest_candidate: bool = True  # Phase 1: fold injection filter; False = DB-only row
    event_date: str = ""  # Date the remembered event occurred (YYYY-MM-DD)
    formation_source: str = "legacy-unknown"  # Auditable minting pathway

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]


def _serialize_embedding(emb: np.ndarray | None) -> bytes | None:
    if emb is None:
        return None
    return emb.astype(EMBEDDING_DTYPE).tobytes()


def _deserialize_embedding(data: bytes | None, dim: int = 1536) -> np.ndarray | None:
    if data is None:
        return None
    return np.frombuffer(data, dtype=EMBEDDING_DTYPE).copy()


def resolve_event_date(event_date: str | None, trigger_text: str | None) -> str:
    """Return the real column value or the legacy fold trigger encoding."""
    if event_date:
        return str(event_date)
    match = _LEGACY_FOLD_EVENT_RE.search(trigger_text or "")
    return match.group(1) if match else ""


def resolve_formation_source(
    formation_source: str | None,
    trigger_text: str | None,
) -> str:
    """Refine the migration default when legacy fold provenance is explicit."""
    source = str(formation_source or "legacy-unknown")
    if source == "legacy-unknown" and _LEGACY_FOLD_EVENT_RE.search(trigger_text or ""):
        return "legacy-nightly-fold"
    return source


class MemoryStore:
    """SQLite-backed persistent memory store for a single agent."""

    def __init__(self, nickname: str, db_dir: str | None = None):
        self.nickname = nickname
        if db_dir is None:
            from ..paths import MEMORY_DIR
            db_dir = str(MEMORY_DIR)
        os.makedirs(db_dir, exist_ok=True)
        self._db_path = os.path.join(db_dir, f"{nickname}.db")
        self._conn: sqlite3.Connection | None = None
        self._open()

    def _open(self) -> None:
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        logger.info(f"Memory store opened: {self._db_path}")

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
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
                project TEXT DEFAULT '',
                digest_candidate INTEGER NOT NULL DEFAULT 1,
                event_date TEXT NOT NULL DEFAULT '',
                formation_source TEXT NOT NULL DEFAULT 'legacy-unknown'
            );

            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS personality (
                key TEXT PRIMARY KEY DEFAULT 'personality',
                value TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scheduled_wakes (
                id TEXT PRIMARY KEY,
                wake_time TEXT NOT NULL,
                prompt TEXT NOT NULL,
                requested_by TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS project_maps (
                id TEXT PRIMARY KEY,
                project_name TEXT NOT NULL UNIQUE,
                content TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                embedding BLOB,
                is_active INTEGER NOT NULL DEFAULT 0,
                project_dir TEXT
            );

            CREATE TABLE IF NOT EXISTS conversation_summary (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                summary_text TEXT NOT NULL,
                messages_summarized INTEGER NOT NULL DEFAULT 0,
                token_estimate INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS formation_cursor (
                id           INTEGER PRIMARY KEY CHECK (id = 1),
                last_index   INTEGER NOT NULL,
                last_ts_utc  TEXT    NOT NULL,
                updated_at   TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_formation_log (
                sequence     INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id    TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS migrations_complete (
                name         TEXT PRIMARY KEY,
                completed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS essays (
                entity_key   TEXT PRIMARY KEY,
                title        TEXT NOT NULL DEFAULT '',
                body         TEXT NOT NULL DEFAULT '',
                citations    TEXT NOT NULL DEFAULT '[]',
                cross_refs   TEXT NOT NULL DEFAULT '[]',
                patch_count  INTEGER NOT NULL DEFAULT 0,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                curated_version INTEGER NOT NULL DEFAULT 0,
                verified_hash TEXT NOT NULL DEFAULT '',
                verified_at TEXT
            );

        """)
        self._conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Run schema migrations."""
        def memory_columns() -> set[str]:
            cursor = self._conn.execute("PRAGMA table_info(memories)")
            return {row[1] for row in cursor.fetchall()}

        columns = memory_columns()
        if "trigger_embedding" in columns and "retrieval_key_embedding" not in columns:
            logger.info("Migrating: trigger_embedding -> retrieval_key_embedding")
            self._conn.execute(
                "ALTER TABLE memories RENAME COLUMN trigger_embedding TO retrieval_key_embedding"
            )
            self._conn.commit()
            columns = memory_columns()

        if "retrieval_key" not in columns:
            logger.info("Migrating: adding retrieval_key column")
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN retrieval_key TEXT NOT NULL DEFAULT ''"
            )
            self._conn.commit()
            columns = memory_columns()
        if "topic_label" not in columns:
            logger.info("Migrating: adding topic_label column")
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN topic_label TEXT DEFAULT ''"
            )
            self._conn.commit()
            columns = memory_columns()

        if "project" not in columns:
            logger.info("Migrating: adding project column to memories")
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN project TEXT DEFAULT ''"
            )
            self._conn.commit()
            columns = memory_columns()

        if "digest_candidate" not in columns:
            logger.info("Migrating: adding digest_candidate column to memories")
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN digest_candidate INTEGER NOT NULL DEFAULT 1"
            )
            self._conn.commit()
            columns = memory_columns()

        if "event_date" not in columns:
            logger.info("Migrating: adding event_date column to memories")
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN event_date TEXT NOT NULL DEFAULT ''"
            )
            self._conn.commit()
            columns = memory_columns()

        if "formation_source" not in columns:
            logger.info("Migrating: adding formation_source column to memories")
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN formation_source "
                "TEXT NOT NULL DEFAULT 'legacy-unknown'"
            )
            self._conn.commit()
            columns = memory_columns()

        cursor_essays = self._conn.execute("PRAGMA table_info(essays)")
        essay_columns = {row[1] for row in cursor_essays.fetchall()}
        if "curated_version" not in essay_columns:
            logger.info("Migrating: adding curated_version column to essays")
            self._conn.execute(
                "ALTER TABLE essays ADD COLUMN "
                "curated_version INTEGER NOT NULL DEFAULT 0"
            )
            self._conn.commit()
        if "verified_hash" not in essay_columns:
            logger.info("Migrating: adding verified_hash column to essays")
            self._conn.execute(
                "ALTER TABLE essays ADD COLUMN "
                "verified_hash TEXT NOT NULL DEFAULT ''"
            )
            self._conn.commit()
        if "verified_at" not in essay_columns:
            logger.info("Migrating: adding verified_at column to essays")
            self._conn.execute(
                "ALTER TABLE essays ADD COLUMN verified_at TEXT"
            )
            self._conn.commit()

        # Increment 2a durable entity registry.  This migration is intentionally
        # schema-only: legacy essays do not seed registry identities or aliases.
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_key       TEXT PRIMARY KEY,
                entity_type      TEXT NOT NULL,
                display_name     TEXT NOT NULL CHECK (trim(display_name) <> ''),
                identity_note    TEXT NOT NULL DEFAULT '',
                status           TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','active','retired')),
                replacement_key  TEXT,
                origin           TEXT NOT NULL,
                evidence_version INTEGER NOT NULL DEFAULT 0
                    CHECK (evidence_version >= 0),
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL,
                activated_at     TEXT,
                retired_at       TEXT,
                CHECK (replacement_key IS NULL OR replacement_key <> entity_key),
                FOREIGN KEY (replacement_key) REFERENCES entities(entity_key)
            );
            CREATE INDEX IF NOT EXISTS idx_entities_status_type
                ON entities(status, entity_type, display_name);

            CREATE TABLE IF NOT EXISTS entity_aliases (
                entity_key       TEXT NOT NULL,
                normalized_alias TEXT NOT NULL CHECK (normalized_alias <> ''),
                display_alias    TEXT NOT NULL,
                source           TEXT NOT NULL,
                created_at       TEXT NOT NULL,
                PRIMARY KEY (entity_key, normalized_alias),
                FOREIGN KEY (entity_key) REFERENCES entities(entity_key)
            );
            CREATE INDEX IF NOT EXISTS idx_entity_aliases_reverse
                ON entity_aliases(normalized_alias, entity_key);

            CREATE TABLE IF NOT EXISTS memory_entities (
                memory_id         TEXT NOT NULL,
                entity_key        TEXT NOT NULL,
                window_key        TEXT,
                assignment_source TEXT NOT NULL,
                assigned_at       TEXT NOT NULL,
                PRIMARY KEY (memory_id, entity_key),
                FOREIGN KEY (memory_id) REFERENCES memories(id),
                FOREIGN KEY (entity_key) REFERENCES entities(entity_key)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_entities_entity
                ON memory_entities(entity_key, assigned_at);
            CREATE INDEX IF NOT EXISTS idx_memory_entities_window
                ON memory_entities(entity_key, window_key);

            CREATE TABLE IF NOT EXISTS entity_group_members (
                group_key  TEXT NOT NULL,
                member_key TEXT NOT NULL,
                role       TEXT NOT NULL DEFAULT '',
                source     TEXT NOT NULL,
                added_at   TEXT NOT NULL,
                PRIMARY KEY (group_key, member_key),
                CHECK (group_key <> member_key),
                FOREIGN KEY (group_key) REFERENCES entities(entity_key),
                FOREIGN KEY (member_key) REFERENCES entities(entity_key)
            );
            CREATE INDEX IF NOT EXISTS idx_entity_group_members_member
                ON entity_group_members(member_key, group_key);

            CREATE TABLE IF NOT EXISTS entity_events (
                sequence          INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type        TEXT NOT NULL,
                entity_key        TEXT,
                memory_id         TEXT,
                old_entity_key    TEXT,
                new_entity_key    TEXT,
                actor_node        TEXT NOT NULL,
                source_message_id TEXT,
                source_author     TEXT,
                reason            TEXT NOT NULL DEFAULT '',
                window_key        TEXT,
                run_key           TEXT,
                details_json      TEXT NOT NULL DEFAULT '{}',
                created_at        TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_entity_events_entity
                ON entity_events(entity_key, sequence);
            CREATE INDEX IF NOT EXISTS idx_entity_events_memory
                ON entity_events(memory_id, sequence);
            CREATE INDEX IF NOT EXISTS idx_entity_events_old_key
                ON entity_events(old_entity_key, sequence);
            CREATE INDEX IF NOT EXISTS idx_entity_events_new_key
                ON entity_events(new_entity_key, sequence);
            -- Per-write-attempt audit (G-004).  ``curation_write_attempt``
            -- rows are read two ways: by turn, to resolve which refused
            -- artifacts later landed, and by (type, time) for a windowed
            -- report.  Both are covering; neither existed before, so the
            -- reporting query would otherwise scan the whole trail.
            CREATE INDEX IF NOT EXISTS idx_entity_events_run
                ON entity_events(run_key, sequence);
            CREATE INDEX IF NOT EXISTS idx_entity_events_type_created
                ON entity_events(event_type, created_at);

            -- Durable pending-additions ledger (T-001 / G-001).  An
            -- over-ceiling write is refused, never truncated; when the model
            -- does not land it inside the turn the addition is written here
            -- and offered back at the top of a later curation turn, so a
            -- refusal is never a terminal drop.  See
            -- ``mesh/memory/pending_additions.py``.
            CREATE TABLE IF NOT EXISTS curation_pending_additions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                agent            TEXT NOT NULL DEFAULT '',
                target_artifact  TEXT NOT NULL,
                tool             TEXT NOT NULL,
                entity_key       TEXT NOT NULL DEFAULT '',
                old_text         TEXT NOT NULL DEFAULT '',
                new_text         TEXT NOT NULL,
                replace_all      INTEGER NOT NULL DEFAULT 0,
                reason           TEXT NOT NULL DEFAULT '',
                measured_tokens  INTEGER,
                budget_tokens    INTEGER,
                origin_turn_id   TEXT NOT NULL DEFAULT '',
                status           TEXT NOT NULL DEFAULT 'pending',
                offers           INTEGER NOT NULL DEFAULT 0,
                resolved_turn_id TEXT,
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL
            );
            -- The drain reads pending rows oldest-first; resolution reads
            -- them by artifact.  Both are covering.
            CREATE INDEX IF NOT EXISTS idx_curation_pending_status
                ON curation_pending_additions(status, id);
            CREATE INDEX IF NOT EXISTS idx_curation_pending_artifact
                ON curation_pending_additions(target_artifact, status);
        """)
        self._conn.commit()

        # Memory v3: allow NULL reflection_embedding for parse-failure fallback
        # placeholders (§2.7.8). Detect via NOT NULL constraint and rebuild table
        # if needed. SQLite can't drop NOT NULL via ALTER, so we copy.
        try:
            cursor_v3 = self._conn.execute(
                "SELECT \"notnull\" FROM pragma_table_info('memories') "
                "WHERE name = 'reflection_embedding'"
            )
            row_v3 = cursor_v3.fetchone()
            if row_v3 and row_v3[0]:  # NOT NULL is set; needs rebuild
                logger.info("Migrating: relaxing NOT NULL on memories.reflection_embedding")
                self._conn.executescript("""
                    BEGIN IMMEDIATE;
                    CREATE TABLE memories_new (
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
                        project TEXT DEFAULT '',
                        digest_candidate INTEGER NOT NULL DEFAULT 1,
                        event_date TEXT NOT NULL DEFAULT '',
                        formation_source TEXT NOT NULL DEFAULT 'legacy-unknown'
                    );
                    INSERT INTO memories_new SELECT
                        id, created_at, summary, reflection, trace, trigger_text,
                        retrieval_key, tags, outcome, reflection_embedding,
                        retrieval_key_embedding, weight, topic_label, project,
                        digest_candidate, event_date, formation_source
                    FROM memories;
                    DROP TABLE memories;
                    ALTER TABLE memories_new RENAME TO memories;
                    COMMIT;
                """)
        except sqlite3.OperationalError:
            # Older SQLite without pragma_table_info — safe to skip; behavior
            # falls back to legacy schema (no fallback entries possible).
            pass

        # project_maps: add project_dir column
        cursor_maps = self._conn.execute("PRAGMA table_info(project_maps)")
        map_columns = {row[1] for row in cursor_maps.fetchall()}
        if "project_dir" not in map_columns:
            logger.info("Migrating: adding project_dir column to project_maps")
            self._conn.execute(
                "ALTER TABLE project_maps ADD COLUMN project_dir TEXT"
            )
            self._conn.commit()

        # project_maps: add summary column for relevance-based injection
        if "summary" not in map_columns:
            logger.info("Migrating: adding summary column to project_maps")
            self._conn.execute(
                "ALTER TABLE project_maps ADD COLUMN summary TEXT NOT NULL DEFAULT ''"
            )
            self._conn.commit()

        # scheduled_wakes: add recurrence column
        cursor_wakes = self._conn.execute("PRAGMA table_info(scheduled_wakes)")
        wake_columns = {row[1] for row in cursor_wakes.fetchall()}
        if "recurrence" not in wake_columns:
            logger.info("Migrating: adding recurrence column to scheduled_wakes")
            self._conn.execute(
                "ALTER TABLE scheduled_wakes ADD COLUMN recurrence TEXT"
            )
            self._conn.commit()

        self._ensure_fts()

    def _ensure_fts(self) -> None:
        """Create and populate FTS5 full-text index if missing."""
        exists = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        ).fetchone()
        if exists:
            return
        logger.info("Creating FTS5 index for memories")
        self._conn.execute(
            "CREATE VIRTUAL TABLE memories_fts USING fts5("
            "  id UNINDEXED, summary, reflection, retrieval_key"
            ")"
        )
        self._conn.execute(
            "INSERT INTO memories_fts (id, summary, reflection, retrieval_key) "
            "SELECT id, summary, reflection, retrieval_key FROM memories"
        )
        self._conn.commit()

    def _fts_upsert(self, entry_id: str, summary: str, reflection: str,
                    retrieval_key: str) -> None:
        """Insert or replace an entry in the FTS index."""
        self._conn.execute(
            "DELETE FROM memories_fts WHERE id = ?", (entry_id,)
        )
        self._conn.execute(
            "INSERT INTO memories_fts (id, summary, reflection, retrieval_key) "
            "VALUES (?, ?, ?, ?)",
            (entry_id, summary, reflection, retrieval_key),
        )

    def _fts_delete(self, entry_id: str) -> None:
        self._conn.execute(
            "DELETE FROM memories_fts WHERE id = ?", (entry_id,)
        )

    def search_fts(self, query: str, limit: int = 50) -> list[tuple[str, float]]:
        """Full-text search using FTS5/BM25. Returns (id, score) pairs.

        Score is negated BM25 (higher = more relevant).
        """
        try:
            cursor = self._conn.execute(
                "SELECT id, -bm25(memories_fts) AS score "
                "FROM memories_fts WHERE memories_fts MATCH ? "
                "ORDER BY score DESC LIMIT ?",
                (query, limit),
            )
            return [(row[0], row[1]) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            return []

    def load(self) -> list[MemoryEntry]:
        """Load all memory entries from the database."""
        cursor = self._conn.execute(
            "SELECT id, created_at, summary, reflection, trace, trigger_text, "
            "retrieval_key, tags, outcome, reflection_embedding, "
            "retrieval_key_embedding, weight, topic_label, project, "
            "digest_candidate, event_date, formation_source "
            "FROM memories ORDER BY created_at"
        )
        entries = []
        for row in cursor.fetchall():
            entries.append(MemoryEntry(
                id=row[0],
                created_at=datetime.fromisoformat(row[1]),
                summary=row[2],
                reflection=row[3],
                trace=row[4],
                trigger=row[5],
                retrieval_key=row[6],
                tags=_parse_tags(row[7]),
                outcome=row[8],
                reflection_embedding=_deserialize_embedding(row[9]),
                retrieval_key_embedding=_deserialize_embedding(row[10]),
                weight=row[11],
                topic_label=row[12] or "",
                project=row[13] or "",
                digest_candidate=bool(row[14]) if row[14] is not None else True,
                event_date=resolve_event_date(row[15], row[5]),
                formation_source=resolve_formation_source(row[16], row[5]),
            ))
        logger.info(f"Loaded {len(entries)} memory entries from {self._db_path}")
        return entries

    def insert(self, entry: MemoryEntry) -> None:
        """Insert or replace a memory entry."""
        self._conn.execute(
            "INSERT INTO memories "
            "(id, created_at, summary, reflection, trace, trigger_text, "
            "retrieval_key, tags, outcome, reflection_embedding, "
            "retrieval_key_embedding, weight, topic_label, project, "
            "digest_candidate, event_date, formation_source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "created_at=excluded.created_at, summary=excluded.summary, "
            "reflection=excluded.reflection, trace=excluded.trace, "
            "trigger_text=excluded.trigger_text, "
            "retrieval_key=excluded.retrieval_key, tags=excluded.tags, "
            "outcome=excluded.outcome, "
            "reflection_embedding=excluded.reflection_embedding, "
            "retrieval_key_embedding=excluded.retrieval_key_embedding, "
            "weight=excluded.weight, topic_label=excluded.topic_label, "
            "project=excluded.project, "
            "digest_candidate=excluded.digest_candidate, "
            "event_date=excluded.event_date, "
            "formation_source=excluded.formation_source",
            (
                entry.id,
                entry.created_at.isoformat(),
                entry.summary,
                entry.reflection,
                entry.trace,
                entry.trigger,
                entry.retrieval_key,
                _serialize_tags(entry.tags),
                entry.outcome,
                _serialize_embedding(entry.reflection_embedding),
                _serialize_embedding(entry.retrieval_key_embedding),
                entry.weight,
                entry.topic_label,
                entry.project,
                1 if entry.digest_candidate else 0,
                entry.event_date,
                entry.formation_source,
            ),
        )
        self._fts_upsert(entry.id, entry.summary, entry.reflection,
                         entry.retrieval_key)
        self._log_authoritative_formation_entry(entry)
        self._conn.commit()

    def _log_authoritative_formation_entry(self, entry: MemoryEntry) -> None:
        """Assign a deletion-safe fold sequence to authoritative live rows."""
        if entry.formation_source != "live-extraction":
            return
        self._conn.execute(
            "INSERT OR IGNORE INTO memory_formation_log (memory_id) VALUES (?)",
            (entry.id,),
        )

    def delete(
        self,
        entry_id: str,
        *,
        actor_node: str | None = None,
        source_message_id: str | None = None,
        source_author: str | None = None,
        reason: str = "memory deleted",
    ) -> bool:
        """Delete a memory and transactionally cascade its entity evidence."""
        from .entities import EntityExecutionContext, EntityService

        context = EntityExecutionContext(
            actor_node=actor_node or f"memory:{self.nickname}",
            source_message_id=source_message_id,
            source_author=source_author,
        )
        service = EntityService(
            self._conn, actor_node=context.actor_node
        )
        return service.delete_memory_transactional(
            entry_id, context=context, reason=reason
        )

    def delete_many(
        self,
        entry_ids: list[str],
        *,
        actor_node: str | None = None,
        reason: str = "memory pool pruning",
    ) -> list[str]:
        """Delete memories and all current entity links in one transaction."""
        from .entities import EntityExecutionContext, EntityService

        context = EntityExecutionContext(
            actor_node=actor_node or f"memory:{self.nickname}"
        )
        return EntityService(
            self._conn, actor_node=context.actor_node
        ).delete_memories_transactional(
            entry_ids, context=context, reason=reason
        )

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Get a single memory entry by ID."""
        cursor = self._conn.execute(
            "SELECT id, created_at, summary, reflection, trace, trigger_text, "
            "retrieval_key, tags, outcome, reflection_embedding, "
            "retrieval_key_embedding, weight, topic_label, project, "
            "digest_candidate, event_date, formation_source "
            "FROM memories WHERE id = ?",
            (entry_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return MemoryEntry(
            id=row[0],
            created_at=datetime.fromisoformat(row[1]),
            summary=row[2],
            reflection=row[3],
            trace=row[4],
            trigger=row[5],
            retrieval_key=row[6],
            tags=_parse_tags(row[7]),
            outcome=row[8],
            reflection_embedding=_deserialize_embedding(row[9]),
            retrieval_key_embedding=_deserialize_embedding(row[10]),
            weight=row[11],
            topic_label=row[12] or "",
            project=row[13] or "",
            digest_candidate=bool(row[14]) if row[14] is not None else True,
            event_date=resolve_event_date(row[15], row[5]),
            formation_source=resolve_formation_source(row[16], row[5]),
        )

    def list_all(self) -> list[MemoryEntry]:
        """Alias for load()."""
        return self.load()

    def list_by_project(
        self,
        project: str | None,
        include_project_empty: bool = True,
    ) -> list[MemoryEntry]:
        """Return entries matching a project, plus optionally project-empty entries.

        project=None returns all entries (no filter).
        """
        if project is None:
            return self.list_all()
        cur = self._conn.execute(
            "SELECT id FROM memories WHERE project = ? OR (? AND project = '')",
            (project, 1 if include_project_empty else 0),
        )
        ids = [r[0] for r in cur.fetchall()]
        out = []
        for eid in ids:
            e = self.get(eid)
            if e is not None:
                out.append(e)
        return out

    def update_entry(
        self,
        entry_id: str,
        *,
        summary: str | None = None,
        reflection: str | None = None,
        retrieval_key: str | None = None,
        tags: list[str] | None = None,
        outcome: str | None = None,
        reflection_embedding: np.ndarray | None = _SENTINEL,
        retrieval_key_embedding: np.ndarray | None = _SENTINEL,
    ) -> bool:
        """Update specified fields of an existing memory entry in place.

        Only non-None arguments are written. Returns True if the row existed.
        """
        sets: list[str] = []
        vals: list = []
        if summary is not None:
            sets.append("summary = ?")
            vals.append(summary)
        if reflection is not None:
            sets.append("reflection = ?")
            vals.append(reflection)
        if retrieval_key is not None:
            sets.append("retrieval_key = ?")
            vals.append(retrieval_key)
        if tags is not None:
            sets.append("tags = ?")
            vals.append(_serialize_tags(tags))
        if outcome is not None:
            sets.append("outcome = ?")
            vals.append(outcome)
        if reflection_embedding is not _SENTINEL:
            sets.append("reflection_embedding = ?")
            vals.append(_serialize_embedding(reflection_embedding))
        if retrieval_key_embedding is not _SENTINEL:
            sets.append("retrieval_key_embedding = ?")
            vals.append(_serialize_embedding(retrieval_key_embedding))
        if not sets:
            return self.get(entry_id) is not None
        vals.append(entry_id)
        cursor = self._conn.execute(
            f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", vals,
        )
        if cursor.rowcount > 0 and any(
            f is not None for f in (summary, reflection, retrieval_key)
        ):
            row = self._conn.execute(
                "SELECT summary, reflection, retrieval_key FROM memories WHERE id = ?",
                (entry_id,),
            ).fetchone()
            if row:
                self._fts_upsert(entry_id, row[0], row[1], row[2])
        self._conn.commit()
        return cursor.rowcount > 0

    def update_weight(self, entry_id: str, weight: float) -> None:
        """Update the cached withholding cost for an entry."""
        self._conn.execute(
            "UPDATE memories SET weight = ? WHERE id = ?",
            (weight, entry_id),
        )
        self._conn.commit()

    def update_weights_batch(self, weights: dict[str, float]) -> None:
        """Batch update withholding costs."""
        cursor = self._conn.cursor()
        for entry_id, weight in weights.items():
            cursor.execute(
                "UPDATE memories SET weight = ? WHERE id = ?",
                (weight, entry_id),
            )
        self._conn.commit()

    def count(self) -> int:
        """Return the number of stored entries."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM memories")
        return cursor.fetchone()[0]

    # ── Personality ──────────────────────────────────────────────

    def get_personality(self) -> str:
        """Get the agent's personality text."""
        cursor = self._conn.execute(
            "SELECT value FROM personality WHERE key = 'personality'"
        )
        row = cursor.fetchone()
        return row[0] if row else ""

    def set_personality(self, text: str) -> None:
        """Set the agent's personality text."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO personality (key, value, updated_at) "
            "VALUES ('personality', ?, ?)",
            (text, now),
        )
        self._conn.commit()

    def personality_updated_at(self) -> str | None:
        """Get the last update timestamp for personality."""
        cursor = self._conn.execute(
            "SELECT updated_at FROM personality WHERE key = 'personality'"
        )
        row = cursor.fetchone()
        return row[0] if row else None

    # ── Scheduled Wakes ────────────────────────────────────────

    def save_wake(self, wake_id: str, wake_time: str, prompt: str,
                  requested_by: str, created_at: str,
                  recurrence: str | None = None) -> None:
        """Persist a scheduled wake."""
        self._conn.execute(
            "INSERT OR REPLACE INTO scheduled_wakes "
            "(id, wake_time, prompt, requested_by, created_at, recurrence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (wake_id, wake_time, prompt, requested_by, created_at, recurrence),
        )
        self._conn.commit()

    def delete_wake(self, wake_id: str) -> None:
        """Delete a scheduled wake by ID."""
        self._conn.execute("DELETE FROM scheduled_wakes WHERE id = ?", (wake_id,))
        self._conn.commit()

    def load_wakes(self) -> list[dict]:
        """Load all scheduled wakes. Returns list of dicts."""
        cursor = self._conn.execute(
            "SELECT id, wake_time, prompt, requested_by, created_at, recurrence "
            "FROM scheduled_wakes"
        )
        return [
            {
                "id": row[0],
                "wake_time": row[1],
                "prompt": row[2],
                "requested_by": row[3],
                "created_at": row[4],
                "recurrence": row[5],
            }
            for row in cursor.fetchall()
        ]

    # ── Project Maps ─────────────────────────────────────────────

    def get_map(self, project_name: str) -> dict | None:
        """Get a project map's metadata by name. Returns dict or None.

        NOTE: content is NOT returned — maps live on disk as PROJECT_MAP.md.
        Use MemorySystemV2.get_map() to read the actual map content from file.
        """
        cursor = self._conn.execute(
            "SELECT id, project_name, updated_at, created_at, "
            "is_active, project_dir FROM project_maps WHERE project_name = ?",
            (project_name,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "project_name": row[1],
            "updated_at": row[2],
            "created_at": row[3],
            "is_active": bool(row[4]),
            "project_dir": row[5],
        }

    def list_maps(self) -> list[dict]:
        """List all project maps. Returns list of dicts (without content)."""
        cursor = self._conn.execute(
            "SELECT id, project_name, updated_at, created_at, is_active "
            "FROM project_maps ORDER BY project_name"
        )
        return [
            {
                "id": row[0],
                "project_name": row[1],
                "updated_at": row[2],
                "created_at": row[3],
                "is_active": bool(row[4]),
            }
            for row in cursor.fetchall()
        ]

    def create_map(
        self, map_id: str, project_name: str, content: str,
        embedding: "np.ndarray | None" = None,
        project_dir: str | None = None,
    ) -> None:
        """Create a new project map."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO project_maps "
            "(id, project_name, content, updated_at, created_at, embedding, is_active, project_dir) "
            "VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
            (map_id, project_name, content, now, now,
             _serialize_embedding(embedding), project_dir),
        )
        self._conn.commit()

    def update_map(self, project_name: str, content: str,
                   embedding: "np.ndarray | None" = None,
                   project_dir: str | None = None) -> bool:
        """Update a project map's content. Returns True if found."""
        now = datetime.now(timezone.utc).isoformat()
        if embedding is not None:
            cursor = self._conn.execute(
                "UPDATE project_maps SET content = ?, updated_at = ?, embedding = ? "
                "WHERE project_name = ?",
                (content, now, _serialize_embedding(embedding), project_name),
            )
        else:
            cursor = self._conn.execute(
                "UPDATE project_maps SET content = ?, updated_at = ? "
                "WHERE project_name = ?",
                (content, now, project_name),
            )
        if project_dir is not None:
            self._conn.execute(
                "UPDATE project_maps SET project_dir = ? WHERE project_name = ?",
                (project_dir, project_name),
            )
        self._conn.commit()
        return cursor.rowcount > 0

    def delete_map(self, project_name: str) -> bool:
        """Delete a project map by name. Returns True if found."""
        cursor = self._conn.execute(
            "DELETE FROM project_maps WHERE project_name = ?",
            (project_name,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def set_active_project(self, project_name: str) -> None:
        """Set a project as active (deactivates all others).

        Both updates run in a single transaction to prevent an intermediate
        state where no project is active (e.g., crash between statements).
        """
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute("UPDATE project_maps SET is_active = 0")
            self._conn.execute(
                "UPDATE project_maps SET is_active = 1 WHERE project_name = ?",
                (project_name,),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def clear_active_project(self) -> None:
        """Clear the active project (set all to inactive)."""
        self._conn.execute("UPDATE project_maps SET is_active = 0")
        self._conn.commit()

    def get_active_project(self) -> str | None:
        """Get the name of the currently active project, or None."""
        cursor = self._conn.execute(
            "SELECT project_name FROM project_maps WHERE is_active = 1"
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def update_map_summary(
        self, project_name: str, summary: str,
        embedding: "np.ndarray | None" = None,
    ) -> bool:
        """Update a project map's summary and embedding."""
        now = datetime.now(timezone.utc).isoformat()
        if embedding is not None:
            cursor = self._conn.execute(
                "UPDATE project_maps SET summary = ?, embedding = ?, updated_at = ? "
                "WHERE project_name = ?",
                (summary, _serialize_embedding(embedding), now, project_name),
            )
        else:
            cursor = self._conn.execute(
                "UPDATE project_maps SET summary = ?, updated_at = ? "
                "WHERE project_name = ?",
                (summary, now, project_name),
            )
        self._conn.commit()
        return cursor.rowcount > 0

    def list_map_embeddings(self) -> list[tuple[str, str, bytes | None]]:
        """Return (project_name, summary, embedding_blob) for all maps."""
        cursor = self._conn.execute(
            "SELECT project_name, summary, embedding FROM project_maps "
            "ORDER BY project_name"
        )
        return [(row[0], row[1], row[2]) for row in cursor.fetchall()]

    def get_project_dir(self, project_name: str) -> str | None:
        """Get the stored project directory for a project, or None."""
        cursor = self._conn.execute(
            "SELECT project_dir FROM project_maps WHERE project_name = ?",
            (project_name,),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    # ── Interpretive Essays ─────────────────────────────────────

    def get_essay(self, entity_key: str) -> dict | None:
        """Get an essay by entity key. Returns dict or None."""
        # Essay retrieval tools run through asyncio.to_thread(); use a fresh
        # read-only connection so the SQLite handle is owned by the worker
        # thread executing this call.
        with sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True) as conn:
            cursor = conn.execute(
                "SELECT entity_key, title, body, citations, cross_refs, "
                "patch_count, created_at, updated_at, curated_version, "
                "verified_hash, verified_at "
                "FROM essays WHERE entity_key = ?",
                (entity_key,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "entity_key": row[0],
            "title": row[1],
            "body": row[2],
            "citations": json.loads(row[3]),
            "cross_refs": json.loads(row[4]),
            "patch_count": row[5],
            "created_at": row[6],
            "updated_at": row[7],
            "curated_version": row[8],
            "verified_hash": row[9],
            "verified_at": row[10],
        }

    def list_essays(self) -> list[dict]:
        """List all essays (without body content for brevity)."""
        # See get_essay(): agent tool calls may execute on worker threads.
        with sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True) as conn:
            cursor = conn.execute(
                "SELECT entity_key, title, patch_count, created_at, updated_at, "
                "curated_version, verified_hash, verified_at "
                "FROM essays ORDER BY entity_key"
            )
            return [
                {
                    "entity_key": row[0],
                    "title": row[1],
                    "patch_count": row[2],
                    "created_at": row[3],
                    "updated_at": row[4],
                    "curated_version": row[5],
                    "verified_hash": row[6],
                    "verified_at": row[7],
                }
                for row in cursor.fetchall()
            ]

    def create_essay(
        self,
        entity_key: str,
        body: str,
        title: str = "",
        citations: list[str] | None = None,
        cross_refs: list[str] | None = None,
    ) -> None:
        """Create a new essay. Raises IntegrityError if key exists."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO essays "
            "(entity_key, title, body, citations, cross_refs, "
            "patch_count, created_at, updated_at, curated_version, "
            "verified_hash, verified_at) "
            "VALUES (?, ?, ?, ?, ?, 0, ?, ?, 0, '', NULL)",
            (
                entity_key,
                title,
                body,
                json.dumps(citations or []),
                json.dumps(cross_refs or []),
                now,
                now,
            ),
        )
        self._conn.commit()

    def update_essay(
        self,
        entity_key: str,
        body: str,
        title: str | None = None,
        citations: list[str] | None = None,
        cross_refs: list[str] | None = None,
    ) -> bool:
        """Replace an essay's body (and optionally title/citations/cross_refs).
        Increments patch_count. Returns True if found."""
        now = datetime.now(timezone.utc).isoformat()
        existing = self.get_essay(entity_key)
        if existing is None:
            return False
        new_title = title if title is not None else existing["title"]
        new_citations = json.dumps(citations) if citations is not None else json.dumps(existing["citations"])
        new_cross_refs = json.dumps(cross_refs) if cross_refs is not None else json.dumps(existing["cross_refs"])
        self._conn.execute(
            "UPDATE essays SET body = ?, title = ?, citations = ?, "
            "cross_refs = ?, patch_count = patch_count + 1, updated_at = ?, "
            "verified_hash = '', verified_at = NULL "
            "WHERE entity_key = ?",
            (body, new_title, new_citations, new_cross_refs, now, entity_key),
        )
        self._conn.commit()
        return True

    def update_essay_if_revision(
        self,
        entity_key: str,
        expected_patch_count: int,
        body: str,
        title: str | None = None,
        citations: list[str] | None = None,
        cross_refs: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Compare-and-swap variant of :meth:`update_essay`.

        ``essay_edit`` reads the body, computes a replacement against that
        snapshot, then writes.  Now that curation turns run concurrently with
        message turns, another writer can land in that window; an unconditional
        UPDATE would silently discard their patch.  Gating the write on the
        ``patch_count`` observed at read time turns the lost update into a
        visible conflict the caller can retry.

        Returns ``(ok, message)``; ``ok=False`` means the row moved (or
        vanished) and nothing was written.
        """
        now = datetime.now(timezone.utc).isoformat()
        existing = self.get_essay(entity_key)
        if existing is None:
            return False, f"No essay found for key '{entity_key}'."
        new_title = title if title is not None else existing["title"]
        new_citations = json.dumps(
            citations if citations is not None else existing["citations"]
        )
        new_cross_refs = json.dumps(
            cross_refs if cross_refs is not None else existing["cross_refs"]
        )
        cursor = self._conn.execute(
            "UPDATE essays SET body = ?, title = ?, citations = ?, "
            "cross_refs = ?, patch_count = patch_count + 1, updated_at = ?, "
            "verified_hash = '', verified_at = NULL "
            "WHERE entity_key = ? AND patch_count = ?",
            (
                body, new_title, new_citations, new_cross_refs, now,
                entity_key, int(expected_patch_count),
            ),
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            current = self.get_essay(entity_key) or {}
            return False, (
                f"Essay '{entity_key}' was modified concurrently "
                f"(expected revision {expected_patch_count}, found "
                f"{current.get('patch_count')}). "
                "Re-read the essay and reapply the edit."
            )
        return True, ""

    def delete_essay(self, entity_key: str) -> bool:
        """Delete an essay by entity key. Returns True if found."""
        cursor = self._conn.execute(
            "DELETE FROM essays WHERE entity_key = ?",
            (entity_key,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def essay_edit(
        self,
        entity_key: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> tuple[bool, str]:
        """Exact string replacement in an essay's body.
        Returns (success, message). Mirrors digest_edit semantics:
        fails if old_text not found or matches multiple times (unless replace_all).
        Increments patch_count on success."""
        essay = self.get_essay(entity_key)
        if essay is None:
            return False, f"No essay found for key '{entity_key}'."
        body = essay["body"]
        count = body.count(old_text)
        if count == 0:
            return False, "old_text not found in essay body."
        if not replace_all and count > 1:
            return False, (
                f"old_text matches {count} locations — "
                f"provide a more specific string or set replace_all=true."
            )
        if replace_all:
            new_body = body.replace(old_text, new_text)
        else:
            new_body = body.replace(old_text, new_text, 1)
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE essays SET body = ?, patch_count = patch_count + 1, "
            "updated_at = ?, verified_hash = '', verified_at = NULL "
            "WHERE entity_key = ?",
            (new_body, now, entity_key),
        )
        self._conn.commit()
        n_replaced = count if replace_all else 1
        return True, f"Essay updated ({n_replaced} replacement{'s' if n_replaced > 1 else ''})."

    def update_essay_citations(
        self,
        entity_key: str,
        citations: list[str],
        cross_refs: list[str] | None = None,
    ) -> bool:
        """Update an essay's citations and optionally cross_refs.
        Does NOT increment patch_count (metadata-only change).
        Returns True if found."""
        now = datetime.now(timezone.utc).isoformat()
        if cross_refs is not None:
            cursor = self._conn.execute(
                "UPDATE essays SET citations = ?, cross_refs = ?, "
                "updated_at = ?, verified_hash = '', verified_at = NULL "
                "WHERE entity_key = ?",
                (json.dumps(citations), json.dumps(cross_refs), now, entity_key),
            )
        else:
            cursor = self._conn.execute(
                "UPDATE essays SET citations = ?, updated_at = ?, "
                "verified_hash = '', verified_at = NULL "
                "WHERE entity_key = ?",
                (json.dumps(citations), now, entity_key),
            )
        self._conn.commit()
        return cursor.rowcount > 0

    # ── Conversation Summary ────────────────────────────────────

    def get_summary(self) -> dict | None:
        """Load conversation summary. Returns dict with summary_text,
        messages_summarized, token_estimate, updated_at — or None."""
        cursor = self._conn.execute(
            "SELECT summary_text, messages_summarized, token_estimate, updated_at "
            "FROM conversation_summary WHERE id = 1"
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "summary_text": row[0],
            "messages_summarized": row[1],
            "token_estimate": row[2],
            "updated_at": row[3],
        }

    def save_summary(
        self, summary_text: str, messages_summarized: int, token_estimate: int
    ) -> None:
        """Upsert conversation summary (single-row table)."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO conversation_summary (id, summary_text, messages_summarized, "
            "token_estimate, updated_at) VALUES (1, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "summary_text=excluded.summary_text, "
            "messages_summarized=excluded.messages_summarized, "
            "token_estimate=excluded.token_estimate, "
            "updated_at=excluded.updated_at",
            (summary_text, messages_summarized, token_estimate, now),
        )
        self._conn.commit()
        logger.info(
            "Conversation summary saved: %d chars, %d messages, ~%d tokens",
            len(summary_text), messages_summarized, token_estimate,
        )

    # ── Formation cursor (Memory v3) ─────────────────────────

    def get_formation_cursor(self) -> tuple[int, str]:
        """Return (last_index, last_ts_utc). (0, "") if uninitialized."""
        cursor = self._conn.execute(
            "SELECT last_index, last_ts_utc FROM formation_cursor WHERE id = 1"
        )
        row = cursor.fetchone()
        if row is None:
            return (0, "")
        return (int(row[0]), row[1] or "")

    def set_formation_cursor(self, last_index: int, last_ts_utc: str) -> None:
        """Upsert the singleton formation cursor row."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO formation_cursor (id, last_index, last_ts_utc, updated_at) "
            "VALUES (1, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "last_index=excluded.last_index, "
            "last_ts_utc=excluded.last_ts_utc, "
            "updated_at=excluded.updated_at",
            (last_index, last_ts_utc, now),
        )
        self._conn.commit()

    def activate_authoritative_formation_contract(self) -> None:
        """Record that the live process loaded the shared extraction contract."""
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (
                FORMATION_CONTRACT_META_KEY,
                AUTHORITATIVE_FORMATION_CONTRACT,
            ),
        )
        self._conn.commit()

    def insert_entry_and_advance_cursor(
        self,
        entries: list[MemoryEntry],
        new_cursor: int,
        new_ts_utc: str,
        entity_mutations: list[dict] | dict | None = None,
        *,
        entity_actor_node: str | None = None,
        entity_resolution_enabled: bool = False,
        entity_activation_window_threshold: int = 3,
        entity_run_key: str | None = None,
    ) -> None:
        """Insert N entries and advance the formation cursor in a single transaction.

        Atomicity guarantees against torn state if the process dies between
        memory entry insert and cursor advance (Risk 10).

        ``entity_mutations`` is optional and empty by default.  Its
        ``window_key`` values are opaque stable identifiers for formation
        windows; this store never parses their contents.  Entity resolution
        runs in a savepoint so a resolver defect cannot discard minted
        memories or prevent cursor advancement.
        """
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            for entry in entries:
                self._conn.execute(
                    "INSERT INTO memories "
                    "(id, created_at, summary, reflection, trace, trigger_text, "
                    "retrieval_key, tags, outcome, reflection_embedding, "
                    "retrieval_key_embedding, weight, topic_label, project, "
                    "digest_candidate, event_date, formation_source) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "created_at=excluded.created_at, summary=excluded.summary, "
                    "reflection=excluded.reflection, trace=excluded.trace, "
                    "trigger_text=excluded.trigger_text, "
                    "retrieval_key=excluded.retrieval_key, tags=excluded.tags, "
                    "outcome=excluded.outcome, "
                    "reflection_embedding=excluded.reflection_embedding, "
                    "retrieval_key_embedding=excluded.retrieval_key_embedding, "
                    "weight=excluded.weight, topic_label=excluded.topic_label, "
                    "project=excluded.project, "
                    "digest_candidate=excluded.digest_candidate, "
                    "event_date=excluded.event_date, "
                    "formation_source=excluded.formation_source",
                    (
                        entry.id,
                        entry.created_at.isoformat(),
                        entry.summary,
                        entry.reflection,
                        entry.trace,
                        entry.trigger,
                        entry.retrieval_key,
                        _serialize_tags(entry.tags),
                        entry.outcome,
                        _serialize_embedding(entry.reflection_embedding),
                        _serialize_embedding(entry.retrieval_key_embedding),
                        entry.weight,
                        entry.topic_label,
                        entry.project,
                        1 if entry.digest_candidate else 0,
                        entry.event_date,
                        entry.formation_source,
                    ),
                )
                self._fts_upsert(entry.id, entry.summary, entry.reflection,
                                 entry.retrieval_key)
                self._log_authoritative_formation_entry(entry)
            if entity_mutations:
                from .entities import (
                    EntityExecutionContext,
                    EntityService,
                )

                self._conn.execute("SAVEPOINT entity_resolution")
                try:
                    context = EntityExecutionContext(
                        actor_node=(
                            entity_actor_node or f"memory:{self.nickname}"
                        )
                    )
                    EntityService(
                        self._conn,
                        actor_node=context.actor_node,
                        activation_window_threshold=(
                            entity_activation_window_threshold
                        ),
                        mutations_enabled=entity_resolution_enabled,
                    ).apply_formation_mutations_in_transaction(
                        entity_mutations,
                        context=context,
                        run_key=entity_run_key,
                    )
                    self._conn.execute("RELEASE SAVEPOINT entity_resolution")
                except Exception as entity_error:
                    self._conn.execute(
                        "ROLLBACK TO SAVEPOINT entity_resolution"
                    )
                    self._conn.execute("RELEASE SAVEPOINT entity_resolution")
                    logger.exception(
                        "Entity resolution failed; committing memories and cursor"
                    )
                    try:
                        EntityService(
                            self._conn,
                            actor_node=(
                                entity_actor_node or f"memory:{self.nickname}"
                            ),
                            activation_window_threshold=(
                                entity_activation_window_threshold
                            ),
                        )._record_event(
                            "entity_resolution_failed",
                            reason=str(entity_error),
                            run_key=entity_run_key,
                            details={
                                "memory_ids": [entry.id for entry in entries]
                            },
                        )
                    except Exception:
                        logger.exception(
                            "Could not record entity resolution failure audit"
                        )
            self._conn.execute(
                "INSERT INTO formation_cursor (id, last_index, last_ts_utc, updated_at) "
                "VALUES (1, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "last_index=excluded.last_index, "
                "last_ts_utc=excluded.last_ts_utc, "
                "updated_at=excluded.updated_at",
                (new_cursor, new_ts_utc, now),
            )
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (
                    FORMATION_CONTRACT_META_KEY,
                    AUTHORITATIVE_FORMATION_CONTRACT,
                ),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ── Migrations tracking ──────────────────────────────────

    def is_migration_complete(self, name: str) -> bool:
        cursor = self._conn.execute(
            "SELECT 1 FROM migrations_complete WHERE name = ?", (name,)
        )
        return cursor.fetchone() is not None

    def mark_migration_complete(self, name: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO migrations_complete (name, completed_at) "
            "VALUES (?, ?)",
            (name, now),
        )
        self._conn.commit()

    def update_reflection_embedding(
        self, entry_id: str, emb: np.ndarray | None
    ) -> None:
        """Update an entry's reflection_embedding in place."""
        self._conn.execute(
            "UPDATE memories SET reflection_embedding = ? WHERE id = ?",
            (_serialize_embedding(emb), entry_id),
        )
        self._conn.commit()

    def bulk_update_reflection_embeddings_and_mark_migration(
        self,
        updates: list[tuple[str, np.ndarray | None]],
        migration_name: str,
    ) -> None:
        """Atomically update reflection_embedding for many entries AND mark a
        migration complete in a single transaction.

        All-or-nothing: if any update fails or the process dies mid-transaction,
        nothing is committed and the migration marker is not written, so the
        migration runs cleanly from a consistent state on retry. (Rev-7 fix to
        code-review Issue 2.)
        """
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            for entry_id, emb in updates:
                self._conn.execute(
                    "UPDATE memories SET reflection_embedding = ? WHERE id = ?",
                    (_serialize_embedding(emb), entry_id),
                )
            self._conn.execute(
                "INSERT OR REPLACE INTO migrations_complete (name, completed_at) "
                "VALUES (?, ?)",
                (migration_name, now),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info(f"Memory store closed: {self._db_path}")


def _serialize_tags(tags: list[str]) -> str:
    """Serialize tags as comma-separated string."""
    return ",".join(tags)


def _parse_tags(tags_str: str) -> list[str]:
    """Parse comma-separated tags string."""
    if not tags_str:
        return []
    return [t.strip() for t in tags_str.split(",") if t.strip()]
