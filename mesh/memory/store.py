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

import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DTYPE = np.float32

_SENTINEL = object()  # distinguishes "not passed" from None for optional embedding args


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


class MemoryStore:
    """SQLite-backed persistent memory store for a single agent."""

    def __init__(self, nickname: str, db_dir: str | None = None):
        if db_dir is None:
            from ..paths import MEMORY_DIR
            db_dir = str(MEMORY_DIR)
        os.makedirs(db_dir, exist_ok=True)
        self._db_path = os.path.join(db_dir, f"{nickname}.db")
        self._conn: sqlite3.Connection | None = None
        self._open()

    def _open(self) -> None:
        self._conn = sqlite3.connect(self._db_path)
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
                reflection_embedding BLOB NOT NULL,
                retrieval_key_embedding BLOB NOT NULL,
                weight REAL NOT NULL DEFAULT 0.0
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

            CREATE TABLE IF NOT EXISTS migrations_complete (
                name         TEXT PRIMARY KEY,
                completed_at TEXT NOT NULL
            );
        """)
        self._conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Run schema migrations."""
        cursor = self._conn.execute("PRAGMA table_info(memories)")
        columns = {row[1] for row in cursor.fetchall()}

        if "trigger_embedding" in columns and "retrieval_key_embedding" not in columns:
            logger.info("Migrating: trigger_embedding -> retrieval_key_embedding")
            self._conn.execute(
                "ALTER TABLE memories RENAME COLUMN trigger_embedding TO retrieval_key_embedding"
            )
            if "retrieval_key" not in columns:
                self._conn.execute(
                    "ALTER TABLE memories ADD COLUMN retrieval_key TEXT NOT NULL DEFAULT ''"
                )
            self._conn.commit()

        if "topic_label" not in columns:
            logger.info("Migrating: adding topic_label column")
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN topic_label TEXT DEFAULT ''"
            )
            self._conn.commit()

        if "project" not in columns:
            logger.info("Migrating: adding project column to memories")
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN project TEXT DEFAULT ''"
            )
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
                        project TEXT DEFAULT ''
                    );
                    INSERT INTO memories_new SELECT
                        id, created_at, summary, reflection, trace, trigger_text,
                        retrieval_key, tags, outcome, reflection_embedding,
                        retrieval_key_embedding, weight, topic_label, project
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

        # Phase 1: digest_candidate column — lowbar formation tags rows as
        # digest-worthy or DB-only; fold injection filter uses this.
        if "digest_candidate" not in columns:
            logger.info("Migrating: adding digest_candidate column to memories")
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN digest_candidate INTEGER NOT NULL DEFAULT 1"
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
            "digest_candidate "
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
            ))
        logger.info(f"Loaded {len(entries)} memory entries from {self._db_path}")
        return entries

    def insert(self, entry: MemoryEntry) -> None:
        """Insert or replace a memory entry."""
        self._conn.execute(
            "INSERT OR REPLACE INTO memories "
            "(id, created_at, summary, reflection, trace, trigger_text, "
            "retrieval_key, tags, outcome, reflection_embedding, "
            "retrieval_key_embedding, weight, topic_label, project, "
            "digest_candidate) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            ),
        )
        self._fts_upsert(entry.id, entry.summary, entry.reflection,
                         entry.retrieval_key)
        self._conn.commit()

    def delete(self, entry_id: str) -> bool:
        """Delete a memory entry by ID. Returns True if found and deleted."""
        self._fts_delete(entry_id)
        cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Get a single memory entry by ID."""
        cursor = self._conn.execute(
            "SELECT id, created_at, summary, reflection, trace, trigger_text, "
            "retrieval_key, tags, outcome, reflection_embedding, "
            "retrieval_key_embedding, weight, topic_label, project, "
            "digest_candidate "
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

    def insert_entry_and_advance_cursor(
        self,
        entries: list[MemoryEntry],
        new_cursor: int,
        new_ts_utc: str,
    ) -> None:
        """Insert N entries and advance the formation cursor in a single transaction.

        Atomicity guarantees against torn state if the process dies between
        memory entry insert and cursor advance (Risk 10).
        """
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            for entry in entries:
                self._conn.execute(
                    "INSERT OR REPLACE INTO memories "
                    "(id, created_at, summary, reflection, trace, trigger_text, "
                    "retrieval_key, tags, outcome, reflection_embedding, "
                    "retrieval_key_embedding, weight, topic_label, project, "
                    "digest_candidate) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    ),
                )
                self._fts_upsert(entry.id, entry.summary, entry.reflection,
                                 entry.retrieval_key)
            self._conn.execute(
                "INSERT INTO formation_cursor (id, last_index, last_ts_utc, updated_at) "
                "VALUES (1, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "last_index=excluded.last_index, "
                "last_ts_utc=excluded.last_ts_utc, "
                "updated_at=excluded.updated_at",
                (new_cursor, new_ts_utc, now),
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
