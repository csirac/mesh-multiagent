# SPDX-License-Identifier: Apache-2.0
"""
Message persistence for the mesh.

Stores undelivered messages so they survive router restarts.
Uses SQLite for simplicity and durability.
"""

from __future__ import annotations

import sqlite3
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Iterator

import hashlib
import secrets
import uuid

from .protocol import Attachment, Message, MessageType


def normalize_timestamp_to_utc(timestamp: str) -> str:
    """
    Normalize an ISO timestamp to UTC with 'Z' suffix.

    This ensures consistent string sorting of timestamps regardless of
    the original timezone offset.

    Examples:
        '2026-02-03T11:27:05-06:00' -> '2026-02-03T17:27:05Z'
        '2026-02-03T17:27:05Z' -> '2026-02-03T17:27:05Z'
        '2026-02-03T17:27:05+00:00' -> '2026-02-03T17:27:05Z'
    """
    if not timestamp:
        return timestamp
    try:
        # Parse the timestamp with fromisoformat
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        # Convert to UTC
        utc_dt = dt.astimezone(timezone.utc)
        # Format with 'Z' suffix (consistent with now_iso())
        return utc_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    except (ValueError, AttributeError):
        # If parsing fails, return original
        return timestamp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttachmentBlobRow:
    """Server-side metadata for a content-addressed attachment blob."""
    id: str
    sha256: str
    size: int
    path: str
    mime_inferred: str
    owner_node: str
    created_at: float
    last_accessed: float
    ref_count: int


class MessageStore:
    """
    Persistent storage for messages awaiting delivery.

    Messages are stored when the target node is offline and
    delivered when the node reconnects.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_messages (
                    id TEXT PRIMARY KEY,
                    to_node TEXT NOT NULL,
                    from_node TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    in_reply_to TEXT,
                    metadata TEXT,
                    attachments_json TEXT NOT NULL DEFAULT '[]',
                    created_at REAL DEFAULT (unixepoch('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pending_to_node
                ON pending_messages(to_node)
            """)
            # FCM push notification tokens
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fcm_tokens (
                    node_id TEXT PRIMARY KEY,
                    fcm_token TEXT NOT NULL,
                    updated_at REAL DEFAULT (unixepoch('now'))
                )
            """)
            # Channels
            conn.execute("""
                CREATE TABLE IF NOT EXISTS channels (
                    name TEXT PRIMARY KEY,
                    description TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL
                )
            """)
            # Channel membership
            conn.execute("""
                CREATE TABLE IF NOT EXISTS channel_members (
                    channel_name TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    joined_at TEXT NOT NULL,
                    PRIMARY KEY (channel_name, node_id),
                    FOREIGN KEY (channel_name) REFERENCES channels(name) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_channel_members_channel
                ON channel_members(channel_name)
            """)
            # Message archive (persistent history)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS message_archive (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    from_node TEXT NOT NULL,
                    to_node TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    in_reply_to TEXT,
                    metadata TEXT,
                    attachments_json TEXT NOT NULL DEFAULT '[]',
                    created_at REAL DEFAULT (unixepoch('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_archive_conversation
                ON message_archive(conversation_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_archive_timestamp
                ON message_archive(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_archive_conv_time
                ON message_archive(conversation_id, timestamp)
            """)
            # Read receipts
            conn.execute("""
                CREATE TABLE IF NOT EXISTS read_receipts (
                    node_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    last_read_timestamp TEXT NOT NULL,
                    updated_at REAL DEFAULT (unixepoch('now')),
                    PRIMARY KEY (node_id, conversation_id)
                )
            """)
            # Conversation summaries (for resuming large conversations)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    conversation_id TEXT PRIMARY KEY,
                    summary_text TEXT NOT NULL,
                    messages_summarized INTEGER NOT NULL,
                    token_estimate INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at REAL DEFAULT (unixepoch('now')),
                    metadata TEXT
                )
            """)
            # Users table for per-user authentication
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    token_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    disabled INTEGER DEFAULT 0,
                    allowed_prefixes TEXT DEFAULT NULL
                )
            """)
            # Migration: add allowed_prefixes column if missing
            cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
            if "allowed_prefixes" not in cols:
                conn.execute("ALTER TABLE users ADD COLUMN allowed_prefixes TEXT DEFAULT NULL")
            # Scratchpad notes (per-conversation)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scratchpad_notes (
                    conversation_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    updated_by TEXT NOT NULL
                )
            """)
            # Per-conversation todo lists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_todos (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    section TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    position INTEGER NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    updated_by TEXT NOT NULL,
                    completed_at TEXT,
                    completed_by TEXT,
                    deleted_at TEXT,
                    version INTEGER NOT NULL DEFAULT 1,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)
            todo_cols = {row[1] for row in conn.execute("PRAGMA table_info(conversation_todos)").fetchall()}
            if "section" not in todo_cols:
                conn.execute("ALTER TABLE conversation_todos ADD COLUMN section TEXT")
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversation_todos_conv_status_pos
                ON conversation_todos(conversation_id, status, position)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversation_todos_conv_section_pos
                ON conversation_todos(conversation_id, section, position)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_todo_settings (
                    conversation_id TEXT PRIMARY KEY,
                    section_order TEXT,
                    updated_at TEXT NOT NULL,
                    updated_by TEXT NOT NULL
                )
            """)
            for table in ("pending_messages", "message_archive"):
                cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
                if "attachments_json" not in cols:
                    conn.execute(
                        f"ALTER TABLE {table} "
                        "ADD COLUMN attachments_json TEXT NOT NULL DEFAULT '[]'"
                    )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attachment_blobs (
                    id TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL UNIQUE,
                    size INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    mime_inferred TEXT NOT NULL,
                    owner_node TEXT NOT NULL,
                    created_at REAL DEFAULT (unixepoch('now')),
                    last_accessed REAL DEFAULT (unixepoch('now')),
                    ref_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_attachment_blobs_sha256
                ON attachment_blobs(sha256)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_attachment_blobs_owner
                ON attachment_blobs(owner_node)
            """)
            conn.commit()
        logger.info(f"Message store initialized at {self.db_path}")

    @staticmethod
    def _attachments_json(msg: Message) -> str:
        """Serialize canonical attachment refs, stripping transient signed URLs."""
        return json.dumps([asdict(a.canonical()) for a in msg.attachments])

    @staticmethod
    def _attachments_from_json(raw: str | None) -> list[Attachment]:
        """Hydrate attachment refs from storage, ignoring malformed entries."""
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return [
            Attachment.from_dict(a)
            for a in data
            if isinstance(a, dict)
        ]

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")  # Enable FK constraints
        conn.execute("PRAGMA journal_mode = WAL")  # WAL mode for better concurrent access
        try:
            yield conn
        finally:
            conn.close()

    def store(self, msg: Message) -> None:
        """Store a message for later delivery."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pending_messages
                (id, to_node, from_node, type, content, timestamp, in_reply_to, metadata, attachments_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg.id,
                    msg.to_node,
                    msg.from_node,
                    msg.type.value,
                    json.dumps(msg.content) if isinstance(msg.content, dict) else msg.content,
                    msg.timestamp,
                    msg.in_reply_to,
                    json.dumps(msg.metadata),
                    self._attachments_json(msg),
                ),
            )
            conn.commit()
        logger.debug(f"Stored message {msg.id} for {msg.to_node}")

    def get_pending(self, node_id: str) -> list[Message]:
        """Get all pending messages for a node."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM pending_messages
                WHERE to_node = ?
                ORDER BY created_at ASC
                """,
                (node_id,),
            ).fetchall()

        messages = []
        for row in rows:
            content = row["content"]
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                pass  # Keep as string

            msg = Message(
                id=row["id"],
                from_node=row["from_node"],
                to_node=row["to_node"],
                type=MessageType(row["type"]),
                content=content,
                timestamp=row["timestamp"],
                in_reply_to=row["in_reply_to"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                attachments=self._attachments_from_json(row["attachments_json"] if "attachments_json" in row.keys() else None),
            )
            messages.append(msg)

        logger.debug(f"Retrieved {len(messages)} pending messages for {node_id}")
        return messages

    def remove(self, msg_id: str) -> None:
        """Remove a message after successful delivery."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM pending_messages WHERE id = ?",
                (msg_id,),
            )
            conn.commit()
        logger.debug(f"Removed message {msg_id}")

    def remove_all(self, node_id: str) -> int:
        """Remove all pending messages for a node. Returns count removed."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM pending_messages WHERE to_node = ?",
                (node_id,),
            )
            conn.commit()
            count = cursor.rowcount
        logger.debug(f"Removed {count} messages for {node_id}")
        return count

    def count_pending(self, node_id: str | None = None) -> int:
        """Count pending messages, optionally filtered by node."""
        with self._connect() as conn:
            if node_id:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM pending_messages WHERE to_node = ?",
                    (node_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM pending_messages"
                ).fetchone()
            return row["cnt"]

    # =========================================================================
    # FCM Token Management
    # =========================================================================

    def set_fcm_token(self, node_id: str, fcm_token: str) -> None:
        """Store or update an FCM token for a node."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO fcm_tokens (node_id, fcm_token, updated_at)
                VALUES (?, ?, unixepoch('now'))
                ON CONFLICT(node_id) DO UPDATE SET
                    fcm_token = excluded.fcm_token,
                    updated_at = excluded.updated_at
                """,
                (node_id, fcm_token),
            )
            conn.commit()
        logger.debug(f"Stored FCM token for {node_id}")

    def get_fcm_token(self, node_id: str) -> str | None:
        """Get the FCM token for a node, or None if not registered."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT fcm_token FROM fcm_tokens WHERE node_id = ?",
                (node_id,),
            ).fetchone()
            return row["fcm_token"] if row else None

    def remove_fcm_token(self, node_id: str) -> None:
        """Remove the FCM token for a node."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM fcm_tokens WHERE node_id = ?",
                (node_id,),
            )
            conn.commit()
        logger.debug(f"Removed FCM token for {node_id}")

    # =========================================================================
    # Channel Management
    # =========================================================================

    def create_channel(
        self, name: str, created_by: str, description: str = ""
    ) -> bool:
        """
        Create a new channel.

        Args:
            name: Channel name (without 'channel:' prefix)
            created_by: Node ID of the creator
            description: Optional description

        Returns:
            True if created, False if channel already exists
        """
        from .protocol import now_iso

        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO channels (name, description, created_at, created_by)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, description, now_iso(), created_by),
                )
                conn.commit()
                logger.info(f"Channel '{name}' created by {created_by}")
                return True
            except sqlite3.IntegrityError:
                logger.debug(f"Channel '{name}' already exists")
                return False

    def delete_channel(self, name: str) -> bool:
        """
        Delete a channel and all its memberships.

        Returns:
            True if deleted, False if channel didn't exist
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM channels WHERE name = ?",
                (name,),
            )
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Channel '{name}' deleted")
            return deleted

    def channel_exists(self, name: str) -> bool:
        """Check if a channel exists."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM channels WHERE name = ?",
                (name,),
            ).fetchone()
            return row is not None

    def get_channel(self, name: str) -> dict | None:
        """
        Get channel info.

        Returns:
            Dict with name, description, created_at, created_by, or None
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM channels WHERE name = ?",
                (name,),
            ).fetchone()
            if row:
                return {
                    "name": row["name"],
                    "description": row["description"],
                    "created_at": row["created_at"],
                    "created_by": row["created_by"],
                }
            return None

    def list_channels(self, for_node: str | None = None) -> list[dict]:
        """
        List all channels with member counts.

        Args:
            for_node: Optional node ID to include is_member flag for

        Returns:
            List of dicts with name, description, member_count, and optionally is_member
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.name, c.description, c.created_at, c.created_by,
                       COUNT(m.node_id) as member_count
                FROM channels c
                LEFT JOIN channel_members m ON c.name = m.channel_name
                GROUP BY c.name
                ORDER BY c.name
                """
            ).fetchall()

            result = []
            for row in rows:
                channel_dict = {
                    "name": row["name"],
                    "description": row["description"],
                    "created_at": row["created_at"],
                    "created_by": row["created_by"],
                    "member_count": row["member_count"],
                }
                if for_node:
                    channel_dict["is_member"] = self.is_channel_member(
                        row["name"], for_node
                    )
                result.append(channel_dict)
            return result

    def join_channel(self, channel_name: str, node_id: str) -> bool:
        """
        Add a node to a channel.

        Returns:
            True if joined, False if already a member or channel doesn't exist
        """
        from .protocol import now_iso

        if not self.channel_exists(channel_name):
            return False

        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO channel_members (channel_name, node_id, joined_at)
                    VALUES (?, ?, ?)
                    """,
                    (channel_name, node_id, now_iso()),
                )
                conn.commit()
                logger.info(f"Node {node_id} joined channel '{channel_name}'")
                return True
            except sqlite3.IntegrityError:
                logger.debug(f"Node {node_id} already in channel '{channel_name}'")
                return False

    def leave_channel(self, channel_name: str, node_id: str) -> bool:
        """
        Remove a node from a channel.

        Returns:
            True if left, False if wasn't a member
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM channel_members
                WHERE channel_name = ? AND node_id = ?
                """,
                (channel_name, node_id),
            )
            conn.commit()
            left = cursor.rowcount > 0
            if left:
                logger.info(f"Node {node_id} left channel '{channel_name}'")
            return left

    def get_channel_members(self, channel_name: str) -> list[str]:
        """
        Get all members of a channel.

        Returns:
            List of node IDs
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT node_id FROM channel_members
                WHERE channel_name = ?
                ORDER BY joined_at
                """,
                (channel_name,),
            ).fetchall()
            return [row["node_id"] for row in rows]

    def is_channel_member(self, channel_name: str, node_id: str) -> bool:
        """Check if a node is a member of a channel."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM channel_members
                WHERE channel_name = ? AND node_id = ?
                """,
                (channel_name, node_id),
            ).fetchone()
            return row is not None

    def get_node_channels(self, node_id: str) -> list[str]:
        """
        Get all channels a node is a member of.

        Returns:
            List of channel names
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT channel_name FROM channel_members
                WHERE node_id = ?
                ORDER BY channel_name
                """,
                (node_id,),
            ).fetchall()
            return [row["channel_name"] for row in rows]

    # =========================================================================
    # Message Archive (Persistent History)
    # =========================================================================

    @staticmethod
    def compute_conversation_id(from_node: str, to_node: str) -> str:
        """
        Compute a consistent conversation ID for two nodes.

        For direct messages: sorted pair "node1,node2"
        For channels: the channel address "channel:name"

        Args:
            from_node: Sender node ID
            to_node: Recipient node ID or channel address

        Returns:
            A stable conversation ID
        """
        if to_node.startswith("channel:"):
            return to_node
        # Sort to ensure consistent ID regardless of direction
        nodes = sorted([from_node, to_node])
        return f"{nodes[0]},{nodes[1]}"

    def archive_message(self, msg: Message, conversation_id: str | None = None) -> None:
        """
        Store a message in the permanent archive.

        Args:
            msg: The message to archive
            conversation_id: Optional explicit conversation ID. If not provided,
                            computed from from_node and to_node.
        """
        if conversation_id is None:
            conversation_id = self.compute_conversation_id(msg.from_node, msg.to_node)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO message_archive
                (id, conversation_id, from_node, to_node, type, content, timestamp, in_reply_to, metadata, attachments_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg.id,
                    conversation_id,
                    msg.from_node,
                    msg.to_node,
                    msg.type.value,
                    json.dumps(msg.content) if isinstance(msg.content, dict) else msg.content,
                    normalize_timestamp_to_utc(msg.timestamp),
                    msg.in_reply_to,
                    json.dumps(msg.metadata),
                    self._attachments_json(msg),
                ),
            )
            if conn.total_changes:
                self._increment_blob_refs(conn, [a.id for a in msg.attachments])
            conn.commit()
        logger.debug(f"Archived message {msg.id} in conversation {conversation_id}")

    def get_conversation_history(
        self,
        conversation_id: str,
        since_timestamp: str | None = None,
        limit: int = 100,
    ) -> list[Message]:
        """
        Get message history for a conversation.

        Args:
            conversation_id: The conversation ID
            since_timestamp: Only return messages after this ISO timestamp
            limit: Maximum number of messages to return

        Returns:
            List of messages, oldest first (but fetches most recent N messages)
        """
        with self._connect() as conn:
            if since_timestamp:
                # When since_timestamp is provided, get messages after that time (oldest first)
                rows = conn.execute(
                    """
                    SELECT * FROM message_archive
                    WHERE conversation_id = ? AND timestamp > ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                    """,
                    (conversation_id, since_timestamp, limit),
                ).fetchall()
            else:
                # When no since_timestamp, get the most recent N messages
                # Use subquery to get newest first, then reverse for chronological order
                rows = conn.execute(
                    """
                    SELECT * FROM (
                        SELECT * FROM message_archive
                        WHERE conversation_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ) ORDER BY timestamp ASC
                    """,
                    (conversation_id, limit),
                ).fetchall()

        return self._rows_to_messages(rows)

    def get_all_history_for_node(
        self,
        node_id: str,
        since_timestamp: str | None = None,
        limit: int = 500,
    ) -> list[Message]:
        """
        Get all message history for a node across all conversations.

        Args:
            node_id: The node ID to get history for
            since_timestamp: Only return messages after this ISO timestamp
            limit: Maximum number of messages to return

        Returns:
            List of messages, oldest first (but fetches most recent N messages)
        """
        with self._connect() as conn:
            if since_timestamp:
                # When since_timestamp is provided, get messages after that time (oldest first)
                rows = conn.execute(
                    """
                    SELECT * FROM message_archive
                    WHERE (from_node = ? OR to_node = ? OR conversation_id LIKE ?
                           OR conversation_id IN (
                               SELECT 'channel:' || channel_name
                               FROM channel_members
                               WHERE node_id = ?
                           ))
                      AND timestamp > ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                    """,
                    (node_id, node_id, f"%{node_id}%", node_id, since_timestamp, limit),
                ).fetchall()
            else:
                # When no since_timestamp, get the most recent N messages
                # Use subquery to get newest first, then reverse for chronological order
                rows = conn.execute(
                    """
                    SELECT * FROM (
                        SELECT * FROM message_archive
                        WHERE from_node = ? OR to_node = ? OR conversation_id LIKE ?
                           OR conversation_id IN (
                               SELECT 'channel:' || channel_name
                               FROM channel_members
                               WHERE node_id = ?
                           )
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ) ORDER BY timestamp ASC
                    """,
                    (node_id, node_id, f"%{node_id}%", node_id, limit),
                ).fetchall()

        return self._rows_to_messages(rows)

    def get_conversations_for_node(self, node_id: str) -> list[dict]:
        """
        Get all conversations a node participates in.

        Args:
            node_id: The node ID

        Returns:
            List of dicts with conversation_id, last_timestamp, message_count
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT conversation_id,
                       MAX(timestamp) as last_timestamp,
                       COUNT(*) as message_count
                FROM message_archive
                WHERE from_node = ? OR to_node = ? OR conversation_id LIKE ?
                GROUP BY conversation_id
                ORDER BY last_timestamp DESC
                """,
                (node_id, node_id, f"%{node_id}%"),
            ).fetchall()

        return [
            {
                "conversation_id": row["conversation_id"],
                "last_timestamp": row["last_timestamp"],
                "message_count": row["message_count"],
            }
            for row in rows
        ]

    def get_all_conversations(self) -> list[dict]:
        """
        Get all conversations in the database.

        Returns:
            List of dicts with conversation_id, last_timestamp, message_count
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT conversation_id,
                       MAX(timestamp) as last_timestamp,
                       COUNT(*) as message_count
                FROM message_archive
                GROUP BY conversation_id
                ORDER BY last_timestamp DESC
                """
            ).fetchall()

        return [
            {
                "conversation_id": row["conversation_id"],
                "last_timestamp": row["last_timestamp"],
                "message_count": row["message_count"],
            }
            for row in rows
        ]

    def _rows_to_messages(self, rows: list) -> list[Message]:
        """Convert database rows to Message objects."""
        messages = []
        for row in rows:
            content = row["content"]
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                pass  # Keep as string

            msg = Message(
                id=row["id"],
                from_node=row["from_node"],
                to_node=row["to_node"],
                type=MessageType(row["type"]),
                content=content,
                timestamp=row["timestamp"],
                in_reply_to=row["in_reply_to"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                attachments=self._attachments_from_json(row["attachments_json"] if "attachments_json" in row.keys() else None),
            )
            messages.append(msg)
        return messages

    # =========================================================================
    # Attachment Blob Management
    # =========================================================================

    def register_blob(
        self,
        blob_id: str,
        sha256: str,
        size: int,
        path: str,
        mime_inferred: str,
        owner_node: str,
    ) -> None:
        """Register a content-addressed attachment blob."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO attachment_blobs
                (id, sha256, size, path, mime_inferred, owner_node)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (blob_id, sha256, size, path, mime_inferred, owner_node),
            )
            conn.commit()

    def get_blob(self, blob_id: str) -> AttachmentBlobRow | None:
        """Return attachment blob metadata, if present."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM attachment_blobs WHERE id = ?",
                (blob_id,),
            ).fetchone()
        if not row:
            return None
        return AttachmentBlobRow(
            id=row["id"],
            sha256=row["sha256"],
            size=row["size"],
            path=row["path"],
            mime_inferred=row["mime_inferred"],
            owner_node=row["owner_node"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            ref_count=row["ref_count"],
        )

    def bump_blob_access(self, blob_id: str) -> None:
        """Update the last-accessed timestamp for an attachment blob."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE attachment_blobs SET last_accessed = unixepoch('now') WHERE id = ?",
                (blob_id,),
            )
            conn.commit()

    def blob_owner_total_bytes(self, node_id: str) -> int:
        """Return total bytes first uploaded by an owner node."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(size), 0) AS total FROM attachment_blobs WHERE owner_node = ?",
                (node_id,),
            ).fetchone()
        return int(row["total"] or 0)

    def increment_blob_refs(self, blob_ids: list[str]) -> None:
        """Increment ref counts for canonical stored message references."""
        with self._connect() as conn:
            self._increment_blob_refs(conn, blob_ids)
            conn.commit()

    @staticmethod
    def _increment_blob_refs(conn: sqlite3.Connection, blob_ids: list[str]) -> None:
        for blob_id in blob_ids:
            conn.execute(
                "UPDATE attachment_blobs SET ref_count = ref_count + 1 WHERE id = ?",
                (blob_id,),
            )

    def node_can_access_blob(self, node_id: str, blob_id: str) -> bool:
        """Check whether a node has an archived visible message referencing a blob."""
        pattern = f'%"id": "{blob_id}"%'
        compact_pattern = f'%"id":"{blob_id}"%'
        with self._connect() as conn:
            direct = conn.execute(
                """
                SELECT 1 FROM message_archive
                WHERE (from_node = ? OR to_node = ?)
                  AND (attachments_json LIKE ? OR attachments_json LIKE ?)
                LIMIT 1
                """,
                (node_id, node_id, pattern, compact_pattern),
            ).fetchone()
            if direct:
                return True

            rows = conn.execute(
                """
                SELECT conversation_id FROM message_archive
                WHERE conversation_id LIKE 'channel:%'
                  AND (attachments_json LIKE ? OR attachments_json LIKE ?)
                """,
                (pattern, compact_pattern),
            ).fetchall()
            for row in rows:
                channel_name = row["conversation_id"][len("channel:"):]
                member = conn.execute(
                    """
                    SELECT 1 FROM channel_members
                    WHERE channel_name = ? AND node_id = ?
                    LIMIT 1
                    """,
                    (channel_name, node_id),
                ).fetchone()
                if member:
                    return True
        return False

    # =========================================================================
    # Read Receipts
    # =========================================================================

    def mark_read(
        self,
        node_id: str,
        conversation_id: str,
        timestamp: str,
    ) -> None:
        """
        Mark messages in a conversation as read up to a timestamp.

        Args:
            node_id: The node marking messages as read
            conversation_id: The conversation ID
            timestamp: ISO timestamp - all messages up to this point are read
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO read_receipts (node_id, conversation_id, last_read_timestamp, updated_at)
                VALUES (?, ?, ?, unixepoch('now'))
                ON CONFLICT(node_id, conversation_id) DO UPDATE SET
                    last_read_timestamp = CASE
                        WHEN excluded.last_read_timestamp > read_receipts.last_read_timestamp
                        THEN excluded.last_read_timestamp
                        ELSE read_receipts.last_read_timestamp
                    END,
                    updated_at = unixepoch('now')
                """,
                (node_id, conversation_id, timestamp),
            )
            conn.commit()
        logger.debug(f"Node {node_id} marked {conversation_id} read up to {timestamp}")

    def get_read_timestamp(
        self,
        node_id: str,
        conversation_id: str,
    ) -> str | None:
        """
        Get the last read timestamp for a node in a conversation.

        Args:
            node_id: The node ID
            conversation_id: The conversation ID

        Returns:
            ISO timestamp or None if never read
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT last_read_timestamp FROM read_receipts
                WHERE node_id = ? AND conversation_id = ?
                """,
                (node_id, conversation_id),
            ).fetchone()
            return row["last_read_timestamp"] if row else None

    def get_all_read_receipts(self, node_id: str) -> dict[str, str]:
        """
        Get all read receipts for a node.

        Args:
            node_id: The node ID

        Returns:
            Dict mapping conversation_id -> last_read_timestamp
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT conversation_id, last_read_timestamp FROM read_receipts
                WHERE node_id = ?
                """,
                (node_id,),
            ).fetchall()
            return {row["conversation_id"]: row["last_read_timestamp"] for row in rows}

    # =========================================================================
    # Conversation Summaries
    # =========================================================================

    def save_summary(
        self,
        conversation_id: str,
        summary_text: str,
        messages_summarized: int,
        token_estimate: int,
        created_at: str,
        metadata: dict | None = None,
    ) -> None:
        """
        Save or update a conversation summary.

        Args:
            conversation_id: The conversation ID
            summary_text: The summarized content
            messages_summarized: Number of messages covered by this summary
            token_estimate: Approximate tokens in the summary
            created_at: ISO timestamp when summary was created
            metadata: Optional metadata dict
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_summaries
                (conversation_id, summary_text, messages_summarized, token_estimate, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    summary_text = excluded.summary_text,
                    messages_summarized = excluded.messages_summarized,
                    token_estimate = excluded.token_estimate,
                    created_at = excluded.created_at,
                    metadata = excluded.metadata,
                    updated_at = unixepoch('now')
                """,
                (
                    conversation_id,
                    summary_text,
                    messages_summarized,
                    token_estimate,
                    created_at,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            conn.commit()
        logger.debug(
            f"Saved summary for {conversation_id}: "
            f"{messages_summarized} messages, ~{token_estimate} tokens"
        )

    def get_summary(self, conversation_id: str) -> dict | None:
        """
        Get the summary for a conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            Dict with summary_text, messages_summarized, token_estimate, created_at, metadata
            or None if no summary exists
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT summary_text, messages_summarized, token_estimate, created_at, metadata
                FROM conversation_summaries
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "summary_text": row["summary_text"],
                "messages_summarized": row["messages_summarized"],
                "token_estimate": row["token_estimate"],
                "created_at": row["created_at"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            }

    def delete_summary(self, conversation_id: str) -> bool:
        """
        Delete a conversation summary.

        Args:
            conversation_id: The conversation ID

        Returns:
            True if a summary was deleted, False if none existed
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM conversation_summaries WHERE conversation_id = ?",
                (conversation_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    # =========================================================================
    # User Management (Per-User Authentication)
    # =========================================================================

    @staticmethod
    def _hash_token(token: str) -> str:
        """Hash a token using SHA-256."""
        return hashlib.sha256(token.encode()).hexdigest()

    def create_user(self, username: str) -> str:
        """
        Create a new user with a randomly generated token.

        Args:
            username: The username (must be unique)

        Returns:
            The generated plaintext token (only shown once!)

        Raises:
            ValueError: If username already exists or is invalid
        """
        from .protocol import now_iso

        # Validate username
        if not username or not username.strip():
            raise ValueError("Username cannot be empty")
        username = username.strip()
        if len(username) > 64:
            raise ValueError("Username too long (max 64 characters)")
        if not username.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username can only contain letters, numbers, hyphens, and underscores")

        # Generate a secure random token (32 bytes = 64 hex chars)
        token = secrets.token_hex(32)
        token_hash = self._hash_token(token)

        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO users (username, token_hash, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (username, token_hash, now_iso()),
                )
                conn.commit()
                logger.info(f"Created user: {username}")
                return token
            except sqlite3.IntegrityError:
                raise ValueError(f"User '{username}' already exists")

    def validate_user_token(self, token: str) -> tuple[str, list[str] | None] | None:
        """
        Validate a token and return the associated username and allowed prefixes.

        Args:
            token: The plaintext token to validate

        Returns:
            (username, allowed_prefixes) if valid and not disabled, None otherwise.
            allowed_prefixes is a list of strings or None (no restriction).
        """
        token_hash = self._hash_token(token.strip())

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT username, allowed_prefixes FROM users
                WHERE token_hash = ? AND disabled = 0
                """,
                (token_hash,),
            ).fetchone()
            if not row:
                return None
            prefixes = None
            if row["allowed_prefixes"]:
                import json
                prefixes = json.loads(row["allowed_prefixes"])
            return (row["username"], prefixes)

    def list_users(self) -> list[dict]:
        """
        List all users.

        Returns:
            List of dicts with id, username, created_at, disabled
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, username, created_at, disabled, allowed_prefixes
                FROM users
                ORDER BY created_at
                """
            ).fetchall()
            import json as _json
            return [
                {
                    "id": row["id"],
                    "username": row["username"],
                    "created_at": row["created_at"],
                    "disabled": bool(row["disabled"]),
                    "allowed_prefixes": _json.loads(row["allowed_prefixes"]) if row["allowed_prefixes"] else None,
                }
                for row in rows
            ]

    def set_allowed_prefixes(self, username: str, prefixes: list[str] | None) -> bool:
        """
        Set identity prefixes a user's token is allowed to register as.

        Args:
            username: The username
            prefixes: List of allowed node_id prefixes (e.g. ["agent:", "test:"])
                      or None to remove restrictions

        Returns:
            True if user was updated, False if user not found
        """
        import json
        value = json.dumps(prefixes) if prefixes is not None else None
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET allowed_prefixes = ? WHERE username = ?",
                (value, username),
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Set allowed_prefixes for {username}: {prefixes}")
                return True
            return False

    def disable_user(self, username: str) -> bool:
        """
        Disable a user (revoke access without deleting).

        Args:
            username: The username to disable

        Returns:
            True if user was disabled, False if user not found
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET disabled = 1 WHERE username = ?",
                (username,),
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Disabled user: {username}")
                return True
            return False

    def enable_user(self, username: str) -> bool:
        """
        Re-enable a disabled user.

        Args:
            username: The username to enable

        Returns:
            True if user was enabled, False if user not found
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET disabled = 0 WHERE username = ?",
                (username,),
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Enabled user: {username}")
                return True
            return False

    def delete_user(self, username: str) -> bool:
        """
        Permanently delete a user.

        Args:
            username: The username to delete

        Returns:
            True if user was deleted, False if user not found
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM users WHERE username = ?",
                (username,),
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Deleted user: {username}")
                return True
            return False

    def regenerate_user_token(self, username: str) -> str | None:
        """
        Generate a new token for an existing user.

        Args:
            username: The username

        Returns:
            The new plaintext token, or None if user not found
        """
        token = secrets.token_hex(32)
        token_hash = self._hash_token(token)

        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET token_hash = ? WHERE username = ?",
                (token_hash, username),
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Regenerated token for user: {username}")
                return token
            return None

    def set_user_token(self, username: str, token: str) -> bool:
        """
        Set a user's token to a specific value.

        Args:
            username: The username
            token: The plaintext token to set

        Returns:
            True if token was set, False if user not found
        """
        token_hash = self._hash_token(token)

        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET token_hash = ? WHERE username = ?",
                (token_hash, username),
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Set custom token for user: {username}")
                return True
            return False

    # =========================================================================
    # Scratchpad Notes
    # =========================================================================

    def get_scratchpad(self, conversation_id: str) -> dict | None:
        """Get a single scratchpad note. Returns {content, updated_at, updated_by} or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content, updated_at, updated_by FROM scratchpad_notes WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if row:
                return {
                    "content": row["content"],
                    "updated_at": row["updated_at"],
                    "updated_by": row["updated_by"],
                }
            return None

    def get_scratchpads(self, conversation_ids: list[str]) -> dict:
        """Get scratchpad notes for specific conversations."""
        result = {}
        with self._connect() as conn:
            for cid in conversation_ids:
                row = conn.execute(
                    "SELECT content, updated_at, updated_by FROM scratchpad_notes WHERE conversation_id = ?",
                    (cid,),
                ).fetchone()
                if row:
                    result[cid] = {
                        "content": row["content"],
                        "updated_at": row["updated_at"],
                        "updated_by": row["updated_by"],
                    }
        return result

    def get_all_scratchpads(self) -> dict:
        """Get all scratchpad notes."""
        result = {}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT conversation_id, content, updated_at, updated_by FROM scratchpad_notes"
            ).fetchall()
            for row in rows:
                result[row["conversation_id"]] = {
                    "content": row["content"],
                    "updated_at": row["updated_at"],
                    "updated_by": row["updated_by"],
                }
        return result

    def set_scratchpad(
        self,
        conversation_id: str,
        content: str,
        updated_by: str,
        client_timestamp: str,
    ) -> tuple[bool, dict]:
        """Set scratchpad with optimistic concurrency.

        Accepts if client_timestamp >= stored updated_at (or no existing record).
        Returns (accepted, current_state).
        """
        from .protocol import now_iso

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT content, updated_at, updated_by FROM scratchpad_notes WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()

            if existing and client_timestamp < existing["updated_at"]:
                return (False, {
                    "content": existing["content"],
                    "updated_at": existing["updated_at"],
                    "updated_by": existing["updated_by"],
                })

            server_now = now_iso()
            conn.execute(
                """
                INSERT INTO scratchpad_notes (conversation_id, content, updated_at, updated_by)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    content = excluded.content,
                    updated_at = excluded.updated_at,
                    updated_by = excluded.updated_by
                """,
                (conversation_id, content, server_now, updated_by),
            )
            conn.commit()

            current = {
                "content": content,
                "updated_at": server_now,
                "updated_by": updated_by,
            }
            logger.debug(f"Scratchpad set for {conversation_id} by {updated_by}")
            return (True, current)

    # =========================================================================
    # Conversation Todos
    # =========================================================================

    _TODO_STATUSES = {"open", "in_progress", "done", "cancelled"}

    @staticmethod
    def _clean_todo_section(section: Any) -> str | None:
        """Normalize a todo section label for storage."""
        if section is None:
            return None
        clean = str(section).strip()
        return clean or None

    @classmethod
    def _clean_todo_section_order(cls, section_order: list[Any] | None) -> list[str]:
        """Normalize and de-duplicate a conversation's section ordering."""
        if section_order is None:
            return []
        if not isinstance(section_order, list):
            raise ValueError("section_order must be a list")
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in section_order:
            section = cls._clean_todo_section(item)
            if section is None:
                continue
            key = section.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(section)
        return cleaned

    @staticmethod
    def _todo_from_row(row: sqlite3.Row | None) -> dict | None:
        """Convert a conversation_todos row to a JSON-serializable dict."""
        if row is None:
            return None
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        return {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "text": row["text"],
            "section": row["section"],
            "status": row["status"],
            "position": row["position"],
            "priority": row["priority"],
            "created_at": row["created_at"],
            "created_by": row["created_by"],
            "updated_at": row["updated_at"],
            "updated_by": row["updated_by"],
            "completed_at": row["completed_at"],
            "completed_by": row["completed_by"],
            "deleted_at": row["deleted_at"],
            "version": row["version"],
            "metadata": metadata,
        }

    def get_todo(self, todo_id: str, include_deleted: bool = False) -> dict | None:
        """Get a single todo row by stable id."""
        where = "id = ?"
        params: list[Any] = [todo_id]
        if not include_deleted:
            where += " AND deleted_at IS NULL"
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM conversation_todos WHERE {where}",
                params,
            ).fetchone()
            return self._todo_from_row(row)

    def list_todos(
        self,
        conversation_id: str,
        include_done: bool = False,
        include_deleted: bool = False,
        limit: int = 100,
    ) -> list[dict]:
        """List todos for a conversation."""
        clauses = ["conversation_id = ?"]
        params: list[Any] = [conversation_id]
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        if not include_done:
            clauses.append("status IN ('open', 'in_progress')")
        safe_limit = max(1, min(int(limit or 100), 1000))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM conversation_todos
                WHERE {' AND '.join(clauses)}
                ORDER BY position ASC, created_at ASC
                LIMIT ?
                """,
                (*params, safe_limit),
            ).fetchall()
            return [todo for row in rows if (todo := self._todo_from_row(row))]

    def get_todo_section_order(self, conversation_id: str) -> list[str]:
        """Return the configured todo section order for a conversation."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT section_order FROM conversation_todo_settings
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
        if not row or not row["section_order"]:
            return []
        try:
            parsed = json.loads(row["section_order"])
        except json.JSONDecodeError:
            return []
        try:
            return self._clean_todo_section_order(parsed)
        except ValueError:
            return []

    def set_todo_section_order(
        self,
        conversation_id: str,
        section_order: list[Any] | None,
        updated_by: str,
    ) -> list[str]:
        """Set or clear the configured todo section order for a conversation."""
        from .protocol import now_iso

        cleaned = self._clean_todo_section_order(section_order)
        now = now_iso()
        with self._connect() as conn:
            if section_order is None:
                conn.execute(
                    "DELETE FROM conversation_todo_settings WHERE conversation_id = ?",
                    (conversation_id,),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO conversation_todo_settings
                    (conversation_id, section_order, updated_at, updated_by)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(conversation_id) DO UPDATE SET
                        section_order = excluded.section_order,
                        updated_at = excluded.updated_at,
                        updated_by = excluded.updated_by
                    """,
                    (conversation_id, json.dumps(cleaned), now, updated_by),
                )
            conn.commit()
        return cleaned

    def add_todo(
        self,
        conversation_id: str,
        text: str,
        created_by: str,
        priority: int = 0,
        position: int | None = None,
        section: str | None = None,
    ) -> dict:
        """Add an open todo to a conversation."""
        from .protocol import now_iso

        clean_text = str(text or "").strip()
        if not clean_text:
            raise ValueError("todo text is required")
        clean_section = self._clean_todo_section(section)

        with self._connect() as conn:
            if position is None:
                row = conn.execute(
                    """
                    SELECT COALESCE(MAX(position), -1) + 1 AS next_position
                    FROM conversation_todos
                    WHERE conversation_id = ? AND deleted_at IS NULL
                    """,
                    (conversation_id,),
                ).fetchone()
                position = int(row["next_position"] if row else 0)
            now = now_iso()
            todo_id = f"todo-{uuid.uuid4().hex[:12]}"
            conn.execute(
                """
                INSERT INTO conversation_todos
                (id, conversation_id, text, section, status, position, priority,
                 created_at, created_by, updated_at, updated_by, metadata)
                VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, '{}')
                """,
                (
                    todo_id,
                    conversation_id,
                    clean_text,
                    clean_section,
                    int(position),
                    int(priority or 0),
                    now,
                    created_by,
                    now,
                    created_by,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM conversation_todos WHERE id = ?",
                (todo_id,),
            ).fetchone()
            return self._todo_from_row(row) or {}

    def update_todo(
        self,
        todo_id: str,
        updated_by: str,
        text: str | None = None,
        status: str | None = None,
        priority: int | None = None,
        position: int | None = None,
        section: str | None = None,
        update_section: bool = False,
        expected_version: int | None = None,
    ) -> tuple[bool, dict]:
        """Update a todo row, returning (accepted, current_state)."""
        from .protocol import now_iso

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM conversation_todos WHERE id = ? AND deleted_at IS NULL",
                (todo_id,),
            ).fetchone()
            current = self._todo_from_row(existing)
            if not current:
                return False, {"error": "todo not found", "id": todo_id}
            if expected_version is not None and int(expected_version) != current["version"]:
                return False, current

            updates = ["updated_at = ?", "updated_by = ?", "version = version + 1"]
            now = now_iso()
            params: list[Any] = [now, updated_by]

            if text is not None:
                clean_text = str(text).strip()
                if not clean_text:
                    return False, {"error": "todo text is required", **current}
                updates.append("text = ?")
                params.append(clean_text)
            if status is not None:
                clean_status = str(status).strip().lower()
                if clean_status not in self._TODO_STATUSES:
                    return False, {
                        "error": f"invalid status '{status}'",
                        "allowed_statuses": sorted(self._TODO_STATUSES),
                        **current,
                    }
                updates.append("status = ?")
                params.append(clean_status)
                if clean_status in {"done", "cancelled"}:
                    updates.extend(["completed_at = ?", "completed_by = ?"])
                    params.extend([now, updated_by])
                else:
                    updates.extend(["completed_at = NULL", "completed_by = NULL"])
            if priority is not None:
                updates.append("priority = ?")
                params.append(int(priority))
            if position is not None:
                updates.append("position = ?")
                params.append(int(position))
            if update_section:
                updates.append("section = ?")
                params.append(self._clean_todo_section(section))

            params.append(todo_id)
            conn.execute(
                f"UPDATE conversation_todos SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM conversation_todos WHERE id = ?",
                (todo_id,),
            ).fetchone()
            return True, self._todo_from_row(row) or {}

    def delete_todo(
        self,
        todo_id: str,
        updated_by: str,
        expected_version: int | None = None,
    ) -> tuple[bool, dict]:
        """Soft-delete a todo row."""
        from .protocol import now_iso

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM conversation_todos WHERE id = ? AND deleted_at IS NULL",
                (todo_id,),
            ).fetchone()
            current = self._todo_from_row(existing)
            if not current:
                return False, {"error": "todo not found", "id": todo_id}
            if expected_version is not None and int(expected_version) != current["version"]:
                return False, current

            now = now_iso()
            conn.execute(
                """
                UPDATE conversation_todos
                SET deleted_at = ?, updated_at = ?, updated_by = ?, version = version + 1
                WHERE id = ?
                """,
                (now, now, updated_by, todo_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM conversation_todos WHERE id = ?",
                (todo_id,),
            ).fetchone()
            return True, self._todo_from_row(row) or {}

    def reorder_todos(
        self,
        conversation_id: str,
        ordered_ids: list[str],
        updated_by: str,
    ) -> list[dict]:
        """Densely reorder live todos for a conversation by the supplied ids."""
        from .protocol import now_iso

        now = now_iso()
        with self._connect() as conn:
            for position, todo_id in enumerate(ordered_ids):
                conn.execute(
                    """
                    UPDATE conversation_todos
                    SET position = ?, updated_at = ?, updated_by = ?, version = version + 1
                    WHERE id = ? AND conversation_id = ? AND deleted_at IS NULL
                    """,
                    (position, now, updated_by, todo_id, conversation_id),
                )
            conn.commit()
        return self.list_todos(conversation_id, include_done=True, limit=1000)
