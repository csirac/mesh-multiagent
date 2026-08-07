"""Durable entity registry and transactional memory-correction service.

The registry is deliberately per-agent: callers pass the SQLite connection
owned by :class:`mesh.memory.store.MemoryStore`.  Public mutation methods own
their transaction unless their name explicitly says ``_in_transaction``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import sqlite3
import unicodedata
from typing import Any, Iterable

from .write_audit import CURATION_WRITE_ATTEMPT_EVENT


ENTITY_TYPES = frozenset({"person", "project", "event", "group"})
ENTITY_STATUSES = frozenset({"pending", "active", "retired"})
MEMORY_EDIT_FIELDS = frozenset(
    {"summary", "reflection", "retrieval_key", "tags", "outcome"}
)
_MEMORY_COLUMNS = (
    "id, summary, reflection, retrieval_key, tags, outcome, "
    "reflection_embedding, retrieval_key_embedding"
)
_ENTITY_COLUMNS = (
    "entity_key, entity_type, display_name, identity_note, status, "
    "replacement_key, origin, evidence_version, created_at, updated_at, "
    "activated_at, retired_at"
)
DEFAULT_ACTIVE_ENTITY_CAP = 1000

logger = logging.getLogger(__name__)


class EntityError(ValueError):
    """Base class for registry validation and correction failures."""


class EntityAuthorityError(EntityError):
    """The execution context does not authorize the requested mutation."""


class ConcurrentMemoryEditError(EntityError):
    """The memory changed after replacement embeddings were prepared."""


@dataclass(frozen=True)
class EntityExecutionContext:
    """Immutable authority and attribution captured at tool-execution time."""

    actor_node: str
    source_message_id: str | None = None
    source_author: str | None = None
    source_content: str = ""
    #: Set only when the mutation runs inside an internal self-curation turn,
    #: to that turn's ``CurationBatch.turn_id``.  Links made in a curation turn
    #: derive their evidence-window key from it, so entities the curation path
    #: builds can activate on the same recurrence rule as windowed formation.
    #: ``None`` on a user-driven interactive correction, which is a single
    #: human act and carries no window provenance to record.
    curation_turn_id: str | None = None


@dataclass(frozen=True)
class RegistryInjection:
    """Deterministic entity-registry payload and its measured envelope."""

    payload: str
    entity_keys: tuple[str, ...]
    statuses: dict[str, str]
    candidates_injected: int
    serialized_token_count: int


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokens(surface: str) -> list[str]:
    """Return normalized Unicode tokens for aliases and slugs.

    Combining marks stay attached to the preceding letter/digit.  This is
    necessary for scripts that encode accents and vowel signs as marks even
    after NFKC normalization; dropping them would collapse distinct names.
    """
    if not isinstance(surface, str):
        raise EntityError("entity names and aliases must be strings")
    normalized = unicodedata.normalize("NFKC", surface).casefold()
    tokens: list[str] = []
    current: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)[:1]
        if category in {"L", "N"}:
            current.append(char)
        elif category == "M" and current:
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def normalize_alias(surface: str) -> str:
    """Normalize an alias using NFKC, casefolding, and separator folding."""
    return " ".join(_tokens(surface))


def make_entity_slug(surface: str) -> str:
    """Return the readable key slug corresponding to ``normalize_alias``."""
    return "-".join(_tokens(surface))


def canonical_dossier_hash(
    title: str,
    body: str,
    citations: list[Any] | str,
    cross_refs: list[Any] | str,
) -> str:
    """SHA-256 of the canonical dossier payload used by synchronization checks."""

    def decoded(value: list[Any] | str) -> list[Any]:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = []
            return parsed if isinstance(parsed, list) else []
        return list(value)

    payload = json.dumps(
        {
            "title": title,
            "body": body,
            "citations": decoded(citations),
            "cross_refs": decoded(cross_refs),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _row_dict(columns: tuple[str, ...], row: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(columns, row))


def _json_details(details: dict[str, Any] | None) -> str:
    return json.dumps(
        details or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


class EntityService:
    """Read and mutation API for one connection's entity registry."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        actor_node: str = "memory-store",
        activation_window_threshold: int = 3,
        mutations_enabled: bool = False,
        active_entity_cap: int = DEFAULT_ACTIVE_ENTITY_CAP,
    ):
        if activation_window_threshold < 1:
            raise EntityError("entity activation window threshold must be at least 1")
        if active_entity_cap < 1:
            raise EntityError("active entity cap must be at least 1")
        self.connection = connection
        self.actor_node = actor_node
        self.activation_window_threshold = activation_window_threshold
        self.mutations_enabled = mutations_enabled
        self.active_entity_cap = active_entity_cap

    # ── Read API ──────────────────────────────────────────────

    def get_entity(self, entity_key: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            f"SELECT {_ENTITY_COLUMNS} FROM entities WHERE entity_key = ?",
            (entity_key,),
        ).fetchone()
        if row is None:
            return None
        return _row_dict(tuple(part.strip() for part in _ENTITY_COLUMNS.split(",")), row)

    def list_registry(
        self,
        statuses: Iterable[str] = ("active",),
        include_aliases: bool = True,
    ) -> list[dict[str, Any]]:
        status_values = tuple(statuses)
        invalid = set(status_values) - ENTITY_STATUSES
        if invalid:
            raise EntityError(f"invalid entity status: {sorted(invalid)[0]}")
        if status_values:
            placeholders = ",".join("?" for _ in status_values)
            rows = self.connection.execute(
                f"SELECT {_ENTITY_COLUMNS} FROM entities "
                f"WHERE status IN ({placeholders}) "
                "ORDER BY entity_type, display_name COLLATE NOCASE, entity_key",
                status_values,
            ).fetchall()
        else:
            rows = self.connection.execute(
                f"SELECT {_ENTITY_COLUMNS} FROM entities "
                "ORDER BY entity_type, display_name COLLATE NOCASE, entity_key"
            ).fetchall()
        columns = tuple(part.strip() for part in _ENTITY_COLUMNS.split(","))
        entities = [_row_dict(columns, row) for row in rows]
        if include_aliases and entities:
            aliases_by_key: dict[str, list[dict[str, Any]]] = {
                entity["entity_key"]: [] for entity in entities
            }
            placeholders = ",".join("?" for _ in entities)
            alias_rows = self.connection.execute(
                "SELECT entity_key, normalized_alias, display_alias, source, created_at "
                f"FROM entity_aliases WHERE entity_key IN ({placeholders}) "
                "ORDER BY entity_key, normalized_alias, display_alias",
                tuple(aliases_by_key),
            ).fetchall()
            for row in alias_rows:
                aliases_by_key[row[0]].append(
                    {
                        "normalized_alias": row[1],
                        "display_alias": row[2],
                        "source": row[3],
                        "created_at": row[4],
                    }
                )
            for entity in entities:
                entity["aliases"] = aliases_by_key[entity["entity_key"]]
        return entities

    def serialize_registry_for_injection(
        self,
        injection_cap: int = DEFAULT_ACTIVE_ENTITY_CAP,
    ) -> RegistryInjection:
        """Serialize active rows, then newest pending rows, under one cap."""
        if injection_cap < 1:
            raise EntityError("entity registry injection cap must be at least 1")
        columns = tuple(part.strip() for part in _ENTITY_COLUMNS.split(","))
        active_rows = self.connection.execute(
            f"SELECT {_ENTITY_COLUMNS} FROM entities "
            "WHERE status = 'active' ORDER BY entity_key"
        ).fetchall()
        remaining = max(0, injection_cap - len(active_rows))
        pending_rows = self.connection.execute(
            f"SELECT {_ENTITY_COLUMNS} FROM entities "
            "WHERE status = 'pending' "
            "ORDER BY created_at DESC, entity_key LIMIT ?",
            (remaining,),
        ).fetchall()
        selected = [
            _row_dict(columns, row)
            for row in [*active_rows[:injection_cap], *pending_rows]
        ]

        aliases_by_key: dict[str, list[str]] = {
            entity["entity_key"]: [] for entity in selected
        }
        if aliases_by_key:
            placeholders = ",".join("?" for _ in aliases_by_key)
            alias_rows = self.connection.execute(
                "SELECT entity_key, display_alias FROM entity_aliases "
                f"WHERE entity_key IN ({placeholders}) "
                "ORDER BY entity_key, normalized_alias, display_alias",
                tuple(aliases_by_key),
            ).fetchall()
            for entity_key, display_alias in alias_rows:
                aliases_by_key[entity_key].append(display_alias)

        lines = [
            "### ENTITY REGISTRY",
            "This is the complete active/pending registry within the configured "
            "injection cap. Use exact keys; absence means no serialized entity.",
        ]
        statuses = {
            row[0]: row[1]
            for row in self.connection.execute(
                "SELECT entity_key, status FROM entities ORDER BY entity_key"
            ).fetchall()
        }
        keys: list[str] = []
        for entity in selected:
            entity_key = entity["entity_key"]
            aliases = ",".join(aliases_by_key[entity_key])[:64]
            row = {
                "key": entity_key,
                "type": entity["entity_type"],
                "display_name": entity["display_name"],
                "aliases": aliases,
                "identity_note": (entity["identity_note"] or "")[:128],
            }
            lines.append(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            keys.append(entity_key)
        if not selected:
            lines.append("No active or pending entities.")
        payload = "\n".join(lines)
        from ..llm import estimate_tokens

        serialized_token_count = estimate_tokens(payload)
        if len(active_rows) > injection_cap:
            logger.error(
                "active entity registry exceeds injection cap: active=%d cap=%d",
                len(active_rows),
                injection_cap,
            )
        return RegistryInjection(
            payload=payload,
            entity_keys=tuple(keys),
            statuses=statuses,
            candidates_injected=len(selected),
            serialized_token_count=serialized_token_count,
        )

    def resolve_alias(
        self,
        surface: str,
        entity_type: str | None = None,
        include_pending: bool = True,
        include_retired: bool = True,
    ) -> list[dict[str, Any]]:
        normalized = normalize_alias(surface)
        if not normalized:
            return []
        if entity_type is not None:
            self._validate_entity_type(entity_type)
        statuses = ["active"]
        if include_pending:
            statuses.append("pending")
        if include_retired:
            statuses.append("retired")
        placeholders = ",".join("?" for _ in statuses)
        params: list[Any] = [normalized, *statuses]
        type_clause = ""
        if entity_type is not None:
            type_clause = " AND e.entity_type = ?"
            params.append(entity_type)
        qualified_columns = ", ".join(
            f"e.{part.strip()}" for part in _ENTITY_COLUMNS.split(",")
        )
        rows = self.connection.execute(
            f"SELECT {qualified_columns}, a.display_alias, a.source "
            "FROM entity_aliases AS a "
            "JOIN entities AS e ON e.entity_key = a.entity_key "
            f"WHERE a.normalized_alias = ? AND e.status IN ({placeholders})"
            f"{type_clause} "
            "ORDER BY e.status, e.entity_type, e.display_name COLLATE NOCASE, "
            "e.entity_key",
            tuple(params),
        ).fetchall()
        entity_columns = tuple(part.strip() for part in _ENTITY_COLUMNS.split(","))
        result = []
        for row in rows:
            entity = _row_dict(entity_columns, row[: len(entity_columns)])
            entity["matched_display_alias"] = row[-2]
            entity["alias_source"] = row[-1]
            result.append(entity)
        return result

    def links_for_memory(self, memory_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT me.memory_id, me.entity_key, me.window_key, "
            "me.assignment_source, me.assigned_at, e.entity_type, "
            "e.display_name, e.identity_note, e.status "
            "FROM memory_entities AS me "
            "JOIN entities AS e ON e.entity_key = me.entity_key "
            "WHERE me.memory_id = ? ORDER BY me.entity_key",
            (memory_id,),
        ).fetchall()
        columns = (
            "memory_id",
            "entity_key",
            "window_key",
            "assignment_source",
            "assigned_at",
            "entity_type",
            "display_name",
            "identity_note",
            "status",
        )
        return [_row_dict(columns, row) for row in rows]

    def memory_ids_for_entity(self, entity_key: str) -> list[str]:
        return [
            row[0]
            for row in self.connection.execute(
                "SELECT memory_id FROM memory_entities WHERE entity_key = ? "
                "ORDER BY assigned_at, memory_id",
                (entity_key,),
            ).fetchall()
        ]

    def group_members(self, group_key: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT gm.group_key, gm.member_key, gm.role, gm.source, gm.added_at, "
            "e.entity_type, e.display_name, e.identity_note, e.status "
            "FROM entity_group_members AS gm "
            "JOIN entities AS e ON e.entity_key = gm.member_key "
            "WHERE gm.group_key = ? "
            "ORDER BY e.display_name COLLATE NOCASE, gm.member_key",
            (group_key,),
        ).fetchall()
        columns = (
            "group_key",
            "member_key",
            "role",
            "source",
            "added_at",
            "entity_type",
            "display_name",
            "identity_note",
            "status",
        )
        return [_row_dict(columns, row) for row in rows]

    def events_for_memory(self, memory_id: str) -> list[dict[str, Any]]:
        return self._events("memory_id = ?", (memory_id,))

    def events_for_entity(self, entity_key: str) -> list[dict[str, Any]]:
        return self._events(
            "(entity_key = ? OR old_entity_key = ? OR new_entity_key = ?)",
            (entity_key, entity_key, entity_key),
        )

    def dossier_needs_work(self, entity_key: str) -> bool:
        row = self.connection.execute(
            "SELECT e.status, e.evidence_version, s.title, s.body, s.citations, "
            "s.cross_refs, s.curated_version, s.verified_hash "
            "FROM entities AS e LEFT JOIN essays AS s "
            "ON s.entity_key = e.entity_key WHERE e.entity_key = ?",
            (entity_key,),
        ).fetchone()
        if row is None or row[0] != "active" or row[2] is None:
            return True
        if row[6] != row[1] or not row[7]:
            return True
        return row[7] != canonical_dossier_hash(row[2], row[3], row[4], row[5])

    # ── Public mutation API ───────────────────────────────────

    def create_pending_entity(
        self,
        entity_type: str,
        display_name: str,
        identity_note: str = "",
        *,
        aliases: Iterable[str] = (),
        origin: str = "formation",
        context: EntityExecutionContext | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        self._require_mutations_enabled()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            affected: set[str] = set()
            entity = self._create_pending(
                entity_type,
                display_name,
                identity_note,
                aliases=aliases,
                origin=origin,
                context=context,
                reason=reason,
                affected=affected,
                reject_retired_alias=origin.startswith(
                    ("formation", "historical", "self-curation")
                ),
            )
            self._bump_entities(affected)
            self.connection.commit()
            return entity
        except Exception:
            self.connection.rollback()
            raise

    def activate_if_eligible(
        self,
        entity_key: str,
        *,
        context: EntityExecutionContext | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        self._require_mutations_enabled()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            affected: set[str] = set()
            activated, collision = self._activate_if_eligible(
                entity_key,
                context=context,
                reason=reason,
                affected=affected,
            )
            self._bump_entities(affected)
            self.connection.commit()
            return {
                "entity": self.get_entity(entity_key),
                "activated": activated,
                "collision": collision,
            }
        except Exception:
            self.connection.rollback()
            raise

    def create_user_named_entity(
        self,
        entity_type: str,
        display_name: str,
        *,
        naming_surface: str,
        identity_note: str = "",
        aliases: Iterable[str] = (),
        origin: str = "user-named",
        context: EntityExecutionContext | None,
        reason: str = "",
    ) -> dict[str, Any]:
        self._require_mutations_enabled()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self._require_active_capacity()
            error = self._user_naming_error(context, naming_surface)
            if error:
                self._record_event(
                    "entity_creation_rejected",
                    context=context,
                    reason=reason or error,
                    details={
                        "display_name": display_name,
                        "entity_type": entity_type,
                        "error": error,
                    },
                )
                self.connection.commit()
                return {"created": False, "error": error}
            affected: set[str] = set()
            entity = self._create_pending(
                entity_type,
                display_name,
                identity_note,
                aliases=aliases,
                origin=origin,
                context=context,
                reason=reason,
                affected=affected,
                reject_retired_alias=False,
            )
            now = _utcnow()
            self.connection.execute(
                "UPDATE entities SET status = 'active', activated_at = ?, "
                "updated_at = ? WHERE entity_key = ?",
                (now, now, entity["entity_key"]),
            )
            affected.add(entity["entity_key"])
            self._record_event(
                "entity_activated",
                entity_key=entity["entity_key"],
                context=context,
                reason=reason or "user named entity with verbatim evidence",
            )
            self._bump_entities(affected)
            self.connection.commit()
            return {"created": True, "entity": self.get_entity(entity["entity_key"])}
        except Exception:
            self.connection.rollback()
            raise

    def add_alias(
        self,
        entity_key: str,
        display_alias: str,
        *,
        source: str,
        context: EntityExecutionContext | None = None,
        reason: str = "",
    ) -> bool:
        self._require_mutations_enabled()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            changed = self._add_alias(
                entity_key,
                display_alias,
                source=source,
                context=context,
                reason=reason,
            )
            self.connection.commit()
            return changed
        except Exception:
            self.connection.rollback()
            raise

    def remove_alias(
        self,
        entity_key: str,
        surface: str,
        *,
        context: EntityExecutionContext | None = None,
        reason: str = "",
    ) -> bool:
        self._require_mutations_enabled()
        normalized = normalize_alias(surface)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.connection.execute(
                "DELETE FROM entity_aliases "
                "WHERE entity_key = ? AND normalized_alias = ?",
                (entity_key, normalized),
            )
            if cursor.rowcount:
                self._record_event(
                    "entity_alias_removed",
                    entity_key=entity_key,
                    context=context,
                    reason=reason,
                    details={"normalized_alias": normalized},
                )
            self.connection.commit()
            return cursor.rowcount > 0
        except Exception:
            self.connection.rollback()
            raise

    def update_entity_details(
        self,
        entity_key: str,
        *,
        display_name: str | None = None,
        identity_note: str | None = None,
        status: str | None = None,
        context: EntityExecutionContext | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        self._require_mutations_enabled()
        if status is not None:
            if status not in {"pending", "active"}:
                raise EntityError("use retire_entity to set retired status")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            current = self._require_entity(entity_key)
            updates: dict[str, Any] = {}
            if display_name is not None and display_name != current["display_name"]:
                if not display_name.strip() or not make_entity_slug(display_name):
                    raise EntityError("display_name must produce a non-empty entity slug")
                updates["display_name"] = display_name
            if identity_note is not None and identity_note != current["identity_note"]:
                updates["identity_note"] = identity_note
            if status is not None and status != current["status"]:
                updates["status"] = status
                if status == "active":
                    updates["activated_at"] = _utcnow()
            if not updates:
                self.connection.commit()
                return {"changed": False, "entity": current}
            updates["updated_at"] = _utcnow()
            assignments = ", ".join(f"{key} = ?" for key in updates)
            self.connection.execute(
                f"UPDATE entities SET {assignments} WHERE entity_key = ?",
                (*updates.values(), entity_key),
            )
            if display_name is not None:
                self._add_alias(
                    entity_key,
                    display_name,
                    source="display-name",
                    context=context,
                    reason=reason,
                )
            self._record_event(
                "entity_details_updated",
                entity_key=entity_key,
                context=context,
                reason=reason,
                details={"fields": sorted(updates.keys() - {"updated_at"})},
            )
            self._bump_entities({entity_key})
            self.connection.commit()
            return {"changed": True, "entity": self.get_entity(entity_key)}
        except Exception:
            self.connection.rollback()
            raise

    def retire_entity(
        self,
        entity_key: str,
        *,
        replacement_key: str | None = None,
        context: EntityExecutionContext | None = None,
        reason: str = "",
    ) -> bool:
        self._require_mutations_enabled()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            current = self._require_entity(entity_key)
            if replacement_key == entity_key:
                raise EntityError("replacement_key must differ from entity_key")
            if replacement_key is not None:
                replacement = self._require_entity(replacement_key)
                if replacement["status"] == "retired":
                    raise EntityError("replacement entity must not be retired")
            if (
                current["status"] == "retired"
                and current["replacement_key"] == replacement_key
            ):
                self.connection.commit()
                return False
            now = _utcnow()
            self.connection.execute(
                "UPDATE entities SET status = 'retired', replacement_key = ?, "
                "retired_at = ?, updated_at = ? WHERE entity_key = ?",
                (replacement_key, now, now, entity_key),
            )
            self._record_event(
                "entity_retired",
                entity_key=entity_key,
                old_entity_key=entity_key,
                new_entity_key=replacement_key,
                context=context,
                reason=reason,
            )
            self._bump_entities({entity_key})
            self.connection.commit()
            return True
        except Exception:
            self.connection.rollback()
            raise

    def link_memory(
        self,
        memory_id: str,
        entity_key: str,
        *,
        window_key: str | None = None,
        assignment_source: str = "manual",
        context: EntityExecutionContext | None = None,
        reason: str = "",
        activate: bool = False,
    ) -> bool:
        self._require_mutations_enabled()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            affected: set[str] = set()
            changed = self._link_memory(
                memory_id,
                entity_key,
                window_key=window_key,
                assignment_source=assignment_source,
                context=context,
                reason=reason,
                affected=affected,
            )
            if activate:
                self._activate_if_eligible(
                    entity_key,
                    context=context,
                    reason=reason,
                    affected=affected,
                )
            self._bump_entities(affected)
            self.connection.commit()
            return changed
        except Exception:
            self.connection.rollback()
            raise

    def unlink_memory(
        self,
        memory_id: str,
        entity_key: str,
        *,
        context: EntityExecutionContext | None = None,
        reason: str = "",
    ) -> bool:
        self._require_mutations_enabled()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            affected: set[str] = set()
            changed = self._unlink_memory(
                memory_id,
                entity_key,
                context=context,
                reason=reason,
                affected=affected,
            )
            self._bump_entities(affected)
            self.connection.commit()
            return changed
        except Exception:
            self.connection.rollback()
            raise

    def record_unresolved(
        self,
        *,
        memory_id: str | None = None,
        entity_key: str | None = None,
        reason: str,
        window_key: str | None = None,
        run_key: str | None = None,
        details: dict[str, Any] | None = None,
        context: EntityExecutionContext | None = None,
    ) -> int:
        self._require_mutations_enabled()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            sequence = self._record_event(
                "entity_unresolved",
                entity_key=entity_key,
                memory_id=memory_id,
                context=context,
                reason=reason,
                window_key=window_key,
                run_key=run_key,
                details=details,
            )
            self.connection.commit()
            return sequence
        except Exception:
            self.connection.rollback()
            raise

    def record_curation_event(
        self,
        event_type: str,
        *,
        run_key: str | None = None,
        reason: str = "",
        details: dict[str, Any] | None = None,
        context: EntityExecutionContext | None = None,
    ) -> int:
        """Record one self-curation lifecycle event in ``entity_events``.

        Curation completion/failure is not an unresolved entity observation.
        Give those events their specified first-class event types so status and
        audit consumers do not have to inspect ``details_json`` to distinguish
        them from genuine ``entity_unresolved`` rows.
        """
        if event_type not in {
            "curation_turn",
            "curation_turn_failed",
            "curation_essays",
            CURATION_WRITE_ATTEMPT_EVENT,
        }:
            raise EntityError(f"unsupported curation event type {event_type!r}")
        self._require_mutations_enabled()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            sequence = self._record_event(
                event_type,
                context=context,
                reason=reason,
                run_key=run_key,
                details=details,
            )
            self.connection.commit()
            return sequence
        except Exception:
            self.connection.rollback()
            raise

    def add_group_member(
        self,
        group_key: str,
        member_key: str,
        *,
        role: str = "",
        source: str = "manual",
        context: EntityExecutionContext | None = None,
        reason: str = "",
    ) -> bool:
        self._require_mutations_enabled()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            group = self._require_entity(group_key)
            self._require_entity(member_key)
            if group["entity_type"] != "group":
                raise EntityError(f"{group_key!r} is not a group entity")
            if group_key == member_key:
                raise EntityError("a group cannot contain itself")
            existing = self.connection.execute(
                "SELECT role, source FROM entity_group_members "
                "WHERE group_key = ? AND member_key = ?",
                (group_key, member_key),
            ).fetchone()
            if existing and existing[0] == role and existing[1] == source:
                self.connection.commit()
                return False
            now = _utcnow()
            if existing and existing[0] == role:
                self.connection.execute(
                    "UPDATE entity_group_members SET source = ?, added_at = ? "
                    "WHERE group_key = ? AND member_key = ?",
                    (source, now, group_key, member_key),
                )
                self._record_event(
                    "entity_group_source_updated",
                    entity_key=group_key,
                    new_entity_key=member_key,
                    context=context,
                    reason=reason,
                    details={"role": role, "source": source},
                )
                self.connection.commit()
                return True
            if existing:
                self.connection.execute(
                    "UPDATE entity_group_members SET role = ?, source = ?, "
                    "added_at = ? WHERE group_key = ? AND member_key = ?",
                    (role, source, now, group_key, member_key),
                )
                event_type = "entity_group_role_changed"
            else:
                self.connection.execute(
                    "INSERT INTO entity_group_members "
                    "(group_key, member_key, role, source, added_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (group_key, member_key, role, source, now),
                )
                event_type = "entity_group_member_added"
            self._record_event(
                event_type,
                entity_key=group_key,
                new_entity_key=member_key,
                context=context,
                reason=reason,
                details={"role": role, "source": source},
            )
            self._invalidate_dossier_verification(group_key)
            self._bump_entities({group_key})
            self.connection.commit()
            return True
        except Exception:
            self.connection.rollback()
            raise

    def remove_group_member(
        self,
        group_key: str,
        member_key: str,
        *,
        context: EntityExecutionContext | None = None,
        reason: str = "",
    ) -> bool:
        self._require_mutations_enabled()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.connection.execute(
                "DELETE FROM entity_group_members "
                "WHERE group_key = ? AND member_key = ?",
                (group_key, member_key),
            )
            if cursor.rowcount:
                self._record_event(
                    "entity_group_member_removed",
                    entity_key=group_key,
                    old_entity_key=member_key,
                    context=context,
                    reason=reason,
                )
                self._invalidate_dossier_verification(group_key)
                self._bump_entities({group_key})
            self.connection.commit()
            return cursor.rowcount > 0
        except Exception:
            self.connection.rollback()
            raise

    # ── Self-curation: merge, dossier publication, groups ─────

    def merge_entities(
        self,
        loser_key: str,
        winner_key: str,
        *,
        context: EntityExecutionContext | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """Fold ``loser_key`` into ``winner_key`` in one transaction.

        A merge is ``retired`` + ``replacement_key`` plus relinking — there is
        no ``merged`` status.  The loser's dossier is retained as historical
        source material: a failed or interrupted prose merge must not destroy
        the only copy.
        """
        self._require_mutations_enabled()
        if loser_key == winner_key:
            raise EntityError("loser_key must differ from winner_key")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            loser = self._require_entity(loser_key)
            winner = self._require_entity(winner_key)
            if loser["status"] == "retired":
                raise EntityError(f"{loser_key!r} is already retired")
            if winner["status"] == "retired":
                raise EntityError(f"{winner_key!r} is retired and cannot win a merge")
            if loser["entity_type"] != winner["entity_type"]:
                raise EntityError(
                    "merge requires compatible entity types "
                    f"({loser['entity_type']!r} vs {winner['entity_type']!r})"
                )

            affected: set[str] = {loser_key, winner_key}

            # 2. Move links without deleting any memory.
            moved_links: list[str] = []
            rows = self.connection.execute(
                "SELECT memory_id, window_key, assignment_source, assigned_at "
                "FROM memory_entities WHERE entity_key = ? ORDER BY memory_id",
                (loser_key,),
            ).fetchall()
            for memory_id, window_key, assignment_source, assigned_at in rows:
                exists = self.connection.execute(
                    "SELECT 1 FROM memory_entities "
                    "WHERE memory_id = ? AND entity_key = ?",
                    (memory_id, winner_key),
                ).fetchone()
                if exists is None:
                    self.connection.execute(
                        "INSERT INTO memory_entities "
                        "(memory_id, entity_key, window_key, assignment_source, "
                        "assigned_at) VALUES (?, ?, ?, ?, ?)",
                        (
                            memory_id,
                            winner_key,
                            window_key,
                            assignment_source or "merge",
                            assigned_at or _utcnow(),
                        ),
                    )
                    moved_links.append(memory_id)
                self.connection.execute(
                    "DELETE FROM memory_entities "
                    "WHERE memory_id = ? AND entity_key = ?",
                    (memory_id, loser_key),
                )

            # 3. Copy aliases with source='merge'.  The loser keeps its own
            # alias rows so the retired-alias recreation guard still sees them.
            copied_aliases: list[str] = []
            for normalized, display in self.connection.execute(
                "SELECT normalized_alias, display_alias FROM entity_aliases "
                "WHERE entity_key = ? ORDER BY normalized_alias",
                (loser_key,),
            ).fetchall():
                present = self.connection.execute(
                    "SELECT 1 FROM entity_aliases "
                    "WHERE entity_key = ? AND normalized_alias = ?",
                    (winner_key, normalized),
                ).fetchone()
                if present is not None:
                    continue
                self.connection.execute(
                    "INSERT INTO entity_aliases "
                    "(entity_key, normalized_alias, display_alias, source, "
                    "created_at) VALUES (?, ?, ?, 'merge', ?)",
                    (winner_key, normalized, display, _utcnow()),
                )
                copied_aliases.append(normalized)

            # 4. Rewrite group membership on both sides, idempotently.
            touched_groups: set[str] = {
                row[0]
                for row in self.connection.execute(
                    "SELECT group_key FROM entity_group_members "
                    "WHERE member_key = ?",
                    (loser_key,),
                ).fetchall()
            }
            self._rewrite_group_rows(loser_key, winner_key)
            touched_groups.discard(loser_key)
            if winner["entity_type"] == "group":
                touched_groups.add(winner_key)
            active_groups: list[str] = []
            for key in sorted(touched_groups):
                row = self.get_entity(key)
                if (
                    row is not None
                    and row["entity_type"] == "group"
                    and row["status"] == "active"
                ):
                    active_groups.append(key)

            # 5. Retire the loser and bump/invalidate the winner.
            now = _utcnow()
            self.connection.execute(
                "UPDATE entities SET status = 'retired', replacement_key = ?, "
                "retired_at = ?, updated_at = ? WHERE entity_key = ?",
                (winner_key, now, now, loser_key),
            )
            self._invalidate_dossier_verification(winner_key)
            for group_key in active_groups:
                affected.add(group_key)
                self._invalidate_dossier_verification(group_key)

            # 6/7. Audit and commit; the loser dossier is left in place.
            loser_dossier = self._dossier_row(loser_key)
            winner_dossier = self._dossier_row(winner_key)
            self._record_event(
                "entity_merged",
                entity_key=winner_key,
                old_entity_key=loser_key,
                new_entity_key=winner_key,
                context=context,
                reason=reason,
                details={
                    "moved_links": moved_links,
                    "copied_aliases": copied_aliases,
                    "groups_touched": sorted(touched_groups),
                },
            )
            self._bump_entities(affected)
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return {
            "merged": True,
            "loser": self.get_entity(loser_key),
            "winner": self.get_entity(winner_key),
            "moved_links": moved_links,
            "copied_aliases": copied_aliases,
            "affected_groups": active_groups,
            "loser_dossier": loser_dossier,
            "winner_dossier": winner_dossier,
        }

    def _rewrite_group_rows(self, loser_key: str, winner_key: str) -> None:
        """Point every membership row at ``winner_key`` without violating CHECK.

        ``entity_group_members`` forbids ``group_key = member_key``, so a
        rewrite that would create a self-membership or a duplicate row drops
        the row instead.
        """
        member_rows = self.connection.execute(
            "SELECT group_key, role, source, added_at FROM entity_group_members "
            "WHERE member_key = ? ORDER BY group_key",
            (loser_key,),
        ).fetchall()
        self.connection.execute(
            "DELETE FROM entity_group_members WHERE member_key = ?",
            (loser_key,),
        )
        for group_key, role, source, added_at in member_rows:
            target_group = winner_key if group_key == loser_key else group_key
            if target_group == winner_key:
                # The winner cannot be a member of itself.
                continue
            existing = self.connection.execute(
                "SELECT 1 FROM entity_group_members "
                "WHERE group_key = ? AND member_key = ?",
                (target_group, winner_key),
            ).fetchone()
            if existing is not None:
                continue
            self.connection.execute(
                "INSERT INTO entity_group_members "
                "(group_key, member_key, role, source, added_at) "
                "VALUES (?, ?, ?, 'merge', ?)",
                (target_group, winner_key, role or "", added_at or _utcnow()),
            )
            del source

        group_rows = self.connection.execute(
            "SELECT member_key, role, source, added_at FROM entity_group_members "
            "WHERE group_key = ? ORDER BY member_key",
            (loser_key,),
        ).fetchall()
        self.connection.execute(
            "DELETE FROM entity_group_members WHERE group_key = ?",
            (loser_key,),
        )
        for member_key, role, source, added_at in group_rows:
            if member_key == winner_key:
                continue
            existing = self.connection.execute(
                "SELECT 1 FROM entity_group_members "
                "WHERE group_key = ? AND member_key = ?",
                (winner_key, member_key),
            ).fetchone()
            if existing is not None:
                continue
            self.connection.execute(
                "INSERT INTO entity_group_members "
                "(group_key, member_key, role, source, added_at) "
                "VALUES (?, ?, ?, 'merge', ?)",
                (winner_key, member_key, role or "", added_at or _utcnow()),
            )
            del source

    def _dossier_row(self, entity_key: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT entity_key, title, body, citations, cross_refs, patch_count, "
            "curated_version, verified_hash, verified_at "
            "FROM essays WHERE entity_key = ?",
            (entity_key,),
        ).fetchone()
        if row is None:
            return None
        return _row_dict(
            (
                "entity_key",
                "title",
                "body",
                "citations",
                "cross_refs",
                "patch_count",
                "curated_version",
                "verified_hash",
                "verified_at",
            ),
            row,
        )

    def _invalidate_dossier_verification(self, entity_key: str) -> None:
        self.connection.execute(
            "UPDATE essays SET verified_hash = '', verified_at = NULL, "
            "updated_at = ? WHERE entity_key = ?",
            (_utcnow(), entity_key),
        )

    def group_bridge_windows(self, group_key: str) -> list[str]:
        """Distinct ``window_key`` values carrying bridge evidence for a group.

        Bridge evidence is a memory linked to at least two currently-active
        members of the group.  ``window_key`` is the durable formation window
        identifier, not the process-local ``window_idx``.
        """
        rows = self.connection.execute(
            "SELECT me.window_key "
            "FROM memory_entities AS me "
            "JOIN entity_group_members AS gm "
            "  ON gm.member_key = me.entity_key AND gm.group_key = ? "
            "JOIN entities AS member "
            "  ON member.entity_key = gm.member_key AND member.status = 'active' "
            "WHERE me.window_key IS NOT NULL "
            "GROUP BY me.memory_id, me.window_key "
            "HAVING COUNT(DISTINCT gm.member_key) >= 2",
            (group_key,),
        ).fetchall()
        windows: list[str] = []
        for (window_key,) in rows:
            if window_key and window_key not in windows:
                windows.append(window_key)
        return sorted(windows)

    def group_bridge_memory_ids(self, group_key: str) -> set[str]:
        """Memory IDs that link at least two currently-active group members."""
        rows = self.connection.execute(
            "SELECT me.memory_id "
            "FROM memory_entities AS me "
            "JOIN entity_group_members AS gm "
            "  ON gm.member_key = me.entity_key AND gm.group_key = ? "
            "JOIN entities AS member "
            "  ON member.entity_key = gm.member_key AND member.status = 'active' "
            "GROUP BY me.memory_id "
            "HAVING COUNT(DISTINCT gm.member_key) >= 2",
            (group_key,),
        ).fetchall()
        return {row[0] for row in rows}

    def active_group_member_count(self, group_key: str) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM entity_group_members AS gm "
                "JOIN entities AS e ON e.entity_key = gm.member_key "
                "WHERE gm.group_key = ? AND e.status = 'active'",
                (group_key,),
            ).fetchone()[0]
        )

    def group_activation_report(self, group_key: str) -> dict[str, Any]:
        """Evaluate the deterministic group-activation gate without mutating."""
        group = self._require_entity(group_key)
        windows = self.group_bridge_windows(group_key)
        active_members = self.active_group_member_count(group_key)
        reasons: list[str] = []
        if group["entity_type"] != "group":
            reasons.append("not a group entity")
        if not (group["identity_note"] or "").strip():
            reasons.append("group purpose (identity_note) is empty")
        if active_members < 2:
            reasons.append(
                f"requires 2 active members, has {active_members}"
            )
        if len(windows) < self.activation_window_threshold:
            reasons.append(
                f"requires bridge evidence across "
                f"{self.activation_window_threshold} distinct windows, "
                f"has {len(windows)}"
            )
        return {
            "group_key": group_key,
            "status": group["status"],
            "eligible": not reasons,
            "blockers": reasons,
            "active_members": active_members,
            "bridge_windows": windows,
        }

    def activate_group_if_eligible(
        self,
        group_key: str,
        *,
        context: EntityExecutionContext | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """Activate a pending group only on deterministic bridge evidence.

        Stricter than :meth:`activate_if_eligible`: a group needs a purpose,
        two active members, and bridge evidence across
        ``activation_window_threshold`` distinct formation windows.
        """
        self._require_mutations_enabled()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            report = self.group_activation_report(group_key)
            if report["status"] != "pending":
                self.connection.commit()
                return {
                    "activated": False,
                    "entity": self.get_entity(group_key),
                    **report,
                }
            if not report["eligible"]:
                self.connection.commit()
                return {
                    "activated": False,
                    "entity": self.get_entity(group_key),
                    **report,
                }
            affected: set[str] = set()
            activated, collisions = self._activate_group(
                group_key,
                context=context,
                reason=reason,
                affected=affected,
                windows=report["bridge_windows"],
            )
            self._bump_entities(affected)
            self.connection.commit()
            return {
                "activated": activated,
                "collision": collisions,
                "entity": self.get_entity(group_key),
                **report,
            }
        except Exception:
            self.connection.rollback()
            raise

    def _activate_group(
        self,
        group_key: str,
        *,
        context: EntityExecutionContext | None,
        reason: str,
        affected: set[str],
        windows: list[str],
    ) -> tuple[bool, list[str]]:
        collisions = [
            row[0]
            for row in self.connection.execute(
                "SELECT DISTINCT other.entity_key "
                "FROM entity_aliases AS mine "
                "JOIN entity_aliases AS theirs "
                "ON theirs.normalized_alias = mine.normalized_alias "
                "JOIN entities AS other ON other.entity_key = theirs.entity_key "
                "WHERE mine.entity_key = ? AND other.entity_key <> ? "
                "AND other.entity_type = 'group' AND other.status = 'active' "
                "ORDER BY other.entity_key",
                (group_key, group_key),
            ).fetchall()
        ]
        if collisions:
            self._record_event(
                "entity_group_activation_collision",
                entity_key=group_key,
                context=context,
                reason=reason,
                details={"candidates": collisions, "bridge_windows": windows},
            )
            return False, collisions
        if not self._has_active_capacity():
            self._record_event(
                "entity_activation_cap_reached",
                entity_key=group_key,
                context=context,
                reason=reason or "active entity admission cap reached",
                details={"active_entity_cap": self.active_entity_cap},
            )
            return False, []
        now = _utcnow()
        self.connection.execute(
            "UPDATE entities SET status = 'active', activated_at = ?, "
            "updated_at = ? WHERE entity_key = ?",
            (now, now, group_key),
        )
        affected.add(group_key)
        self._record_event(
            "entity_group_activated",
            entity_key=group_key,
            context=context,
            reason=reason,
            details={"bridge_windows": windows},
        )
        return True, []

    def reconcile_group_membership(
        self,
        group_key: str,
        *,
        context: EntityExecutionContext | None = None,
        reason: str = "",
        token_budget: int = 4000,
        measure: Any | None = None,
    ) -> dict[str, Any]:
        """Follow replacements, reconcile the protected roster, report degradation.

        A group is never auto-retired.  Dropping below two active members is
        surfaced as ``degraded`` so the model can merge or explicitly retire it.
        For an active group with an existing dossier, the roster block is
        regenerated and republished through :meth:`publish_dossier`.  If the
        dossier no longer passes citation/roster/budget validation, membership
        remains committed but its already-invalidated verification stays empty
        and ``roster_error`` reports the partial failure.
        """
        self._require_mutations_enabled()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            rewritten: list[dict[str, str]] = []
            dropped: list[str] = []
            rows = self.connection.execute(
                "SELECT gm.member_key, gm.role, gm.added_at, e.status, "
                "e.replacement_key FROM entity_group_members AS gm "
                "JOIN entities AS e ON e.entity_key = gm.member_key "
                "WHERE gm.group_key = ? ORDER BY gm.member_key",
                (group_key,),
            ).fetchall()
            for member_key, role, added_at, status, replacement_key in rows:
                if status != "retired" or not replacement_key:
                    continue
                self.connection.execute(
                    "DELETE FROM entity_group_members "
                    "WHERE group_key = ? AND member_key = ?",
                    (group_key, member_key),
                )
                if replacement_key == group_key:
                    dropped.append(member_key)
                    continue
                existing = self.connection.execute(
                    "SELECT 1 FROM entity_group_members "
                    "WHERE group_key = ? AND member_key = ?",
                    (group_key, replacement_key),
                ).fetchone()
                if existing is not None:
                    dropped.append(member_key)
                    continue
                self.connection.execute(
                    "INSERT INTO entity_group_members "
                    "(group_key, member_key, role, source, added_at) "
                    "VALUES (?, ?, ?, 'reconcile', ?)",
                    (group_key, replacement_key, role or "", added_at or _utcnow()),
                )
                rewritten.append({"from": member_key, "to": replacement_key})
            if rewritten or dropped:
                self._invalidate_dossier_verification(group_key)
                self._record_event(
                    "entity_group_membership_reconciled",
                    entity_key=group_key,
                    context=context,
                    reason=reason or "retired member followed to replacement",
                    details={"rewritten": rewritten, "dropped": dropped},
                )
                self._bump_entities({group_key})
            active_members = self.active_group_member_count(group_key)
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

        roster_reconciled = False
        roster_error = ""
        group = self.get_entity(group_key)
        dossier = self._dossier_row(group_key)
        if (
            group is not None
            and group["entity_type"] == "group"
            and group["status"] == "active"
            and dossier is not None
        ):
            from .curation import render_roster_block, roster_block_of
            if measure is None:
                from ..llm import estimate_tokens

                measure = estimate_tokens
            live_roster = render_roster_block(self.group_members(group_key))
            stored_roster = roster_block_of(dossier["body"] or "")
            if stored_roster is None:
                roster_error = (
                    "active group dossier has no protected roster block"
                )
            elif stored_roster != live_roster:
                candidate = (dossier["body"] or "").replace(
                    stored_roster, live_roster, 1
                )
                try:
                    raw_cross_refs = dossier.get("cross_refs") or "[]"
                    cross_refs = (
                        json.loads(raw_cross_refs)
                        if isinstance(raw_cross_refs, str)
                        else list(raw_cross_refs)
                    )
                    self.publish_dossier(
                        group_key,
                        body=candidate,
                        title=dossier.get("title") or "",
                        cross_refs=list(cross_refs or []),
                        expected_evidence_version=int(
                            self.get_entity(group_key)["evidence_version"]
                        ),
                        expected_entity_type="group",
                        expected_patch_count=int(dossier["patch_count"]),
                        token_budget=int(token_budget),
                        measure=measure,
                        context=context,
                        reason=reason or "protected roster reconciliation",
                    )
                    roster_reconciled = True
                except Exception as exc:
                    roster_error = str(exc)
        return {
            "group_key": group_key,
            "rewritten": rewritten,
            "dropped": dropped,
            "active_members": active_members,
            "degraded": active_members < 2,
            "roster_reconciled": roster_reconciled,
            "roster_error": roster_error,
        }

    def preflight_dossier_addition(
        self, entity_key: str, addition: str,
    ) -> dict[str, Any]:
        """Reject an impossible dossier addition before composition.

        Curation calls this before it derives the full candidate body.  The
        authoritative :meth:`publish_dossier` validation repeats the checks on
        the completed body under its transaction, because a preflight cannot
        replace a commit-time guard.
        """
        from .curation import extract_citations, find_loose_citations

        entity = self._preflight_dossier_target(entity_key)
        loose = find_loose_citations(addition)
        if loose:
            raise EntityError(
                f"malformed memory reference {loose[0]!r}; use [m_<id>]"
            )
        self._validate_dossier_citations(entity, extract_citations(addition))
        return entity

    def _preflight_dossier_target(self, entity_key: str) -> dict[str, Any]:
        """Return a dossier target only when it is already publishable."""
        entity = self._require_entity(entity_key)
        if entity["status"] != "active":
            raise EntityError(
                f"dossier target {entity_key!r} is {entity['status']}: "
                "entity not yet active; only active entities are publishable"
            )
        return entity

    def _validate_dossier_citations(
        self, entity: dict[str, Any], citations: Iterable[str],
    ) -> None:
        """Validate resolvability and entity evidence scope for citations."""
        entity_key = str(entity["entity_key"])
        if entity["entity_type"] == "group":
            allowed = self.group_bridge_memory_ids(entity_key)
            scope_note = (
                "group dossier citations must be bridge evidence linking "
                "at least two active members"
            )
        else:
            allowed = set(self.memory_ids_for_entity(entity_key))
            scope_note = f"citation must be linked to {entity_key!r}"
        for citation in citations:
            # ``memories.id`` and the link tables store the bare hex ID; the
            # ``m_`` prefix belongs to the citation surface only.
            bare = citation.removeprefix("m_")
            row = self.connection.execute(
                "SELECT project FROM memories WHERE id = ?", (bare,)
            ).fetchone()
            if row is None:
                raise EntityError(f"unresolvable citation [{citation}]")
            if bare not in allowed:
                raise EntityError(
                    f"out-of-scope citation [{citation}]: {scope_note}"
                )

    def publish_dossier(
        self,
        entity_key: str,
        *,
        body: str,
        title: str | None = None,
        cross_refs: list[str] | None = None,
        expected_evidence_version: int | None = None,
        expected_entity_type: str | None = None,
        expected_patch_count: int | None = None,
        token_budget: int,
        measure: Any,
        context: EntityExecutionContext | None = None,
        reason: str = "",
        validate_only: bool = False,
    ) -> dict[str, Any]:
        """Validate then upsert one dossier inside a single transaction.

        Validation order is fixed (§6.3): target, citations, bracket tokens,
        roster/cross-references, budget.  Any failure rolls back the whole
        publication; nothing is truncated.

        ``validate_only`` runs the identical validators and then rolls back
        instead of writing — the shadow-mode path, so a dry run can never
        diverge from the authoritative one.
        """
        from .curation import (
            extract_citations,
            find_bracket_tokens,
            find_loose_citations,
            render_roster_block,
            roster_block_of,
        )

        self._require_mutations_enabled()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            # 1. Target.
            entity = self._preflight_dossier_target(entity_key)
            if (
                expected_entity_type is not None
                and entity["entity_type"] != expected_entity_type
            ):
                raise EntityError(
                    "entity type changed between compose and commit"
                )
            if (
                expected_evidence_version is not None
                and int(entity["evidence_version"]) != int(expected_evidence_version)
            ):
                raise EntityError(
                    "evidence_version changed between compose and commit "
                    f"({expected_evidence_version} -> {entity['evidence_version']}); "
                    "re-read the dossier and recompose"
                )

            is_group = entity["entity_type"] == "group"

            # 2. Citations, derived from the body rather than caller metadata.
            citations = extract_citations(body)
            loose = find_loose_citations(body)
            if loose:
                raise EntityError(
                    f"malformed memory reference {loose[0]!r}; use [m_<id>]"
                )
            self._validate_dossier_citations(entity, citations)

            # 3. Bracket tokens.
            tokens = find_bracket_tokens(body)
            if tokens:
                raise EntityError(
                    f"unexpanded placeholder token {tokens[0]!r} in dossier body"
                )

            # 4. Roster and cross-references (groups only).
            resolved_cross_refs = list(cross_refs or [])
            if is_group:
                members = self.group_members(entity_key)
                # Always derive the protected roster from the authoritative
                # membership table.  Accepting a caller-supplied override would
                # let an internal caller certify a stale or fabricated roster.
                live_block = render_roster_block(members)
                candidate_block = roster_block_of(body)
                if candidate_block is None:
                    raise EntityError(
                        "group dossier body must contain the protected roster block"
                    )
                if candidate_block != live_block:
                    raise EntityError(
                        "roster block does not match the live membership table"
                    )
                member_keys = [member["member_key"] for member in members]
                missing = [
                    key for key in member_keys if key not in resolved_cross_refs
                ]
                resolved_cross_refs.extend(missing)

            # 5. Budget.
            measured = int(measure(body))
            if measured > int(token_budget):
                # The refusal carries the budget-pressure order itself
                # (T-001).  "compress before publishing" named the goal but
                # not the procedure, and the curation turn that reads this
                # does not have the fold constitution in context.
                from .ceiling_rules import dossier_ceiling_refusal

                refusal = EntityError(
                    dossier_ceiling_refusal(measured, int(token_budget))
                )
                # Carry the numbers structurally so the write auditor does not
                # have to scrape them back out of this prose (G-004).  The
                # refusal itself is unchanged.
                refusal.measured_tokens = measured
                refusal.budget_tokens = int(token_budget)
                raise refusal

            existing = self._dossier_row(entity_key)
            # Optimistic concurrency: the caller composed ``body`` against a
            # revision it read outside this transaction.  Curation turns now
            # run concurrently with message turns, so another writer may have
            # published in between — refuse rather than overwrite them.
            if expected_patch_count is not None:
                found = int((existing or {}).get("patch_count") or 0)
                if existing is None or found != int(expected_patch_count):
                    raise EntityError(
                        f"dossier {entity_key!r} was modified concurrently "
                        f"(expected revision {int(expected_patch_count)}, "
                        f"found {found if existing is not None else 'no row'}); "
                        "re-read the dossier and reapply the edit"
                    )
            resolved_title = (
                title
                if title is not None
                else (existing["title"] if existing else "")
            )
            citations_json = json.dumps(citations, ensure_ascii=False)
            cross_refs_json = json.dumps(resolved_cross_refs, ensure_ascii=False)
            verified_hash = canonical_dossier_hash(
                resolved_title, body, citations, resolved_cross_refs
            )
            now = _utcnow()
            if validate_only:
                self.connection.rollback()
                return {
                    "entity_key": entity_key,
                    "title": resolved_title,
                    "citations": citations,
                    "cross_refs": resolved_cross_refs,
                    "tokens": measured,
                    "token_budget": int(token_budget),
                    "verified_hash": verified_hash,
                    "verified_at": None,
                    "curated_version": int(entity["evidence_version"]),
                    "validated_only": True,
                }
            if existing is None:
                self.connection.execute(
                    "INSERT INTO essays (entity_key, title, body, citations, "
                    "cross_refs, patch_count, created_at, updated_at, "
                    "curated_version, verified_hash, verified_at) "
                    "VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)",
                    (
                        entity_key,
                        resolved_title,
                        body,
                        citations_json,
                        cross_refs_json,
                        now,
                        now,
                        int(entity["evidence_version"]),
                        verified_hash,
                        now,
                    ),
                )
            else:
                self.connection.execute(
                    "UPDATE essays SET title = ?, body = ?, citations = ?, "
                    "cross_refs = ?, patch_count = patch_count + 1, "
                    "updated_at = ?, curated_version = ?, verified_hash = ?, "
                    "verified_at = ? WHERE entity_key = ?",
                    (
                        resolved_title,
                        body,
                        citations_json,
                        cross_refs_json,
                        now,
                        int(entity["evidence_version"]),
                        verified_hash,
                        now,
                        entity_key,
                    ),
                )
            self._record_event(
                "entity_dossier_published",
                entity_key=entity_key,
                context=context,
                reason=reason,
                details={
                    "citations": citations,
                    "cross_refs": resolved_cross_refs,
                    "tokens": measured,
                    "token_budget": int(token_budget),
                },
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return {
            "entity_key": entity_key,
            "title": resolved_title,
            "citations": citations,
            "cross_refs": resolved_cross_refs,
            "tokens": measured,
            "token_budget": int(token_budget),
            "verified_hash": verified_hash,
            "verified_at": now,
            "curated_version": int(entity["evidence_version"]),
        }

    def edit_memory_transactional(
        self,
        memory_id: str,
        memory_patch: dict[str, Any],
        *,
        prepared_snapshot: dict[str, Any] | None = None,
        reflection_embedding: Any = None,
        retrieval_key_embedding: Any = None,
        embeddings_prepared: bool = False,
        context: EntityExecutionContext | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            affected: set[str] = set()
            result = self._apply_memory_patch(
                memory_id,
                memory_patch,
                prepared_snapshot=prepared_snapshot,
                reflection_embedding=reflection_embedding,
                retrieval_key_embedding=retrieval_key_embedding,
                embeddings_prepared=embeddings_prepared,
                context=context,
                reason=reason,
                affected=affected,
            )
            self._bump_entities(affected)
            self.connection.commit()
            result["affected_entity_keys"] = sorted(affected)
            return result
        except Exception:
            self.connection.rollback()
            raise

    def correct_link_transactional(
        self,
        memory_id: str,
        *,
        reason: str,
        context: EntityExecutionContext,
        prepared_snapshot: dict[str, Any],
        memory_patch: dict[str, Any] | None = None,
        remove_entity_key: str | None = None,
        add_entity_key: str | None = None,
        new_entity_type: str | None = None,
        new_display_name: str | None = None,
        new_identity_note: str = "",
        aliases: Iterable[str] = (),
        naming_surface: str | None = None,
        reflection_embedding: Any = None,
        retrieval_key_embedding: Any = None,
        embeddings_prepared: bool = False,
    ) -> dict[str, Any]:
        self._require_mutations_enabled()
        new_spec_present = any(
            value is not None
            for value in (new_entity_type, new_display_name)
        )
        if add_entity_key and new_spec_present:
            raise EntityError(
                "exactly one of add_entity_key or a new-entity specification may be supplied"
            )
        if new_spec_present and (not new_entity_type or not new_display_name):
            raise EntityError(
                "new_entity_type and new_display_name are both required"
            )
        if not reason.strip():
            raise EntityError("reason is required")
        if new_entity_type:
            self._validate_entity_type(new_entity_type)

        self.connection.execute("BEGIN IMMEDIATE")
        try:
            if new_spec_present:
                authority_error = self._user_naming_error(
                    context, naming_surface or ""
                )
                if authority_error:
                    self._record_event(
                        "entity_creation_rejected",
                        memory_id=memory_id,
                        context=context,
                        reason=reason,
                        details={
                            "display_name": new_display_name,
                            "entity_type": new_entity_type,
                            "error": authority_error,
                        },
                    )
                    self.connection.commit()
                    return {"changed": False, "error": authority_error}

            affected: set[str] = set()
            source_result = self._apply_memory_patch(
                memory_id,
                memory_patch or {},
                prepared_snapshot=prepared_snapshot,
                reflection_embedding=reflection_embedding,
                retrieval_key_embedding=retrieval_key_embedding,
                embeddings_prepared=embeddings_prepared,
                context=context,
                reason=reason,
                affected=affected,
            )
            link_changes: list[str] = []
            if remove_entity_key:
                if self._unlink_memory(
                    memory_id,
                    remove_entity_key,
                    context=context,
                    reason=reason,
                    affected=affected,
                ):
                    link_changes.append(f"removed:{remove_entity_key}")

            resolved_add_key = add_entity_key
            if add_entity_key:
                add_entity = self._require_entity(add_entity_key)
                if add_entity["status"] == "retired":
                    raise EntityError("cannot link a memory to a retired entity")
            elif new_spec_present:
                self._require_active_capacity()
                created = self._create_pending(
                    new_entity_type or "",
                    new_display_name or "",
                    new_identity_note,
                    aliases=aliases,
                    origin="user-named-correction",
                    context=context,
                    reason=reason,
                    affected=affected,
                    reject_retired_alias=False,
                )
                resolved_add_key = created["entity_key"]
                now = _utcnow()
                self.connection.execute(
                    "UPDATE entities SET status = 'active', activated_at = ?, "
                    "updated_at = ? WHERE entity_key = ?",
                    (now, now, resolved_add_key),
                )
                affected.add(resolved_add_key)
                self._record_event(
                    "entity_activated",
                    entity_key=resolved_add_key,
                    context=context,
                    reason=reason,
                )

            if resolved_add_key:
                if self._link_memory(
                    memory_id,
                    resolved_add_key,
                    window_key=self._context_window_key(context),
                    assignment_source="interactive-correction",
                    context=context,
                    reason=reason,
                    affected=affected,
                ):
                    link_changes.append(f"added:{resolved_add_key}")

            if not source_result["changed"] and not link_changes:
                raise EntityError("at least one effective source or link mutation is required")

            # Evidence just changed for the linked entity, so re-run the same
            # admission rule the formation path runs after its links land.
            # Without this a curation turn could accumulate evidence across any
            # number of batches and the entity would stay pending forever: the
            # only other evaluators are ``apply_formation_mutations`` (a path
            # self-curation never takes) and the explicit ``activate_if_eligible``
            # entry point (which nothing calls on this path).  The rule itself is
            # unchanged — ``_activate_if_eligible`` is a no-op unless the entity
            # is pending and already has enough distinct evidence windows.
            activated_keys: list[str] = []
            if resolved_add_key and link_changes:
                activated, _collisions = self._activate_if_eligible(
                    resolved_add_key,
                    context=context,
                    reason=reason or "evidence threshold reached",
                    affected=affected,
                )
                if activated:
                    activated_keys.append(resolved_add_key)

            self._record_event(
                "entity_link_corrected",
                memory_id=memory_id,
                old_entity_key=remove_entity_key,
                new_entity_key=resolved_add_key,
                context=context,
                reason=reason,
                details={
                    "source_fields": source_result["changed_fields"],
                    "link_changes": link_changes,
                },
            )
            self._bump_entities(affected)
            self.connection.commit()
            return {
                "changed": True,
                "memory_changed": source_result["changed"],
                "changed_fields": source_result["changed_fields"],
                "link_changes": link_changes,
                "added_entity_key": resolved_add_key,
                "activated_entity_keys": activated_keys,
                "affected_entity_keys": sorted(affected),
            }
        except Exception:
            self.connection.rollback()
            raise

    def delete_memory_transactional(
        self,
        memory_id: str,
        *,
        context: EntityExecutionContext | None = None,
        reason: str = "memory deleted",
    ) -> bool:
        result = self.delete_memories_transactional(
            [memory_id], context=context, reason=reason
        )
        return bool(result)

    def delete_memories_transactional(
        self,
        memory_ids: Iterable[str],
        *,
        context: EntityExecutionContext | None = None,
        reason: str = "memory deleted",
    ) -> list[str]:
        ordered_ids = list(dict.fromkeys(memory_ids))
        if not ordered_ids:
            return []
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            affected: set[str] = set()
            deleted: list[str] = []
            for memory_id in ordered_ids:
                exists = self.connection.execute(
                    "SELECT 1 FROM memories WHERE id = ?", (memory_id,)
                ).fetchone()
                if exists is None:
                    continue
                link_rows = self.connection.execute(
                    "SELECT entity_key, window_key, assignment_source "
                    "FROM memory_entities WHERE memory_id = ? ORDER BY entity_key",
                    (memory_id,),
                ).fetchall()
                for entity_key, window_key, assignment_source in link_rows:
                    self.connection.execute(
                        "DELETE FROM memory_entities "
                        "WHERE memory_id = ? AND entity_key = ?",
                        (memory_id, entity_key),
                    )
                    affected.add(entity_key)
                    self._record_event(
                        "memory_link_removed_for_delete",
                        entity_key=entity_key,
                        memory_id=memory_id,
                        old_entity_key=entity_key,
                        context=context,
                        reason=reason,
                        window_key=window_key,
                        details={"assignment_source": assignment_source},
                    )
                if self._fts_exists():
                    self.connection.execute(
                        "DELETE FROM memories_fts WHERE id = ?", (memory_id,)
                    )
                self.connection.execute(
                    "DELETE FROM memories WHERE id = ?", (memory_id,)
                )
                self._record_event(
                    "memory_deleted",
                    memory_id=memory_id,
                    context=context,
                    reason=reason,
                )
                deleted.append(memory_id)
            self._bump_entities(affected)
            self.connection.commit()
            return deleted
        except Exception:
            self.connection.rollback()
            raise

    def apply_formation_mutations_in_transaction(
        self,
        mutations: list[dict[str, Any]] | dict[str, Any] | None,
        *,
        context: EntityExecutionContext | None = None,
        run_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Apply an entity batch inside the caller's already-open transaction.

        ``window_key`` is an opaque, stable formation-window identifier.  The
        service compares it only for equality and derives recurrence with
        ``COUNT(DISTINCT window_key)``.
        """
        self._require_mutations_enabled()
        if not self.connection.in_transaction:
            raise EntityError(
                "apply_formation_mutations_in_transaction requires an open transaction"
            )
        if not mutations:
            return []
        if isinstance(mutations, dict):
            mutation_list = mutations.get("mutations")
            if mutation_list is None:
                mutation_list = [mutations]
        else:
            mutation_list = mutations
        if not isinstance(mutation_list, list):
            raise EntityError("formation entity mutations must be a list")

        affected: set[str] = set()
        results: list[dict[str, Any]] = []
        for mutation in mutation_list:
            if not isinstance(mutation, dict):
                raise EntityError("each formation entity mutation must be an object")
            op = mutation.get("op", "resolve_and_link")
            memory_id = mutation.get("memory_id")
            window_key = mutation.get("window_key")
            if op == "record_unresolved":
                self._record_event(
                    "entity_unresolved",
                    entity_key=mutation.get("entity_key"),
                    memory_id=memory_id,
                    context=context,
                    reason=mutation.get("reason", "formation unresolved"),
                    window_key=window_key,
                    run_key=run_key,
                    details=mutation.get("details"),
                )
                results.append({"status": "unresolved"})
                continue
            if op == "raise":
                raise RuntimeError(mutation.get("message", "forced resolver failure"))
            if not memory_id:
                raise EntityError("formation entity mutations require memory_id")

            if op == "link":
                entity_key = mutation.get("entity_key")
                if not entity_key:
                    raise EntityError("link mutation requires entity_key")
            elif op in {"resolve_and_link", "create_and_link"}:
                entity_type = mutation.get("entity_type")
                display_name = mutation.get("display_name")
                if not entity_type or not display_name:
                    raise EntityError(
                        "resolve_and_link requires entity_type and display_name"
                    )
                self._validate_entity_type(entity_type)
                candidates = self.resolve_alias(
                    display_name,
                    entity_type=entity_type,
                    include_pending=True,
                    include_retired=True,
                )
                live_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate["status"] != "retired"
                ]
                retired_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate["status"] == "retired"
                ]
                if retired_candidates and not live_candidates:
                    self._record_event(
                        "entity_unresolved",
                        memory_id=memory_id,
                        context=context,
                        reason="retired entity alias cannot be recreated from formation",
                        window_key=window_key,
                        run_key=run_key,
                        details={
                            "surface": display_name,
                            "candidates": [
                                item["entity_key"] for item in retired_candidates
                            ],
                        },
                    )
                    results.append({"status": "retired-alias-rejected"})
                    continue
                if len(live_candidates) > 1:
                    self._record_event(
                        "entity_unresolved",
                        memory_id=memory_id,
                        context=context,
                        reason="ambiguous entity alias",
                        window_key=window_key,
                        run_key=run_key,
                        details={
                            "surface": display_name,
                            "candidates": [
                                {
                                    "entity_key": item["entity_key"],
                                    "identity_note": item["identity_note"],
                                }
                                for item in live_candidates
                            ],
                        },
                    )
                    results.append({"status": "ambiguous"})
                    continue
                if live_candidates:
                    entity_key = live_candidates[0]["entity_key"]
                else:
                    entity = self._create_pending(
                        entity_type,
                        display_name,
                        mutation.get("identity_note", ""),
                        aliases=mutation.get("aliases") or (),
                        origin=mutation.get("origin", "formation"),
                        context=context,
                        reason=mutation.get("reason", ""),
                        affected=affected,
                        reject_retired_alias=True,
                    )
                    entity_key = entity["entity_key"]
            else:
                raise EntityError(f"unsupported formation mutation op {op!r}")

            changed = self._link_memory(
                memory_id,
                entity_key,
                window_key=window_key,
                assignment_source=mutation.get(
                    "assignment_source", "formation"
                ),
                context=context,
                reason=mutation.get("reason", ""),
                affected=affected,
                run_key=run_key,
            )
            activated, collision = self._activate_if_eligible(
                entity_key,
                context=context,
                reason=mutation.get("reason", ""),
                affected=affected,
                run_key=run_key,
            )
            results.append(
                {
                    "status": "linked" if changed else "already-linked",
                    "entity_key": entity_key,
                    "activated": activated,
                    "collision": collision,
                }
            )
        self._bump_entities(affected)
        return results

    # ── Internal helpers ──────────────────────────────────────

    def _events(self, where: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        columns = (
            "sequence",
            "event_type",
            "entity_key",
            "memory_id",
            "old_entity_key",
            "new_entity_key",
            "actor_node",
            "source_message_id",
            "source_author",
            "reason",
            "window_key",
            "run_key",
            "details_json",
            "created_at",
        )
        rows = self.connection.execute(
            f"SELECT {', '.join(columns)} FROM entity_events "
            f"WHERE {where} ORDER BY sequence",
            params,
        ).fetchall()
        events = []
        for row in rows:
            event = _row_dict(columns, row)
            try:
                event["details"] = json.loads(event.pop("details_json"))
            except json.JSONDecodeError:
                event["details"] = {}
                event.pop("details_json", None)
            events.append(event)
        return events

    def _validate_entity_type(self, entity_type: str) -> None:
        if entity_type not in ENTITY_TYPES:
            raise EntityError(
                f"entity_type must be one of {', '.join(sorted(ENTITY_TYPES))}"
            )

    def _require_mutations_enabled(self) -> None:
        if not self.mutations_enabled:
            raise EntityAuthorityError(
                "entity resolution mutations are disabled for this agent"
            )

    def _require_entity(self, entity_key: str) -> dict[str, Any]:
        entity = self.get_entity(entity_key)
        if entity is None:
            raise EntityError(f"unknown entity key {entity_key!r}")
        return entity

    def _next_entity_key(self, entity_type: str, display_name: str) -> str:
        self._validate_entity_type(entity_type)
        slug = make_entity_slug(display_name)
        if not slug:
            raise EntityError("display_name must produce a non-empty entity slug")
        base = f"{entity_type}:{slug}"
        if self.get_entity(base) is None:
            return base
        ordinal = 2
        while self.get_entity(f"{base}-{ordinal}") is not None:
            ordinal += 1
        return f"{base}-{ordinal}"

    def _create_pending(
        self,
        entity_type: str,
        display_name: str,
        identity_note: str,
        *,
        aliases: Iterable[str],
        origin: str,
        context: EntityExecutionContext | None,
        reason: str,
        affected: set[str],
        reject_retired_alias: bool,
    ) -> dict[str, Any]:
        self._validate_entity_type(entity_type)
        if not isinstance(display_name, str) or not display_name.strip():
            raise EntityError("display_name must not be empty")
        normalized = normalize_alias(display_name)
        if not normalized:
            raise EntityError("display_name must produce a non-empty entity slug")
        if isinstance(aliases, (str, bytes)):
            raise EntityError("aliases must be a list of strings")
        aliases = list(aliases)
        if not all(isinstance(alias, str) for alias in aliases):
            raise EntityError("aliases must be a list of strings")
        if reject_retired_alias:
            for surface in (display_name, *aliases):
                surface_slug = make_entity_slug(surface)
                base_entity = (
                    self.get_entity(f"{entity_type}:{surface_slug}")
                    if surface_slug
                    else None
                )
                retired = [
                    item
                    for item in self.resolve_alias(
                        surface,
                        entity_type=entity_type,
                        include_pending=False,
                        include_retired=True,
                    )
                    if item["status"] == "retired"
                ]
                if retired or (
                    base_entity is not None
                    and base_entity["status"] == "retired"
                ):
                    raise EntityError(
                        "retired entity alias cannot be recreated from historical text"
                    )
        entity_key = self._next_entity_key(entity_type, display_name)
        now = _utcnow()
        self.connection.execute(
            "INSERT INTO entities "
            "(entity_key, entity_type, display_name, identity_note, status, "
            "replacement_key, origin, evidence_version, created_at, updated_at, "
            "activated_at, retired_at) "
            "VALUES (?, ?, ?, ?, 'pending', NULL, ?, 0, ?, ?, NULL, NULL)",
            (entity_key, entity_type, display_name, identity_note, origin, now, now),
        )
        self._add_alias(
            entity_key,
            display_name,
            source="display-name",
            context=context,
            reason=reason,
        )
        for alias in aliases:
            self._add_alias(
                entity_key,
                alias,
                source=origin,
                context=context,
                reason=reason,
            )
        self._record_event(
            "entity_created_pending",
            entity_key=entity_key,
            context=context,
            reason=reason,
            details={"origin": origin},
        )
        # Creation itself is version zero.  Later status/link evidence changes
        # add the key to ``affected``.
        return self._require_entity(entity_key)

    def _add_alias(
        self,
        entity_key: str,
        display_alias: str,
        *,
        source: str,
        context: EntityExecutionContext | None,
        reason: str,
    ) -> bool:
        self._require_entity(entity_key)
        normalized = normalize_alias(display_alias)
        if not normalized:
            raise EntityError("alias must normalize to at least one letter or digit")
        existing = self.connection.execute(
            "SELECT display_alias, source FROM entity_aliases "
            "WHERE entity_key = ? AND normalized_alias = ?",
            (entity_key, normalized),
        ).fetchone()
        if existing == (display_alias, source):
            return False
        now = _utcnow()
        self.connection.execute(
            "INSERT INTO entity_aliases "
            "(entity_key, normalized_alias, display_alias, source, created_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(entity_key, normalized_alias) DO UPDATE SET "
            "display_alias=excluded.display_alias, source=excluded.source",
            (entity_key, normalized, display_alias, source, now),
        )
        self._record_event(
            "entity_alias_added" if existing is None else "entity_alias_display_updated",
            entity_key=entity_key,
            context=context,
            reason=reason,
            details={
                "normalized_alias": normalized,
                "display_alias": display_alias,
                "source": source,
            },
        )
        return True

    @staticmethod
    def _context_window_key(
        context: EntityExecutionContext | None,
    ) -> str | None:
        """Evidence-window key for a link made under ``context``, if any.

        Returns the curation batch's derived window key inside a self-curation
        turn and ``None`` outside one.  Keeping the derivation here means the
        forward path and the historical backfill compute the identical key
        from the identical input (the batch ``turn_id``), so a backfilled row
        and a freshly written one are indistinguishable.
        """
        turn_id = getattr(context, "curation_turn_id", None)
        if not turn_id:
            return None
        from .curation import curation_window_key

        return curation_window_key(str(turn_id))

    def _link_memory(
        self,
        memory_id: str,
        entity_key: str,
        *,
        window_key: str | None,
        assignment_source: str,
        context: EntityExecutionContext | None,
        reason: str,
        affected: set[str],
        run_key: str | None = None,
    ) -> bool:
        entity = self._require_entity(entity_key)
        if entity["status"] == "retired":
            raise EntityError("cannot link a memory to a retired entity")
        if self.connection.execute(
            "SELECT 1 FROM memories WHERE id = ?", (memory_id,)
        ).fetchone() is None:
            raise EntityError(f"unknown memory ID {memory_id!r}")
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO memory_entities "
            "(memory_id, entity_key, window_key, assignment_source, assigned_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (memory_id, entity_key, window_key, assignment_source, _utcnow()),
        )
        if not cursor.rowcount:
            return False
        affected.add(entity_key)
        self._record_event(
            "memory_entity_linked",
            entity_key=entity_key,
            memory_id=memory_id,
            new_entity_key=entity_key,
            context=context,
            reason=reason,
            window_key=window_key,
            run_key=run_key,
            details={"assignment_source": assignment_source},
        )
        return True

    def _unlink_memory(
        self,
        memory_id: str,
        entity_key: str,
        *,
        context: EntityExecutionContext | None,
        reason: str,
        affected: set[str],
    ) -> bool:
        row = self.connection.execute(
            "SELECT window_key, assignment_source FROM memory_entities "
            "WHERE memory_id = ? AND entity_key = ?",
            (memory_id, entity_key),
        ).fetchone()
        if row is None:
            return False
        self.connection.execute(
            "DELETE FROM memory_entities WHERE memory_id = ? AND entity_key = ?",
            (memory_id, entity_key),
        )
        affected.add(entity_key)
        self._record_event(
            "memory_entity_unlinked",
            entity_key=entity_key,
            memory_id=memory_id,
            old_entity_key=entity_key,
            context=context,
            reason=reason,
            window_key=row[0],
            details={"assignment_source": row[1]},
        )
        return True

    def _activate_if_eligible(
        self,
        entity_key: str,
        *,
        context: EntityExecutionContext | None,
        reason: str,
        affected: set[str],
        run_key: str | None = None,
    ) -> tuple[bool, list[str]]:
        entity = self._require_entity(entity_key)
        if entity["status"] != "pending":
            return False, []
        count = self.connection.execute(
            "SELECT COUNT(DISTINCT window_key) FROM memory_entities "
            "WHERE entity_key = ? AND window_key IS NOT NULL",
            (entity_key,),
        ).fetchone()[0]
        if count < self.activation_window_threshold:
            return False, []
        collisions = [
            row[0]
            for row in self.connection.execute(
                "SELECT DISTINCT other.entity_key "
                "FROM entity_aliases AS mine "
                "JOIN entity_aliases AS theirs "
                "ON theirs.normalized_alias = mine.normalized_alias "
                "JOIN entities AS other ON other.entity_key = theirs.entity_key "
                "WHERE mine.entity_key = ? AND other.entity_key <> ? "
                "AND other.entity_type = ? AND other.status = 'active' "
                "ORDER BY other.entity_key",
                (entity_key, entity_key, entity["entity_type"]),
            ).fetchall()
        ]
        if collisions:
            self._record_event(
                "entity_activation_collision",
                entity_key=entity_key,
                context=context,
                reason=reason,
                run_key=run_key,
                details={"candidates": collisions, "distinct_windows": count},
            )
            return False, collisions
        if not self._has_active_capacity():
            self._record_event(
                "entity_activation_cap_reached",
                entity_key=entity_key,
                context=context,
                reason=reason or "active entity admission cap reached",
                run_key=run_key,
                details={
                    "active_entity_cap": self.active_entity_cap,
                    "distinct_windows": count,
                },
            )
            logger.error(
                "entity activation blocked at active cap=%d for %s",
                self.active_entity_cap,
                entity_key,
            )
            return False, []
        now = _utcnow()
        self.connection.execute(
            "UPDATE entities SET status = 'active', activated_at = ?, "
            "updated_at = ? WHERE entity_key = ?",
            (now, now, entity_key),
        )
        affected.add(entity_key)
        self._record_event(
            "entity_activated",
            entity_key=entity_key,
            context=context,
            reason=reason,
            run_key=run_key,
            details={"distinct_windows": count},
        )
        return True, []

    def _has_active_capacity(self) -> bool:
        active_count = self.connection.execute(
            "SELECT COUNT(*) FROM entities WHERE status = 'active'"
        ).fetchone()[0]
        return active_count < self.active_entity_cap

    def _require_active_capacity(self) -> None:
        if not self._has_active_capacity():
            raise EntityError(
                f"active entity cap of {self.active_entity_cap} reached"
            )

    def _memory_snapshot(self, memory_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        columns = tuple(part.strip() for part in _MEMORY_COLUMNS.split(","))
        snapshot = _row_dict(columns, row)
        snapshot["tags"] = [
            tag.strip() for tag in (snapshot["tags"] or "").split(",") if tag.strip()
        ]
        return snapshot

    @staticmethod
    def comparable_memory_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            field: snapshot.get(field)
            for field in ("id", "summary", "reflection", "retrieval_key", "tags", "outcome")
        }

    def _apply_memory_patch(
        self,
        memory_id: str,
        memory_patch: dict[str, Any],
        *,
        prepared_snapshot: dict[str, Any] | None,
        reflection_embedding: Any,
        retrieval_key_embedding: Any,
        embeddings_prepared: bool,
        context: EntityExecutionContext | None,
        reason: str,
        affected: set[str],
    ) -> dict[str, Any]:
        if not isinstance(memory_patch, dict):
            raise EntityError("memory_patch must be an object")
        unknown = set(memory_patch) - MEMORY_EDIT_FIELDS
        if unknown:
            raise EntityError(f"unsupported memory patch field {sorted(unknown)[0]!r}")
        current = self._memory_snapshot(memory_id)
        if current is None:
            raise EntityError(f"unknown memory ID {memory_id!r}")
        if prepared_snapshot is not None:
            expected = self.comparable_memory_snapshot(prepared_snapshot)
            observed = self.comparable_memory_snapshot(current)
            if expected != observed:
                raise ConcurrentMemoryEditError(
                    "memory changed while correction embeddings were prepared"
                )

        normalized_patch = dict(memory_patch)
        if "tags" in normalized_patch:
            tags = normalized_patch["tags"]
            if isinstance(tags, str):
                tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
            if not isinstance(tags, list) or not all(
                isinstance(tag, str) for tag in tags
            ):
                raise EntityError("memory_patch.tags must be a list of strings")
            normalized_patch["tags"] = tags
        if "outcome" in normalized_patch and normalized_patch["outcome"] not in {
            "success",
            "partial",
            "failure",
        }:
            raise EntityError("memory_patch.outcome must be success, partial, or failure")
        for text_field in ("summary", "reflection", "retrieval_key"):
            if text_field in normalized_patch and not isinstance(
                normalized_patch[text_field], str
            ):
                raise EntityError(f"memory_patch.{text_field} must be a string")

        changed_fields = [
            field
            for field, value in normalized_patch.items()
            if value != current[field]
        ]
        if not changed_fields:
            return {"changed": False, "changed_fields": [], "memory": current}

        assignments: list[str] = []
        values: list[Any] = []
        for field in changed_fields:
            assignments.append(f"{field if field != 'tags' else 'tags'} = ?")
            value = normalized_patch[field]
            values.append(",".join(value) if field == "tags" else value)
        text_changed = bool(
            {"summary", "reflection", "retrieval_key"} & set(changed_fields)
        )
        if text_changed:
            if not embeddings_prepared:
                raise EntityError(
                    "replacement embeddings must be prepared before the write transaction"
                )
            assignments.extend(
                ["reflection_embedding = ?", "retrieval_key_embedding = ?"]
            )
            values.extend(
                [
                    self._serialize_embedding(reflection_embedding),
                    self._serialize_embedding(retrieval_key_embedding),
                ]
            )
        values.append(memory_id)
        self.connection.execute(
            f"UPDATE memories SET {', '.join(assignments)} WHERE id = ?", values
        )
        updated = self._memory_snapshot(memory_id)
        if updated is None:
            raise EntityError(f"memory {memory_id!r} disappeared during correction")
        if text_changed and self._fts_exists():
            self.connection.execute(
                "DELETE FROM memories_fts WHERE id = ?", (memory_id,)
            )
            self.connection.execute(
                "INSERT INTO memories_fts "
                "(id, summary, reflection, retrieval_key) VALUES (?, ?, ?, ?)",
                (
                    memory_id,
                    updated["summary"],
                    updated["reflection"],
                    updated["retrieval_key"],
                ),
            )
        linked = {
            row[0]
            for row in self.connection.execute(
                "SELECT entity_key FROM memory_entities WHERE memory_id = ?",
                (memory_id,),
            ).fetchall()
        }
        affected.update(linked)
        self._record_event(
            "memory_source_edited",
            memory_id=memory_id,
            context=context,
            reason=reason,
            details={"changed_fields": sorted(changed_fields)},
        )
        return {
            "changed": True,
            "changed_fields": sorted(changed_fields),
            "memory": updated,
        }

    @staticmethod
    def _serialize_embedding(embedding: Any) -> bytes | None:
        if embedding is None:
            return None
        if isinstance(embedding, (bytes, bytearray, memoryview)):
            return bytes(embedding)
        try:
            import numpy as np

            return np.asarray(embedding, dtype=np.float32).tobytes()
        except Exception as exc:
            raise EntityError(f"invalid replacement embedding: {exc}") from exc

    def _fts_exists(self) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='memories_fts'"
            ).fetchone()
            is not None
        )

    def _user_naming_error(
        self,
        context: EntityExecutionContext | None,
        naming_surface: str,
    ) -> str | None:
        if context is None:
            return "entity correction requires an in-process execution context"
        if not (context.source_author or "").startswith("user:"):
            return "immediate entity creation requires a user-authored source message"
        if not naming_surface:
            return "naming_surface is required for immediate entity creation"
        if naming_surface not in context.source_content:
            return "naming_surface must appear verbatim in the source message"
        return None

    def _bump_entities(self, entity_keys: set[str]) -> None:
        if not entity_keys:
            return
        now = _utcnow()
        for entity_key in sorted(entity_keys):
            self.connection.execute(
                "UPDATE entities SET evidence_version = evidence_version + 1, "
                "updated_at = ? WHERE entity_key = ?",
                (now, entity_key),
            )

    def _record_event(
        self,
        event_type: str,
        *,
        entity_key: str | None = None,
        memory_id: str | None = None,
        old_entity_key: str | None = None,
        new_entity_key: str | None = None,
        context: EntityExecutionContext | None = None,
        reason: str = "",
        window_key: str | None = None,
        run_key: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> int:
        actor_node = context.actor_node if context is not None else self.actor_node
        source_message_id = (
            context.source_message_id if context is not None else None
        )
        source_author = context.source_author if context is not None else None
        cursor = self.connection.execute(
            "INSERT INTO entity_events "
            "(event_type, entity_key, memory_id, old_entity_key, new_entity_key, "
            "actor_node, source_message_id, source_author, reason, window_key, "
            "run_key, details_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_type,
                entity_key,
                memory_id,
                old_entity_key,
                new_entity_key,
                actor_node,
                source_message_id,
                source_author,
                reason,
                window_key,
                run_key,
                _json_details(details),
                _utcnow(),
            ),
        )
        return int(cursor.lastrowid)


# Module-level façade for callers that prefer a connection-first service API.
# The class remains useful when several operations share actor/threshold state.

def _for_connection(
    connection: sqlite3.Connection,
    *,
    actor_node: str = "memory-store",
    activation_window_threshold: int = 3,
    mutations_enabled: bool = False,
) -> EntityService:
    return EntityService(
        connection,
        actor_node=actor_node,
        activation_window_threshold=activation_window_threshold,
        mutations_enabled=mutations_enabled,
    )


def get_entity(connection: sqlite3.Connection, entity_key: str):
    return _for_connection(connection).get_entity(entity_key)


def list_registry(
    connection: sqlite3.Connection,
    statuses: Iterable[str] = ("active",),
    include_aliases: bool = True,
):
    return _for_connection(connection).list_registry(statuses, include_aliases)


def serialize_registry_for_injection(
    service_or_connection: EntityService | sqlite3.Connection,
    injection_cap: int = DEFAULT_ACTIVE_ENTITY_CAP,
) -> RegistryInjection:
    service = (
        service_or_connection
        if isinstance(service_or_connection, EntityService)
        else _for_connection(service_or_connection)
    )
    return service.serialize_registry_for_injection(injection_cap)


def resolve_alias(
    connection: sqlite3.Connection,
    surface: str,
    entity_type: str | None = None,
    include_pending: bool = True,
    include_retired: bool = True,
):
    return _for_connection(connection).resolve_alias(
        surface, entity_type, include_pending, include_retired
    )


def links_for_memory(connection: sqlite3.Connection, memory_id: str):
    return _for_connection(connection).links_for_memory(memory_id)


def memory_ids_for_entity(connection: sqlite3.Connection, entity_key: str):
    return _for_connection(connection).memory_ids_for_entity(entity_key)


def group_members(connection: sqlite3.Connection, group_key: str):
    return _for_connection(connection).group_members(group_key)


def events_for_memory(connection: sqlite3.Connection, memory_id: str):
    return _for_connection(connection).events_for_memory(memory_id)


def events_for_entity(connection: sqlite3.Connection, entity_key: str):
    return _for_connection(connection).events_for_entity(entity_key)


def dossier_needs_work(connection: sqlite3.Connection, entity_key: str):
    return _for_connection(connection).dossier_needs_work(entity_key)


def create_pending_entity(
    connection: sqlite3.Connection,
    *args,
    mutations_enabled: bool = False,
    **kwargs,
):
    return _for_connection(
        connection, mutations_enabled=mutations_enabled
    ).create_pending_entity(*args, **kwargs)


def activate_if_eligible(
    connection: sqlite3.Connection,
    *args,
    mutations_enabled: bool = False,
    activation_window_threshold: int = 3,
    **kwargs,
):
    return _for_connection(
        connection,
        mutations_enabled=mutations_enabled,
        activation_window_threshold=activation_window_threshold,
    ).activate_if_eligible(*args, **kwargs)


def create_user_named_entity(
    connection: sqlite3.Connection,
    *args,
    mutations_enabled: bool = False,
    **kwargs,
):
    return _for_connection(
        connection, mutations_enabled=mutations_enabled
    ).create_user_named_entity(*args, **kwargs)


def add_alias(
    connection: sqlite3.Connection,
    *args,
    mutations_enabled: bool = False,
    **kwargs,
):
    return _for_connection(
        connection, mutations_enabled=mutations_enabled
    ).add_alias(*args, **kwargs)


def remove_alias(
    connection: sqlite3.Connection,
    *args,
    mutations_enabled: bool = False,
    **kwargs,
):
    return _for_connection(
        connection, mutations_enabled=mutations_enabled
    ).remove_alias(*args, **kwargs)


def update_entity_details(
    connection: sqlite3.Connection,
    *args,
    mutations_enabled: bool = False,
    **kwargs,
):
    return _for_connection(
        connection, mutations_enabled=mutations_enabled
    ).update_entity_details(*args, **kwargs)


def retire_entity(
    connection: sqlite3.Connection,
    *args,
    mutations_enabled: bool = False,
    **kwargs,
):
    return _for_connection(
        connection, mutations_enabled=mutations_enabled
    ).retire_entity(*args, **kwargs)


def link_memory(
    connection: sqlite3.Connection,
    *args,
    mutations_enabled: bool = False,
    **kwargs,
):
    return _for_connection(
        connection, mutations_enabled=mutations_enabled
    ).link_memory(*args, **kwargs)


def unlink_memory(
    connection: sqlite3.Connection,
    *args,
    mutations_enabled: bool = False,
    **kwargs,
):
    return _for_connection(
        connection, mutations_enabled=mutations_enabled
    ).unlink_memory(*args, **kwargs)


def record_unresolved(
    connection: sqlite3.Connection,
    *,
    mutations_enabled: bool = False,
    **kwargs,
):
    return _for_connection(
        connection, mutations_enabled=mutations_enabled
    ).record_unresolved(**kwargs)


def add_group_member(
    connection: sqlite3.Connection,
    *args,
    mutations_enabled: bool = False,
    **kwargs,
):
    return _for_connection(
        connection, mutations_enabled=mutations_enabled
    ).add_group_member(*args, **kwargs)


def remove_group_member(
    connection: sqlite3.Connection,
    *args,
    mutations_enabled: bool = False,
    **kwargs,
):
    return _for_connection(
        connection, mutations_enabled=mutations_enabled
    ).remove_group_member(*args, **kwargs)


def edit_memory_transactional(
    connection: sqlite3.Connection,
    *args,
    **kwargs,
):
    return _for_connection(connection).edit_memory_transactional(*args, **kwargs)


def correct_link_transactional(
    connection: sqlite3.Connection,
    *args,
    mutations_enabled: bool = False,
    activation_window_threshold: int = 3,
    **kwargs,
):
    return _for_connection(
        connection,
        mutations_enabled=mutations_enabled,
        activation_window_threshold=activation_window_threshold,
    ).correct_link_transactional(*args, **kwargs)


def apply_formation_mutations_in_transaction(
    connection: sqlite3.Connection,
    *args,
    mutations_enabled: bool = False,
    activation_window_threshold: int = 3,
    **kwargs,
):
    return _for_connection(
        connection,
        mutations_enabled=mutations_enabled,
        activation_window_threshold=activation_window_threshold,
    ).apply_formation_mutations_in_transaction(*args, **kwargs)
