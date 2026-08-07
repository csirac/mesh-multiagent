"""Single recall-oriented contract for all memory formation.

The canonical prompt remains in ``mesh/fold_driver/prompts`` because the
standing-digest tooling already treats that file as a deployed artifact.
Live formation and fold compatibility code import the prompt and validator
from this module; no caller carries a private parser or a second prompt.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_MESH_DIR = Path(__file__).resolve().parents[1]
FORMATION_PROMPT_PATH = (
    _MESH_DIR
    / "fold_driver"
    / "prompts"
    / "standing_digest"
    / "form_memories_lowbar.txt"
)
CONSTITUTION_PATH = (
    _MESH_DIR
    / "fold_driver"
    / "prompts"
    / "standing_digest"
    / "constitution.txt"
)

FORMATION_PROMPT = FORMATION_PROMPT_PATH.read_text(encoding="utf-8")
FORMATION_CONTRACT_VERSION = 2
FORMATION_FIELDS = frozenset({
    "summary",
    "reflection",
    "trace",
    "retrieval_key",
    "tags",
    "outcome",
    "topic_label",
    "project",
    "event_date",
    "digest_candidate",
})
_EVENT_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_VALID_OUTCOMES = {"success", "partial", "failure", ""}
_ENTITY_TYPES = frozenset({"person", "project", "event", "group"})
_ENTITY_MODES = frozenset({"shadow", "write"})
_ENTITY_CONTRACT = """\
## ENTITY ASSIGNMENT

The full active/pending entity registry appears after the known-project block.
For every memory record, emit an "entity" object with ALL of these fields:

- "existing_keys": exact registry keys for entities confidently supported by
  this memory. Never invent or alter a key.
- "new_entities": proposals not present anywhere in the full registry. Each
  proposal has "entity_type" (person|project|event|group), "display_name",
  "identity_note", and "aliases" (a JSON list of strings). Do not propose a
  duplicate merely because spelling differs.
- "unresolved": abstentions. Each has "surface", "candidates" (exact registry
  keys considered), and "reason". Use this when identity is uncertain.

The service generates keys for proposals. Never emit a proposed slug or key.
Every record must include all three entity lists, including empty lists.
"""
# The contract is spliced in at this anchor rather than carried as a
# ``{entity_contract}`` placeholder: the prompt file is also rendered by the
# out-of-repo fold driver, whose ``.format()`` call passes only agent_label
# and constitution. A placeholder would make that call raise KeyError.
_ENTITY_CONTRACT_ANCHOR = "## Output format"


class FormationContractError(ValueError):
    """The formation model returned data outside the shared contract."""


def render_formation_prompt(
    *,
    agent_label: str,
    budget_tokens: int = 32_000,
    entity_resolution_mode: str = "off",
) -> str:
    """Render the one canonical prompt with the standing constitution."""
    mode = str(entity_resolution_mode or "off").strip().lower()
    if mode not in {"off", *_ENTITY_MODES}:
        raise ValueError(
            "entity_resolution_mode must be one of: off, shadow, write"
        )
    constitution = CONSTITUTION_PATH.read_text(encoding="utf-8").format(
        budget_tokens=f"{budget_tokens:,}",
        budget_chars=f"{budget_tokens * 4:,}",
    )
    prompt = FORMATION_PROMPT.format(
        agent_label=agent_label,
        constitution=constitution,
    )
    if mode not in _ENTITY_MODES:
        return prompt
    anchor = prompt.find(_ENTITY_CONTRACT_ANCHOR)
    if anchor == -1:
        return f"{prompt.rstrip()}\n\n{_ENTITY_CONTRACT}"
    return f"{prompt[:anchor]}{_ENTITY_CONTRACT}\n{prompt[anchor:]}"


def _json_array(raw: str) -> list[Any]:
    if not isinstance(raw, str) or not raw.strip():
        raise FormationContractError("formation response is empty")
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"\A```(?:json)?\s*|\s*```\Z",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
    match = _JSON_ARRAY_RE.search(cleaned)
    payload = match.group(0) if match else cleaned
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise FormationContractError(f"formation response is not JSON: {exc}") from exc
    if not isinstance(value, list):
        raise FormationContractError("formation response must be a JSON array")
    return value


def _text(record: dict[str, Any], field: str) -> str:
    value = record[field]
    if value is None:
        return ""
    if not isinstance(value, str):
        raise FormationContractError(
            f"{field} must be a string, got {type(value).__name__}"
        )
    return value.strip()


def _tags(value: Any) -> list[str]:
    if isinstance(value, str):
        return [tag.strip() for tag in value.split(",") if tag.strip()]
    if isinstance(value, list):
        return [str(tag).strip() for tag in value if str(tag).strip()]
    raise FormationContractError(
        f"tags must be a comma-separated string or list, got {type(value).__name__}"
    )


def _entity_metadata(
    value: Any,
    *,
    index: int,
    known_entity_statuses: dict[str, str] | None,
    allowed_entity_keys: set[str] | frozenset[str] | None,
) -> tuple[dict[str, list[dict[str, Any]] | list[str]], list[str]]:
    """Normalize one parsed entity block without discarding core memory data."""
    empty: dict[str, list[dict[str, Any]] | list[str]] = {
        "existing_keys": [],
        "new_entities": [],
        "unresolved": [],
    }
    failures: list[str] = []
    if not isinstance(value, dict):
        return empty, [f"record {index} entity must be an object"]

    missing = {"existing_keys", "new_entities", "unresolved"} - value.keys()
    failures.extend(
        f"record {index} entity missing field {field}"
        for field in sorted(missing)
    )

    existing_keys = value.get("existing_keys", [])
    if not isinstance(existing_keys, list):
        failures.append(f"record {index} entity existing_keys must be a list")
        existing_keys = []
    for item_index, key in enumerate(existing_keys):
        if not isinstance(key, str) or not key.strip():
            failures.append(
                f"record {index} entity existing_keys[{item_index}] "
                "must be a non-empty string"
            )
            continue
        key = key.strip()
        status = (
            known_entity_statuses.get(key)
            if known_entity_statuses is not None
            else None
        )
        allowed = (
            key in allowed_entity_keys
            if allowed_entity_keys is not None
            else status is not None
        )
        if status == "retired":
            failures.append(
                f"record {index} retired entity key {key!r}"
            )
            continue
        if not allowed or status is None:
            failures.append(
                f"record {index} unknown entity key {key!r}"
            )
            continue
        if status not in {"active", "pending"}:
            failures.append(
                f"record {index} entity key {key!r} has invalid status {status!r}"
            )
            continue
        empty["existing_keys"].append(key)

    proposals = value.get("new_entities", [])
    if not isinstance(proposals, list):
        failures.append(f"record {index} entity new_entities must be a list")
        proposals = []
    for item_index, proposal in enumerate(proposals):
        prefix = f"record {index} entity new_entities[{item_index}]"
        if not isinstance(proposal, dict):
            failures.append(f"{prefix} must be an object")
            continue
        entity_type = proposal.get("entity_type")
        display_name = proposal.get("display_name")
        identity_note = proposal.get("identity_note")
        aliases = proposal.get("aliases")
        if entity_type not in _ENTITY_TYPES:
            failures.append(f"{prefix} has invalid entity_type {entity_type!r}")
            continue
        if not isinstance(display_name, str) or not display_name.strip():
            failures.append(f"{prefix} display_name must be a non-empty string")
            continue
        if not isinstance(identity_note, str):
            failures.append(f"{prefix} identity_note must be a string")
            continue
        if (
            not isinstance(aliases, list)
            or not all(isinstance(alias, str) for alias in aliases)
        ):
            failures.append(f"{prefix} aliases must be a list of strings")
            continue
        empty["new_entities"].append({
            "entity_type": entity_type,
            "display_name": display_name.strip(),
            "identity_note": identity_note.strip(),
            "aliases": [
                alias.strip() for alias in aliases if alias.strip()
            ],
        })

    unresolved_items = value.get("unresolved", [])
    if not isinstance(unresolved_items, list):
        failures.append(f"record {index} entity unresolved must be a list")
        unresolved_items = []
    for item_index, unresolved in enumerate(unresolved_items):
        prefix = f"record {index} entity unresolved[{item_index}]"
        if not isinstance(unresolved, dict):
            failures.append(f"{prefix} must be an object")
            continue
        surface = unresolved.get("surface")
        candidates = unresolved.get("candidates")
        reason = unresolved.get("reason")
        if not isinstance(surface, str) or not surface.strip():
            failures.append(f"{prefix} surface must be a non-empty string")
            continue
        if (
            not isinstance(candidates, list)
            or not all(isinstance(key, str) for key in candidates)
        ):
            failures.append(f"{prefix} candidates must be a list of strings")
            continue
        if not isinstance(reason, str) or not reason.strip():
            failures.append(f"{prefix} reason must be a non-empty string")
            continue
        valid_candidates: list[str] = []
        for key in candidates:
            key = key.strip()
            status = (
                known_entity_statuses.get(key)
                if known_entity_statuses is not None
                else None
            )
            allowed = (
                key in allowed_entity_keys
                if allowed_entity_keys is not None
                else status is not None
            )
            if not key:
                failures.append(f"{prefix} contains an empty candidate key")
            elif status == "retired":
                failures.append(f"{prefix} has retired candidate key {key!r}")
            elif not allowed or status is None:
                failures.append(f"{prefix} has unknown candidate key {key!r}")
            elif status in {"active", "pending"}:
                valid_candidates.append(key)
            else:
                failures.append(
                    f"{prefix} candidate {key!r} has invalid status {status!r}"
                )
        empty["unresolved"].append({
            "surface": surface.strip(),
            "candidates": valid_candidates,
            "reason": reason.strip(),
        })
    return empty, failures


def parse_formation_response(
    raw: str,
    *,
    entity_resolution_mode: str = "off",
    known_entity_statuses: dict[str, str] | None = None,
    allowed_entity_keys: set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Parse and normalize one formation response.

    Validation intentionally matches the surviving fold contract: every field
    is required, summary and model-selected evidence are non-empty, event_date
    is an ISO calendar date, and digest_candidate is a real boolean.
    """
    mode = str(entity_resolution_mode or "off").strip().lower()
    if mode not in {"off", *_ENTITY_MODES}:
        raise ValueError(
            "entity_resolution_mode must be one of: off, shadow, write"
        )
    entity_enabled = mode in _ENTITY_MODES
    normalized: list[dict[str, Any]] = []
    for index, value in enumerate(_json_array(raw)):
        if not isinstance(value, dict):
            raise FormationContractError(f"record {index} must be an object")
        required_fields = (
            FORMATION_FIELDS | {"entity"}
            if entity_enabled
            else FORMATION_FIELDS
        )
        missing = required_fields - value.keys()
        if missing:
            raise FormationContractError(
                f"record {index} missing fields {sorted(missing)}"
            )

        summary = _text(value, "summary")
        trace = _text(value, "trace")
        if not summary:
            raise FormationContractError(f"record {index} summary is empty")
        if not trace:
            raise FormationContractError(f"record {index} trace is empty")

        event_date = _text(value, "event_date")
        if not _EVENT_DATE_RE.fullmatch(event_date):
            raise FormationContractError(
                f"record {index} has invalid event_date {event_date!r}"
            )
        digest_candidate = value["digest_candidate"]
        if not isinstance(digest_candidate, bool):
            raise FormationContractError(
                f"record {index} digest_candidate must be bool"
            )

        outcome = _text(value, "outcome").lower()
        if outcome not in _VALID_OUTCOMES:
            raise FormationContractError(
                f"record {index} has invalid outcome {outcome!r}"
            )

        record = {
            "summary": summary,
            "reflection": _text(value, "reflection"),
            "trace": trace,
            "retrieval_key": _text(value, "retrieval_key"),
            "tags": _tags(value["tags"]),
            "outcome": outcome,
            "topic_label": _text(value, "topic_label")[:120],
            "project": _text(value, "project"),
            "event_date": event_date,
            "digest_candidate": digest_candidate,
        }
        if entity_enabled:
            entity, failures = _entity_metadata(
                value["entity"],
                index=index,
                known_entity_statuses=known_entity_statuses,
                allowed_entity_keys=allowed_entity_keys,
            )
            record["entity"] = entity
            record["entity_validation_failures"] = failures
        normalized.append(record)
    return normalized
