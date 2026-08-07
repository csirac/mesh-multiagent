"""Governed procedural memory (skill cards).

Skill cards are advisory, human-approved procedure records.  This module owns
their filesystem schema, compact index, deterministic retrieval, injection
format, append-only outcome receipts, and lightweight fold/meta-review hooks.

The runtime deliberately has no activation API.  It can index active cards,
select them, and append receipts; proposed/active/retired transitions remain a
human-controlled file review operation.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import tempfile
import threading
from typing import Any, Iterator
import uuid

import yaml

from .paths import SKILLS_DIR


SCHEMA_VERSION = 1
MAX_SELECTED_CARDS = 3
DEFAULT_SELECTION_THRESHOLD = 0.05
UNKNOWN_ONLY_TEXT_THRESHOLD = 2.0
SKILL_DRAFT_TRACE_MAX_CHARS = 60_000
WORKER_TRACE_ROOT = SKILLS_DIR.parent / "worker_traces"

_CARD_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MEMORY_ID_RE = re.compile(r"^(?:m_)?[0-9a-f]{12}$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_EXPLICIT_FACET_RE = re.compile(
    r"\b(host|service|endpoint|port|model|artifact|action|agent)\s*"
    r"(?:=|:|\bis\b)\s*([^\s,;]+)",
    re.IGNORECASE,
)
_ON_HOST_RE = re.compile(r"\bon\s+([a-z0-9][a-z0-9._-]*)\b", re.IGNORECASE)

_REQUIRED_FIELDS = {
    "schema_version", "id", "version", "status", "owner_agent",
    "purpose", "triggers", "preconditions", "authority",
    "procedure_source", "required_invariants", "verification", "rollback",
    "evidence", "proposed_by", "approved_by", "approved_at",
    "supersedes_version", "last_reviewed_at", "outcomes",
}
_VALID_STATUSES = {"proposed", "active", "retired"}
_VALID_PRECONDITION_OPERATORS = {"equals", "one_of", "present", "absent"}

_PROCESS_LOCK = threading.RLock()


class SkillCardError(ValueError):
    """A card, index, or governed transition failed validation."""


@dataclass(frozen=True)
class SkillSelection:
    """One selected card and the deterministic evidence for its score."""

    card: dict[str, Any]
    score: float
    text_score: float
    precondition_factor: float
    matched: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()
    task_facets: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class FormationSignal:
    """Lightweight fold signal for a structurally recurring procedure."""

    eligible: bool
    success_signal: bool
    likely_recurring: bool
    detail_count: int
    details: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class SkillReviewFinding:
    """One triaged skill-card meta-review finding."""

    tier: int
    card_id: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class SkillDraftPackage:
    """Immutable handoff assembled for one on-demand drafting worker."""

    run_id: str
    task: str
    staging_path: Path
    source_files: tuple[Path, ...]
    trace_path: Path | None = None
    trace_fingerprint: str | None = None
    trace_truncated: bool = False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat()


def _tokens(text: Any) -> list[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


def _normalize_phrase(value: Any) -> str:
    return " ".join(_tokens(value))


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the canonical SHA-256 fingerprint for a local file."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def _validate_owner_agent(owner_agent: str) -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", owner_agent or ""):
        raise SkillCardError(f"invalid owner agent {owner_agent!r}")


def persist_completed_worker_trace(
    owner_agent: str,
    worker_id: str,
    trace_text: str,
    *,
    task_summary: str = "",
    kind: str = "worker",
    trace_root: str | Path | None = None,
    completed_at: datetime | None = None,
) -> Path:
    """Persist one full completed-worker trace as a private source artifact.

    Prompt-capture logs describe launch context, not execution, and the router's
    ``WorkerResult`` trace is otherwise ephemeral.  This archive is therefore
    the canonical source used by later ``skill_draft`` calls.  Drafting-worker
    traces are retained for audit but skipped by automatic latest-trace lookup
    to avoid recursively teaching a card from a prior card-drafting run.
    """
    _validate_owner_agent(owner_agent)
    if not isinstance(worker_id, str) or not worker_id.strip():
        raise SkillCardError("worker_id is required for trace persistence")
    if not isinstance(trace_text, str) or not trace_text.strip():
        raise SkillCardError("completed worker trace is empty")
    if kind not in {"worker", "skill_draft"}:
        raise SkillCardError(f"invalid worker trace kind {kind!r}")

    timestamp = completed_at or _utc_now()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    safe_worker_id = re.sub(r"[^A-Za-z0-9._-]+", "-", worker_id).strip("-.")
    safe_worker_id = safe_worker_id or "worker"
    trace_dir = Path(trace_root or WORKER_TRACE_ROOT) / owner_agent
    trace_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(trace_dir, 0o700)
    except OSError:
        pass
    filename = (
        timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
        + f"-{safe_worker_id}-{uuid.uuid4().hex[:8]}.trace.txt"
    )
    destination = trace_dir / filename
    metadata = {
        "schema_version": 1,
        "record_type": "completed_worker_trace",
        "owner_agent": owner_agent,
        "worker_id": worker_id,
        "kind": kind,
        "completed_at": timestamp.isoformat(),
        "task_summary": " ".join(str(task_summary or "").split()),
    }
    payload = (
        json.dumps(metadata, sort_keys=True, ensure_ascii=False)
        + "\n--- BEGIN FULL WORKER TRACE ---\n"
        + trace_text.rstrip()
        + "\n--- END FULL WORKER TRACE ---\n"
    )
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(trace_dir)
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
    return destination


def _worker_trace_metadata(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline()
        raw = json.loads(first_line)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def locate_completed_worker_trace(
    owner_agent: str,
    *,
    trace_path: str | Path | None = None,
    trace_root: str | Path | None = None,
) -> Path | None:
    """Resolve an explicit trace or the latest substantive completed trace."""
    _validate_owner_agent(owner_agent)
    if trace_path is not None:
        path = Path(trace_path)
        if not path.is_absolute():
            raise SkillCardError("skill_draft.trace_path must be an absolute path")
        try:
            path = path.resolve(strict=True)
        except OSError as exc:
            raise SkillCardError(f"cannot resolve worker trace {path}: {exc}") from exc
        if not path.is_file():
            raise SkillCardError(f"worker trace is not a file: {path}")
        return path

    trace_dir = Path(trace_root or WORKER_TRACE_ROOT) / owner_agent
    if not trace_dir.is_dir():
        return None
    candidates: list[Path] = []
    for path in trace_dir.glob("*.trace.txt"):
        metadata = _worker_trace_metadata(path)
        if metadata.get("record_type") != "completed_worker_trace":
            continue
        if metadata.get("owner_agent") != owner_agent:
            continue
        if metadata.get("kind") == "skill_draft":
            continue
        candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


_TRACE_ERROR_RE = re.compile(
    r"(?:\berror\b|\bfailed\b|\bfailure\b|\bexception\b|\btraceback\b|"
    r"permission denied|exit (?:code )?[1-9][0-9]*|\bwarning\b)",
    re.IGNORECASE,
)


def bounded_worker_trace(
    trace_text: str,
    *,
    max_chars: int = SKILL_DRAFT_TRACE_MAX_CHARS,
) -> tuple[str, bool]:
    """Return a tail-weighted trace view with explicit error evidence.

    The full archive remains fingerprinted on disk.  When the drafting handoff
    needs truncation, detected error/failure lines are preserved in a compact
    evidence section and the remainder of the budget favors the final working
    sequence.  The returned text never exceeds ``max_chars``.
    """
    if max_chars < 512:
        raise SkillCardError("skill-draft trace cap must be at least 512 characters")
    if len(trace_text) <= max_chars:
        return trace_text, False

    marker = (
        f"[TRACE VIEW TRUNCATED: full archive has {len(trace_text)} characters; "
        f"handoff cap is {max_chars}. Error evidence and the final working "
        "sequence are prioritized.]\n"
    )
    error_lines: list[str] = []
    for line_number, line in enumerate(trace_text.splitlines(), 1):
        if _TRACE_ERROR_RE.search(line):
            compact = line.strip()
            if len(compact) > 600:
                compact = compact[:600] + " ... [line truncated]"
            error_lines.append(f"L{line_number}: {compact}")

    error_budget = max_chars // 3
    error_header = "\n[DETECTED ERROR / FAILED-ATTEMPT EVIDENCE]\n"
    error_text = "\n".join(error_lines)
    if len(error_text) > error_budget:
        error_text = (
            error_text[: max(0, error_budget - 80)]
            + "\n[additional detected error evidence omitted by the handoff cap]"
        )
    evidence = error_header + (error_text or "(no explicit error markers detected)")
    tail_header = "\n\n[FINAL WORKING-SEQUENCE TAIL]\n"
    tail_budget = max_chars - len(marker) - len(evidence) - len(tail_header)
    if tail_budget < 0:
        evidence = evidence[: max(0, max_chars - len(marker) - len(tail_header))]
        tail_budget = max_chars - len(marker) - len(evidence) - len(tail_header)
    tail = trace_text[-tail_budget:] if tail_budget > 0 else ""
    rendered = marker + evidence + tail_header + tail
    return rendered[:max_chars], True


def _markdown_section(text: str, heading: str, next_heading: str | None) -> str:
    """Return one top-level Markdown section, including its heading."""
    start = text.find(heading)
    if start < 0:
        raise SkillCardError(f"missing specification section {heading!r}")
    if next_heading is None:
        return text[start:].strip()
    end = text.find(next_heading, start + len(heading))
    if end < 0:
        raise SkillCardError(
            f"missing specification boundary {next_heading!r} after {heading!r}"
        )
    return text[start:end].strip()


def build_skill_draft_package(
    owner_agent: str,
    task_summary: str,
    *,
    recent_context: str,
    source_files: list[str | Path] | tuple[str | Path, ...] = (),
    trace_path: str | Path | None = None,
    trace_root: str | Path | None = None,
    trace_max_chars: int = SKILL_DRAFT_TRACE_MAX_CHARS,
    staging_root: str | Path | None = None,
    run_id: str | None = None,
    spec_path: str | Path | None = None,
    example_path: str | Path | None = None,
    isolation_policy=None,
) -> SkillDraftPackage:
    """Package authoritative context and a fail-closed persistence contract.

    The returned task is run by the ordinary worker machinery.  The worker may
    draft freely in the isolated staging directory, but the only supported
    promotion path is ``scripts/write_skill_proposal.py``, which validates the
    complete card before atomically writing under ``.proposals``.
    """
    _validate_owner_agent(owner_agent)
    summary = " ".join(str(task_summary or "").split())
    if not summary:
        raise SkillCardError("skill_draft.task_summary is required")

    repo_root = Path(__file__).resolve().parents[1]
    resolved_spec = Path(spec_path) if spec_path else (
        repo_root / "docs/governed-procedural-memory.md"
    )
    resolved_example = Path(example_path) if example_path else (
        repo_root
        / "docs/examples/skill_cards/example/model-service-recovery.yaml"
    )
    try:
        spec_text = resolved_spec.read_text(encoding="utf-8")
        example_text = resolved_example.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SkillCardError(f"cannot package skill-draft reference: {exc}") from exc

    spec_excerpt = "\n\n".join((
        _markdown_section(spec_text, "## 3. Card Schema", "## 4. Directory Layout"),
        _markdown_section(spec_text, "## 5. Lifecycle", "## 6."),
        _markdown_section(spec_text, "## 10. Governance and Safety", "## 11."),
    ))

    # Phase 2B: skill_draft reads whatever files the model names and embeds
    # their full contents in a worker prompt, so an unvalidated source list is
    # an exfiltration primitive. ``None`` (unisolated) skips every check.
    policy = (
        isolation_policy
        if (isolation_policy is not None and getattr(isolation_policy, "enabled", False))
        else None
    )

    def _require_in_boundary(candidate: Path, label: str) -> None:
        if policy is None:
            return
        if not policy.contains(candidate):
            roots = ", ".join(str(p) for p in policy.workspaces)
            raise SkillCardError(
                f"{label} {candidate} is outside this agent's isolation "
                f"boundary. Allowed roots: {roots}"
            )
        if policy.is_protected_state(candidate):
            raise SkillCardError(
                f"{label} {candidate} is protected agent state and may not be "
                f"packaged into a skill draft"
            )

    resolved_sources: list[Path] = []
    source_blocks: list[str] = []
    for raw_path in source_files:
        path = Path(raw_path)
        if not path.is_absolute():
            raise SkillCardError(f"source_files must use absolute paths: {raw_path}")
        try:
            path = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise SkillCardError(f"cannot read skill source {path}: {exc}") from exc
        # Check after resolution so a symlink inside the workspace pointing
        # out of it is caught, and before reading so a refused file is never
        # opened.
        _require_in_boundary(path, "skill source")
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise SkillCardError(f"cannot read skill source {path}: {exc}") from exc
        if not path.is_file():
            raise SkillCardError(f"skill source is not a file: {path}")
        resolved_sources.append(path)
        source_blocks.append(
            f"<source_file path={str(path)!r} fingerprint={sha256_file(path)!r}>\n"
            f"{content}\n</source_file>"
        )

    resolved_trace = locate_completed_worker_trace(
        owner_agent,
        trace_path=trace_path,
        trace_root=trace_root,
    )
    trace_fingerprint: str | None = None
    trace_truncated = False
    if resolved_trace is None:
        trace_context = (
            "No completed worker trace was available. This may be a "
            "router-executed procedure. Use the recent conversation and "
            "authoritative sources, and record the missing trace explicitly "
            "in caveats.trace_assessment and caveats.unverified_steps."
        )
    else:
        try:
            resolved_trace = resolved_trace.resolve()
        except (OSError, RuntimeError) as exc:
            raise SkillCardError(
                f"cannot read completed worker trace {resolved_trace}: {exc}"
            ) from exc
        # A caller-supplied trace_path is as arbitrary as a source file.
        # The agent's own trace root lives inside state_root, which is
        # protected state, so check containment only — not protection.
        if policy is not None and not policy.contains(resolved_trace):
            roots = ", ".join(str(p) for p in policy.workspaces)
            raise SkillCardError(
                f"worker trace {resolved_trace} is outside this agent's "
                f"isolation boundary. Allowed roots: {roots}"
            )
        try:
            full_trace = resolved_trace.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise SkillCardError(
                f"cannot read completed worker trace {resolved_trace}: {exc}"
            ) from exc
        trace_fingerprint = sha256_file(resolved_trace)
        trace_view, trace_truncated = bounded_worker_trace(
            full_trace,
            max_chars=trace_max_chars,
        )
        trace_context = (
            f"<completed_worker_trace_source path={str(resolved_trace)!r} "
            f"fingerprint={trace_fingerprint!r} "
            f"full_chars={len(full_trace)!r} "
            f"included_chars={len(trace_view)!r} "
            f"truncated={str(trace_truncated).lower()!r}>\n"
            f"{trace_view}\n"
            "</completed_worker_trace_source>"
        )

    generated_run_id = run_id or (
        f"skill-draft-{_utc_now().strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    if not re.fullmatch(r"[A-Za-z0-9._-]+", generated_run_id):
        raise SkillCardError(f"invalid skill-draft run id {generated_run_id!r}")
    root = (
        Path(staging_root)
        if staging_root is not None
        else SKILLS_DIR.parent / "skill_drafts"
    )
    staging_dir = root / owner_agent / generated_run_id
    try:
        staging_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise SkillCardError(
            f"cannot create skill-draft staging directory {staging_dir}: {exc}"
        ) from exc
    staging_path = staging_dir / "card.yaml"

    source_args = "".join(
        f" --source-file {shlex.quote(str(path))}" for path in resolved_sources
    )
    source_context = "\n\n".join(source_blocks) or (
        "No source_files were supplied. Identify current authoritative local "
        "runbooks or implementation files from the episode, include their exact "
        "absolute paths in procedure_source, and pass each one to the validator."
    )
    task = f"""\
Draft one governed procedural-memory card proposal for agent `{owner_agent}`.

USER-REQUESTED SCOPE
{summary}

This is an on-demand formation task after a procedure that is expected to recur.
Recover exact flags, paths, values, failed-attempt lessons, verification probes,
and authority boundaries from the recent episode. Do not generalize beyond the
evidence. The card is advisory procedural memory, not executable authority.

MANDATORY LIFECYCLE
- schema_version: 1
- id: stable lowercase kebab-case
- version: 1
- status: proposed
- owner_agent: {owner_agent}
- approved_by: null
- approved_at: null
- supersedes_version: null
- last_reviewed_at: null
- outcomes: []
- proposed_by.mechanism: user_requested_tool
- proposed_by.agent: {owner_agent}
- proposed_by.fold_round: null
- proposed_by.run_id: {generated_run_id}
- proposed_by.proposed_at: current ISO-8601 timestamp

AUTHORING REQUIREMENTS
1. Author realistic triggers and typed preconditions that distinguish similar
   tasks. Mark genuinely load-bearing facets required.
2. State exact invariants, observable verification probes, rollback, and bounded
   authority. Current user authority is not expanded by the card.
3. Cite canonical evidence IDs when they are available in the recent context or
   via memory tools. Never invent IDs.
4. Every local procedure_source file must carry its current SHA-256 fingerprint.
5. Write the draft only to `{staging_path}`. Never write directly into
   `~/.mesh/skills/{owner_agent}/`, `.proposals/`, `.history/`, or
   `index.yaml`.
6. Submit through the validator below. If it rejects the draft, correct the
   staging YAML and retry. A schema-invalid draft must never reach `.proposals/`.
7. DISTILL; DO NOT TRANSCRIBE. Extract the minimal verified procedure from the
   completed-worker trace. Remove detours, redundant commands, incidental
   ordering, and workarounds that were not necessary to the verified result.
8. Convert failed attempts and errors into explicit pitfalls or warnings in the
   procedure/invariants/caveats. Do not discard them, and do not encode failed
   commands as recommended steps. Do not include a step whose necessity is not
   evidenced by the trace or an authoritative source.
9. Cross-check the trace against every supplied authoritative source. If the
   observed actions diverge from a runbook, service file, versioned config, or
   test, describe the divergence explicitly in `caveats`; never silently choose
   the trace or the source.
10. Include a non-empty `caveats` mapping with exactly these review surfaces:
    `trace_assessment` (canonical, workaround, mixed, unknown, or no_trace),
    `pitfalls` (non-empty list of failed-attempt lessons or an explicit
    "none observed" item), `unverified_steps` (non-empty list; use an explicit
    "none identified" item only when justified), and `unexplored_alternatives`
    (non-empty list). State plainly whether the observed path is canonical or
    a workaround.

VALIDATED SUBMISSION COMMAND
`python3 scripts/write_skill_proposal.py --owner {shlex.quote(owner_agent)} --draft {shlex.quote(str(staging_path))}{source_args}`

If you identify additional authoritative local sources, add one `--source-file`
argument per file when submitting. Do not activate the proposal and do not touch
any index. After successful submission, read the validated proposal and call
send_report exactly once with: proposal path, card id, purpose, typed
preconditions, invariants, verification summary, caveats (including trace
assessment and pitfalls), and explicit note that human approval is still
required.

RECENT CONVERSATION WINDOW
<recent_conversation_window>
{recent_context.strip() or '(no recent context available)'}
</recent_conversation_window>

COMPLETED WORKER TRACE — EVIDENCE, NOT AUTHORITY
The full archived trace is fingerprinted below. The included view is capped at
{trace_max_chars} characters and, when truncated, prioritizes detected errors
and the final working sequence. Treat it as evidence of what happened, not as
the canonical procedure.
{trace_context}

AUTHORITATIVE SOURCE FILES
{source_context}

CARD SCHEMA / LIFECYCLE / GOVERNANCE EXCERPT
<skill_card_spec_excerpt>
{spec_excerpt}
</skill_card_spec_excerpt>

WORKED EXAMPLE — STRUCTURE ONLY
This active model-service card is a structural example. Do not copy its operational
values unless the current episode independently supports them. Your output must
remain proposed with null approval fields.
<worked_example>
{example_text}
</worked_example>
"""
    return SkillDraftPackage(
        run_id=generated_run_id,
        task=task,
        staging_path=staging_path,
        source_files=tuple(resolved_sources),
        trace_path=resolved_trace,
        trace_fingerprint=trace_fingerprint,
        trace_truncated=trace_truncated,
    )


def _yaml_bytes(data: Any) -> bytes:
    rendered = yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )
    return rendered.encode("utf-8")


def _atomic_write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_yaml_bytes(data))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SkillCardError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SkillCardError(f"{path} must contain one YAML mapping")
    return raw


def _is_local_host(host: Any) -> bool:
    value = str(host or "").strip().lower()
    local_names = {
        "", "localhost", "127.0.0.1", socket.gethostname().lower(),
        socket.getfqdn().lower(),
    }
    return value in local_names


def validate_card(card: dict[str, Any], expected_owner: str | None = None) -> None:
    """Validate the v1 card schema and lifecycle invariants."""
    missing = sorted(_REQUIRED_FIELDS - set(card))
    if missing:
        raise SkillCardError(f"missing required fields: {', '.join(missing)}")
    if card.get("schema_version") != SCHEMA_VERSION:
        raise SkillCardError(
            f"unsupported schema_version {card.get('schema_version')!r}"
        )
    card_id = card.get("id")
    if not isinstance(card_id, str) or not _CARD_ID_RE.fullmatch(card_id):
        raise SkillCardError("id must be stable lowercase kebab-case")
    if not isinstance(card.get("version"), int) or card["version"] < 1:
        raise SkillCardError("version must be a positive integer")
    if card.get("status") not in _VALID_STATUSES:
        raise SkillCardError("status must be proposed, active, or retired")
    owner = card.get("owner_agent")
    if not isinstance(owner, str) or not owner.strip():
        raise SkillCardError("owner_agent must be non-empty")
    if expected_owner and owner != expected_owner:
        raise SkillCardError(
            f"owner_agent {owner!r} does not match directory {expected_owner!r}"
        )
    if not isinstance(card.get("purpose"), str) or not card["purpose"].strip():
        raise SkillCardError("purpose must be non-empty")
    if not isinstance(card.get("triggers"), list) or not card["triggers"]:
        raise SkillCardError("triggers must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in card["triggers"]):
        raise SkillCardError("every trigger must be non-empty text")
    if not isinstance(card.get("preconditions"), list):
        raise SkillCardError("preconditions must be a list")
    for condition in card["preconditions"]:
        if not isinstance(condition, dict) or not condition.get("key"):
            raise SkillCardError("each precondition must have a key")
        if condition.get("operator") not in _VALID_PRECONDITION_OPERATORS:
            raise SkillCardError(
                f"invalid precondition operator {condition.get('operator')!r}"
            )
        if not isinstance(condition.get("required", False), bool):
            raise SkillCardError("precondition.required must be boolean")
    for key in (
        "authority", "rollback", "proposed_by",
    ):
        if not isinstance(card.get(key), dict):
            raise SkillCardError(f"{key} must be a mapping")
    caveats = card.get("caveats")
    if caveats is not None:
        if not isinstance(caveats, dict):
            raise SkillCardError("caveats must be a mapping")
        assessment = caveats.get("trace_assessment")
        if assessment not in {
            "canonical", "workaround", "mixed", "unknown", "no_trace",
        }:
            raise SkillCardError(
                "caveats.trace_assessment must be canonical, workaround, "
                "mixed, unknown, or no_trace"
            )
        for field_name in (
            "pitfalls", "unverified_steps", "unexplored_alternatives",
        ):
            values = caveats.get(field_name)
            if not isinstance(values, list) or not values:
                raise SkillCardError(f"caveats.{field_name} must be a non-empty list")
            if not all(isinstance(item, str) and item.strip() for item in values):
                raise SkillCardError(
                    f"every caveats.{field_name} item must be non-empty text"
                )
    for key in (
        "procedure_source", "required_invariants", "verification",
        "evidence", "outcomes",
    ):
        if not isinstance(card.get(key), list):
            raise SkillCardError(f"{key} must be a list")
    if not card["procedure_source"]:
        raise SkillCardError("procedure_source must name authoritative artifacts")
    for source in card["procedure_source"]:
        if not isinstance(source, dict):
            raise SkillCardError("procedure_source entries must be mappings")
        if not source.get("kind") or not source.get("path"):
            raise SkillCardError("procedure_source entries need kind and path")
    if not card["required_invariants"] or not card["verification"]:
        raise SkillCardError("required_invariants and verification cannot be empty")
    for memory_id in card["evidence"]:
        if not isinstance(memory_id, str) or not _MEMORY_ID_RE.fullmatch(memory_id):
            raise SkillCardError(f"invalid canonical memory ID {memory_id!r}")
    if card["status"] == "active":
        if card.get("approved_by") != "user:approver" or not card.get("approved_at"):
            raise SkillCardError(
                "active cards require v1 approval by user:approver and an approval timestamp"
            )
        for source in card["procedure_source"]:
            fingerprint = source.get("approved_fingerprint")
            if not isinstance(fingerprint, str) or not fingerprint.startswith("sha256:"):
                raise SkillCardError(
                    "active cards require approved fingerprints for every source"
                )


def extract_task_facets(task: str) -> dict[str, set[str]]:
    """Extract deterministic typed facets from explicit task syntax."""
    facets: dict[str, set[str]] = {}
    for match in _EXPLICIT_FACET_RE.finditer(task):
        key = match.group(1).lower()
        value = _normalize_phrase(match.group(2))
        if value:
            facets.setdefault(key, set()).add(value)
    for match in _ON_HOST_RE.finditer(task):
        value = _normalize_phrase(match.group(1))
        if value and value not in {"port", "the", "this", "that", "all"}:
            facets.setdefault("host", set()).add(value)
    return facets


def _condition_terms(condition: dict[str, Any]) -> set[str]:
    values: list[Any] = []
    value = condition.get("value")
    if isinstance(value, list):
        values.extend(value)
    elif value is not None:
        values.append(value)
    aliases = condition.get("aliases") or []
    if isinstance(aliases, list):
        values.extend(aliases)
    return {normalized for item in values if (normalized := _normalize_phrase(item))}


def _condition_state(
    condition: dict[str, Any],
    task: str,
    task_facets: dict[str, set[str]],
) -> str:
    key = str(condition.get("key") or "").lower()
    operator = condition.get("operator")
    terms = _condition_terms(condition)
    normalized_task = _normalize_phrase(task)
    explicit = task_facets.get(key, set())

    # Explicit typed facets are authoritative.  Evaluate them before loose
    # alias matching so text such as ``service=model-tunnel.service``
    # cannot satisfy a production-vLLM card merely because it contains
    # "qwen".  A required typed mismatch is a hard contradiction.
    if explicit:
        if operator == "absent":
            return "contradicted"
        if operator == "present":
            return "matched"
        if any(value in terms for value in explicit):
            return "matched"
        return "contradicted"

    if operator == "absent":
        return "contradicted" if any(term in normalized_task for term in terms) else "matched"
    if operator == "present":
        return "matched" if any(term in normalized_task for term in terms) else "unknown"

    if any(term and term in normalized_task for term in terms):
        return "matched"
    return "unknown"


def _bm25_scores(query: str, documents: list[str]) -> list[float]:
    if not documents:
        return []
    query_terms = _tokens(query)
    if not query_terms:
        return [0.0] * len(documents)
    tokenized = [_tokens(document) for document in documents]
    average_length = sum(len(doc) for doc in tokenized) / max(1, len(tokenized))
    document_frequency: Counter[str] = Counter()
    for document in tokenized:
        document_frequency.update(set(document))
    scores: list[float] = []
    k1 = 1.5
    b = 0.75
    n_docs = len(tokenized)
    for document in tokenized:
        frequencies = Counter(document)
        score = 0.0
        for term in set(query_terms):
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (
                1 - b + b * len(document) / max(1.0, average_length)
            )
            score += inverse_frequency * (frequency * (k1 + 1)) / denominator
        scores.append(score)
    return scores


def detect_formation_signal(text: str) -> FormationSignal:
    """Detect a first-success procedural-memory candidate for the fold.

    This is intentionally a candidate detector, not an autonomous card author.
    The fold can log or report the result; a human still decides whether a
    proposal should be written or activated.
    """
    lower = text.lower()
    success_signal = bool(re.search(
        r"\b(completed|succeeded|successful|verified|passed|restored|recovered|healthy|working)\b",
        lower,
    ))
    likely_recurring = bool(re.search(
        r"\b(restart|relaunch|recover|restore|deploy|deployment|migrate|migration|"
        r"reconfigure|configuration|provision|rotate|fold|service)\b",
        lower,
    ))
    details = set(re.findall(r"--[a-z0-9][a-z0-9-]*", lower))
    details.update(re.findall(r"(?:/[a-zA-Z0-9._~-]+){2,}", text))
    details.update(re.findall(r"\b[A-Z][A-Z0-9_]{2,}=", text))
    details.update(re.findall(
        r"\b(?:host|service|port|model|backend|context|tp|path)\s*[=:]\s*[^\s,;]+",
        text,
        re.IGNORECASE,
    ))
    detail_count = len(details)
    eligible = success_signal and likely_recurring and detail_count >= 2
    if eligible:
        rationale = (
            "verified success for a task recurring by nature with multiple "
            "flags, paths, or typed variables"
        )
    else:
        missing = []
        if not success_signal:
            missing.append("verified-success signal")
        if not likely_recurring:
            missing.append("structural recurrence signal")
        if detail_count < 2:
            missing.append("multiple fragile details")
        rationale = "missing " + ", ".join(missing)
    return FormationSignal(
        eligible=eligible,
        success_signal=success_signal,
        likely_recurring=likely_recurring,
        detail_count=detail_count,
        details=tuple(sorted(details)),
        rationale=rationale,
    )


class SkillStore:
    """Per-agent governed skill-card store."""

    def __init__(self, owner_agent: str, root: str | Path | None = None):
        if not owner_agent or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", owner_agent):
            raise SkillCardError(f"invalid owner agent {owner_agent!r}")
        self.owner_agent = owner_agent
        self.root = Path(root) if root is not None else SKILLS_DIR
        self.agent_dir = self.root / owner_agent
        self.index_path = self.agent_dir / "index.yaml"

    def ensure_layout(self) -> None:
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        (self.agent_dir / ".proposals").mkdir(exist_ok=True)
        (self.agent_dir / ".history").mkdir(exist_ok=True)

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        self.ensure_layout()
        lock_path = self.agent_dir / ".lock"
        with _PROCESS_LOCK, lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _card_paths(self) -> list[Path]:
        if not self.agent_dir.exists():
            return []
        return sorted(
            path for path in self.agent_dir.glob("*.yaml")
            if path.name != "index.yaml" and not path.name.startswith(".")
        )

    def load_card(self, card_id: str) -> dict[str, Any]:
        if not _CARD_ID_RE.fullmatch(card_id):
            raise SkillCardError(f"invalid card id {card_id!r}")
        path = self.agent_dir / f"{card_id}.yaml"
        card = _load_yaml(path)
        validate_card(card, self.owner_agent)
        if card["id"] != card_id:
            raise SkillCardError(f"filename/card id mismatch for {path}")
        return card

    def load_cards(self) -> list[dict[str, Any]]:
        cards = []
        for path in self._card_paths():
            card = _load_yaml(path)
            validate_card(card, self.owner_agent)
            if path.name != f"{card['id']}.yaml":
                raise SkillCardError(f"filename/card id mismatch for {path}")
            cards.append(card)
        return cards

    def write_proposal(
        self,
        card: dict[str, Any],
        *,
        source_files: list[str | Path] | tuple[str | Path, ...] = (),
    ) -> Path:
        """Validate and atomically stage a new v1 card proposal.

        This method intentionally has no activation behavior: it writes only
        beneath ``.proposals`` and never rebuilds or mutates ``index.yaml``.
        """
        candidate = copy.deepcopy(card)
        if candidate.get("version") != 1:
            raise SkillCardError("on-demand new-card proposals must use version 1")
        if candidate.get("status") != "proposed":
            raise SkillCardError("skill proposals must have status proposed")
        if candidate.get("approved_by") is not None or candidate.get("approved_at") is not None:
            raise SkillCardError("skill proposals must have null approval fields")
        if candidate.get("supersedes_version") is not None:
            raise SkillCardError("new-card proposals must not supersede a version")
        if candidate.get("outcomes") != []:
            raise SkillCardError("new-card proposals must start with no outcomes")
        proposed_by = candidate.get("proposed_by")
        if not isinstance(proposed_by, dict) or proposed_by.get("mechanism") != "user_requested_tool":
            raise SkillCardError(
                "on-demand proposals require proposed_by.mechanism=user_requested_tool"
            )
        if proposed_by.get("agent") != self.owner_agent:
            raise SkillCardError("proposed_by.agent must match the skill owner")
        if not str(proposed_by.get("run_id") or "").strip():
            raise SkillCardError("proposed_by.run_id is required")
        if not str(proposed_by.get("proposed_at") or "").strip():
            raise SkillCardError("proposed_by.proposed_at is required")
        if not isinstance(candidate.get("caveats"), dict):
            raise SkillCardError(
                "skill_draft proposals require a non-empty caveats mapping"
            )

        validate_card(candidate, self.owner_agent)
        expected_sources: dict[Path, str] = {}
        for raw_path in source_files:
            path = Path(raw_path)
            if not path.is_absolute():
                raise SkillCardError(f"source_files must use absolute paths: {raw_path}")
            try:
                path = path.resolve(strict=True)
            except OSError as exc:
                raise SkillCardError(f"cannot resolve source file {path}: {exc}") from exc
            if not path.is_file():
                raise SkillCardError(f"skill source is not a file: {path}")
            expected_sources[path] = sha256_file(path)

        observed_sources: dict[Path, str] = {}
        for source in candidate["procedure_source"]:
            fingerprint = source.get("approved_fingerprint")
            if not isinstance(fingerprint, str) or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", fingerprint
            ):
                raise SkillCardError(
                    "on-demand procedure sources require exact SHA-256 fingerprints"
                )
            if source.get("kind") not in {"file", "runbook", "test"}:
                continue
            if not _is_local_host(source.get("host")):
                continue
            path = Path(str(source.get("path") or ""))
            if not path.is_absolute():
                raise SkillCardError(
                    f"local procedure_source paths must be absolute: {path}"
                )
            try:
                path = path.resolve(strict=True)
            except OSError as exc:
                raise SkillCardError(f"cannot resolve procedure source {path}: {exc}") from exc
            if not path.is_file():
                raise SkillCardError(f"procedure source is not a file: {path}")
            actual = sha256_file(path)
            if fingerprint != actual:
                raise SkillCardError(
                    f"procedure source fingerprint mismatch for {path}: expected {actual}"
                )
            observed_sources[path] = actual

        missing_sources = sorted(str(path) for path in expected_sources if path not in observed_sources)
        if missing_sources:
            raise SkillCardError(
                "proposal omits supplied authoritative source(s): "
                + ", ".join(missing_sources)
            )

        card_id = str(candidate["id"])
        with self._write_lock():
            active_path = self.agent_dir / f"{card_id}.yaml"
            proposal_path = self.agent_dir / ".proposals" / f"{card_id}.yaml"
            if active_path.exists():
                raise SkillCardError(
                    f"card {card_id!r} already exists; revisions require the separate human workflow"
                )
            if proposal_path.exists():
                raise SkillCardError(f"proposal already exists: {proposal_path}")
            _atomic_write_yaml(proposal_path, candidate)
            persisted = _load_yaml(proposal_path)
            validate_card(persisted, self.owner_agent)
            if persisted != candidate:
                proposal_path.unlink(missing_ok=True)
                raise SkillCardError("proposal changed during atomic persistence")
            return proposal_path

    def write_proposal_file(
        self,
        draft_path: str | Path,
        *,
        source_files: list[str | Path] | tuple[str | Path, ...] = (),
    ) -> Path:
        """Load a staged YAML draft and promote it through ``write_proposal``."""
        return self.write_proposal(
            _load_yaml(Path(draft_path)),
            source_files=source_files,
        )

    def _source_issues(self, card: dict[str, Any]) -> list[str]:
        issues: list[str] = []
        for source in card.get("procedure_source", []):
            fingerprint = source.get("approved_fingerprint")
            label = f"{source.get('host', 'local')}:{source.get('path', '')}"
            if not isinstance(fingerprint, str) or not fingerprint.startswith("sha256:"):
                issues.append(f"missing approved fingerprint for {label}")
                continue
            if not _is_local_host(source.get("host")):
                # Remote sources are verified by the fold/meta-review against
                # the target host. Retrieval remains network-free and checks
                # the last human-approved fingerprint carried by the card.
                continue
            if source.get("kind") not in {"file", "runbook", "test"}:
                continue
            source_path = Path(str(source.get("path")))
            if not source_path.is_file():
                issues.append(f"missing local source {source_path}")
                continue
            actual = sha256_file(source_path)
            if actual != fingerprint:
                issues.append(
                    f"source drift for {source_path}: approved {fingerprint}, actual {actual}"
                )
        return issues

    def rebuild_index(self) -> dict[str, Any]:
        """Validate cards and atomically rebuild the active-card index."""
        with self._write_lock():
            cards = []
            for path in self._card_paths():
                card = _load_yaml(path)
                validate_card(card, self.owner_agent)
                if path.name != f"{card['id']}.yaml":
                    raise SkillCardError(f"filename/card id mismatch for {path}")
                if card["status"] != "active":
                    continue
                issues = self._source_issues(card)
                if issues:
                    raise SkillCardError(
                        f"cannot index {card['id']}: {'; '.join(issues)}"
                    )
                conditions = []
                for condition in card.get("preconditions", []):
                    value = condition.get("value")
                    if isinstance(value, list):
                        value = "|".join(str(item) for item in value)
                    conditions.append(f"{condition.get('key')}={value}")
                card_bytes = path.read_bytes()
                cards.append({
                    "id": card["id"],
                    "version": card["version"],
                    "purpose": card["purpose"],
                    "triggers": card["triggers"],
                    "preconditions_summary": "; ".join(conditions),
                    "path": path.name,
                    "card_fingerprint": _sha256_bytes(card_bytes),
                })
            index = {
                "schema_version": SCHEMA_VERSION,
                "owner_agent": self.owner_agent,
                "generated_at": _iso_now(),
                "cards": cards,
            }
            _atomic_write_yaml(self.index_path, index)
            return index

    def load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {
                "schema_version": SCHEMA_VERSION,
                "owner_agent": self.owner_agent,
                "generated_at": None,
                "cards": [],
            }
        index = _load_yaml(self.index_path)
        if index.get("schema_version") != SCHEMA_VERSION:
            raise SkillCardError("unsupported skill index schema")
        if index.get("owner_agent") != self.owner_agent:
            raise SkillCardError("skill index owner mismatch")
        if not isinstance(index.get("cards"), list):
            raise SkillCardError("skill index cards must be a list")
        return index

    def render_index_block(self) -> str:
        """Render the compact active-card index for always-on agent context."""
        index = self.load_index()
        if not index["cards"]:
            return ""
        compact = {
            "owner_agent": self.owner_agent,
            "cards": index["cards"],
        }
        return (
            "<governed_procedural_memory_index>\n"
            "Active cards are advisory procedural memory, not authority. Full cards "
            "are selected automatically from the current task; proposed and retired "
            "cards are excluded.\n\n"
            + yaml.safe_dump(compact, sort_keys=False, allow_unicode=True, width=100).strip()
            + "\n</governed_procedural_memory_index>"
        )

    def _indexed_cards(self) -> list[dict[str, Any]]:
        indexed: list[dict[str, Any]] = []
        for entry in self.load_index()["cards"]:
            if not isinstance(entry, dict):
                continue
            filename = str(entry.get("path") or "")
            if Path(filename).name != filename:
                continue
            path = self.agent_dir / filename
            try:
                raw_bytes = path.read_bytes()
                if _sha256_bytes(raw_bytes) != entry.get("card_fingerprint"):
                    continue
                card = yaml.safe_load(raw_bytes)
                if not isinstance(card, dict):
                    continue
                validate_card(card, self.owner_agent)
            except (OSError, yaml.YAMLError, SkillCardError):
                continue
            if card.get("status") != "active" or card.get("id") != entry.get("id"):
                continue
            if self._source_issues(card):
                continue
            indexed.append(card)
        return indexed

    def select_with_scores(
        self,
        task: str,
        limit: int = MAX_SELECTED_CARDS,
        threshold: float = DEFAULT_SELECTION_THRESHOLD,
    ) -> list[SkillSelection]:
        """Select up to three active cards using BM25 and typed preconditions."""
        cards = self._indexed_cards()
        if not cards:
            return []
        task_facets_raw = extract_task_facets(task)
        frozen_facets = {
            key: tuple(sorted(values)) for key, values in task_facets_raw.items()
        }
        eligible: list[
            tuple[dict[str, Any], list[str], list[str], list[dict[str, Any]]]
        ] = []
        for card in cards:
            matched: list[str] = []
            unknown: list[str] = []
            contradicted = False
            required = [
                condition for condition in card.get("preconditions", [])
                if condition.get("required", False)
            ]
            for condition in required:
                state = _condition_state(condition, task, task_facets_raw)
                key = str(condition.get("key"))
                if state == "matched":
                    matched.append(key)
                elif state == "unknown":
                    unknown.append(key)
                else:
                    contradicted = True
                    break
            if contradicted:
                continue
            eligible.append((card, matched, unknown, required))

        # Text relevance is computed only after typed contradictions have
        # removed ineligible cards.  This preserves the selector's documented
        # evidence order and prevents contradictory cards from influencing BM25
        # document-frequency statistics for eligible cards.
        documents = [
            " ".join([card["purpose"], *card.get("triggers", [])])
            for card, _, _, _ in eligible
        ]
        text_scores = _bm25_scores(task, documents)
        selections: list[SkillSelection] = []
        for (card, matched, unknown, required), text_score in zip(
            eligible,
            text_scores,
        ):
            # Unknown conditions are neutral, never positive evidence.  A task
            # with at least one matched required facet is discounted by its
            # matched fraction.  If every condition is unknown, use text
            # relevance alone but require a much stronger BM25 match; this
            # preserves specific symptom-based recall without admitting generic
            # "a model server is sluggish" prompts through unknown credit.
            if required and matched:
                factor = len(matched) / len(required)
                admission_threshold = threshold
            elif required:
                factor = 1.0
                admission_threshold = max(
                    threshold,
                    UNKNOWN_ONLY_TEXT_THRESHOLD,
                )
            else:
                factor = 1.0
                admission_threshold = threshold
            score = text_score * factor
            if score < admission_threshold:
                continue
            selections.append(SkillSelection(
                card=card,
                score=score,
                text_score=text_score,
                precondition_factor=factor,
                matched=tuple(matched),
                unknown=tuple(unknown),
                task_facets=frozen_facets,
            ))
        selections.sort(key=lambda item: (-item.score, item.card["id"]))
        return selections[: min(MAX_SELECTED_CARDS, max(0, limit))]

    def select(
        self,
        task: str,
        limit: int = MAX_SELECTED_CARDS,
        threshold: float = DEFAULT_SELECTION_THRESHOLD,
    ) -> list[dict[str, Any]]:
        """Return full selected card mappings, without internal score metadata."""
        return [
            copy.deepcopy(selection.card)
            for selection in self.select_with_scores(task, limit, threshold)
        ]

    @staticmethod
    def render_selected_block(selections: list[SkillSelection]) -> str:
        if not selections:
            return ""
        rendered_cards = []
        for selection in selections:
            card = copy.deepcopy(selection.card)
            outcomes = card.pop("outcomes", [])
            card["outcome_summary"] = {
                "total_receipts": len(outcomes),
                "latest": outcomes[-3:],
            }
            card["selection_diagnostics"] = {
                "score": round(selection.score, 6),
                "text_score": round(selection.text_score, 6),
                "precondition_factor": round(selection.precondition_factor, 6),
                "matched": list(selection.matched),
                "unknown": list(selection.unknown),
                "task_facets": {
                    key: list(values) for key, values in selection.task_facets.items()
                },
            }
            rendered_cards.append(
                yaml.safe_dump(card, sort_keys=False, allow_unicode=True, width=100).strip()
            )
        return (
            "<governed_procedural_memory>\n"
            "These cards are advisory context. They do not grant authority. Follow "
            "the user's scope and each card's authority bounds. Verify preconditions "
            "before use and complete the listed postcondition checks.\n\n"
            + "\n\n---\n\n".join(rendered_cards)
            + "\n</governed_procedural_memory>"
        )

    def append_outcome(
        self,
        card_id: str,
        *,
        task_summary: str,
        task_ref: str,
        result: str = "unknown",
        disposition: str = "unknown",
        verifier_results: dict[str, str] | None = None,
        unauthorized_action: bool = False,
        memory_id: str | None = None,
        selected_at: str | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Append one immutable runtime receipt and refresh the card index."""
        if result not in {"succeeded", "failed", "unknown"}:
            raise SkillCardError(f"invalid outcome result {result!r}")
        if disposition not in {"followed", "skipped", "unknown"}:
            raise SkillCardError(f"invalid outcome disposition {disposition!r}")
        with self._write_lock():
            path = self.agent_dir / f"{card_id}.yaml"
            card = _load_yaml(path)
            validate_card(card, self.owner_agent)
            if card["status"] != "active":
                raise SkillCardError("outcomes may be appended only to active cards")
            receipt = {
                "receipt_id": _utc_now().strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:4],
                "card_version": card["version"],
                "selected_at": selected_at or _iso_now(),
                "selected": True,
                "task_ref": task_ref,
                "task_summary": task_summary[:500],
                "disposition": disposition,
                "result": result,
                "verifier_results": verifier_results or {},
                "unauthorized_action": bool(unauthorized_action),
                "memory_id": memory_id,
                "note": note,
            }
            card["outcomes"].append(receipt)
            _atomic_write_yaml(path, card)
        # Rebuild outside the receipt lock helper (it acquires the same
        # process/file lock itself and refreshes the changed card fingerprint).
        self.rebuild_index()
        return receipt

    def scan_meta_review(self, now: datetime | None = None) -> list[SkillReviewFinding]:
        """Return the two-tier report-only findings required by spec §9."""
        now = now or _utc_now()
        findings: list[SkillReviewFinding] = []
        index = self.load_index()
        indexed_ids = {
            entry.get("id") for entry in index.get("cards", []) if isinstance(entry, dict)
        }
        valid_cards: list[dict[str, Any]] = []
        for path in self._card_paths():
            card_id = path.stem
            try:
                card = _load_yaml(path)
                validate_card(card, self.owner_agent)
                valid_cards.append(card)
            except SkillCardError as exc:
                findings.append(SkillReviewFinding(
                    1, card_id, "failed schema validation", str(exc)
                ))
                continue
            status = card["status"]
            if status == "active":
                issues = self._source_issues(card)
                for issue in issues:
                    findings.append(SkillReviewFinding(
                        1, card_id, "source drift or inaccessible source", issue
                    ))
                if card_id not in indexed_ids:
                    findings.append(SkillReviewFinding(
                        1, card_id, "active card missing from index"
                    ))
                outcomes = card.get("outcomes", [])
                if len(outcomes) >= 2 and all(
                    outcome.get("result") == "failed" for outcome in outcomes[-2:]
                ):
                    findings.append(SkillReviewFinding(
                        1, card_id, "at least two consecutive verified failures"
                    ))
                timestamps = [outcome.get("selected_at") for outcome in outcomes]
                latest = None
                for timestamp in reversed(timestamps):
                    if not timestamp:
                        continue
                    try:
                        latest = datetime.fromisoformat(timestamp)
                    except ValueError:
                        continue
                    break
                if latest and latest.tzinfo is None:
                    latest = latest.replace(tzinfo=timezone.utc)
                if latest and now - latest > timedelta(days=30):
                    findings.append(SkillReviewFinding(
                        2, card_id, "card not selected in more than 30 days"
                    ))
                if any(outcome.get("disposition") == "skipped" for outcome in outcomes[-3:]):
                    findings.append(SkillReviewFinding(
                        2, card_id, "recent skip requires human review"
                    ))
            elif card_id in indexed_ids:
                findings.append(SkillReviewFinding(
                    1, card_id, "non-active card present in index"
                ))
            if status == "proposed":
                findings.append(SkillReviewFinding(
                    2, card_id, "new proposal awaiting human approval"
                ))
            if status == "retired" and card.get("outcomes"):
                findings.append(SkillReviewFinding(
                    2, card_id, "retired card has outcome history; audit retrieval logs"
                ))

        active = [card for card in valid_cards if card.get("status") == "active"]
        for position, card in enumerate(active):
            left = set(_tokens(" ".join(card.get("triggers", []))))
            for other in active[position + 1:]:
                right = set(_tokens(" ".join(other.get("triggers", []))))
                union = left | right
                overlap = len(left & right) / len(union) if union else 0.0
                if overlap >= 0.75:
                    findings.append(SkillReviewFinding(
                        2,
                        card["id"],
                        f"possible duplicate of {other['id']}",
                        f"trigger-token Jaccard={overlap:.2f}",
                    ))
        return findings

    @staticmethod
    def format_meta_review(findings: list[SkillReviewFinding]) -> str:
        if not findings:
            return "## Skill cards\n\nNo findings."
        sections = ["## Skill cards"]
        for tier, title in ((1, "Requires attention"), (2, "For review")):
            group = [finding for finding in findings if finding.tier == tier]
            if not group:
                continue
            sections.append(f"### Tier {tier} — {title} ({len(group)})")
            for finding in sorted(group, key=lambda item: (item.card_id, item.reason)):
                line = f"- **{finding.card_id}**: {finding.reason}"
                if finding.detail:
                    line += f" — {finding.detail}"
                sections.append(line)
        return "\n\n".join(sections)
