"""Project dossier, session report, and worker-budget primitives.

Implements the durable-state layer of ``docs/plans/autonomous-agent-mode.md``:

* **Project dossier** — one markdown file per project entity, holding the
  seven constitutional sections (Identity, Goals, Tasks, Timeline, Narrative,
  Standing decisions, Open threads).  The autonomous controller is its sole
  routine writer.  Edits are exact string replacements (the standing-digest
  mechanism), validated against the constitution before they land.
* **Session report** — one immutable markdown file per autonomous session.
  Create-once: identical bytes are idempotent, different bytes are refused.
  Writing a report also appends a link entry to the dossier Timeline.
* **Budget ledger** — a per-project JSON file counting admitted worker
  attempts against a daily limit.

Path derivation (never model-supplied):

    entity key      project:mesh-infra
    safe key        project-mesh-infra          (":" → "-")
    dossier         ~/.mesh/digests/project-mesh-infra.md
    report          ~/.mesh/reports/project-mesh-infra-{date}-{seq}.md
    budget          ~/.mesh/budget/project-mesh-infra.json

The ``project-`` filename prefix in the plan (§2.2) *is* the ``project:``
entity-key prefix with its colon normalized, so ``{safe_key}`` and
``project-{slug}`` denote the same name.

All roots resolve through :mod:`mesh.paths`, which reads ``/etc/passwd``
rather than ``$HOME`` — CC processes run with a synthetic home.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date, datetime, timedelta
import json
import os
from pathlib import Path
import re
import tempfile

from .paths import BUDGET_DIR, DIGESTS_DIR, REPORTS_DIR

__all__ = [
    "DEFAULT_DOSSIER_TOKEN_BUDGET",
    "DEFAULT_WORKERS_PER_DAY",
    "PROJECT_KEY_RE",
    "REQUIRED_SECTIONS",
    "DossierError",
    "BudgetExhausted",
    "safe_entity_key",
    "dossier_path",
    "report_path",
    "budget_path",
    "render_skeleton",
    "section_errors",
    "_validate_project_model_block",
    "validate_dossier",
    "init_dossier",
    "read_dossier",
    "edit_dossier",
    "write_report",
    "check_budget",
    "spend_budget",
    "reset_budget",
]


# The seven level-two sections, in required order (§3.2).
REQUIRED_SECTIONS: tuple[str, ...] = (
    "Identity",
    "Goals",
    "Tasks",
    "Timeline",
    "Narrative",
    "Standing decisions",
    "Open threads",
)

TASKS_BEGIN = "<!-- tasks:begin schema=1 -->"
TASKS_END = "<!-- tasks:end -->"

# §3.7 — one global hard ceiling, no per-section hard caps.
DEFAULT_DOSSIER_TOKEN_BUDGET = 12000

# §8.6 pilot default.  Overridable per project via the dossier frontmatter key
# ``max_workers_per_day``.
DEFAULT_WORKERS_PER_DAY = 4

#: Active mode is opt-in.  A project only self-schedules its next session once
#: an operator has armed it with ``/auto active <project> on``; a dossier with
#: no ``active`` frontmatter line reads as off.
DEFAULT_ACTIVE = False

_ENTITY_KEY_RE = re.compile(r"^project:[a-z0-9][a-z0-9_-]*$")
# Public alias: the single shape authority for a project entity key.  The
# router's admission guard validates model-supplied keys against this rather
# than re-deriving the grammar.
PROJECT_KEY_RE = _ENTITY_KEY_RE
_SECTION_RE = re.compile(r"^##[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PROJECT_MODEL_HEADING_RE = re.compile(
    r"^###[ \t]+Project Model[ \t]*$", re.MULTILINE
)
_PROJECT_MODEL_ITEM_RE = re.compile(
    r"^-[ \t]+`\[([^\]]*)\]`[ \t]+`([^`]*)`[ \t]+`([^`]*)`"
    r"[ \t]+—[ \t]+(.+?)[ \t]*$"
)
_PROJECT_MODEL_STATES = ("verified", "inferred", "uncertain", "superseded")
_PROJECT_MODEL_TOKEN_BUDGET = 1200


class DossierError(Exception):
    """Raised when a dossier / report / budget operation is refused."""


class BudgetExhausted(DossierError):
    """Raised when a worker admission would exceed the project's limit."""


# ─────────────────────────────────────────────────────────────────────
# Identity and paths
# ─────────────────────────────────────────────────────────────────────


def safe_entity_key(entity_key: str) -> str:
    """Return the filesystem-safe form of a project entity key.

    ``project:mesh-infra`` → ``project-mesh-infra``.  Validates the key shape
    first so a caller can never steer the derived path outside its directory.
    """
    key = (entity_key or "").strip()
    if not _ENTITY_KEY_RE.match(key):
        raise DossierError(
            f"Invalid project entity key {entity_key!r} — expected "
            "'project:<slug>' where slug matches [a-z0-9][a-z0-9_-]*."
        )
    return key.replace(":", "-")


# ─────────────────────────────────────────────────────────────────────
# Path derivation
# ─────────────────────────────────────────────────────────────────────
#
# Every path function takes an optional ``state_paths``.  When it is omitted
# the module-level globals are read *at call time*, which is what keeps the
# legacy path working and keeps the existing tests able to monkeypatch
# ``DIGESTS_DIR``/``REPORTS_DIR``/``BUDGET_DIR`` on this module.


def _digests_root(state_paths=None) -> Path:
    return state_paths.digests_dir if state_paths is not None else DIGESTS_DIR


def _reports_root(state_paths=None) -> Path:
    return state_paths.reports_dir if state_paths is not None else REPORTS_DIR


def _budget_root(state_paths=None) -> Path:
    return state_paths.budget_dir if state_paths is not None else BUDGET_DIR


def dossier_path(entity_key: str, state_paths=None) -> Path:
    """Canonical dossier path for a project entity key."""
    return _digests_root(state_paths) / f"{safe_entity_key(entity_key)}.md"


def report_path(
    entity_key: str, date: str, seq: int | str, state_paths=None
) -> Path:
    """Canonical immutable session-report path."""
    safe = safe_entity_key(entity_key)
    date = str(date).strip()
    if not _DATE_RE.match(date):
        raise DossierError(f"Invalid report date {date!r} — expected YYYY-MM-DD.")
    try:
        seq_n = int(seq)
    except (TypeError, ValueError):
        raise DossierError(f"Invalid report sequence {seq!r} — expected an integer.")
    if seq_n < 1:
        raise DossierError(f"Invalid report sequence {seq_n} — must be >= 1.")
    return _reports_root(state_paths) / f"{safe}-{date}-{seq_n}.md"


def budget_path(entity_key: str, state_paths=None) -> Path:
    """Canonical budget-ledger path for a project entity key."""
    return _budget_root(state_paths) / f"{safe_entity_key(entity_key)}.json"


# ─────────────────────────────────────────────────────────────────────
# Skeleton and validation
# ─────────────────────────────────────────────────────────────────────


def render_skeleton(
    entity_key: str,
    owner_agent: str = "",
    token_budget: int = DEFAULT_DOSSIER_TOKEN_BUDGET,
    max_workers_per_day: int = DEFAULT_WORKERS_PER_DAY,
    active: bool = DEFAULT_ACTIVE,
) -> str:
    """Render the seven-section dossier skeleton for a new project."""
    slug = safe_entity_key(entity_key).split("-", 1)[1]
    return f"""---
schema_version: 1
entity_key: {entity_key}
owner_agent: {owner_agent or "unassigned"}
token_budget: {int(token_budget)}
max_workers_per_day: {int(max_workers_per_day)}
active: {"true" if active else "false"}
---

# Project dossier: {slug}

## Identity

Project `{entity_key}`. Repository/worktree path: (not yet recorded.)
One-paragraph description pending initial seeding.

## Goals

- `G-001` **(placeholder)** Define the first strategic objective.
  Exit condition: (not yet defined.)

## Tasks

{TASKS_BEGIN}
- `T-001` [pending] [P2] goal=`G-001` — Seed this dossier with real Identity,
  Goals, and Tasks content. Evidence: none yet.
{TASKS_END}

## Timeline

(No autonomous sessions recorded yet.)

## Narrative

This dossier was initialized from the autonomous-agent-mode skeleton. No
interpretive history has accrued yet.

## Standing decisions

- `D-001` The autonomous controller is the sole routine writer of this
  dossier. Workers may read it; they must never edit it.

## Open threads

- Dossier is a fresh skeleton and needs seeding before autonomous work begins.
"""


def _parse_frontmatter(text: str) -> dict[str, str]:
    match = _FRONTMATTER_RE.match(text or "")
    if not match:
        return {}
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        name, _, value = line.partition(":")
        fields[name.strip()] = value.strip()
    return fields


def _section_bodies(text: str) -> list[tuple[str, str]]:
    """Return ``[(heading, body), ...]`` in document order."""
    matches = list(_SECTION_RE.finditer(text or ""))
    out: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((m.group(1).strip(), text[m.end():end].strip()))
    return out


def _section_span(text: str, heading: str) -> tuple[int, int] | None:
    """Return ``(start, end)`` offsets of ``heading``'s body, stripped of padding.

    ``start`` is the first character of the body proper (blank lines after the
    heading skipped); ``end`` is one past its last non-whitespace character.
    """
    matches = list(_SECTION_RE.finditer(text or ""))
    for i, m in enumerate(matches):
        if m.group(1).strip() != heading:
            continue
        raw_start = m.end()
        raw_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[raw_start:raw_end]
        lead = len(body) - len(body.lstrip())
        trail = len(body) - len(body.rstrip())
        return raw_start + lead, raw_end - trail
    return None


def section_errors(text: str) -> list[str]:
    """Return constitution violations in ``text``; empty list means valid.

    Checks presence, exact order, no duplicates, no empty bodies, and a
    well-formed Tasks block.
    """
    errors: list[str] = []
    sections = _section_bodies(text)
    headings = [h for h, _ in sections]

    seen: set[str] = set()
    for heading in headings:
        if heading in REQUIRED_SECTIONS and heading in seen:
            errors.append(f"duplicate section: {heading}")
        seen.add(heading)

    for required in REQUIRED_SECTIONS:
        if required not in headings:
            errors.append(f"missing required section: {required}")

    # Order is only meaningful once every required section is present.
    if not errors:
        present = [h for h in headings if h in REQUIRED_SECTIONS]
        if present != list(REQUIRED_SECTIONS):
            errors.append(
                "sections out of order: expected "
                f"{' → '.join(REQUIRED_SECTIONS)}, found {' → '.join(present)}"
            )

    for heading, body in sections:
        if heading in REQUIRED_SECTIONS and not body:
            errors.append(f"empty required section: {heading}")

    if TASKS_BEGIN not in (text or ""):
        errors.append("missing Tasks block start marker")
    if TASKS_END not in (text or ""):
        errors.append("missing Tasks block end marker")
    if TASKS_BEGIN in (text or "") and TASKS_END in (text or ""):
        if text.index(TASKS_END) < text.index(TASKS_BEGIN):
            errors.append("Tasks block markers are inverted")

    return errors


def _estimate_tokens(text: str) -> int:
    """Token count via ``mesh.llm.estimate_tokens`` with a local fallback.

    The import is deferred: ``mesh.llm`` pulls in backend clients, and the
    dossier layer must remain importable in a bare test process.
    """
    try:
        from .llm import estimate_tokens

        return estimate_tokens(text)
    except Exception:
        return max(1, len(text) // 4)


def _validate_project_model_block(narrative_body: str) -> list[str]:
    """Validate the optional bounded Project Model nested in Narrative."""
    match = _PROJECT_MODEL_HEADING_RE.search(narrative_body or "")
    if match is None:
        return []

    following = narrative_body[match.end():]
    next_heading = re.search(r"^###[ \t]+.+?[ \t]*$", following, re.MULTILINE)
    block_end = match.end() + next_heading.start() if next_heading else len(narrative_body)
    block = narrative_body[match.start():block_end]
    errors: list[str] = []

    tokens = _estimate_tokens(block)
    if tokens > _PROJECT_MODEL_TOKEN_BUDGET:
        errors.append(
            f"Project Model block is over budget: {tokens} tokens > "
            f"{_PROJECT_MODEL_TOKEN_BUDGET} ceiling"
        )

    required_parts = ("Observations", "Current approach")
    seen_parts: set[str] = set()
    item_counts = {part: 0 for part in required_parts}
    current_part: str | None = None

    for line_number, line in enumerate(block.splitlines(), start=1):
        if line_number == 1 or not line.strip():
            continue
        if line in ("#### Observations", "#### Current approach"):
            current_part = line.removeprefix("#### ")
            seen_parts.add(current_part)
            continue
        if re.fullmatch(r"#{3,4}[ \t]+\S.*", line):
            current_part = None
            continue

        item = _PROJECT_MODEL_ITEM_RE.fullmatch(line)
        if item is None:
            errors.append(f"Project Model line {line_number}: unrecognized format")
            continue
        epistemic, timestamp, evidence, statement = item.groups()
        if current_part is not None:
            item_counts[current_part] += 1
        if epistemic not in _PROJECT_MODEL_STATES:
            errors.append(
                f"Project Model line {line_number}: invalid epistemic state "
                f"{epistemic!r} (expected verified/inferred/uncertain/superseded)"
            )
        if not _DATE_RE.fullmatch(timestamp):
            errors.append(
                f"Project Model line {line_number}: invalid timestamp "
                f"{timestamp!r} (expected YYYY-MM-DD)"
            )
        if not evidence.strip():
            errors.append(
                f"Project Model line {line_number}: missing evidence pointer"
            )
        if epistemic == "superseded" and not re.search(
            r"\bsuperseded by\s+\S", statement, re.IGNORECASE
        ):
            errors.append(
                f"Project Model line {line_number}: superseded item must name "
                "what superseded it"
            )
        if current_part == "Current approach":
            final_clause = re.split(r";\s+|(?<=[.!?])\s+", statement.strip())[-1]
            if not re.fullmatch(
                r"Reconsider (?:if|when):?\s+\S.*[.!?]?",
                final_clause,
                re.IGNORECASE,
            ):
                errors.append(
                    f"Project Model line {line_number}: Current approach item is "
                    "missing a final reconsideration trigger"
                )

    for part in required_parts:
        if part not in seen_parts:
            errors.append(
                f"Project Model block is missing required sub-section: {part}"
            )
        elif item_counts[part] == 0:
            errors.append(f"Project Model sub-section is empty: {part}")

    return errors


def validate_dossier(entity_key: str, text: str) -> list[str]:
    """Full validation: frontmatter identity, sections, and token ceiling."""
    errors = section_errors(text)

    fm = _parse_frontmatter(text)
    declared = fm.get("entity_key", "")
    if not declared:
        errors.append("frontmatter is missing entity_key")
    elif declared != entity_key:
        errors.append(
            f"frontmatter entity_key {declared!r} does not match "
            f"requested project {entity_key!r}"
        )

    try:
        budget = int(fm.get("token_budget", DEFAULT_DOSSIER_TOKEN_BUDGET))
    except ValueError:
        budget = DEFAULT_DOSSIER_TOKEN_BUDGET
        errors.append("frontmatter token_budget is not an integer")
    tokens = _estimate_tokens(text)
    if tokens > budget:
        errors.append(f"dossier is over budget: {tokens} tokens > {budget} ceiling")

    narrative_body = next(
        (body for heading, body in _section_bodies(text) if heading == "Narrative"),
        "",
    )
    errors.extend(_validate_project_model_block(narrative_body))

    return errors


# ─────────────────────────────────────────────────────────────────────
# Atomic write helper
# ─────────────────────────────────────────────────────────────────────


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via same-directory temp file + fsync + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ─────────────────────────────────────────────────────────────────────
# Dossier operations
# ─────────────────────────────────────────────────────────────────────


def init_dossier(
    entity_key: str,
    owner_agent: str = "",
    token_budget: int = DEFAULT_DOSSIER_TOKEN_BUDGET,
    max_workers_per_day: int = DEFAULT_WORKERS_PER_DAY,
    state_paths=None,
) -> Path:
    """Create the dossier skeleton if it does not exist. Never overwrites.

    Returns the canonical path in both the created and already-present cases.
    """
    path = dossier_path(entity_key, state_paths)
    if path.exists():
        return path
    body = render_skeleton(
        entity_key,
        owner_agent=owner_agent,
        token_budget=token_budget,
        max_workers_per_day=max_workers_per_day,
    )
    errors = validate_dossier(entity_key, body)
    if errors:  # pragma: no cover — the skeleton is validated by tests
        raise DossierError("generated skeleton is invalid: " + "; ".join(errors))
    _atomic_write(path, body)
    return path


def read_dossier(entity_key: str, state_paths=None) -> str:
    """Return the dossier body. Fails closed when the file is absent."""
    path = dossier_path(entity_key, state_paths)
    try:
        return path.read_text()
    except FileNotFoundError:
        raise DossierError(
            f"No dossier at {path}. Initialize it before autonomous work begins."
        )
    except OSError as e:
        raise DossierError(f"Error reading dossier at {path}: {e}")


@dataclass
class EditResult:
    """Diagnostics returned by a successful dossier mutation."""

    path: Path
    replacements: int
    tokens: int
    token_budget: int


def edit_dossier(
    entity_key: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
    state_paths=None,
) -> EditResult:
    """Exact string replacement in the dossier, validated before it lands.

    On any validation failure the old dossier remains byte-identical.
    """
    if not isinstance(old_text, str) or not old_text:
        raise DossierError("old_text must be a non-empty string.")
    if not isinstance(new_text, str):
        raise DossierError("new_text must be a string.")

    path = dossier_path(entity_key, state_paths)
    content = read_dossier(entity_key, state_paths)

    count = content.count(old_text)
    if count == 0:
        raise DossierError("old_text not found in dossier.")
    if not replace_all and count > 1:
        raise DossierError(
            f"old_text matches {count} locations — provide a more specific "
            "string or set replace_all=true."
        )

    updated = (
        content.replace(old_text, new_text)
        if replace_all
        else content.replace(old_text, new_text, 1)
    )

    errors = validate_dossier(entity_key, updated)
    if errors:
        raise DossierError(
            "edit refused — it would violate the dossier constitution: "
            + "; ".join(errors)
        )

    _atomic_write(path, updated)
    fm = _parse_frontmatter(updated)
    try:
        budget = int(fm.get("token_budget", DEFAULT_DOSSIER_TOKEN_BUDGET))
    except ValueError:
        budget = DEFAULT_DOSSIER_TOKEN_BUDGET
    return EditResult(
        path=path,
        replacements=count if replace_all else 1,
        tokens=_estimate_tokens(updated),
        token_budget=budget,
    )


# ─────────────────────────────────────────────────────────────────────
# Session reports
# ─────────────────────────────────────────────────────────────────────


def _timeline_entry(
    entity_key: str, path: Path, date: str, seq: int, state_paths=None
) -> str:
    rel = os.path.relpath(path, dossier_path(entity_key, state_paths).parent)
    return f"- **{date} — autonomous session {seq}.** Report: [{path.name}]({rel})."


def write_report(
    entity_key: str,
    date: str,
    seq: int | str,
    content: str,
    state_paths=None,
) -> Path:
    """Create an immutable session report and link it from the dossier Timeline.

    Create-once semantics: identical bytes are an idempotent success, different
    bytes are refused.  The dossier must exist — a report that cannot be linked
    is not a usable audit trail.
    """
    if not isinstance(content, str) or not content.strip():
        raise DossierError("Report content must be a non-empty string.")

    path = report_path(entity_key, date, seq, state_paths)
    # Fail closed before writing anything if the dossier cannot be linked.
    dossier = read_dossier(entity_key, state_paths)

    if path.exists():
        existing = path.read_text()
        if existing == content:
            return path
        raise DossierError(
            f"Session report {path} already exists with different content. "
            "Reports are immutable — write an amendment as a new report."
        )

    entry = _timeline_entry(
        entity_key, path, str(date).strip(), int(seq), state_paths
    )
    if entry in dossier:
        updated = dossier
    else:
        span = _section_span(dossier, "Timeline")
        if span is None:
            raise DossierError("Dossier has no Timeline section to link the report from.")
        start, end = span
        # Newest first (§3.6).  Splice at the section span so the anchor is
        # positional, never a substring match that could land elsewhere.
        updated = dossier[:start] + f"{entry}\n" + dossier[start:end] + dossier[end:]
        errors = validate_dossier(entity_key, updated)
        if errors:
            raise DossierError(
                "Timeline update refused — it would violate the dossier "
                "constitution: " + "; ".join(errors)
            )

    _atomic_write(path, content)
    try:
        if updated != dossier:
            _atomic_write(dossier_path(entity_key, state_paths), updated)
    except OSError as e:
        raise DossierError(
            f"Report written to {path} but the Timeline link failed: {e}. "
            "The report is an orphan pending reconciliation."
        )
    return path


# ─────────────────────────────────────────────────────────────────────
# Budget ledger
# ─────────────────────────────────────────────────────────────────────


def _today() -> str:
    return _date.today().isoformat()


def _next_midnight_iso() -> str:
    tomorrow = datetime.now().replace(
        hour=0, minute=0, second=0, microsecond=0
    ) + timedelta(days=1)
    return tomorrow.isoformat()


def _configured_limit(entity_key: str, state_paths=None) -> int:
    """Daily worker limit, from the dossier frontmatter when available."""
    try:
        fm = _parse_frontmatter(read_dossier(entity_key, state_paths))
    except DossierError:
        return DEFAULT_WORKERS_PER_DAY
    raw = fm.get("max_workers_per_day", "")
    try:
        limit = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_WORKERS_PER_DAY
    return limit if limit >= 0 else DEFAULT_WORKERS_PER_DAY


def active_flag(entity_key: str, state_paths=None) -> bool:
    """Whether the project is in active mode, from the dossier frontmatter.

    Fails closed: a missing dossier, a missing line, or an unparseable value
    all read as ``False``.  Active mode arms an automatic scheduler, so an
    ambiguous dossier must never be treated as armed.
    """
    try:
        fm = _parse_frontmatter(read_dossier(entity_key, state_paths))
    except DossierError:
        return DEFAULT_ACTIVE
    raw = str(fm.get("active", "")).strip().lower()
    if raw in ("true", "yes", "on", "1"):
        return True
    return DEFAULT_ACTIVE


def _load_ledger(entity_key: str, state_paths=None) -> dict:
    """Read the ledger, rolling the day bucket over when the date changed."""
    path = budget_path(entity_key, state_paths)
    limit = _configured_limit(entity_key, state_paths)
    today = _today()
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("ledger is not an object")
    except (FileNotFoundError, ValueError, OSError):
        data = {}

    if data.get("date") != today:
        data = {"date": today, "used": 0}
    try:
        used = max(0, int(data.get("used", 0)))
    except (TypeError, ValueError):
        used = 0

    return {
        "entity_key": entity_key,
        "date": today,
        "used": used,
        "limit": limit,
        "remaining": max(0, limit - used),
        "resets_at": _next_midnight_iso(),
    }


def check_budget(entity_key: str, state_paths=None) -> dict:
    """Return ``{remaining, used, limit, resets_at, ...}`` without mutating."""
    return _load_ledger(entity_key, state_paths)


def spend_budget(entity_key: str, count: int = 1, state_paths=None) -> dict:
    """Charge ``count`` worker admissions; refuse if it would exceed the limit.

    Raises :class:`BudgetExhausted` without touching the ledger when the spend
    does not fit.  Budgets bound *attempts*, so callers charge before dispatch.
    """
    try:
        count = int(count)
    except (TypeError, ValueError):
        raise DossierError(f"Invalid budget count {count!r} — expected an integer.")
    if count < 1:
        raise DossierError(f"Invalid budget count {count} — must be >= 1.")

    state = _load_ledger(entity_key, state_paths)
    if state["used"] + count > state["limit"]:
        raise BudgetExhausted(
            f"Autonomous worker budget exhausted for {entity_key}: "
            f"used {state['used']} of {state['limit']} today; requested {count}. "
            f"Resets at {state['resets_at']}."
        )

    state["used"] += count
    state["remaining"] = max(0, state["limit"] - state["used"])
    _atomic_write(
        budget_path(entity_key, state_paths),
        json.dumps({"date": state["date"], "used": state["used"]}, indent=2) + "\n",
    )
    return state


def reset_budget(entity_key: str, state_paths=None) -> dict:
    """Clear today's spent-worker counter without changing its configured limit."""
    _load_ledger(entity_key, state_paths)
    today = _today()
    _atomic_write(
        budget_path(entity_key, state_paths),
        json.dumps({"date": today, "used": 0}, indent=2) + "\n",
    )
    return _load_ledger(entity_key, state_paths)
