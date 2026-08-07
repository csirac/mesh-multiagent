"""Per-agent filesystem and capability isolation policy.

This module is the single source of truth for:

* the normalized, immutable :class:`IsolationPolicy` derived from YAML/CLI;
* the derived :class:`StatePaths` that every agent-owned directory hangs off;
* the :class:`WorkerIsolationScope` that a parent process hands to a worker
  subprocess;
* tool capability classification and the one authorization function.

Design contract
---------------

``enabled: false`` (and an absent ``isolation`` block) must reproduce today's
behaviour byte for byte: state lives under ``~/.mesh``, no tool is filtered,
and no path is canonicalized on the hot path.  :meth:`IsolationPolicy.legacy`
is that fast path and it performs no filesystem work.

``enabled: true`` fails closed.  Every workspace root is canonicalized with
``realpath`` at construction time, ``state_root`` must resolve inside one of
those roots, and anything that cannot be proven contained is rejected with an
actionable startup error rather than silently narrowed or broadened.

Nothing here reads module globals at call time, so two agents in one process
(tests, future multi-agent hosting) cannot leak policy into each other.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "FilesystemMode",
    "ToolCapability",
    "IsolationConfig",
    "IsolationPolicy",
    "StatePaths",
    "WorkerIsolationScope",
    "Authorization",
    "IsolationConfigError",
    "authorize_tool",
    "resolve_workspace_path",
    "is_path_contained",
    "STATE_SUBDIRS",
    "SCRATCH_SUBDIR",
    "WORKER_SCOPE_ENV_VAR",
]


class IsolationConfigError(ValueError):
    """Raised when an isolation block cannot be normalized into a policy.

    This is a startup error by design: an agent that asked to be isolated and
    cannot be must not come up unisolated.
    """


class FilesystemMode(str, Enum):
    """How the declared workspace roots may be touched by raw tools."""

    WORKSPACE_WRITE = "workspace-write"
    READ_ONLY = "read-only"

    @classmethod
    def parse(cls, value: "str | FilesystemMode | None") -> "FilesystemMode":
        if value is None:
            return cls.WORKSPACE_WRITE
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower().replace("_", "-")
        for member in cls:
            if member.value == text:
                return member
        allowed = ", ".join(m.value for m in cls)
        raise IsolationConfigError(
            f"isolation.filesystem_mode {value!r} is not recognized — expected one of: {allowed}"
        )


class ToolCapability(str, Enum):
    """What a tool needs in order to run.

    ``LOCAL``
        Reads/writes only inside the policy's declared roots.
    ``EXTERNAL_NETWORK``
        Reaches a third-party data plane (Exa, Gmail, arXiv, notes, ...).
    ``MESH_CONTROL``
        The agent's own control plane — ``send_message``, ``send_report``,
        worker lifecycle, todos.  Never gated by ``allow_network``, because
        cutting it off does not contain the agent, it only silences it.
    ``CREDENTIAL``
        Reads or rewrites host credential material.  Always requires an
        explicit name in ``allowed_credential_tools``, even when network is
        allowed.
    """

    LOCAL = "local"
    EXTERNAL_NETWORK = "external_network"
    MESH_CONTROL = "mesh_control"
    CREDENTIAL = "credential"

    @classmethod
    def parse(cls, value: "str | ToolCapability") -> "ToolCapability":
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower()
        for member in cls:
            if member.value == text:
                return member
        allowed = ", ".join(m.value for m in cls)
        raise IsolationConfigError(
            f"unknown tool capability {value!r} — expected one of: {allowed}"
        )


#: Subdirectory layout under ``state_root``.  ``StatePaths`` derives every
#: agent-owned location from this table so a future per-category override can
#: be added without touching a single consumer.
STATE_SUBDIRS: dict[str, str] = {
    "memory_dir": "memory",
    "maps_dir": "memory/maps",
    "digests_dir": "digests",
    "reports_dir": "reports",
    "budget_dir": "budget",
    "history_dir": "history",
    "skills_dir": "skills",
    "skill_drafts_dir": "skill_drafts",
    "worker_traces_dir": "worker_traces",
    "fixed_tool_runs_dir": "fixed_tool_runs",
    "pev_reports_dir": "pev_reports",
    "harness_crashes_dir": "harness-crashes",
    "cc_sessions_dir": "sessions",
    "tmp_dir": "tmp",
}

#: The one state subdirectory raw tools may write even under ``protect_state``.
#:
#: An isolated agent loses the implicit host ``/tmp`` allowance, so it needs
#: *some* scratch space or ordinary work (unpacking, staging, redirecting shell
#: output) becomes impossible.  Scratch is not agent state — nothing here feeds
#: memory, budget, or history integrity — so it is deliberately carved out of
#: the protected subtree rather than granting a second root outside it.
SCRATCH_SUBDIR = STATE_SUBDIRS["tmp_dir"]

#: Environment variable used to hand a scope to a worker subprocess.
WORKER_SCOPE_ENV_VAR = "MESH_WORKER_ISOLATION_SCOPE"

#: Private HOME handed to an isolated Codex subprocess, relative to the state
#: root.  Codex resolves ``~/.codex/auth.json`` from HOME, so an isolated
#: worker needs a config location of its own rather than the operator's real
#: home directory.  Phase 4 provisions the credential material inside it.
CODEX_HOME_SUBDIR = "codex-home"


# ─────────────────────────────────────────────────────────────────────
# Canonical path helpers
# ─────────────────────────────────────────────────────────────────────


def resolve_workspace_path(path: str | Path) -> Path:
    """Canonicalize ``path`` for containment decisions.

    ``~`` expands through :mod:`mesh.paths` (the real home from ``/etc/passwd``,
    not ``$HOME``, which is a synthetic Claude Code account home when the
    process was launched from a CC session).  Symlinks are resolved so that a
    link pointing outside a workspace cannot be used to smuggle a path back in.
    """
    from .paths import resolve_path as _expand_home

    text = _expand_home(str(path))
    return Path(text).expanduser().resolve()


def is_path_contained(path: str | Path, workspaces: Iterable[str | Path]) -> bool:
    """Return True when ``path`` resolves inside one of ``workspaces``.

    A root contains itself.  Both sides are canonicalized first, so ``..``
    segments and symlink components cannot produce a false positive.
    """
    try:
        candidate = resolve_workspace_path(path)
    except (OSError, RuntimeError):
        return False
    for workspace in workspaces:
        try:
            root = resolve_workspace_path(workspace)
        except (OSError, RuntimeError):
            continue
        if candidate == root or root in candidate.parents:
            return True
    return False


def _legacy_state_root() -> Path:
    """The historical global state root, ``<real home>/.mesh``."""
    from .paths import MESH_DIR

    return Path(MESH_DIR)


# ─────────────────────────────────────────────────────────────────────
# Raw configuration shape
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class IsolationConfig:
    """Raw ``isolation:`` block exactly as written in YAML.

    No validation happens here — this is the transport shape.  All checks live
    in :meth:`IsolationPolicy.from_config` so there is exactly one place where
    a policy can be born.
    """

    enabled: bool = False
    workspace: str | None = None
    additional_workspaces: tuple[str, ...] = ()
    state_root: str | None = None
    filesystem_mode: str = FilesystemMode.WORKSPACE_WRITE.value
    protect_state: bool = True
    allow_network: bool = False
    #: ``None`` means "every external-network tool" when ``allow_network`` is
    #: true; ``()`` means none; a populated tuple is a name allowlist.
    allowed_network_tools: tuple[str, ...] | None = ()
    allowed_credential_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "protect_state", bool(self.protect_state))
        object.__setattr__(self, "allow_network", bool(self.allow_network))
        object.__setattr__(
            self, "additional_workspaces", _as_str_tuple(self.additional_workspaces)
        )
        if self.allowed_network_tools is not None:
            object.__setattr__(
                self, "allowed_network_tools", _as_str_tuple(self.allowed_network_tools)
            )
        object.__setattr__(
            self, "allowed_credential_tools", _as_str_tuple(self.allowed_credential_tools)
        )

    @classmethod
    def from_dict(cls, data: dict | None) -> "IsolationConfig":
        """Build from a YAML mapping, rejecting unknown keys."""
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise IsolationConfigError(
                f"isolation must be a mapping, got {type(data).__name__}"
            )
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = sorted(set(data) - known)
        if unknown:
            raise IsolationConfigError(
                f"unknown isolation option(s): {', '.join(unknown)} — "
                f"supported: {', '.join(sorted(known))}"
            )
        return cls(**data)

    def to_dict(self) -> dict:
        """Round-trippable mapping for ``MeshConfig.to_dict()``."""
        return {
            "enabled": self.enabled,
            "workspace": self.workspace,
            "additional_workspaces": list(self.additional_workspaces),
            "state_root": self.state_root,
            "filesystem_mode": self.filesystem_mode,
            "protect_state": self.protect_state,
            "allow_network": self.allow_network,
            "allowed_network_tools": (
                None
                if self.allowed_network_tools is None
                else list(self.allowed_network_tools)
            ),
            "allowed_credential_tools": list(self.allowed_credential_tools),
        }


def _as_str_tuple(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Path)):
        return (str(value),)
    return tuple(str(item) for item in value)


# ─────────────────────────────────────────────────────────────────────
# Normalized policy
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class IsolationPolicy:
    """A validated, immutable isolation decision for one agent."""

    enabled: bool = False
    #: Canonical primary workspace.  ``None`` on the legacy path.
    workspace: Path | None = None
    #: Every canonical writable/readable root, primary first.
    workspaces: tuple[Path, ...] = ()
    state_root: Path = field(default_factory=_legacy_state_root)
    filesystem_mode: FilesystemMode = FilesystemMode.WORKSPACE_WRITE
    protect_state: bool = False
    allow_network: bool = True
    allowed_network_tools: tuple[str, ...] | None = None
    allowed_credential_tools: tuple[str, ...] | None = None
    #: Where this policy came from, for error messages and diagnostics.
    source: str = "legacy"

    # -- construction -------------------------------------------------

    @classmethod
    def legacy(cls, source: str = "legacy") -> "IsolationPolicy":
        """The disabled policy: current paths, current tools, no checks.

        ``allowed_network_tools``/``allowed_credential_tools`` are ``None``
        ("no allowlist configured"), which combined with ``enabled=False``
        means :func:`authorize_tool` allows everything without touching the
        filesystem.
        """
        return cls(
            enabled=False,
            workspace=None,
            workspaces=(),
            state_root=_legacy_state_root(),
            filesystem_mode=FilesystemMode.WORKSPACE_WRITE,
            protect_state=False,
            allow_network=True,
            allowed_network_tools=None,
            allowed_credential_tools=None,
            source=source,
        )

    @classmethod
    def from_config(
        cls,
        config: "IsolationConfig | dict | None",
        *,
        source: str = "config",
    ) -> "IsolationPolicy":
        """Normalize and validate a raw config block into a policy."""
        cfg = IsolationConfig.from_dict(config) if not isinstance(
            config, IsolationConfig
        ) else config

        if not cfg.enabled:
            # Fast path: do not touch the filesystem, do not canonicalize.
            return cls.legacy(source=source)

        if not cfg.workspace or not str(cfg.workspace).strip():
            raise IsolationConfigError(
                "isolation.enabled is true but isolation.workspace is not set — "
                "an isolated agent needs exactly one primary workspace root"
            )

        primary = _require_existing_dir(cfg.workspace, "isolation.workspace")

        roots: list[Path] = [primary]
        for raw in cfg.additional_workspaces:
            extra = _require_existing_dir(raw, "isolation.additional_workspaces")
            if extra == primary or extra in roots:
                raise IsolationConfigError(
                    f"isolation.additional_workspaces contains a duplicate root: {extra}"
                )
            # Nested roots are rejected rather than merged: silently collapsing
            # them makes the effective boundary differ from the written one.
            for existing in roots:
                if existing in extra.parents:
                    raise IsolationConfigError(
                        f"isolation workspace {extra} is nested inside {existing} — "
                        "declare only the outer root"
                    )
                if extra in existing.parents:
                    raise IsolationConfigError(
                        f"isolation workspace {extra} contains already-declared root "
                        f"{existing} — declaring the parent would broaden the policy"
                    )
            roots.append(extra)

        workspaces = tuple(roots)

        if cfg.state_root and str(cfg.state_root).strip():
            state_root = _resolve_creatable_path(cfg.state_root, "isolation.state_root")
        else:
            state_root = primary / ".mesh"

        if not is_path_contained(state_root, workspaces):
            raise IsolationConfigError(
                f"isolation.state_root {state_root} resolves outside every declared "
                f"workspace root ({', '.join(str(w) for w in workspaces)}) — "
                "an isolated agent may not keep state outside its boundary"
            )

        mode = FilesystemMode.parse(cfg.filesystem_mode)

        return cls(
            enabled=True,
            workspace=primary,
            workspaces=workspaces,
            state_root=state_root,
            filesystem_mode=mode,
            protect_state=bool(cfg.protect_state),
            allow_network=bool(cfg.allow_network),
            allowed_network_tools=(
                None
                if cfg.allowed_network_tools is None
                else tuple(cfg.allowed_network_tools)
            ),
            allowed_credential_tools=tuple(cfg.allowed_credential_tools),
            source=source,
        )

    # -- CLI narrowing ------------------------------------------------

    def narrow(
        self,
        *,
        allow_network: bool | None = None,
        filesystem_mode: "str | FilesystemMode | None" = None,
    ) -> "IsolationPolicy":
        """Apply a CLI override that may only make the policy stricter.

        A run-time flag must never broaden an enabled YAML policy: an operator
        typo should not be the thing that grants network to an isolated agent.
        Requests that would broaden are ignored, not raised, so ``--no-network``
        on an already-offline agent stays a no-op.
        """
        updates: dict = {}

        if allow_network is False and self.allow_network:
            updates["allow_network"] = False
        # allow_network=True on an enabled policy would broaden — ignored.
        elif allow_network is True and not self.enabled:
            updates["allow_network"] = True

        if filesystem_mode is not None:
            requested = FilesystemMode.parse(filesystem_mode)
            if (
                requested is FilesystemMode.READ_ONLY
                and self.filesystem_mode is not FilesystemMode.READ_ONLY
            ):
                updates["filesystem_mode"] = requested

        if not updates:
            return self
        return replace(self, **updates)

    # -- queries ------------------------------------------------------

    @property
    def writable(self) -> bool:
        """Whether raw tools may write inside the declared roots."""
        return self.filesystem_mode is FilesystemMode.WORKSPACE_WRITE

    def contains(self, path: str | Path) -> bool:
        """Whether ``path`` is inside the policy boundary.

        A disabled policy contains everything — that is what preserves current
        behaviour for unisolated agents.
        """
        if not self.enabled:
            return True
        return is_path_contained(path, self.workspaces)

    def is_protected_state(self, path: str | Path) -> bool:
        """Whether ``path`` is state that raw tools must not touch.

        The scratch subtree (:data:`SCRATCH_SUBDIR`) is excluded: it replaces
        the host ``/tmp`` allowance an isolated agent gives up and holds no
        integrity-bearing state.  Everything else beneath ``state_root`` —
        memory, budget, history, digests — stays closed to raw file tools so a
        worker cannot rewrite its own ledger.
        """
        if not self.enabled or not self.protect_state:
            return False
        if not is_path_contained(path, (self.state_root,)):
            return False
        return not is_path_contained(path, (self.scratch_dir,))

    @property
    def scratch_dir(self) -> Path:
        """The agent's private temp directory, replacing host ``/tmp``."""
        return self.state_root / SCRATCH_SUBDIR

    def worker_scope(self) -> "WorkerIsolationScope":
        """The serializable subset a worker subprocess is allowed to see."""
        return WorkerIsolationScope(
            enabled=self.enabled,
            workspaces=tuple(str(p) for p in self.workspaces),
            state_root=str(self.state_root),
            filesystem_mode=self.filesystem_mode.value,
            protect_state=self.protect_state,
            allow_network=self.allow_network,
        )

    def describe(self) -> str:
        """One-line human summary for startup logs."""
        if not self.enabled:
            return f"isolation disabled (state_root={self.state_root})"
        roots = ", ".join(str(p) for p in self.workspaces)
        return (
            f"isolation enabled mode={self.filesystem_mode.value} "
            f"roots=[{roots}] state_root={self.state_root} "
            f"protect_state={self.protect_state} allow_network={self.allow_network}"
        )


def _require_existing_dir(raw: str | Path, label: str) -> Path:
    text = str(raw).strip()
    if not text:
        raise IsolationConfigError(f"{label} must not be empty")

    from .paths import resolve_path as _expand_home

    expanded = _expand_home(text)
    if not os.path.isabs(expanded):
        raise IsolationConfigError(
            f"{label} must be an absolute path, got {raw!r}"
        )
    path = Path(expanded)
    if not path.exists():
        raise IsolationConfigError(
            f"{label} {path} does not exist — create the workspace before "
            "enabling isolation for this agent"
        )
    if not path.is_dir():
        raise IsolationConfigError(f"{label} {path} is not a directory")
    return path.resolve()


def _resolve_creatable_path(raw: str | Path, label: str) -> Path:
    """Resolve a path that may not exist yet but whose parent must.

    ``state_root`` is allowed to be created on first run, but its existing
    ancestry has to be real so a symlink component cannot appear later and
    move it out of the boundary between validation and use.
    """
    text = str(raw).strip()
    if not text:
        raise IsolationConfigError(f"{label} must not be empty")

    from .paths import resolve_path as _expand_home

    expanded = _expand_home(text)
    if not os.path.isabs(expanded):
        raise IsolationConfigError(f"{label} must be an absolute path, got {raw!r}")

    path = Path(expanded)
    if path.exists():
        if not path.is_dir():
            raise IsolationConfigError(f"{label} {path} is not a directory")
        return path.resolve()

    # Walk up to the nearest existing ancestor and resolve from there, so
    # symlinked ancestry is canonicalized even though the leaf is absent.
    existing = path
    tail: list[str] = []
    while not existing.exists():
        if existing.parent == existing:
            raise IsolationConfigError(
                f"{label} {path} has no existing ancestor directory"
            )
        tail.append(existing.name)
        existing = existing.parent
    if not existing.is_dir():
        raise IsolationConfigError(
            f"{label} {path} cannot be created — {existing} is not a directory"
        )
    resolved = existing.resolve()
    for part in reversed(tail):
        resolved = resolved / part
    return resolved


# ─────────────────────────────────────────────────────────────────────
# Derived state paths
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StatePaths:
    """Every agent-owned directory, derived once from a policy.

    Consumers receive this object instead of importing module globals, which
    is what makes two agents in one process safe and what makes an isolated
    agent's state provably inside its boundary.
    """

    state_root: Path
    memory_dir: Path
    maps_dir: Path
    digests_dir: Path
    reports_dir: Path
    budget_dir: Path
    history_dir: Path
    skills_dir: Path
    skill_drafts_dir: Path
    worker_traces_dir: Path
    fixed_tool_runs_dir: Path
    pev_reports_dir: Path
    harness_crashes_dir: Path
    cc_sessions_dir: Path
    #: Private scratch space; the isolated replacement for host ``/tmp``.
    tmp_dir: Path
    #: True when these paths came from the historical globals in mesh.paths.
    legacy: bool = True

    @classmethod
    def for_state_root(
        cls, state_root: str | Path, *, legacy: bool = False
    ) -> "StatePaths":
        """Derive the full layout beneath ``state_root`` without creating it."""
        root = Path(state_root)
        return cls(
            state_root=root,
            legacy=legacy,
            **{name: root / sub for name, sub in STATE_SUBDIRS.items()},
        )

    @classmethod
    def legacy_paths(cls) -> "StatePaths":
        """The historical layout, using the exact constants in mesh.paths.

        These are read from :mod:`mesh.paths` rather than re-derived so a
        disabled agent is guaranteed to land on the same directories it uses
        today, even if the two layouts ever drift.
        """
        from . import paths as _paths

        mesh_dir = Path(_paths.MESH_DIR)
        return cls(
            state_root=mesh_dir,
            memory_dir=Path(_paths.MEMORY_DIR),
            maps_dir=Path(_paths.MAPS_DIR),
            digests_dir=Path(_paths.DIGESTS_DIR),
            reports_dir=Path(_paths.REPORTS_DIR),
            budget_dir=Path(_paths.BUDGET_DIR),
            history_dir=Path(_paths.HISTORY_DIR),
            skills_dir=Path(_paths.SKILLS_DIR),
            skill_drafts_dir=mesh_dir / "skill_drafts",
            worker_traces_dir=mesh_dir / "worker_traces",
            fixed_tool_runs_dir=mesh_dir / "fixed_tool_runs",
            pev_reports_dir=mesh_dir / "pev_reports",
            harness_crashes_dir=mesh_dir / "harness-crashes",
            cc_sessions_dir=mesh_dir / "sessions",
            # Never used on the legacy path: a disabled agent keeps the host
            # ``/tmp`` allowance and this directory is never created.
            tmp_dir=mesh_dir / SCRATCH_SUBDIR,
            legacy=True,
        )

    @classmethod
    def from_policy(
        cls, policy: IsolationPolicy | None, *, create: bool = True
    ) -> "StatePaths":
        """Build the layout for ``policy``, creating directories by default.

        A disabled policy returns the legacy layout and creates nothing new —
        those directories already exist and creating them would be the first
        behavioural difference on the supposedly inert path.
        """
        if policy is None or not policy.enabled:
            return cls.legacy_paths()
        paths = cls.for_state_root(policy.state_root, legacy=False)
        if create:
            paths.ensure()
        return paths

    def ensure(self) -> "StatePaths":
        """Create every derived directory.  Idempotent."""
        for name in STATE_SUBDIRS:
            getattr(self, name).mkdir(parents=True, exist_ok=True)
        return self

    # -- derived files ------------------------------------------------

    def agent_history_file(self, nickname: str) -> Path:
        """History file for ``nickname``, matching the current naming."""
        return self.history_dir / f"agent-{nickname}.json"

    def router_history_file(self, nickname: str) -> Path:
        return self.history_dir / f"router-{nickname}.json"

    def standing_digest_file(self, nickname: str) -> Path:
        return self.digests_dir / f"agent-{nickname}.md"

    def memory_db_file(self, nickname: str) -> Path:
        return self.memory_dir / f"{nickname}.db"


# ─────────────────────────────────────────────────────────────────────
# Worker scope
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WorkerIsolationScope:
    """The immutable execution scope a parent hands to a worker.

    Deliberately a strict subset of the policy: a worker learns its roots and
    its ceilings, never the tool allowlists, which stay a parent-side decision
    so a compromised worker cannot read off what it is missing.
    """

    enabled: bool = False
    workspaces: tuple[str, ...] = ()
    state_root: str = ""
    filesystem_mode: str = FilesystemMode.WORKSPACE_WRITE.value
    protect_state: bool = False
    allow_network: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspaces", _as_str_tuple(self.workspaces))

    @property
    def primary_workspace(self) -> str | None:
        return self.workspaces[0] if self.workspaces else None

    @property
    def writable(self) -> bool:
        return self.filesystem_mode == FilesystemMode.WORKSPACE_WRITE.value

    @property
    def scratch_dir(self) -> str:
        """Scratch space for the worker; empty when the scope is disabled."""
        return str(Path(self.state_root) / SCRATCH_SUBDIR) if self.state_root else ""

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "workspaces": list(self.workspaces),
            "state_root": self.state_root,
            "filesystem_mode": self.filesystem_mode,
            "protect_state": self.protect_state,
            "allow_network": self.allow_network,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "WorkerIsolationScope":
        if not data:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", False)),
            workspaces=_as_str_tuple(data.get("workspaces") or ()),
            state_root=str(data.get("state_root") or ""),
            filesystem_mode=str(
                data.get("filesystem_mode") or FilesystemMode.WORKSPACE_WRITE.value
            ),
            protect_state=bool(data.get("protect_state", False)),
            allow_network=bool(data.get("allow_network", True)),
        )

    def to_env(self) -> dict[str, str]:
        """Environment fragment for a subprocess.

        A disabled scope contributes nothing, so an unisolated worker's
        environment is unchanged.
        """
        if not self.enabled:
            return {}
        return {WORKER_SCOPE_ENV_VAR: json.dumps(self.to_dict(), sort_keys=True)}

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "WorkerIsolationScope":
        """Recover a scope a parent placed in the environment."""
        source = os.environ if env is None else env
        raw = source.get(WORKER_SCOPE_ENV_VAR)
        if not raw:
            return cls()
        try:
            data = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise IsolationConfigError(
                f"{WORKER_SCOPE_ENV_VAR} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise IsolationConfigError(
                f"{WORKER_SCOPE_ENV_VAR} must decode to an object"
            )
        return cls.from_dict(data)

    @classmethod
    def from_policy(
        cls, policy: "IsolationPolicy | None"
    ) -> "WorkerIsolationScope":
        """Derive the worker-visible subset of a parent's policy.

        A disabled (or absent) policy yields the disabled scope, so every
        caller can hand the result straight to a launch path without first
        checking whether isolation is on.
        """
        if policy is None or not policy.enabled:
            return cls()
        return cls(
            enabled=True,
            workspaces=tuple(str(root) for root in policy.workspaces),
            state_root=str(policy.state_root),
            filesystem_mode=FilesystemMode.parse(policy.filesystem_mode).value,
            protect_state=policy.protect_state,
            allow_network=policy.allow_network,
        )


# ─────────────────────────────────────────────────────────────────────
# Worker launch (Phase 3)
# ─────────────────────────────────────────────────────────────────────

#: Codex arguments a scoped run must never inherit from operator config.
#: ``--dangerously-bypass-approvals-and-sandbox`` disables the sandbox
#: outright; ``--add-dir`` would grant a root the policy never declared;
#: ``--sandbox`` is re-derived from the scope so it cannot be broadened.
_CODEX_NON_BROADENABLE_FLAGS = frozenset({
    "--dangerously-bypass-approvals-and-sandbox",
    "--add-dir",
    "--sandbox",
    "-s",
    # A second working-directory flag appended after the scope-derived ``-C``
    # can repoint Codex outside the declared workspace just as effectively as
    # an undeclared ``--add-dir``.
    "-C",
    "--cd",
})

_CODEX_FLAGS_WITH_VALUES = frozenset({
    "--add-dir",
    "--sandbox",
    "-s",
    "-C",
    "--cd",
})

_CODEX_CONFIG_FLAGS = frozenset({"-c", "--config"})

#: Codex's own name for "no writes anywhere", used when either the scope or
#: operator configuration asks for read-only.
_CODEX_READ_ONLY = FilesystemMode.READ_ONLY.value
_CODEX_WORKSPACE_WRITE = FilesystemMode.WORKSPACE_WRITE.value


def _requested_codex_sandbox(extra_args: Sequence[str]) -> str | None:
    """The sandbox mode an operator asked for in ``codex_extra_args``."""
    args = list(extra_args)
    for index, arg in enumerate(args):
        if arg.startswith("--sandbox=") or arg.startswith("-s="):
            return arg.split("=", 1)[1].strip()
        if arg in {"--sandbox", "-s"} and index + 1 < len(args):
            return str(args[index + 1]).strip()
    return None


def _codex_config_broadens_scope(value: str) -> bool:
    """Return whether one ``-c key=value`` override controls containment."""
    key = str(value).split("=", 1)[0].strip().lower().replace("-", "_")
    return key == "sandbox_mode" or key.startswith("sandbox_")


def strip_codex_broadening_args(extra_args: Sequence[str] | None) -> list[str]:
    """Remove flags that could widen an isolated Codex run's authority.

    Only used on the enabled path.  A disabled scope keeps operator arguments
    exactly as configured.
    """
    args = [str(arg) for arg in list(extra_args or ())]
    remaining: list[str] = []
    index = 0
    while index < len(args):
        text = args[index]
        if text in _CODEX_NON_BROADENABLE_FLAGS:
            index += 2 if text in _CODEX_FLAGS_WITH_VALUES else 1
            continue
        if any(
            text.startswith(f"{flag}=")
            for flag in _CODEX_NON_BROADENABLE_FLAGS
            if flag.startswith("--") or flag == "-s"
        ):
            index += 1
            continue
        if text.startswith("-C") and text != "-C":
            index += 1
            continue
        if text in _CODEX_CONFIG_FLAGS and index + 1 < len(args):
            value = args[index + 1]
            if _codex_config_broadens_scope(value):
                index += 2
                continue
            remaining.extend([text, value])
            index += 2
            continue
        if text.startswith("--config="):
            if _codex_config_broadens_scope(text.split("=", 1)[1]):
                index += 1
                continue
        remaining.append(text)
        index += 1
    return remaining


def codex_sandbox_mode(
    scope: "WorkerIsolationScope", extra_args: Sequence[str] | None = None
) -> str:
    """The ``--sandbox`` mode for an isolated Codex run.

    The scope sets the ceiling; an operator may only narrow it further, so a
    ``read-only`` request from either side wins.
    """
    mode = _CODEX_WORKSPACE_WRITE if scope.writable else _CODEX_READ_ONLY
    requested = _requested_codex_sandbox(extra_args or ())
    if requested == _CODEX_READ_ONLY:
        return _CODEX_READ_ONLY
    return mode


def build_codex_isolation_args(
    scope: "WorkerIsolationScope", extra_args: Sequence[str] | None = None
) -> list[str]:
    """Scope-derived Codex CLI arguments, in a fixed order.

    Returns ``[]`` for a disabled scope so the legacy command stays
    byte-identical.  Shared by :meth:`mesh.llm.LLMClient._complete_codex` and
    ``mesh.tool_implementations._run_mesh_qwen_codex`` so the two Codex launch
    paths cannot drift apart.
    """
    if not scope.enabled:
        return []
    if not scope.workspaces:
        raise IsolationConfigError(
            "an enabled worker scope must declare at least one workspace "
            "before a Codex worker can be launched"
        )
    args = [
        "--sandbox",
        codex_sandbox_mode(scope, extra_args),
        # Operator ``~/.codex/config.toml`` is outside the boundary and is not
        # reviewed policy, so an isolated run must not read it.
        "--ignore-user-config",
    ]
    for additional in scope.workspaces[1:]:
        args.extend(["--add-dir", additional])
    return args


def codex_home_dir(scope: "WorkerIsolationScope", *, create: bool = False) -> str:
    """The private HOME for an isolated Codex subprocess.

    Empty for a disabled scope, whose HOME handling is unchanged.
    """
    if not scope.enabled or not scope.state_root:
        return ""
    home = Path(scope.state_root) / CODEX_HOME_SUBDIR
    if create:
        home.mkdir(parents=True, exist_ok=True)
    return str(home)


def scope_workspace_cwd(scope: "WorkerIsolationScope") -> str:
    """The working directory an isolated worker subprocess must start in."""
    if not scope.enabled:
        return ""
    workspace = scope.primary_workspace
    if not workspace:
        raise IsolationConfigError(
            "an enabled worker scope must declare at least one workspace"
        )
    return workspace


def assert_cwd_in_scope(scope: "WorkerIsolationScope", cwd: str | Path) -> str:
    """Validate a requested worker ``cwd`` against the scope's workspaces.

    Returns the canonical path.  A disabled scope accepts anything and returns
    the input unchanged, preserving today's behaviour exactly.
    """
    if not scope.enabled:
        return str(cwd)
    resolved = resolve_workspace_path(cwd)
    if not is_path_contained(resolved, scope.workspaces):
        allowed = ", ".join(scope.workspaces) or "(none)"
        raise PermissionError(
            f"worker cwd '{cwd}' is outside this agent's isolation boundary. "
            f"Allowed roots: {allowed}"
        )
    return str(resolved)


def apply_worker_scope_env(
    env: dict[str, str], scope: "WorkerIsolationScope"
) -> dict[str, str]:
    """Merge the serialized scope into a subprocess environment in place.

    A disabled scope contributes nothing, so an unisolated worker's
    environment is byte-identical to today's.
    """
    env.update(scope.to_env())
    return env


# ─────────────────────────────────────────────────────────────────────
# Authorization
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Authorization:
    """The result of one tool authorization decision."""

    allowed: bool
    tool_name: str = ""
    reason: str = ""
    denied_capability: ToolCapability | None = None
    policy_source: str = ""

    def __bool__(self) -> bool:
        return self.allowed

    def as_refusal(self) -> dict:
        """Stable structured refusal.

        Names the tool and the capability that was denied and nothing else —
        no credential paths, no list of roots the caller does not already know.
        """
        return {
            "error": "isolation_denied",
            "tool": self.tool_name,
            "capability": (
                self.denied_capability.value if self.denied_capability else None
            ),
            "reason": self.reason,
            "policy_source": self.policy_source,
        }


def authorize_tool(
    policy: IsolationPolicy | None,
    tool_name: str,
    capabilities: "Sequence[str | ToolCapability] | str | ToolCapability | None" = None,
) -> Authorization:
    """Decide whether ``tool_name`` may run under ``policy``.

    Rules, in order:

    1. A disabled (or absent) policy allows everything.  This is the inert
       legacy path and it does no work.
    2. ``mesh_control`` is never denied — cutting the control plane silences
       an agent without containing it.
    3. ``credential`` requires an explicit name in ``allowed_credential_tools``,
       regardless of network policy.
    4. ``external_network`` requires ``allow_network`` and, when
       ``allowed_network_tools`` is a list, membership in it.  ``None`` means
       "all external-network tools" and only applies when network is allowed.

    A tool carrying several capabilities must satisfy all of them.
    """
    if policy is None or not policy.enabled:
        return Authorization(allowed=True, tool_name=tool_name)

    caps = _normalize_capabilities(capabilities)
    source = policy.source

    if ToolCapability.CREDENTIAL in caps:
        allowed_creds = policy.allowed_credential_tools or ()
        if tool_name not in allowed_creds:
            return Authorization(
                allowed=False,
                tool_name=tool_name,
                reason=(
                    f"{tool_name} is credential-bearing and is not listed in "
                    "isolation.allowed_credential_tools"
                ),
                denied_capability=ToolCapability.CREDENTIAL,
                policy_source=source,
            )

    if ToolCapability.EXTERNAL_NETWORK in caps:
        if not policy.allow_network:
            return Authorization(
                allowed=False,
                tool_name=tool_name,
                reason=(
                    f"{tool_name} requires external network access and "
                    "isolation.allow_network is false"
                ),
                denied_capability=ToolCapability.EXTERNAL_NETWORK,
                policy_source=source,
            )
        allowlist = policy.allowed_network_tools
        if allowlist is not None and tool_name not in allowlist:
            return Authorization(
                allowed=False,
                tool_name=tool_name,
                reason=(
                    f"{tool_name} is not listed in isolation.allowed_network_tools"
                ),
                denied_capability=ToolCapability.EXTERNAL_NETWORK,
                policy_source=source,
            )

    return Authorization(allowed=True, tool_name=tool_name, policy_source=source)


def _normalize_capabilities(
    capabilities: "Sequence[str | ToolCapability] | str | ToolCapability | None",
) -> frozenset[ToolCapability]:
    if capabilities is None:
        return frozenset({ToolCapability.LOCAL})
    if isinstance(capabilities, (str, ToolCapability)):
        capabilities = [capabilities]
    parsed = {ToolCapability.parse(item) for item in capabilities}
    return frozenset(parsed or {ToolCapability.LOCAL})
