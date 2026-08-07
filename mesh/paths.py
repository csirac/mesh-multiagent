"""
Central path resolution for mesh agents.

Some subprocess-backed agents run with a synthetic HOME, but all mesh state
(history, memory DBs, plans, config) must persist at the real user home.

This module provides the canonical home directory from /etc/passwd,
bypassing the $HOME environment variable entirely.
"""

import os
import pwd
from pathlib import Path


def real_home() -> Path:
    """Return the real user home directory from /etc/passwd, ignoring $HOME."""
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def resolve_path(path: str) -> str:
    """Expand ~ to the real user home, not the CC synthetic home.

    For ~otheruser, falls back to standard os.path.expanduser().
    """
    path = str(path)
    if path == "~":
        return str(real_home())
    if path.startswith("~/"):
        return str(real_home() / path[2:])
    # ~otheruser — use standard expansion
    if path.startswith("~"):
        return os.path.expanduser(path)
    return path


# Canonical directories — always under the real home
MESH_DIR = real_home() / ".mesh"
HISTORY_DIR = MESH_DIR / "history"
MEMORY_DIR = MESH_DIR / "memory"
MAPS_DIR = MEMORY_DIR / "maps"
SKILLS_DIR = MESH_DIR / "skills"

# Autonomous agent mode: project dossiers, immutable session reports, and the
# per-project worker-admission budget ledger.
DIGESTS_DIR = MESH_DIR / "digests"
REPORTS_DIR = MESH_DIR / "reports"
BUDGET_DIR = MESH_DIR / "budget"


# ─────────────────────────────────────────────────────────────────────
# Policy-aware resolution
# ─────────────────────────────────────────────────────────────────────
#
# The constants above are the legacy layout and must never change: an agent
# with isolation disabled has to keep landing on exactly those directories.
# The helper below is the only supported way to get a *scoped* layout, and it
# returns the legacy constants unchanged whenever isolation is off.


def resolve_state_paths(policy=None, *, create: bool = True):
    """Return the ``StatePaths`` layout for ``policy``.

    ``policy`` may be ``None`` or a disabled :class:`~mesh.isolation.IsolationPolicy`,
    in which case the historical constants in this module are returned verbatim
    and nothing is created.  An enabled policy yields a layout beneath its
    validated ``state_root``.

    Imported lazily so that ``mesh.paths`` stays dependency-free at import time.
    """
    from .isolation import StatePaths

    return StatePaths.from_policy(policy, create=create)


def legacy_state_paths():
    """The historical ``~/.mesh`` layout as a ``StatePaths`` object."""
    from .isolation import StatePaths

    return StatePaths.legacy_paths()
