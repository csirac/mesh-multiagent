"""
Central path resolution for mesh agents.

CC agents run with HOME=~/.claude-acctN, but all mesh state
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
