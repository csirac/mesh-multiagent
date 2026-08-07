"""Locked read/modify/write access to the standing digest file.

The digest is a single file mutated by several independent writers: the
``digest_edit`` tool on a message turn, the curation drain loop, and the
nightly fold driver.  Once curation stops sharing ``_router_turn_lock`` with
message processing those writers genuinely overlap, so a read-then-write pair
that is not held under one exclusive ``flock`` loses updates: both writers
snapshot the same baseline, each computes a replacement against it, and the
last writer's ``truncate``/``write`` erases the other's edit.

Every function here therefore does the *whole* read-modify-write inside a
single ``flock`` critical section, and every reader takes a shared lock so it
can never observe a half-written file.
"""

from __future__ import annotations

import fcntl
import os
from typing import IO


class DigestNotFound(OSError):
    """The digest path does not exist."""


def _read_locked(handle: IO[str]) -> str:
    """Read an already-open handle under a shared lock."""
    fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
    try:
        handle.seek(0)
        return handle.read()
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_digest(path: str) -> str:
    """Return the digest text, read under a shared lock.

    Raises ``FileNotFoundError`` if the path is absent and ``OSError`` for any
    other IO failure, so callers keep their existing error branches.
    """
    with open(path, encoding="utf-8") as handle:
        return _read_locked(handle)


def read_digest_or_empty(path: str) -> str:
    """Best-effort locked read; "" when the digest is missing or unreadable.

    Prompt-composition callers degrade to their pre-digest pathway on "", so a
    transient IO error must never raise into prompt assembly.
    """
    if not path:
        return ""
    try:
        return read_digest(path)
    except OSError:
        return ""


def edit_digest(
    path: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
) -> tuple[bool, str, int]:
    """Exact string replacement in the digest, atomically.

    The read, the match check, the replacement and the write all happen inside
    one ``LOCK_EX`` critical section, so a concurrent editor either runs wholly
    before or wholly after this call and neither loses its edit.

    Returns ``(ok, message, n_replaced)``.  ``ok=False`` carries a
    user-facing error message and ``n_replaced=0``; nothing was written.
    """
    try:
        handle = open(path, "r+", encoding="utf-8")
    except FileNotFoundError:
        return False, f"Error: digest file not found at {path}.", 0
    except OSError as exc:
        return False, f"Error reading digest: {exc}", 0

    try:
        with handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                # Re-read *inside* the lock.  Anything read before acquiring it
                # may already be stale by the time the lock is granted.
                handle.seek(0)
                content = handle.read()

                count = content.count(old_text)
                if count == 0:
                    return False, "Error: old_text not found in digest.", 0
                if not replace_all and count > 1:
                    return (
                        False,
                        f"Error: old_text matches {count} locations in digest — "
                        f"provide a more specific string or set replace_all=true.",
                        0,
                    )

                if replace_all:
                    new_content = content.replace(old_text, new_text)
                    n_replaced = count
                else:
                    new_content = content.replace(old_text, new_text, 1)
                    n_replaced = 1

                handle.seek(0)
                handle.write(new_content)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        return False, f"Error writing digest: {exc}", 0

    return True, "", n_replaced
