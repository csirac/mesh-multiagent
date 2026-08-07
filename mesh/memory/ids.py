"""Canonical memory-ID normalization at the tool-argument boundary.

``memories.id`` stores a bare lowercase 12-hex string (``uuid4().hex[:12]``).
The ``m_`` prefix and the surrounding brackets belong to the *citation
surface* only — the form a memory is rendered in for the model, ``[m_3f723ef2c694]``.
Curation instructs the model to copy that handle exactly as shown, so every
tool argument naming a memory arrives in citation form while every lookup is
an exact match against the bare ID.  This module is the single place that
bridges the two.

Two properties matter more than convenience:

* **Shape-only.**  Normalization strips the citation surface and nothing else.
  It never repairs a mistyped ID, so a transcription typo still fails the
  existence check downstream instead of silently resolving to some *other*
  memory.
* **Closed accept-list.**  Exactly three forms are accepted — bare
  ``0123456789ab``, the handle ``m_0123456789ab``, and the fully bracketed
  ``[m_0123456789ab]``.  Everything else (wrong length, non-hex, uppercase,
  other prefixes such as ``mem-``, a half-bracketed token) is rejected
  explicitly rather than substring-stripped into something that happens to
  match a row.

Sibling validators that already do this by hand: the digest citation gate in
``mesh/agent_node.py`` and the essay citation validator in
``mesh/memory/entities.py``.
"""

from __future__ import annotations

import re

__all__ = [
    "MEMORY_ID_RE",
    "MemoryIdError",
    "normalize_memory_id",
    "try_normalize_memory_id",
]


#: The stored shape: bare, lowercase, exactly 12 hex digits.
MEMORY_ID_RE = re.compile(r"\A[0-9a-f]{12}\Z")

#: The complete set of accepted argument shapes.  Anchored, no alternation
#: that could match a substring of a longer token.
_ACCEPTED_MEMORY_ID_RE = re.compile(
    r"\A(?:"
    r"\[m_(?P<bracketed>[0-9a-f]{12})\]"  # [m_0123456789ab]
    r"|m_(?P<handle>[0-9a-f]{12})"        # m_0123456789ab
    r"|(?P<bare>[0-9a-f]{12})"            # 0123456789ab
    r")\Z"
)


class MemoryIdError(ValueError):
    """A memory-ID argument was not one of the accepted canonical forms.

    Subclasses ``ValueError`` so existing ``except Exception`` handlers on the
    correction path surface it as an ordinary tool error string.
    """


def try_normalize_memory_id(value: object) -> str | None:
    """Return the bare 12-hex ID for an accepted form, else ``None``.

    Surrounding whitespace is trimmed first; that cannot change *which*
    memory is named, only whether the token is well-formed at all.
    """
    if not isinstance(value, str):
        return None
    match = _ACCEPTED_MEMORY_ID_RE.match(value.strip())
    if match is None:
        return None
    return match.group("bracketed") or match.group("handle") or match.group("bare")


def normalize_memory_id(value: object, *, field: str = "memory_id") -> str:
    """Return the bare 12-hex ID, raising :class:`MemoryIdError` otherwise.

    Use this wherever a model-supplied memory reference crosses into a lookup.
    A rejected value is a malformed *argument*, which is a different failure
    from a well-formed ID that names no row — the caller reports that second
    case as "unknown memory ID" after the existence check.
    """
    normalized = try_normalize_memory_id(value)
    if normalized is None:
        raise MemoryIdError(
            f"malformed {field} {value!r}; expected a bare 12-hex memory ID "
            f"or its citation handle m_<id> / [m_<id>]"
        )
    return normalized
