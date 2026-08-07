"""Shared rendering helpers for multi-worker status lines.

The heartbeat payload carries ``worker_count`` and a ``workers`` array, but the
operator-facing surfaces historically printed only the primary compatibility
worker (``worker_elapsed_s``).  With more than one slot in flight that display
is actively misleading: it shows one worker while several are running.

These helpers are pure functions over the already-decoded status summary so
that every surface -- the curses TUI (``run_user_tui.py``), the ``mesh_status``
tool, and the GTK Linux client -- renders concurrency the same way.

Single-worker output is byte-for-byte identical to the previous behavior, so
capacity-1 agents (the current fleet default) see no change at all.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = [
    "worker_entries",
    "format_worker_state",
    "format_worker_detail_lines",
]


def _coerce_seconds(value: Any) -> float | None:
    """Best-effort numeric coercion; status payloads cross a JSON boundary."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def worker_entries(summary: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Return the ``workers`` array from a status summary, defensively.

    Older agents predate the array and send only ``worker_elapsed_s``.  In that
    case synthesize a single entry so callers have one uniform shape to render.
    """
    if not summary:
        return []

    raw = summary.get("workers")
    entries: list[dict[str, Any]] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if isinstance(item, Mapping):
                entries.append(dict(item))
    if entries:
        return entries

    # Legacy agent: fall back to the singleton compat view.
    elapsed = _coerce_seconds(summary.get("worker_elapsed_s"))
    if elapsed is not None:
        return [{"worker_id": summary.get("worker_id") or "worker",
                 "elapsed_s": elapsed,
                 "task_description": ""}]
    return []


def format_worker_state(
    summary: Mapping[str, Any] | None,
    state_upper: str,
) -> str:
    """Render the BUSY state token, accounting for concurrent workers.

    One worker  -> ``BUSY (45s)``            (unchanged legacy form)
    Two or more -> ``BUSY · 2 workers (45s, 12s)``
    """
    entries = worker_entries(summary)
    count = 0
    if summary is not None:
        raw_count = summary.get("worker_count")
        if isinstance(raw_count, int) and not isinstance(raw_count, bool):
            count = raw_count
    if not count:
        count = len(entries)

    elapsed_values = [
        _coerce_seconds(entry.get("elapsed_s")) for entry in entries
    ]
    elapsed_values = [v for v in elapsed_values if v is not None]

    if count <= 1:
        if elapsed_values:
            return f"{state_upper} ({int(elapsed_values[0])}s)"
        fallback = _coerce_seconds(
            summary.get("worker_elapsed_s") if summary else None
        )
        if fallback is not None:
            return f"{state_upper} ({int(fallback)}s)"
        return state_upper

    if elapsed_values:
        times = ", ".join(f"{int(v)}s" for v in elapsed_values)
        return f"{state_upper} · {count} workers ({times})"
    return f"{state_upper} · {count} workers"


def format_worker_detail_lines(
    summary: Mapping[str, Any] | None,
    indent: str = "",
) -> list[str]:
    """Render one line per active worker for verbose status surfaces.

    Returns an empty list when fewer than two workers are active -- the state
    token already says everything there is to say in that case.
    """
    entries = worker_entries(summary)
    if len(entries) < 2:
        return []

    lines: list[str] = []
    for entry in entries:
        worker_id = str(entry.get("worker_id") or "worker")
        elapsed = _coerce_seconds(entry.get("elapsed_s"))
        head = f"{worker_id} ({int(elapsed)}s)" if elapsed is not None else worker_id
        task = " ".join(str(entry.get("task_description") or "").split())
        lines.append(f"{indent}{head}: {task}" if task else f"{indent}{head}")
    return lines
