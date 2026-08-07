"""get_context tool — read a bounded line window around one target line."""

from __future__ import annotations

import os

from ...tools import ToolParameter, tool

MAX_CONTEXT_RADIUS = 200


@tool(
    name="get_context",
    description=(
        "Read a numbered window around one 1-indexed line in a text file. "
        "Use after grep to inspect nearby code without calculating a file_read range."
    ),
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the text file (relative or absolute)",
            required=True,
        ),
        ToolParameter(
            name="line",
            type="integer",
            description="1-indexed line at the center of the window",
            required=True,
        ),
        ToolParameter(
            name="radius",
            type="integer",
            description=(
                "Lines above and below the target (default 20, maximum "
                f"{MAX_CONTEXT_RADIUS})"
            ),
            required=False,
            default=20,
        ),
    ],
)
def get_context(path: str, line: int, radius: int = 20) -> str:
    """Return numbered lines from ``line - radius`` through ``line + radius``."""
    if isinstance(line, bool) or not isinstance(line, int) or line < 1:
        return "Error: line must be a positive integer"
    if (
        isinstance(radius, bool)
        or not isinstance(radius, int)
        or not 0 <= radius <= MAX_CONTEXT_RADIUS
    ):
        return (
            "Error: radius must be a non-negative integer no greater than "
            f"{MAX_CONTEXT_RADIUS}"
        )

    from ...paths import resolve_path as _resolve_home

    resolved = _resolve_home(path)
    if not os.path.isabs(resolved):
        resolved = os.path.join(os.getcwd(), resolved)

    if not os.path.exists(resolved):
        return f"Error: File not found: {resolved}"
    if os.path.isdir(resolved):
        return f"Error: {resolved} is a directory, not a file. Use list_dir instead."

    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError as exc:
        return f"Error reading file: {exc}"

    total = len(lines)
    if line > total:
        return f"Error: line {line} out of range (file has {total} lines)"

    start = max(1, line - radius)
    end = min(total, line + radius)
    numbered = [
        f"{number:4d}|{lines[number - 1].rstrip(chr(10))}"
        for number in range(start, end + 1)
    ]
    return "\n".join(numbered) + f"\n\n({total} lines total)"
