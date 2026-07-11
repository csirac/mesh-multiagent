"""
file_read tool — read files with explicit line ranges.

All range parameters are required so the model must always specify
exactly which slice it needs, preventing accidental full-file reads
that blow context budget.
"""

from __future__ import annotations

import os

from ...tools import tool, ToolParameter


@tool(
    name="file_read",
    description=(
        "Read a file's contents with line numbers. "
        "You MUST specify start_line and either num_lines or end_line (or both). "
        "start_line is 1-indexed. end_line is inclusive. "
        "If both num_lines and end_line are given, the range ends at whichever comes first."
    ),
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to read (relative or absolute)",
            required=True,
        ),
        ToolParameter(
            name="start_line",
            type="integer",
            description="Starting line number, 1-indexed",
            required=True,
        ),
        ToolParameter(
            name="num_lines",
            type="integer",
            description="Number of lines to read from start_line",
            required=True,
        ),
        ToolParameter(
            name="end_line",
            type="integer",
            description="Ending line number, inclusive (if both num_lines and end_line are given, the range ends at whichever comes first)",
            required=False,
        ),
    ],
)
def file_read(path: str, start_line: int = 1, num_lines: int = 200, end_line: int | None = None) -> str:
    """Read a file with line numbers."""
    from ...paths import resolve_path as _resolve_home
    path = _resolve_home(path)
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)

    if not os.path.exists(path):
        return f"Error: File not found: {path}"
    if os.path.isdir(path):
        return f"Error: {path} is a directory, not a file. Use list_dir instead."

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading file: {e}"

    total = len(lines)
    start_idx = max(0, int(start_line) - 1)

    end_from_num = start_idx + int(num_lines)
    if end_line is not None:
        end_from_end = int(end_line)
        end_idx = min(total, end_from_num, end_from_end)
    else:
        end_idx = min(total, end_from_num)

    selected = lines[start_idx:end_idx]
    line_offset = start_idx

    numbered = []
    for i, line in enumerate(selected, start=line_offset + 1):
        line = line.rstrip("\n")
        numbered.append(f"{i:4d}|{line}")

    result = "\n".join(numbered)
    result += f"\n\n({total} lines total)"
    return result
