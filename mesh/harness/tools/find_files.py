"""
find_files tool — glob-based file discovery.

Read-only search tool for the observer phase. Uses pathlib.glob
to find files matching a pattern, returning sorted paths.
"""

from __future__ import annotations

import os
from pathlib import Path

from ...tools import tool, ToolParameter

MAX_RESULTS = 200


@tool(
    name="find_files",
    description=(
        "Find files matching a glob pattern. Returns sorted file paths. "
        "Use this to discover project structure, find test files, locate "
        "config files, etc. Supports recursive globs with '**'."
    ),
    parameters=[
        ToolParameter(
            name="pattern",
            type="string",
            description="Glob pattern (e.g. '*.py', '**/*.test.js', 'src/**/*.ts')",
            required=True,
        ),
        ToolParameter(
            name="path",
            type="string",
            description="Directory to search in (default: current directory)",
            required=False,
            default=".",
        ),
    ],
)
def find_files(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern."""
    from ...paths import resolve_path as _resolve_home
    path = _resolve_home(path)
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)

    root = Path(path)
    if not root.is_dir():
        return f"Error: Not a directory: {path}"

    try:
        matches = sorted(str(p) for p in root.glob(pattern) if p.is_file())
    except Exception as e:
        return f"Error: {e}"

    total = len(matches)
    if total == 0:
        return "No files found matching the pattern."

    if total > MAX_RESULTS:
        matches = matches[:MAX_RESULTS]
        output = "\n".join(matches)
        output += f"\n\n({total} total files, showing first {MAX_RESULTS}. Narrow your pattern.)"
    else:
        output = "\n".join(matches)
        output += f"\n\n({total} files)"

    return output
