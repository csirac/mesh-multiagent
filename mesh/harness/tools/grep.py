"""
grep tool — search file contents by regex pattern.

Read-only search tool for the observer phase. Wraps subprocess grep
to find matching lines across files, returning file:line:match triples.
"""

from __future__ import annotations

import os
import subprocess

from ...tools import tool, ToolParameter

MAX_RESULTS = 100


@tool(
    name="grep",
    description=(
        "Search file contents for a regex pattern. Returns matching lines with "
        "file paths and line numbers. Use this to find functions, classes, imports, "
        "or any text pattern across the codebase without reading entire files."
    ),
    parameters=[
        ToolParameter(
            name="pattern",
            type="string",
            description="Regex pattern to search for (passed to grep -E)",
            required=True,
        ),
        ToolParameter(
            name="path",
            type="string",
            description="File or directory to search in (default: current directory)",
            required=False,
            default=".",
        ),
        ToolParameter(
            name="include",
            type="string",
            description="Glob pattern to filter files (e.g. '*.py', '*.js')",
            required=False,
        ),
    ],
)
def grep(pattern: str, path: str = ".", include: str | None = None) -> str:
    """Search files for a regex pattern."""
    from ...paths import resolve_path as _resolve_home
    path = _resolve_home(path)
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)

    if not os.path.exists(path):
        return f"Error: Path not found: {path}"

    cmd = ["grep", "-rn", "-E", "--color=never"]
    if include:
        cmd.extend(["--include", include])
    cmd.extend(["--", pattern, path])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds. Try a more specific pattern or path."
    except Exception as e:
        return f"Error running grep: {e}"

    if result.returncode == 1:
        return "No matches found."
    if result.returncode != 0 and result.returncode != 1:
        err = result.stderr.strip()
        return f"Error: grep returned exit code {result.returncode}: {err}"

    lines = result.stdout.splitlines()
    total = len(lines)
    if total > MAX_RESULTS:
        lines = lines[:MAX_RESULTS]
        output = "\n".join(lines)
        output += f"\n\n({total} total matches, showing first {MAX_RESULTS}. Narrow your search with a more specific pattern or --include.)"
    else:
        output = "\n".join(lines)
        output += f"\n\n({total} matches)"

    return output
