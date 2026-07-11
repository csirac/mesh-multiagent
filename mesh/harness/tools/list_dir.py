"""
list_dir tool — tree-style directory listing.

Mirrors Codex's list_dir: shows directory contents with indentation,
type suffixes (/ for dirs, @ for symlinks), and pagination.
"""

from __future__ import annotations

import os
from pathlib import Path

from ...tools import tool, ToolParameter


def _list_tree(
    dir_path: str,
    depth: int = 2,
    offset: int = 1,
    limit: int = 25,
) -> str:
    """Build a tree listing of a directory."""
    root = Path(dir_path).resolve()
    if not root.is_dir():
        return f"Error: Not a directory: {dir_path}"

    entries: list[str] = []
    truncated = False

    def _walk(path: Path, current_depth: int, prefix: str) -> None:
        nonlocal truncated
        if current_depth > depth:
            return
        if truncated:
            return

        try:
            children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            entries.append(f"{prefix}[permission denied]")
            return

        for child in children:
            if len(entries) >= offset - 1 + limit + 1:
                truncated = True
                return

            name = child.name
            if child.is_symlink():
                suffix = "@"
            elif child.is_dir():
                suffix = "/"
            elif not child.is_file():
                suffix = "?"
            else:
                suffix = ""

            entries.append(f"{prefix}{name}{suffix}")

            if child.is_dir() and not child.is_symlink() and current_depth < depth:
                _walk(child, current_depth + 1, prefix + "  ")

    _walk(root, 1, "  ")

    # Apply pagination
    start_idx = offset - 1
    page = entries[start_idx:start_idx + limit]

    lines = [f"Absolute path: {root}"]
    lines.extend(page)
    if truncated:
        lines.append(f"  (More than {limit} entries found, use offset to paginate)")
    return "\n".join(lines)


@tool(
    name="list_dir",
    description=(
        "List the contents of a directory as an indented tree. "
        "Shows files and subdirectories with type suffixes: / for directories, "
        "@ for symlinks. Supports depth control and pagination via offset/limit."
    ),
    parameters=[
        ToolParameter(
            name="dir_path",
            type="string",
            description="Path to the directory to list",
            required=True,
        ),
        ToolParameter(
            name="depth",
            type="integer",
            description="Maximum depth to recurse (default 2)",
            required=False,
            default=2,
        ),
        ToolParameter(
            name="offset",
            type="integer",
            description="1-indexed offset for pagination (default 1)",
            required=False,
            default=1,
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="Maximum entries to return (default 25)",
            required=False,
            default=25,
        ),
    ],
)
def list_dir(dir_path: str, depth: int = 2, offset: int = 1, limit: int = 25) -> str:
    """List directory contents as an indented tree."""
    from ...paths import resolve_path as _resolve_home
    dir_path = _resolve_home(dir_path)
    if not os.path.isabs(dir_path):
        dir_path = os.path.join(os.getcwd(), dir_path)
    return _list_tree(dir_path, depth=depth, offset=offset, limit=limit)
