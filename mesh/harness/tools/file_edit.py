"""
file_edit tool — exact string replacement in files.

CC-style Edit tool: find old_string, replace with new_string.
Requires exact byte-for-byte match — no fuzzy matching.
Safety property: refuses if old_string matches more than once
(unless replace_all=True).
"""

from __future__ import annotations

import os

from ...tools import tool, ToolParameter


@tool(
    name="file_edit",
    description=(
        "Replace exact text in a file. "
        "Finds old_string in the file and replaces it with new_string. "
        "By default, old_string must appear exactly once in the file "
        "(to prevent accidental edits to the wrong location). "
        "Set replace_all=true to replace every occurrence. "
        "Use file_read first to see the file content and find the right text to replace."
    ),
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to edit (relative or absolute)",
            required=True,
        ),
        ToolParameter(
            name="old_string",
            type="string",
            description="The exact text to find and replace",
            required=True,
        ),
        ToolParameter(
            name="new_string",
            type="string",
            description="The replacement text",
            required=True,
        ),
        ToolParameter(
            name="replace_all",
            type="boolean",
            description="If true, replace all occurrences. Default: false (requires unique match).",
            required=False,
            default=False,
        ),
    ],
)
def file_edit(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Replace exact text in a file."""
    from ...paths import resolve_path as _resolve_home
    path = _resolve_home(path)
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)

    if not os.path.exists(path):
        return f"Error: File not found: {path}"
    if os.path.isdir(path):
        return f"Error: {path} is a directory, not a file."

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return f"Error reading file: {e}"

    count = content.count(old_string)

    if count == 0:
        preview = old_string[:80]
        if len(old_string) > 80:
            preview += "..."
        return (
            f"Error: old_string not found in {os.path.basename(path)}. "
            f"Searched for: {repr(preview)}"
        )

    if not replace_all and count > 1:
        return (
            f"Error: old_string found {count} times in {os.path.basename(path)}. "
            f"Use replace_all=true to replace all occurrences, "
            f"or provide more context to make the match unique."
        )

    if replace_all:
        new_content = content.replace(old_string, new_string)
    else:
        new_content = content.replace(old_string, new_string, 1)

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        return f"Error writing file: {e}"

    if replace_all and count > 1:
        return f"Replaced {count} occurrences in {os.path.basename(path)}"
    return f"Updated {os.path.basename(path)}"
