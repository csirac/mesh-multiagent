"""Formatting and history hygiene for the human-only tool audit block."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
import re


TOOLS_CALLED_RE = re.compile(r"\n*\[Tools called: [^\]]*\]\s*$")


_HARNESS_TOOL_ALIASES = {
    "bash": "bash_exec",
    "shell": "bash_exec",
    "read": "file_read",
    "edit": "file_edit",
    "write": "file_write",
    "patch": "apply_patch",
}


def normalize_tool_visibility_name(name: str) -> str:
    """Return a compact, source-independent name for an observed tool call."""
    normalized = str(name or "unknown")
    for prefix in ("cc:", "codex:", "harness:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    if normalized.startswith("mcp:"):
        normalized = normalized[len("mcp:"):]
    if "__" in normalized:
        normalized = normalized.rsplit("__", 1)[-1]
    return _HARNESS_TOOL_ALIASES.get(normalized.lower(), normalized)


def format_tools_called_block(
    tools: Iterable[str | tuple[str, str]],
) -> str:
    """Format a deduplicated Contract §5 visibility block.

    Counts preserve first-appearance ordering. Argument summaries are normally
    omitted; the resolved ``worker_launch`` selection stamp is the sole
    exception because task-type resolution and user overrides are auditable
    authority decisions.
    """
    counts: OrderedDict[str, int] = OrderedDict()
    selection_stamps: dict[str, str] = {}
    for item in tools:
        raw_name = item[0] if isinstance(item, tuple) else item
        name = normalize_tool_visibility_name(raw_name)
        counts[name] = counts.get(name, 0) + 1
        if (
            name == "worker_launch"
            and isinstance(item, tuple)
            and str(item[1]).startswith(("type=", "backend="))
        ):
            selection_stamps[name] = str(item[1])
    if not counts:
        return ""
    parts = []
    for name, count in counts.items():
        label = f"{name} (x{count})" if count > 1 else name
        if name in selection_stamps:
            label = f"{label} — {selection_stamps[name]}"
        parts.append(label)
    return f"[Tools called: {', '.join(parts)}]"


def append_tools_called_block(
    content: str,
    tools: Iterable[str | tuple[str, str]],
) -> str:
    """Append one visibility block unless the content already has one."""
    block = format_tools_called_block(tools)
    if not block or TOOLS_CALLED_RE.search(content):
        return content
    return f"{content.rstrip()}\n\n{block}"


def strip_tools_called_block(content: str) -> str:
    """Remove the human-only audit block before model context construction."""
    return TOOLS_CALLED_RE.sub("", content)
