"""
Tool implementations for mesh agents.

This module provides tool implementations for mesh agents.
Tool clients are now self-contained in the mesh.clients package.

Tools are organized into categories:
- Bash/Shell
- File operations
- Web search (Exa)
- Email (Gmail)
- Calendar
- Notes
- Browser
- Account management
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from .tools import tool, ToolParameter, get_registry

# =============================================================================
# Client singletons (lazy initialization)
# =============================================================================

_bash_tools = None
_bash_working_directory = None  # Working directory for bash commands
_memory_system = None  # MemorySystem singleton, set by AgentNode on init
_memory_search_mode = "hybrid"  # default search mode, set from agent config
_exa_client = None
_tool_host = None
_browser_client = None
_scholar_client = None

# Sandbox settings (module-level, set by agent_node)
_sandboxed: bool = False
_allowed_dirs: list[str] = []
_allow_network: bool = True


def configure_sandbox(
    sandboxed: bool = False,
    allowed_dirs: list[str] | None = None,
    allow_network: bool = True
) -> None:
    """
    Configure sandbox settings for file and bash tools.

    Called by AgentNode when initializing with sandbox options.
    """
    global _sandboxed, _allowed_dirs, _allow_network, _bash_tools
    _sandboxed = sandboxed
    _allowed_dirs = allowed_dirs or []
    _allow_network = allow_network

    # Reset bash_tools so it gets recreated with new settings
    _bash_tools = None


def _validate_path(path: str, require_write: bool = False) -> str:
    """
    Validate and resolve a path against sandbox restrictions.

    Args:
        path: The path to validate
        require_write: If True, path must be in allowed_dirs (for write operations)

    Returns:
        The resolved absolute path

    Raises:
        PermissionError: If sandboxed and path is not in allowed directories
    """
    # Expand ~ and resolve to absolute path (using real home, not CC home)
    from .paths import resolve_path as _rp
    expanded = _rp(path)
    resolved = Path(expanded).resolve()

    if not _sandboxed:
        return str(resolved)

    # In sandbox mode, check if path is in allowed directories
    for allowed in _allowed_dirs:
        allowed_resolved = Path(_rp(allowed)).resolve()
        try:
            resolved.relative_to(allowed_resolved)
            return str(resolved)
        except ValueError:
            continue

    # Also allow /tmp
    try:
        resolved.relative_to(Path("/tmp").resolve())
        return str(resolved)
    except ValueError:
        pass

    if require_write:
        raise PermissionError(
            f"Path '{path}' is not in allowed directories. "
            f"Allowed: {_allowed_dirs + ['/tmp']}"
        )
    else:
        # For read operations, we're more permissive but still log
        # Actually, let's be consistent - sandbox means sandbox
        raise PermissionError(
            f"Path '{path}' is not in allowed directories. "
            f"Allowed: {_allowed_dirs + ['/tmp']}"
        )


def _get_bash_tools():
    """Get or create BashTools instance."""
    global _bash_tools
    if _bash_tools is None:
        from .clients.bash_tools import BashTools
        _bash_tools = BashTools(
            user_confirm=False,  # No CLI confirmation in mesh context
            timeout_sec=30.0,
            max_output_chars=100000,
            sandboxed=_sandboxed,
            allowed_dirs=_allowed_dirs,
            allow_network=_allow_network,
        )
    return _bash_tools


def _get_exa_client():
    """Get or create ExaSearchClient instance.

    Re-checks API key on each call and reinitializes if key changed.
    """
    global _exa_client
    from .clients.exa_client import ExaSearchClient
    api_key = os.environ.get("EXA_API_KEY")

    # Always reinitialize if no client or if key changed
    if _exa_client is None:
        _exa_client = ExaSearchClient(api_key)
    elif api_key and not _exa_client.is_available():
        # Key wasn't set before but is now - reinitialize
        _exa_client = ExaSearchClient(api_key)

    return _exa_client


def _get_tool_host():
    """Get or create ToolHost instance for Gmail/Calendar/Notes."""
    global _tool_host
    if _tool_host is None:
        from .clients.account_manager import ToolHost
        # Use pwd-based home to avoid CC fallback HOME override pollution
        import pwd
        real_home = pwd.getpwuid(os.getuid()).pw_dir
        config_path = os.path.join(real_home, ".config", "mesh", "accounts.json")
        if os.path.exists(config_path):
            _tool_host = ToolHost(config_path, confirmation_mode="cli")
        else:
            _tool_host = None
    return _tool_host


def _get_browser_client():
    """Get or create BrowserClient instance."""
    global _browser_client
    if _browser_client is None:
        # BrowserClient requires async initialization, so we just store a marker
        # and create it on first async use
        _browser_client = "pending"
    return _browser_client


async def _ensure_browser_client():
    """Ensure browser client is initialized (async)."""
    global _browser_client
    if _browser_client == "pending" or _browser_client is None:
        from .clients.browser_client_minimal import BrowserClient
        _browser_client = BrowserClient()
    return _browser_client


def _get_scholar_client():
    """Get or create ScholarToolClient instance for literature search."""
    global _scholar_client
    if _scholar_client is None:
        from .literature import ScholarToolClient
        _scholar_client = ScholarToolClient()
    return _scholar_client


# =============================================================================
# TOOL HELP
# =============================================================================


@tool(
    name="sleep",
    description=(
        "Logical no-op indicating the agent intentionally chose not to send "
        "any messages in response to the current trigger. Useful for channel "
        "messages or agent-only logs where no user-facing reply is needed."
    ),
    parameters=[
        ToolParameter(
            name="reason",
            type="string",
            description=(
                "Short explanation of why no response was needed (for logs only)."
            ),
            required=True,
        ),
    ],
)
def sleep(reason: str) -> str:
    """Record an intentional decision to stay quiet (no-op)."""
    # We don't actually delay execution; this is a logical marker only.
    return json.dumps({"status": "ok", "reason": reason})


@tool(
    name="tool_help",
    description="Get detailed help and syntax for a specific tool, or list all available tools.",
    parameters=[
        ToolParameter(
            name="tool_name",
            type="string",
            description="Name of the tool to get help for. Use 'list' to see all available tools.",
            required=True,
        ),
    ],
)
def tool_help(tool_name: str) -> str:
    """Get detailed help for a tool."""
    registry = get_registry()

    if tool_name.lower() == "list":
        names = sorted(registry.list_names())
        return "Available tools:\n" + "\n".join(f"- {n}" for n in names)

    return registry.get_tool_help(tool_name)


# =============================================================================
# SCHEDULED WAKES
# =============================================================================
# These tools are intercepted and executed by the agent directly (like send_message).
# The handlers here are stubs - actual execution happens in AgentNode.


@tool(
    name="schedule_wake",
    description=(
        "Schedule a future wake-up with a prompt. At the specified time, "
        "you will receive a message with the prompt, triggering LLM processing "
        "with full conversation context. Use this for reminders, delayed tasks, "
        "or time-sensitive checks.\n\n"
        "Time formats supported:\n"
        "- ISO 8601: '2026-01-26T17:00:00-06:00'\n"
        "- Relative: 'in 30 minutes', 'in 2 hours', 'in 1 day'\n"
        "- Natural time: '5pm', '17:00', '5:30pm' (uses local timezone)\n\n"
        "Optional recurrence makes the wake repeat automatically. "
        "Supported rules: 'daily', 'weekly', 'weekdays', 'hourly', "
        "'every N minutes', 'every N hours', 'every N days'. "
        "Cancel the wake ID to stop the series."
    ),
    parameters=[
        ToolParameter(
            name="wake_time",
            type="string",
            description=(
                "When to wake. Accepts ISO 8601 timestamps, relative times "
                "(e.g., 'in 30 minutes'), or natural times (e.g., '5pm')."
            ),
            required=True,
        ),
        ToolParameter(
            name="prompt",
            type="string",
            description="The prompt to receive at wake time. Include enough context for meaningful action.",
            required=True,
        ),
        ToolParameter(
            name="recurrence",
            type="string",
            description=(
                "Optional. Makes the wake recurring. Supported: 'daily', 'weekly', "
                "'weekdays', 'hourly', 'every N minutes', 'every N hours', 'every N days'."
            ),
            required=False,
        ),
    ],
)
def schedule_wake(wake_time: str, prompt: str, recurrence: str = "") -> str:
    """Schedule a wake-up (stub - intercepted by agent)."""
    return json.dumps({"status": "error", "error": "schedule_wake must be executed by an agent"})


@tool(
    name="schedule_list",
    description="List all pending scheduled wakes. Shows wake IDs, times, and prompt previews.",
    parameters=[],
)
def schedule_list() -> str:
    """List scheduled wakes (stub - intercepted by agent)."""
    return json.dumps({"status": "error", "error": "schedule_list must be executed by an agent"})


@tool(
    name="schedule_cancel",
    description="Cancel a scheduled wake by its ID. Use schedule_list to see pending wake IDs.",
    parameters=[
        ToolParameter(
            name="wake_id",
            type="string",
            description="The ID of the scheduled wake to cancel (e.g., 'wake-abc123').",
            required=True,
        ),
    ],
)
def schedule_cancel(wake_id: str) -> str:
    """Cancel a scheduled wake (stub - intercepted by agent)."""
    return json.dumps({"status": "error", "error": "schedule_cancel must be executed by an agent"})


# =============================================================================
# BASH TOOL
# =============================================================================

@tool(
    name="set_working_directory",
    description="Set the working directory for subsequent bash commands. All bash_exec calls will run from this directory.",
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="The directory path to use as working directory",
            required=True,
        ),
    ],
)
def set_working_directory(path: str) -> str:
    """Set the working directory for bash commands."""
    global _bash_working_directory
    from .paths import resolve_path as _rp
    expanded = _rp(path)

    if not os.path.isdir(expanded):
        return json.dumps({"error": f"Directory does not exist: {path}"})

    _bash_working_directory = os.path.abspath(expanded)
    return json.dumps({"working_directory": _bash_working_directory})


@tool(
    name="get_working_directory",
    description="Get the current working directory for bash commands.",
    parameters=[],
)
def get_working_directory() -> str:
    """Get the current working directory."""
    global _bash_working_directory
    if _bash_working_directory:
        return json.dumps({"working_directory": _bash_working_directory})
    else:
        return json.dumps({"working_directory": os.getcwd(), "note": "Using process default"})


@tool(
    name="bash_exec",
    description=(
        "Execute a bash shell command and return stdout/stderr/exit code. "
        "Runs in the directory set by set_working_directory, or the process default. "
        "Note: the returncode in the result is the shell exit code, NOT a tool status. "
        "Exit code 0 = success; exit code 1 often means 'no match' or a false condition "
        "(e.g. grep finding no matches, test/diff returning false), not a failure; "
        "exit code 2+ usually indicates an actual error. A non-zero returncode does not "
        "mean the tool is broken — re-running the identical command will give the same result."
    ),
    parameters=[
        ToolParameter(
            name="command",
            type="string",
            description="The shell command to execute",
            required=True,
        ),
        ToolParameter(
            name="timeout",
            type="number",
            description="Timeout in seconds (default 30)",
            required=False,
            default=30,
        ),
    ],
)
def bash_exec(command: str, timeout: float = 30) -> str:
    """Execute a bash command."""
    global _bash_working_directory
    bt = _get_bash_tools()
    bt.timeout_sec = float(timeout)

    # Prepend cd if working directory is set
    if _bash_working_directory:
        command = f'cd {_bash_working_directory!r} && {command}'

    # Use sandboxed execution if enabled
    if bt.sandboxed:
        result = bt._run_sandboxed_command(command)
    else:
        result = bt._run_command(command)
    return json.dumps(result, ensure_ascii=False)


# =============================================================================
# FILE TOOLS
# =============================================================================

def _resolve_path(path: str, require_write: bool = False) -> str:
    """Resolve a file path, respecting the current working directory and sandbox.

    - Expands ~ to home directory
    - If path is relative and a working directory is set, resolves relative to it
    - Returns an absolute path
    - If sandboxed, validates path is in allowed directories

    Args:
        path: The path to resolve
        require_write: If True, indicates this is a write operation (for clearer errors)

    Raises:
        PermissionError: If sandboxed and path is not in allowed directories
    """
    global _bash_working_directory
    from .paths import resolve_path as _rp

    # First expand ~ (using real home, not CC home)
    expanded = _rp(path)

    # Resolve to absolute path
    if os.path.isabs(expanded):
        resolved = Path(expanded).resolve()
    elif _bash_working_directory:
        resolved = Path(os.path.join(_bash_working_directory, expanded)).resolve()
    else:
        resolved = Path(expanded).resolve()

    # Apply sandbox validation if enabled
    if _sandboxed:
        allowed = False

        # Check against allowed directories
        for allowed_dir in _allowed_dirs:
            allowed_resolved = Path(_rp(allowed_dir)).resolve()
            try:
                resolved.relative_to(allowed_resolved)
                allowed = True
                break
            except ValueError:
                continue

        # Also allow /tmp
        if not allowed:
            try:
                resolved.relative_to(Path("/tmp").resolve())
                allowed = True
            except ValueError:
                pass

        if not allowed:
            op = "write to" if require_write else "access"
            raise PermissionError(
                f"Cannot {op} '{path}': not in allowed directories. "
                f"Allowed: {_allowed_dirs + ['/tmp']}"
            )

    return str(resolved)


@tool(
    name="file_read",
    description="Read a file with line numbers. Use instead of cat/head/tail. "
                "You MUST specify start_line and either num_lines or end_line (or both). "
                "start_line is 1-indexed. end_line is inclusive. "
                "If both num_lines and end_line are given, the range ends at whichever comes first.",
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to read",
            required=True,
        ),
        ToolParameter(
            name="start_line",
            type="integer",
            description="Starting line number (1-indexed)",
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
    path = _resolve_path(path)

    if not os.path.exists(path):
        return f"Error: File not found: {path}"

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading file: {e}"

    total_lines = len(lines)
    start_idx = max(0, int(start_line) - 1)

    end_from_num = start_idx + int(num_lines)
    if end_line is not None:
        end_from_end = int(end_line)
        end_idx = min(total_lines, end_from_num, end_from_end)
    else:
        end_idx = min(total_lines, end_from_num)

    selected_lines = lines[start_idx:end_idx]
    line_offset = start_idx

    numbered = []
    for i, line in enumerate(selected_lines, start=line_offset + 1):
        line = line.rstrip('\n')
        numbered.append(f"{i:4d}│{line}")

    result = "\n".join(numbered)
    result += f"\n\n({total_lines} lines total)"
    return result


@tool(
    name="file_edit",
    description="Perform exact string replacement in a file. Use file_read first to see the content.",
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to edit",
            required=True,
        ),
        ToolParameter(
            name="old_string",
            type="string",
            description="The exact string to find and replace",
            required=True,
        ),
        ToolParameter(
            name="new_string",
            type="string",
            description="The replacement string",
            required=True,
        ),
        ToolParameter(
            name="replace_all",
            type="boolean",
            description="Replace all occurrences (default false)",
            required=False,
            default=False,
        ),
    ],
)
def file_edit(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Edit a file by replacing exact strings."""
    path = _resolve_path(path, require_write=True)

    if not os.path.exists(path):
        return f"Error: File not found: {path}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"Error reading file: {e}"

    # Count occurrences
    count = content.count(old_string)

    if count == 0:
        return f"Error: old_string not found in file. Make sure you're using the exact string."

    if count > 1 and not replace_all:
        return f"Error: old_string found {count} times. Use replace_all=true to replace all, or provide more context to make it unique."

    # Perform replacement
    if replace_all:
        new_content = content.replace(old_string, new_string)
    else:
        new_content = content.replace(old_string, new_string, 1)

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        return f"Error writing file: {e}"

    return f"Successfully replaced {count if replace_all else 1} occurrence(s) in {path}"


@tool(
    name="file_create",
    description="Create a new file with the given content.",
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path for the new file",
            required=True,
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Content to write to the file",
            required=True,
        ),
    ],
)
def file_create(path: str, content: str) -> str:
    """Create a new file."""
    path = _resolve_path(path, require_write=True)

    if os.path.exists(path):
        return f"Error: File already exists: {path}. Use file_edit to modify it."

    # Create parent directories if needed
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        return f"Error creating file: {e}"

    return f"Successfully created {path}"


@tool(
    name="file_write",
    description="""Write content to a file, creating it if it doesn't exist or overwriting if it does.

Use this tool when you need to:
- Replace an entire file with new content
- Create a new file (same as file_create but allows overwriting)
- Rewrite a file after major changes

For small edits to existing files, prefer file_edit.""",
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to write to (will overwrite if exists)",
            required=True,
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Content to write to the file",
            required=True,
        ),
    ],
)
def file_write(path: str, content: str) -> str:
    """Write content to file, overwriting if exists."""
    path = _resolve_path(path, require_write=True)

    # Create parent directories if needed
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    existed = os.path.exists(path)

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        return f"Error writing file: {e}"

    action = "Overwrote" if existed else "Created"
    return f"Successfully {action.lower()} {path} ({len(content)} bytes)"


@tool(
    name="file_diff",
    description="""Apply a unified diff patch to a file.

Accepts standard unified diff format (like `diff -u` or `git diff` output).
Supports fuzzy context matching for minor whitespace differences.

Example diff format:
```
--- a/file.py
+++ b/file.py
@@ -10,4 +10,5 @@
 def hello():
-    print("old")
+    print("new")
+    return True

```

Use this tool when:
- Making multiple related edits to a file
- The exact string match required by file_edit is too strict
- You want to express changes in a familiar diff format

The tool will report which hunks succeeded/failed with context.""",
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to patch",
            required=True,
        ),
        ToolParameter(
            name="diff",
            type="string",
            description="Unified diff content to apply",
            required=True,
        ),
        ToolParameter(
            name="fuzz",
            type="integer",
            description="Fuzz factor for context matching (0=exact, 1=ignore leading/trailing whitespace, 2=normalize all whitespace). Default 1.",
            required=False,
            default=1,
        ),
    ],
)
def file_diff(path: str, diff: str, fuzz: int = 1) -> str:
    """Apply unified diff to a file with fuzzy matching."""
    import re
    from difflib import SequenceMatcher

    path = _resolve_path(path, require_write=True)

    if not os.path.exists(path):
        return f"Error: File does not exist: {path}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            original_content = f.read()
    except Exception as e:
        return f"Error reading file: {e}"

    original_lines = original_content.splitlines(keepends=True)
    # Ensure last line has newline for consistent matching
    if original_lines and not original_lines[-1].endswith('\n'):
        original_lines[-1] += '\n'

    # Parse unified diff into hunks
    hunks = _parse_unified_diff(diff)
    if isinstance(hunks, str):  # Error message
        return hunks

    if not hunks:
        return "Error: No valid hunks found in diff"

    # Apply hunks (track offset as we modify)
    result_lines = list(original_lines)
    offset = 0
    applied = []
    failed = []

    for i, hunk in enumerate(hunks):
        success, new_lines, new_offset, msg = _apply_hunk(
            result_lines, hunk, offset, fuzz
        )
        if success:
            result_lines = new_lines
            offset = new_offset
            applied.append(i + 1)
        else:
            failed.append((i + 1, msg))

    if failed and not applied:
        # All hunks failed - don't modify file
        error_details = "\n".join(f"  Hunk {n}: {msg}" for n, msg in failed)
        return f"Error: All hunks failed to apply:\n{error_details}"

    # Write result
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(result_lines))
    except Exception as e:
        return f"Error writing file: {e}"

    # Build result message
    result_msg = f"Patched {path}: {len(applied)} hunk(s) applied"
    if failed:
        result_msg += f", {len(failed)} failed"
        for n, msg in failed:
            result_msg += f"\n  Hunk {n} failed: {msg}"

    return result_msg


def _parse_unified_diff(diff_text: str) -> list | str:
    """Parse unified diff into list of hunks.

    Returns list of hunks or error string.
    Each hunk is dict with: start_line, context_before, removals, additions, context_after
    """
    import re

    lines = diff_text.splitlines(keepends=True)
    # Ensure lines end with newline
    lines = [l if l.endswith('\n') else l + '\n' for l in lines]

    hunks = []
    i = 0

    # Skip header lines (---, +++, etc)
    while i < len(lines):
        line = lines[i]
        if line.startswith('@@'):
            break
        i += 1

    # Parse hunks
    hunk_header_re = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')

    while i < len(lines):
        line = lines[i]
        match = hunk_header_re.match(line)
        if not match:
            i += 1
            continue

        old_start = int(match.group(1))
        old_count = int(match.group(2)) if match.group(2) else 1
        new_start = int(match.group(3))
        new_count = int(match.group(4)) if match.group(4) else 1

        i += 1

        # Collect hunk lines
        context_before = []
        removals = []
        additions = []
        context_after = []
        in_change = False

        while i < len(lines) and not lines[i].startswith('@@'):
            hunk_line = lines[i]
            if hunk_line.startswith(' '):
                content = hunk_line[1:]
                if in_change:
                    context_after.append(content)
                else:
                    context_before.append(content)
            elif hunk_line.startswith('-'):
                in_change = True
                context_after = []  # Reset - context_after only after all changes
                removals.append(hunk_line[1:])
            elif hunk_line.startswith('+'):
                in_change = True
                context_after = []
                additions.append(hunk_line[1:])
            elif hunk_line.startswith('\\'):
                # "\ No newline at end of file" - skip
                pass
            else:
                # Unknown line, stop parsing this hunk
                break
            i += 1

        hunks.append({
            'old_start': old_start,
            'old_count': old_count,
            'context_before': context_before,
            'removals': removals,
            'additions': additions,
            'context_after': context_after,
        })

    return hunks


def _normalize_for_match(line: str, fuzz: int) -> str:
    """Normalize a line for fuzzy matching."""
    if fuzz == 0:
        return line
    elif fuzz == 1:
        return line.strip() + '\n'
    else:  # fuzz >= 2
        return ' '.join(line.split()) + '\n'


def _find_hunk_location(lines: list, hunk: dict, offset: int, fuzz: int) -> int | None:
    """Find where a hunk should be applied.

    Returns line index (0-based) or None if not found.
    """
    # Lines to match: context_before + removals + context_after
    # (all lines that appear in the "old" side of the diff)
    match_lines = hunk['context_before'] + hunk['removals'] + hunk['context_after']
    if not match_lines:
        # Pure addition - use line number hint
        return max(0, hunk['old_start'] - 1 + offset)

    # Normalize for matching
    norm_match = [_normalize_for_match(l, fuzz) for l in match_lines]

    # Start searching near expected location
    expected = hunk['old_start'] - 1 + offset
    search_range = 50  # Look up to 50 lines away

    for delta in range(search_range):
        for sign in [0, -1, 1] if delta == 0 else [-1, 1]:
            pos = expected + delta * sign
            if pos < 0 or pos + len(match_lines) > len(lines):
                continue

            # Check if lines match
            candidate = lines[pos:pos + len(match_lines)]
            norm_candidate = [_normalize_for_match(l, fuzz) for l in candidate]

            if norm_candidate == norm_match:
                return pos

    return None


def _apply_hunk(lines: list, hunk: dict, offset: int, fuzz: int) -> tuple:
    """Apply a single hunk to lines.

    Returns (success, new_lines, new_offset, message)
    """
    pos = _find_hunk_location(lines, hunk, offset, fuzz)

    if pos is None:
        # Build diagnostic message
        match_lines = hunk['context_before'] + hunk['removals'] + hunk['context_after']
        if match_lines:
            expected = "".join(match_lines[:3])
            if len(match_lines) > 3:
                expected += "..."
            return (False, lines, offset, f"Could not find context:\n{expected}")
        else:
            return (False, lines, offset, "Empty hunk with no context")

    # Calculate what to remove (context_before + removals + context_after)
    # and what to replace it with (context_before + additions + context_after)
    old_lines = hunk['context_before'] + hunk['removals'] + hunk['context_after']
    new_lines_content = hunk['context_before'] + hunk['additions'] + hunk['context_after']

    remove_count = len(old_lines)

    # Apply
    result_lines = lines[:pos] + new_lines_content + lines[pos + remove_count:]

    # Calculate offset change (only additions vs removals matter, context stays same)
    offset_change = len(hunk['additions']) - len(hunk['removals'])

    return (True, result_lines, offset + offset_change, "OK")


# =============================================================================
# EXA (WEB SEARCH) TOOLS
# =============================================================================

@tool(
    name="exa_search",
    description="Search the web using Exa API. Returns snippets and URLs.",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="The search query",
            required=True,
        ),
        ToolParameter(
            name="num_results",
            type="integer",
            description="Number of results (default 8, max 12)",
            required=False,
            default=8,
        ),
    ],
)
def exa_search(query: str, num_results: int = 8) -> str:
    """Search the web using Exa."""
    client = _get_exa_client()
    if not client.is_available():
        return "Error: Exa API not available (no EXA_API_KEY set)"

    num_results = min(int(num_results), 12)
    return client.search(query, num_results)


@tool(
    name="exa_fetch_full",
    description="Fetch full content of a URL using Exa API.",
    parameters=[
        ToolParameter(
            name="url",
            type="string",
            description="The URL to fetch",
            required=True,
        ),
    ],
)
def exa_fetch_full(url: str) -> str:
    """Fetch full content of a URL."""
    client = _get_exa_client()
    if not client.is_available():
        return "Error: Exa API not available (no EXA_API_KEY set)"

    return client.fetch_full_content_by_url(url)


# =============================================================================
# GMAIL TOOLS
# =============================================================================

@tool(
    name="gmail_list_from_date",
    description="List emails received on a specific date.",
    parameters=[
        ToolParameter(
            name="date",
            type="string",
            description="Date in YYYY-MM-DD format",
            required=True,
        ),
    ],
)
def gmail_list_from_date(date: str) -> str:
    """List emails from a specific date."""
    host = _get_tool_host()
    if host is None:
        return "Error: Gmail not configured (accounts.json not found)"

    gmail = host.gmail_client()
    if not gmail.ready:
        return "Error: Gmail client not initialized"

    result = gmail.list_emails_from_date(date)
    return json.dumps(result, ensure_ascii=False, default=str)


@tool(
    name="gmail_get_email",
    description="Get full content of an email by ID.",
    parameters=[
        ToolParameter(
            name="message_id",
            type="string",
            description="The Gmail message ID",
            required=True,
        ),
    ],
)
def gmail_get_email(message_id: str) -> str:
    """Get full email content."""
    host = _get_tool_host()
    if host is None:
        return "Error: Gmail not configured"

    gmail = host.gmail_client()
    if not gmail.ready:
        return "Error: Gmail client not initialized"

    result = gmail.get_email(message_id)
    if result is None:
        return f"Error: Could not fetch email {message_id}"

    return json.dumps(result, ensure_ascii=False, default=str)


@tool(
    name="gmail_send_message",
    description="Send an email. Requires confirmation.",
    parameters=[
        ToolParameter(
            name="to",
            type="string",
            description="Recipient email address",
            required=True,
        ),
        ToolParameter(
            name="subject",
            type="string",
            description="Email subject",
            required=True,
        ),
        ToolParameter(
            name="body",
            type="string",
            description="Email body text",
            required=True,
        ),
        ToolParameter(
            name="cc",
            type="string",
            description="CC recipients (comma-separated)",
            required=False,
        ),
        ToolParameter(
            name="attachments",
            type="array",
            description=(
                "List of file attachments. Each item is an object with: "
                "'path' (required, absolute file path), "
                "'filename' (optional, display name), "
                "'mimeType' (optional, e.g. 'application/pdf'). "
                "Example: [{\"path\": \"/tmp/report.pdf\"}]"
            ),
            required=False,
        ),
    ],
    requires_confirmation=True,
)
def gmail_send_message(to: str, subject: str, body: str, cc: str = None, attachments: list = None) -> str:
    """Send an email. Confirmation is handled at mesh level via requires_confirmation."""
    host = _get_tool_host()
    if host is None:
        return "Error: Gmail not configured"

    gmail = host.gmail_client()
    if not gmail.ready:
        return "Error: Gmail client not initialized"

    cc_list = [c.strip() for c in cc.split(",")] if cc else None
    # Call _send_email_raw directly - mesh agent already confirmed via requires_confirmation=True
    result = gmail._send_email_raw(
        to=to,
        subject=subject,
        body_text=body,
        cc=cc_list,
        attachments=attachments,
    )

    if result is None:
        return "Error: Failed to send email"

    return json.dumps(result, ensure_ascii=False, default=str)


@tool(
    name="gmail_reply_to",
    description="Reply to an existing email. Requires confirmation.",
    parameters=[
        ToolParameter(
            name="message_id",
            type="string",
            description="The Gmail message ID to reply to",
            required=True,
        ),
        ToolParameter(
            name="body",
            type="string",
            description="Reply body text",
            required=True,
        ),
        ToolParameter(
            name="cc",
            type="string",
            description="CC recipients (comma-separated)",
            required=False,
        ),
        ToolParameter(
            name="attachments",
            type="array",
            description=(
                "List of file attachments. Each item is an object with: "
                "'path' (required, absolute file path), "
                "'filename' (optional, display name), "
                "'mimeType' (optional, e.g. 'application/pdf'). "
                "Example: [{\"path\": \"/tmp/report.pdf\"}]"
            ),
            required=False,
        ),
    ],
    requires_confirmation=True,
)
def gmail_reply_to(message_id: str, body: str, cc: str = None, attachments: list = None) -> str:
    """Reply to an email. Confirmation is handled at mesh level via requires_confirmation."""
    host = _get_tool_host()
    if host is None:
        return "Error: Gmail not configured"

    gmail = host.gmail_client()
    if not gmail.ready:
        return "Error: Gmail client not initialized"

    # Fetch original message to get reply headers
    try:
        original = (
            gmail.service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
    except Exception as e:
        return f"Error: Failed to fetch original message {message_id}: {e}"

    headers = gmail._extract_headers(original)
    thread_id = original.get("threadId")

    to = headers.get("Reply-To") or headers.get("From")
    if not to:
        return "Error: Original message has no From/Reply-To; cannot determine recipient"

    subject = headers.get("Subject") or ""
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"

    orig_message_id = headers.get("Message-ID")
    references = headers.get("References")
    if orig_message_id:
        references = (references + " " + orig_message_id).strip() if references else orig_message_id

    cc_list = [c.strip() for c in cc.split(",")] if cc else None

    # Call _send_email_raw directly - mesh agent already confirmed via requires_confirmation=True
    result = gmail._send_email_raw(
        to=to,
        subject=subject,
        body_text=body,
        thread_id=thread_id,
        in_reply_to=orig_message_id,
        references=references,
        cc=cc_list,
        attachments=attachments,
    )

    if result is None:
        return "Error: Failed to send reply"

    return json.dumps(result, ensure_ascii=False, default=str)


@tool(
    name="gmail_create_draft",
    description=(
        "Create a Gmail draft (does NOT send anything). The draft lands in the "
        "Drafts folder for the user to review, edit, and send from Gmail."
    ),
    parameters=[
        ToolParameter(
            name="to",
            type="string",
            description="Recipient email address",
            required=True,
        ),
        ToolParameter(
            name="subject",
            type="string",
            description="Email subject",
            required=True,
        ),
        ToolParameter(
            name="body",
            type="string",
            description="Email body text",
            required=True,
        ),
        ToolParameter(
            name="cc",
            type="string",
            description="CC recipients (comma-separated)",
            required=False,
        ),
        ToolParameter(
            name="attachments",
            type="array",
            description=(
                "List of file attachments. Each item is an object with: "
                "'path' (required, absolute file path), "
                "'filename' (optional, display name), "
                "'mimeType' (optional, e.g. 'application/pdf'). "
                "Example: [{\"path\": \"/tmp/report.pdf\"}]"
            ),
            required=False,
        ),
    ],
)
def gmail_create_draft(to: str, subject: str, body: str, cc: str = None, attachments: list = None) -> str:
    """Create a Gmail draft. No confirmation needed — nothing is sent."""
    host = _get_tool_host()
    if host is None:
        return "Error: Gmail not configured"

    gmail = host.gmail_client()
    if not gmail.ready:
        return "Error: Gmail client not initialized"

    cc_list = [c.strip() for c in cc.split(",")] if cc else None
    result = gmail._create_draft_raw(
        to=to,
        subject=subject,
        body_text=body,
        cc=cc_list,
        attachments=attachments,
    )

    if result is None:
        return "Error: Failed to create draft"

    return json.dumps(result, ensure_ascii=False, default=str)


@tool(
    name="gmail_draft_reply",
    description=(
        "Create a reply draft to an existing email (does NOT send anything). "
        "The draft threads under the original message and lands in the Drafts "
        "folder for the user to review, edit, and send from Gmail."
    ),
    parameters=[
        ToolParameter(
            name="message_id",
            type="string",
            description="The Gmail message ID to draft a reply to",
            required=True,
        ),
        ToolParameter(
            name="body",
            type="string",
            description="Reply body text",
            required=True,
        ),
        ToolParameter(
            name="cc",
            type="string",
            description="CC recipients (comma-separated)",
            required=False,
        ),
        ToolParameter(
            name="attachments",
            type="array",
            description=(
                "List of file attachments. Each item is an object with: "
                "'path' (required, absolute file path), "
                "'filename' (optional, display name), "
                "'mimeType' (optional, e.g. 'application/pdf'). "
                "Example: [{\"path\": \"/tmp/report.pdf\"}]"
            ),
            required=False,
        ),
    ],
)
def gmail_draft_reply(message_id: str, body: str, cc: str = None, attachments: list = None) -> str:
    """Create a threaded reply draft. No confirmation needed — nothing is sent."""
    host = _get_tool_host()
    if host is None:
        return "Error: Gmail not configured"

    gmail = host.gmail_client()
    if not gmail.ready:
        return "Error: Gmail client not initialized"

    # Fetch original message to get reply headers (same derivation as gmail_reply_to)
    try:
        original = (
            gmail.service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
    except Exception as e:
        return f"Error: Failed to fetch original message {message_id}: {e}"

    headers = gmail._extract_headers(original)
    thread_id = original.get("threadId")

    to = headers.get("Reply-To") or headers.get("From")
    if not to:
        return "Error: Original message has no From/Reply-To; cannot determine recipient"

    subject = headers.get("Subject") or ""
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"

    orig_message_id = headers.get("Message-ID")
    references = headers.get("References")
    if orig_message_id:
        references = (references + " " + orig_message_id).strip() if references else orig_message_id

    cc_list = [c.strip() for c in cc.split(",")] if cc else None

    result = gmail._create_draft_raw(
        to=to,
        subject=subject,
        body_text=body,
        thread_id=thread_id,
        in_reply_to=orig_message_id,
        references=references,
        cc=cc_list,
        attachments=attachments,
    )

    if result is None:
        return "Error: Failed to create reply draft"

    return json.dumps(result, ensure_ascii=False, default=str)


@tool(
    name="gmail_search_emails",
    description="Search emails using Gmail query syntax.",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Gmail search query (e.g., 'from:alice subject:meeting')",
            required=True,
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="Maximum results (default 20)",
            required=False,
            default=20,
        ),
    ],
)
def gmail_search_emails(query: str, limit: int = 20) -> str:
    """Search emails."""
    host = _get_tool_host()
    if host is None:
        return "Error: Gmail not configured"

    gmail = host.gmail_client()
    if not gmail.ready:
        return "Error: Gmail client not initialized"

    result = gmail.search_emails(query, limit=int(limit))
    return json.dumps(result, ensure_ascii=False, default=str)


@tool(
    name="gmail_list_recent",
    description=(
        "List the N most recent emails, newest first. "
        "No query or date needed — use this when browsing the inbox."
    ),
    parameters=[
        ToolParameter(
            name="limit",
            type="integer",
            description="Maximum number of emails to return (default 20)",
            required=False,
            default=20,
        ),
    ],
)
def gmail_list_recent(limit: int = 20) -> str:
    """List recent emails."""
    host = _get_tool_host()
    if host is None:
        return "Error: Gmail not configured"

    gmail = host.gmail_client()
    if not gmail.ready:
        return "Error: Gmail client not initialized"

    result = gmail.list_recent_emails(limit=int(limit), priority_inbox=False)
    return json.dumps(result, ensure_ascii=False, default=str)


@tool(
    name="gmail_list_unread",
    description=(
        "List unread emails. Use this to check what's new in the inbox."
    ),
    parameters=[
        ToolParameter(
            name="limit",
            type="integer",
            description="Maximum number of unread emails to return (default 20)",
            required=False,
            default=20,
        ),
    ],
)
def gmail_list_unread(limit: int = 20) -> str:
    """List unread emails."""
    host = _get_tool_host()
    if host is None:
        return "Error: Gmail not configured"

    gmail = host.gmail_client()
    if not gmail.ready:
        return "Error: Gmail client not initialized"

    result = gmail.search_emails("is:unread", limit=int(limit))
    return json.dumps(result, ensure_ascii=False, default=str)


# =============================================================================
# CALENDAR TOOLS
# =============================================================================

@tool(
    name="calendar_list_on_date",
    description="List calendar events on a specific date.",
    parameters=[
        ToolParameter(
            name="date",
            type="string",
            description="Date in YYYY-MM-DD format",
            required=True,
        ),
        ToolParameter(
            name="timezone",
            type="string",
            description="IANA timezone (default America/Chicago)",
            required=False,
            default="America/Chicago",
        ),
        ToolParameter(
            name="account",
            type="string",
            description="Optional account context, e.g. work or personal.",
            required=False,
        ),
    ],
)
def calendar_list_on_date(date: str, timezone: str = "America/Chicago", account: str = None) -> str:
    """List calendar events on a date."""
    host = _get_tool_host()
    if host is None:
        return "Error: Calendar not configured"

    previous_account = host.get_current_account()
    if account:
        try:
            host.set_current_account(account)
        except Exception as e:
            return f"Error: Calendar account '{account}' unavailable: {e}"

    try:
        calendar = host.calendar_client()
        if not calendar.ready:
            return "Error: Calendar client not initialized"

        result = calendar.list_events_on_date(date, timezone=timezone)
        return json.dumps(result, ensure_ascii=False, default=str)
    finally:
        if account and previous_account and previous_account != account:
            try:
                host.set_current_account(previous_account)
            except Exception:
                pass


@tool(
    name="calendar_create_event",
    description="Create a calendar event. Requires confirmation.",
    parameters=[
        ToolParameter(
            name="summary",
            type="string",
            description="Event title",
            required=True,
        ),
        ToolParameter(
            name="start",
            type="string",
            description="Start time in YYYY-MM-DDTHH:MM:SS format",
            required=True,
        ),
        ToolParameter(
            name="end",
            type="string",
            description="End time in YYYY-MM-DDTHH:MM:SS format",
            required=True,
        ),
        ToolParameter(
            name="description",
            type="string",
            description="Event description",
            required=False,
        ),
        ToolParameter(
            name="location",
            type="string",
            description="Event location",
            required=False,
        ),
        ToolParameter(
            name="timezone",
            type="string",
            description="IANA timezone (default America/Chicago)",
            required=False,
            default="America/Chicago",
        ),
        ToolParameter(
            name="attendees",
            type="array",
            description="List of attendee email addresses to invite",
            required=False,
        ),
        ToolParameter(
            name="recurrence",
            type="array",
            description="List of RRULE strings for recurring events. Examples: ['RRULE:FREQ=WEEKLY;BYDAY=TH'] for weekly on Thursday, ['RRULE:FREQ=DAILY;COUNT=5'] for daily 5 times, ['RRULE:FREQ=WEEKLY;UNTIL=20261231'] for weekly until end of year.",
            required=False,
        ),
    ],
    requires_confirmation=True,
)
def calendar_create_event(
    summary: str,
    start: str,
    end: str,
    description: str = None,
    location: str = None,
    timezone: str = "America/Chicago",
    attendees: list = None,
    recurrence: list = None,
) -> str:
    """Create a calendar event. Confirmation is handled at mesh level via requires_confirmation."""
    host = _get_tool_host()
    if host is None:
        return "Error: Calendar not configured"

    calendar = host.calendar_client()
    if not calendar.ready:
        return "Error: Calendar client not initialized"

    # Build event body and call API directly - mesh agent already confirmed
    event_body = {
        "summary": summary,
        "description": description,
        "location": location,
        "start": {"dateTime": start, "timeZone": timezone},
        "end": {"dateTime": end, "timeZone": timezone},
    }

    # Add attendees if provided
    if attendees:
        event_body["attendees"] = [{"email": a} for a in attendees]

    # Add recurrence rules if provided
    if recurrence:
        event_body["recurrence"] = recurrence

    try:
        result = (
            calendar.service.events()
            .insert(calendarId="primary", body=event_body)
            .execute()
        )
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error: Failed to create event: {e}"


@tool(
    name="calendar_delete_event",
    description="Delete a calendar event. Requires confirmation.",
    parameters=[
        ToolParameter(
            name="event_id",
            type="string",
            description="The calendar event ID",
            required=True,
        ),
    ],
    requires_confirmation=True,
)
def calendar_delete_event(event_id: str) -> str:
    """Delete a calendar event. Confirmation is handled at mesh level via requires_confirmation."""
    host = _get_tool_host()
    if host is None:
        return "Error: Calendar not configured"

    calendar = host.calendar_client()
    if not calendar.ready:
        return "Error: Calendar client not initialized"

    # Call API directly - mesh agent already confirmed via requires_confirmation=True
    try:
        calendar.service.events().delete(
            calendarId="primary",
            eventId=event_id,
        ).execute()
        return json.dumps({
            "status": "deleted",
            "event_id": event_id,
        })
    except Exception as e:
        return f"Error: Failed to delete event: {e}"


# =============================================================================
# NOTES TOOLS (7 tools - using HTTP API to remote server)
# =============================================================================

# Notes HTTP API helpers

def _get_notes_server() -> tuple[str, dict]:
    """Get notes server base URL and auth headers.

    Returns (server_base, headers) or raises error.
    """
    server_base = os.environ.get("RN_SERVER_BASE")
    if not server_base:
        raise ValueError("RN_SERVER_BASE environment variable not set")

    token = os.environ.get("RN_API_TOKEN")
    if not token:
        raise ValueError("RN_API_TOKEN environment variable not set")

    headers = {"X-API-Token": token}
    return server_base.rstrip("/"), headers


def _notes_http_get(url: str, headers: dict) -> Any:
    """Make HTTP GET request and return JSON."""
    import urllib.request
    import urllib.error

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = resp.read().decode("utf-8")
            if not data:
                return None
            return json.loads(data)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        detail = e.read().decode("utf-8", errors="replace")
        raise ValueError(f"HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise ValueError(f"Could not reach server: {e}")


def _notes_http_post(url: str, headers: dict, payload: dict) -> Any:
    """Make HTTP POST request with JSON body."""
    import urllib.request
    import urllib.error

    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers)
    try:
        with urllib.request.urlopen(req) as resp:
            resp_data = resp.read().decode("utf-8")
            if not resp_data:
                return {}
            return json.loads(resp_data)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise ValueError(f"HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise ValueError(f"Could not reach server: {e}")


def _notes_http_put(url: str, headers: dict, payload: dict) -> Any:
    """Make HTTP PUT request with JSON body."""
    import urllib.request
    import urllib.error

    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers, method="PUT")
    try:
        with urllib.request.urlopen(req) as resp:
            resp_data = resp.read().decode("utf-8")
            if not resp_data:
                return {}
            return json.loads(resp_data)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise ValueError(f"HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise ValueError(f"Could not reach server: {e}")


def _notes_http_delete(url: str, headers: dict) -> Any:
    """Make HTTP DELETE request."""
    import urllib.request
    import urllib.error

    req = urllib.request.Request(url, headers=headers, method="DELETE")
    try:
        with urllib.request.urlopen(req) as resp:
            resp_data = resp.read().decode("utf-8")
            if not resp_data:
                return {"success": True}
            return json.loads(resp_data)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        detail = e.read().decode("utf-8", errors="replace")
        raise ValueError(f"HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise ValueError(f"Could not reach server: {e}")


def _validate_notes_db(db: str) -> str:
    """Validate and normalize db name. Returns 'work' or 'personal'."""
    db = db.lower().strip()
    if db in ("work", "w"):
        return "work"
    if db in ("personal", "p", "pers"):
        return "personal"
    raise ValueError(f"Invalid db '{db}'. Use 'work' or 'personal'.")


@tool(
    name="notes_search",
    description="Full-text search notes.",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Search query",
            required=True,
        ),
        ToolParameter(
            name="db",
            type="string",
            description="Database: 'work' or 'personal'",
            required=True,
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="Maximum results",
            required=False,
        ),
        ToolParameter(
            name="date_from",
            type="string",
            description="Start date filter (YYYY-MM-DD)",
            required=False,
        ),
        ToolParameter(
            name="date_to",
            type="string",
            description="End date filter (YYYY-MM-DD)",
            required=False,
        ),
    ],
)
def notes_search(
    query: str,
    db: str,
    limit: int = None,
    date_from: str = None,
    date_to: str = None,
) -> str:
    """Search notes using full-text search via HTTP API."""
    from urllib.parse import urlencode

    try:
        db_name = _validate_notes_db(db)
        server_base, headers = _get_notes_server()
    except ValueError as e:
        return f"Error: {e}"

    params = {"query": query}
    if limit:
        params["limit"] = int(limit)
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to

    url = f"{server_base}/{db_name}/search?{urlencode(params)}"

    try:
        results = _notes_http_get(url, headers) or []
        return json.dumps(results, ensure_ascii=False)
    except ValueError as e:
        return f"Error: {e}"


@tool(
    name="notes_get",
    description="Get a note by ID.",
    parameters=[
        ToolParameter(
            name="id",
            type="integer",
            description="Note ID",
            required=True,
        ),
        ToolParameter(
            name="db",
            type="string",
            description="Database: 'work' or 'personal'",
            required=True,
        ),
    ],
)
def notes_get(id: int, db: str) -> str:
    """Get a note by ID via HTTP API."""
    try:
        db_name = _validate_notes_db(db)
        server_base, headers = _get_notes_server()
    except ValueError as e:
        return f"Error: {e}"

    url = f"{server_base}/{db_name}/notes/{int(id)}"

    try:
        note = _notes_http_get(url, headers)
        if note is None:
            return f"Error: Note {id} not found"
        return json.dumps(note, ensure_ascii=False)
    except ValueError as e:
        return f"Error: {e}"


@tool(
    name="notes_list",
    description="List notes with filters (recent, date, tag, source).",
    parameters=[
        ToolParameter(
            name="db",
            type="string",
            description="Database: 'work' or 'personal'",
            required=True,
        ),
        ToolParameter(
            name="recent",
            type="integer",
            description="Number of recent notes to list",
            required=False,
        ),
        ToolParameter(
            name="date",
            type="string",
            description="List notes on this date (YYYY-MM-DD)",
            required=False,
        ),
        ToolParameter(
            name="tag",
            type="string",
            description="Filter by tag",
            required=False,
        ),
    ],
)
def notes_list(
    db: str,
    recent: int = None,
    date: str = None,
    tag: str = None,
) -> str:
    """List notes with filters via HTTP API."""
    from urllib.parse import urlencode

    try:
        db_name = _validate_notes_db(db)
        server_base, headers = _get_notes_server()
    except ValueError as e:
        return f"Error: {e}"

    # Determine the right endpoint
    if date:
        # Single date - use between endpoint with same start/end
        params = {"start": date, "end": date}
        path = f"/{db_name}/notes/between"
    else:
        # Recent notes
        limit = int(recent) if recent else 10
        params = {"limit": limit}
        path = f"/{db_name}/notes/recent"

    url = f"{server_base}{path}?{urlencode(params)}"

    try:
        results = _notes_http_get(url, headers) or []

        # Client-side tag filtering if server doesn't support it
        if tag and isinstance(results, list):
            results = [n for n in results if tag in (n.get("tags") or [])]

        return json.dumps(results, ensure_ascii=False)
    except ValueError as e:
        return f"Error: {e}"


@tool(
    name="notes_add",
    description="Create a new note.",
    parameters=[
        ToolParameter(
            name="body",
            type="string",
            description="Note body content",
            required=True,
        ),
        ToolParameter(
            name="db",
            type="string",
            description="Database: 'work' or 'personal'",
            required=True,
        ),
        ToolParameter(
            name="title",
            type="string",
            description="Note title",
            required=False,
        ),
        ToolParameter(
            name="tags",
            type="array",
            description="List of tags",
            required=False,
        ),
        ToolParameter(
            name="date",
            type="string",
            description="Date for the note (YYYY-MM-DD)",
            required=False,
        ),
    ],
)
def notes_add(
    body: str,
    db: str,
    title: str = None,
    tags: list = None,
    date: str = None,
) -> str:
    """Create a new note via HTTP API."""
    from datetime import datetime

    try:
        db_name = _validate_notes_db(db)
        server_base, headers = _get_notes_server()
    except ValueError as e:
        return f"Error: {e}"

    # Build payload
    when = datetime.now()
    if date:
        try:
            from datetime import date as date_cls, time as time_cls
            d = date_cls.fromisoformat(date)
            when = datetime.combine(d, time_cls())
        except ValueError:
            pass

    payload = {
        "title": title,
        "body": body.rstrip("\n"),
        "tags": tags or [],
        "meta": {
            "client_source": "mesh-agent",
            "when": when.isoformat(timespec="seconds"),
        },
    }

    url = f"{server_base}/{db_name}/notes"

    try:
        result = _notes_http_post(url, headers, payload)
        return json.dumps({"success": True, "id": result.get("id")}, ensure_ascii=False)
    except ValueError as e:
        return f"Error: {e}"


@tool(
    name="notes_delete",
    description="Delete a note by ID.",
    parameters=[
        ToolParameter(
            name="id",
            type="integer",
            description="Note ID to delete",
            required=True,
        ),
        ToolParameter(
            name="db",
            type="string",
            description="Database: 'work' or 'personal'",
            required=True,
        ),
    ],
)
def notes_delete(id: int, db: str) -> str:
    """Delete a note via HTTP API."""
    try:
        db_name = _validate_notes_db(db)
        server_base, headers = _get_notes_server()
    except ValueError as e:
        return f"Error: {e}"

    url = f"{server_base}/{db_name}/notes/{int(id)}"

    try:
        result = _notes_http_delete(url, headers)
        if result is None:
            return f"Error: Note {id} not found"
        return json.dumps({"success": True, "deleted_id": id})
    except ValueError as e:
        return f"Error: {e}"


@tool(
    name="notes_read",
    description="Read note with line numbers (for editing).",
    parameters=[
        ToolParameter(
            name="id",
            type="integer",
            description="Note ID",
            required=True,
        ),
        ToolParameter(
            name="db",
            type="string",
            description="Database: 'work' or 'personal'",
            required=True,
        ),
        ToolParameter(
            name="start_line",
            type="integer",
            description="Starting line (1-indexed)",
            required=False,
        ),
        ToolParameter(
            name="end_line",
            type="integer",
            description="Ending line (inclusive)",
            required=False,
        ),
    ],
)
def notes_read(
    id: int,
    db: str,
    start_line: int = None,
    end_line: int = None,
) -> str:
    """Read note with line numbers via HTTP API."""
    try:
        db_name = _validate_notes_db(db)
        server_base, headers = _get_notes_server()
    except ValueError as e:
        return f"Error: {e}"

    url = f"{server_base}/{db_name}/notes/{int(id)}"

    try:
        note = _notes_http_get(url, headers)
        if note is None:
            return f"Error: Note {id} not found"

        body = note.get("body") or ""
        lines = body.split("\n")
        total_lines = len(lines)

        if start_line is not None or end_line is not None:
            start = (int(start_line) - 1) if start_line else 0
            end = int(end_line) if end_line else total_lines
            selected_lines = lines[start:end]
            line_offset = start
        else:
            selected_lines = lines
            line_offset = 0

        # Format with line numbers
        numbered = []
        for i, line in enumerate(selected_lines, start=line_offset + 1):
            numbered.append(f"{i:4d}: {line}")

        return json.dumps({
            "id": note.get("id"),
            "title": note.get("title"),
            "date": note.get("date"),
            "tags": note.get("tags"),
            "total_lines": total_lines,
            "content": "\n".join(numbered),
        }, ensure_ascii=False)
    except ValueError as e:
        return f"Error: {e}"


@tool(
    name="notes_edit",
    description="Perform exact string replacement in a note (like file_edit).",
    parameters=[
        ToolParameter(
            name="id",
            type="integer",
            description="Note ID",
            required=True,
        ),
        ToolParameter(
            name="db",
            type="string",
            description="Database: 'work' or 'personal'",
            required=True,
        ),
        ToolParameter(
            name="old_string",
            type="string",
            description="The exact string to find and replace",
            required=True,
        ),
        ToolParameter(
            name="new_string",
            type="string",
            description="The replacement string",
            required=True,
        ),
        ToolParameter(
            name="replace_all",
            type="boolean",
            description="Replace all occurrences (default false)",
            required=False,
            default=False,
        ),
    ],
)
def notes_edit(
    id: int,
    db: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Edit a note by replacing exact strings via HTTP API."""
    try:
        db_name = _validate_notes_db(db)
        server_base, headers = _get_notes_server()
    except ValueError as e:
        return f"Error: {e}"

    url = f"{server_base}/{db_name}/notes/{int(id)}"

    try:
        # First fetch the note
        note = _notes_http_get(url, headers)
        if note is None:
            return f"Error: Note {id} not found"

        body = note.get("body") or ""
        count = body.count(old_string)

        if count == 0:
            return "Error: old_string not found in note."

        if count > 1 and not replace_all:
            return f"Error: old_string found {count} times. Use replace_all=true or provide more context."

        if replace_all:
            new_body = body.replace(old_string, new_string)
        else:
            new_body = body.replace(old_string, new_string, 1)

        # Update the note
        payload = {"body": new_body}
        _notes_http_put(url, headers, payload)

        return json.dumps({
            "success": True,
            "replaced_count": count if replace_all else 1,
        })
    except ValueError as e:
        return f"Error: {e}"


# =============================================================================
# ACCOUNT TOOLS
# =============================================================================

@tool(
    name="account_get_current",
    description="Get the current account context (work/personal).",
    parameters=[],
)
def account_get_current() -> str:
    """Get current account."""
    host = _get_tool_host()
    if host is None:
        return json.dumps({"error": "Account system not configured"})
    return json.dumps({"current_account": host.get_current_account()})


@tool(
    name="account_list",
    description="List available accounts.",
    parameters=[],
)
def account_list() -> str:
    """List accounts."""
    host = _get_tool_host()
    if host is None:
        return json.dumps({"error": "Account system not configured"})
    return json.dumps({
        "accounts": host.list_accounts(),
        "current_account": host.get_current_account(),
    })


@tool(
    name="account_set_current",
    description="Switch to a different account context.",
    parameters=[
        ToolParameter(
            name="account",
            type="string",
            description="Account name to switch to",
            required=True,
        ),
    ],
)
def account_set_current(account: str) -> str:
    """Set current account."""
    host = _get_tool_host()
    if host is None:
        return json.dumps({"error": "Account system not configured"})

    try:
        host.set_current_account(account)
        return json.dumps({
            "success": True,
            "current_account": host.get_current_account(),
        })
    except KeyError as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# BROWSER TOOLS (async)
# =============================================================================

@tool(
    name="browser_session_status",
    description="Check if a browser session is currently open.",
    parameters=[],
)
async def browser_session_status() -> str:
    """Check browser session status."""
    client = await _ensure_browser_client()
    is_open = await client.is_open()
    url = None
    if is_open:
        try:
            url = await client.get_url()
        except Exception:
            pass
    return json.dumps({"open": is_open, "url": url})


@tool(
    name="browser_session_open",
    description="Open a browser session with a Chrome profile.",
    parameters=[
        ToolParameter(
            name="user_data_dir",
            type="string",
            description="Path to Chrome user data directory",
            required=True,
        ),
        ToolParameter(
            name="profile_directory",
            type="string",
            description="Profile directory name (default 'Default')",
            required=False,
            default="Default",
        ),
        ToolParameter(
            name="headless",
            type="boolean",
            description="Run in headless mode (default false)",
            required=False,
            default=False,
        ),
    ],
)
async def browser_session_open(
    user_data_dir: str,
    profile_directory: str = "Default",
    headless: bool = False,
) -> str:
    """Open a browser session."""
    from .paths import resolve_path as _rp
    client = await _ensure_browser_client()

    if await client.is_open():
        return json.dumps({"error": "Browser session already open"})

    try:
        await client.open(
            user_data_dir=_rp(user_data_dir),
            profile_directory=profile_directory,
            headless=headless,
        )
        return json.dumps({"success": True, "url": await client.get_url()})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="browser_session_close",
    description="Close the current browser session.",
    parameters=[],
)
async def browser_session_close() -> str:
    """Close browser session."""
    client = await _ensure_browser_client()

    if not await client.is_open():
        return json.dumps({"error": "No browser session open"})

    try:
        await client.close()
        return json.dumps({"success": True})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="browser_goto",
    description="Navigate to a URL.",
    parameters=[
        ToolParameter(
            name="url",
            type="string",
            description="The URL to navigate to",
            required=True,
        ),
    ],
)
async def browser_goto(url: str) -> str:
    """Navigate to a URL."""
    client = await _ensure_browser_client()

    if not await client.is_open():
        return json.dumps({"error": "No browser session open"})

    try:
        await client.goto(url)
        return json.dumps({"success": True, "url": await client.get_url()})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="browser_get_url",
    description="Get the current page URL.",
    parameters=[],
)
async def browser_get_url() -> str:
    """Get current URL."""
    client = await _ensure_browser_client()

    if not await client.is_open():
        return json.dumps({"error": "No browser session open"})

    return json.dumps({"url": await client.get_url()})


@tool(
    name="browser_snapshot_controls",
    description="Capture actionable UI controls on the page.",
    parameters=[
        ToolParameter(
            name="filter",
            type="string",
            description="Optional filter to match control labels",
            required=False,
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="Maximum controls to return (default 200)",
            required=False,
            default=200,
        ),
    ],
)
async def browser_snapshot_controls(filter: str = None, limit: int = 200) -> str:
    """Snapshot UI controls."""
    client = await _ensure_browser_client()

    if not await client.is_open():
        return json.dumps({"error": "No browser session open"})

    try:
        result = await client.snapshot_fast(
            limit=int(limit),
            filter=filter,
            delta=False,
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="browser_read_text",
    description="Read text content from the page.",
    parameters=[
        ToolParameter(
            name="filter",
            type="string",
            description="Optional filter to match text content",
            required=False,
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="Maximum text elements (default 200)",
            required=False,
            default=200,
        ),
    ],
)
async def browser_read_text(filter: str = None, limit: int = 200) -> str:
    """Read text from page."""
    client = await _ensure_browser_client()

    if not await client.is_open():
        return json.dumps({"error": "No browser session open"})

    try:
        result = await client.snapshot_text(
            limit=int(limit),
            filter=filter,
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="browser_click",
    description="Click an element by its snapshot ID.",
    parameters=[
        ToolParameter(
            name="id",
            type="integer",
            description="Element snapshot ID from browser_snapshot_controls",
            required=True,
        ),
    ],
)
async def browser_click(id: int) -> str:
    """Click an element."""
    client = await _ensure_browser_client()

    if not await client.is_open():
        return json.dumps({"error": "No browser session open"})

    try:
        loc = await client._loc_for_id(int(id))
        await client._move_mouse_to_element(loc)
        await loc.click()
        await client._post_action()
        return json.dumps({"success": True})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="browser_fill",
    description="Fill an input element with text.",
    parameters=[
        ToolParameter(
            name="id",
            type="integer",
            description="Element snapshot ID",
            required=True,
        ),
        ToolParameter(
            name="value",
            type="string",
            description="Value to fill",
            required=True,
        ),
    ],
)
async def browser_fill(id: int, value: str) -> str:
    """Fill an input element."""
    client = await _ensure_browser_client()

    if not await client.is_open():
        return json.dumps({"error": "No browser session open"})

    try:
        loc = await client._loc_for_id(int(id))
        await client._move_mouse_to_element(loc)
        await loc.fill(value)
        await client._post_action()
        return json.dumps({"success": True})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="browser_type",
    description="Type text into an element with keystrokes.",
    parameters=[
        ToolParameter(
            name="id",
            type="integer",
            description="Element snapshot ID",
            required=True,
        ),
        ToolParameter(
            name="text",
            type="string",
            description="Text to type",
            required=True,
        ),
        ToolParameter(
            name="delay",
            type="integer",
            description="Delay between keystrokes in ms (default 50)",
            required=False,
            default=50,
        ),
    ],
)
async def browser_type(id: int, text: str, delay: int = 50) -> str:
    """Type text into an element."""
    client = await _ensure_browser_client()

    if not await client.is_open():
        return json.dumps({"error": "No browser session open"})

    try:
        loc = await client._loc_for_id(int(id))
        await client._move_mouse_to_element(loc)
        await loc.type(text, delay=int(delay))
        await client._post_action()
        return json.dumps({"success": True})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="browser_press",
    description="Press a keyboard key on an element.",
    parameters=[
        ToolParameter(
            name="id",
            type="integer",
            description="Element snapshot ID",
            required=True,
        ),
        ToolParameter(
            name="key",
            type="string",
            description="Key to press (e.g., 'Enter', 'Tab', 'Escape')",
            required=True,
        ),
    ],
)
async def browser_press(id: int, key: str) -> str:
    """Press a key on an element."""
    client = await _ensure_browser_client()

    if not await client.is_open():
        return json.dumps({"error": "No browser session open"})

    try:
        loc = await client._loc_for_id(int(id))
        await client._move_mouse_to_element(loc)
        await loc.press(key)
        # Longer jitter for navigation-like keys
        if key.lower() in ("enter", "return", "tab"):
            await client._post_nav()
        else:
            await client._post_action()
        return json.dumps({"success": True})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="browser_select",
    description="Select an option in a dropdown.",
    parameters=[
        ToolParameter(
            name="id",
            type="integer",
            description="Element snapshot ID of the select element",
            required=True,
        ),
        ToolParameter(
            name="value",
            type="string",
            description="Option value to select",
            required=False,
        ),
        ToolParameter(
            name="label",
            type="string",
            description="Option label to select (alternative to value)",
            required=False,
        ),
    ],
)
async def browser_select(id: int, value: str = None, label: str = None) -> str:
    """Select a dropdown option."""
    client = await _ensure_browser_client()

    if not await client.is_open():
        return json.dumps({"error": "No browser session open"})

    if not value and not label:
        return json.dumps({"error": "Must provide either value or label"})

    try:
        loc = await client._loc_for_id(int(id))
        await client._move_mouse_to_element(loc)

        if value:
            await loc.select_option(value=value)
        else:
            await loc.select_option(label=label)

        await client._post_action()
        return json.dumps({"success": True})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="browser_back",
    description="Navigate back in browser history.",
    parameters=[],
)
async def browser_back() -> str:
    """Navigate back."""
    client = await _ensure_browser_client()

    if not await client.is_open():
        return json.dumps({"error": "No browser session open"})

    try:
        await client.back()
        return json.dumps({"success": True, "url": await client.get_url()})
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# LITERATURE SEARCH TOOLS
# =============================================================================

@tool(
    name="literature_search",
    description="""Search academic literature across multiple sources (arXiv, PubMed, Semantic Scholar).
Automatically routes queries to the best sources based on domain:
- CS/ML queries → arXiv, Semantic Scholar
- Biomedical queries → PubMed, Semantic Scholar
- General queries → All sources

Returns paper metadata including titles, authors, abstracts, and identifiers.""",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Search query for academic papers",
            required=True,
        ),
        ToolParameter(
            name="max_results",
            type="integer",
            description="Maximum number of results (default 10)",
            required=False,
            default=10,
        ),
        ToolParameter(
            name="sources",
            type="string",
            description="Comma-separated list of sources: arxiv, pubmed, semantic_scholar (default: auto-detect)",
            required=False,
        ),
    ],
)
def literature_search(query: str, max_results: int = 10, sources: str = None) -> str:
    """Search academic literature across multiple sources."""
    client = _get_scholar_client()

    args = {
        "query": query,
        "max_results": max_results,
    }
    if sources:
        args["sources"] = [s.strip() for s in sources.split(",")]

    try:
        result = client.handle_literature_search(args)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="literature_fulltext",
    description="""Get full text of a paper by ID. Tries multiple sources in cascade:
1. arXiv PDF (if arxiv_id provided)
2. PMC full text (if pmcid available)
3. Publisher HTML
4. Abstract fallback

Provide at least one identifier: arxiv_id, pmid, or doi.""",
    parameters=[
        ToolParameter(
            name="arxiv_id",
            type="string",
            description="arXiv paper ID (e.g., '1706.03762')",
            required=False,
        ),
        ToolParameter(
            name="pmid",
            type="string",
            description="PubMed ID (numeric)",
            required=False,
        ),
        ToolParameter(
            name="doi",
            type="string",
            description="Digital Object Identifier",
            required=False,
        ),
        ToolParameter(
            name="max_tokens",
            type="integer",
            description="Maximum number of tokens to return (optional)",
            required=False,
        ),
        ToolParameter(
            name="max_pages",
            type="integer",
            description="Limit PDF extraction to first N pages (optional)",
            required=False,
        ),
    ],
)
def literature_fulltext(
    arxiv_id: str = None,
    pmid: str = None,
    doi: str = None,
    max_tokens: int = None,
    max_pages: int = None,
) -> str:
    """Get full text of a paper."""
    client = _get_scholar_client()

    args = {}
    if arxiv_id:
        args["arxiv_id"] = arxiv_id
    if pmid:
        args["pmid"] = pmid
    if doi:
        args["doi"] = doi
    if max_tokens:
        args["max_tokens"] = max_tokens
    if max_pages:
        args["max_pages"] = max_pages

    if not any([arxiv_id, pmid, doi]):
        return json.dumps({"error": "Provide at least one identifier: arxiv_id, pmid, or doi"})

    try:
        result = client.handle_literature_fulltext(args)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="arxiv_search",
    description="Search arXiv for papers in CS, ML, Physics, Math, etc. Returns metadata with links to PDFs.",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Search query",
            required=True,
        ),
        ToolParameter(
            name="max_results",
            type="integer",
            description="Maximum number of results (default 10)",
            required=False,
            default=10,
        ),
        ToolParameter(
            name="search_field",
            type="string",
            description="Search field: 'all', 'ti' (title), 'au' (author), 'abs' (abstract). Default: all",
            required=False,
            default="all",
        ),
    ],
)
def arxiv_search(query: str, max_results: int = 10, search_field: str = "all") -> str:
    """Search arXiv for papers."""
    client = _get_scholar_client()

    args = {
        "query": query,
        "max_results": max_results,
        "search_field": search_field,
    }

    try:
        result = client.handle_arxiv_search(args)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="arxiv_get",
    description="Fetch a specific arXiv paper by ID. Returns full metadata including abstract.",
    parameters=[
        ToolParameter(
            name="arxiv_id",
            type="string",
            description="arXiv paper ID (e.g., '1706.03762' or 'arxiv:1706.03762')",
            required=True,
        ),
    ],
)
def arxiv_get(arxiv_id: str) -> str:
    """Get a specific arXiv paper."""
    client = _get_scholar_client()

    try:
        result = client.handle_arxiv_get({"arxiv_id": arxiv_id})
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="arxiv_fulltext",
    description="Download and extract full text from an arXiv paper PDF.",
    parameters=[
        ToolParameter(
            name="arxiv_id",
            type="string",
            description="arXiv paper ID (e.g., '1706.03762')",
            required=True,
        ),
        ToolParameter(
            name="max_pages",
            type="integer",
            description="Maximum number of pages to extract (optional, default: all)",
            required=False,
        ),
    ],
)
def arxiv_fulltext(arxiv_id: str, max_pages: int = None) -> str:
    """Extract full text from arXiv paper."""
    client = _get_scholar_client()

    args = {"arxiv_id": arxiv_id}
    if max_pages:
        args["max_pages"] = max_pages

    try:
        result = client.handle_arxiv_fulltext(args)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="pubmed_search",
    description="Search PubMed for biomedical literature. Supports PubMed query syntax.",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Search query (supports PubMed syntax like author:[name], journal:[name])",
            required=True,
        ),
        ToolParameter(
            name="max_results",
            type="integer",
            description="Maximum number of results (default 10)",
            required=False,
            default=10,
        ),
        ToolParameter(
            name="sort",
            type="string",
            description="Sort order: 'relevance', 'pub_date', 'first_author'. Default: relevance",
            required=False,
            default="relevance",
        ),
        ToolParameter(
            name="min_date",
            type="string",
            description="Minimum publication date (YYYY or YYYY/MM/DD)",
            required=False,
        ),
        ToolParameter(
            name="max_date",
            type="string",
            description="Maximum publication date (YYYY or YYYY/MM/DD)",
            required=False,
        ),
    ],
)
def pubmed_search(
    query: str,
    max_results: int = 10,
    sort: str = "relevance",
    min_date: str = None,
    max_date: str = None,
) -> str:
    """Search PubMed for papers."""
    client = _get_scholar_client()

    args = {
        "query": query,
        "max_results": max_results,
        "sort": sort,
    }
    if min_date:
        args["min_date"] = min_date
    if max_date:
        args["max_date"] = max_date

    try:
        result = client.handle_pubmed_search(args)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="pubmed_get",
    description="Fetch a specific PubMed paper by ID. Returns full metadata including MeSH terms.",
    parameters=[
        ToolParameter(
            name="pmid",
            type="string",
            description="PubMed ID (numeric)",
            required=True,
        ),
    ],
)
def pubmed_get(pmid: str) -> str:
    """Get a specific PubMed paper."""
    client = _get_scholar_client()

    try:
        result = client.handle_pubmed_get({"pmid": pmid})
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="pubmed_fulltext",
    description="""Get full text from PMC Open Access. Only works for papers in the PMC Open Access Subset.
Provide either a PMID or PMCID.""",
    parameters=[
        ToolParameter(
            name="pmid",
            type="string",
            description="PubMed ID",
            required=False,
        ),
        ToolParameter(
            name="pmcid",
            type="string",
            description="PMC ID (e.g., 'PMC1234567')",
            required=False,
        ),
    ],
)
def pubmed_fulltext(pmid: str = None, pmcid: str = None) -> str:
    """Get full text from PMC."""
    client = _get_scholar_client()

    if not pmid and not pmcid:
        return json.dumps({"error": "Provide either pmid or pmcid"})

    args = {}
    if pmid:
        args["pmid"] = pmid
    if pmcid:
        args["pmcid"] = pmcid

    try:
        result = client.handle_pubmed_fulltext(args)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="pubmed_related",
    description="Find papers related to a given PubMed paper.",
    parameters=[
        ToolParameter(
            name="pmid",
            type="string",
            description="PubMed ID of the source paper",
            required=True,
        ),
        ToolParameter(
            name="max_results",
            type="integer",
            description="Maximum number of related papers (default 10)",
            required=False,
            default=10,
        ),
    ],
)
def pubmed_related(pmid: str, max_results: int = 10) -> str:
    """Find related PubMed papers."""
    client = _get_scholar_client()

    try:
        result = client.handle_pubmed_related({
            "pmid": pmid,
            "max_results": max_results,
        })
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="extract_url",
    description="Extract full text from any PDF or HTML URL. Works with arXiv, publisher sites, local files, etc.",
    parameters=[
        ToolParameter(
            name="url",
            type="string",
            description="URL to PDF or HTML page",
            required=True,
        ),
        ToolParameter(
            name="max_tokens",
            type="integer",
            description="Maximum number of tokens to return (optional)",
            required=False,
        ),
        ToolParameter(
            name="max_pages",
            type="integer",
            description="Limit PDF extraction to first N pages (optional)",
            required=False,
        ),
    ],
)
def extract_url(url: str, max_tokens: int = None, max_pages: int = None) -> str:
    """Extract text from a URL."""
    client = _get_scholar_client()

    args = {"url": url}
    if max_tokens:
        args["max_tokens"] = max_tokens
    if max_pages:
        args["max_pages"] = max_pages

    try:
        result = client.handle_extract_url(args)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


_openalex_client = None

def _get_openalex_client():
    global _openalex_client
    if _openalex_client is None:
        from .literature.openalex import OpenAlexClient
        _openalex_client = OpenAlexClient()
    return _openalex_client


@tool(
    name="openalex_search",
    description="""Search OpenAlex for academic papers across all disciplines (~250M works).
Supports date-range filtering. Higher rate limits than Semantic Scholar (10 req/s vs 1 req/s).
Use for broad keyword sweeps, date-filtered searches, and when S2 is rate-limited.""",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Search query for academic papers",
            required=True,
        ),
        ToolParameter(
            name="max_results",
            type="integer",
            description="Maximum number of results (default 20, max 200)",
            required=False,
            default=20,
        ),
        ToolParameter(
            name="date_from",
            type="string",
            description="Start date filter (YYYY-MM-DD format, e.g. '2024-01-01')",
            required=False,
        ),
        ToolParameter(
            name="date_to",
            type="string",
            description="End date filter (YYYY-MM-DD format, e.g. '2026-12-31')",
            required=False,
        ),
    ],
)
def openalex_search(query: str, max_results: int = 20, date_from: str = None, date_to: str = None) -> str:
    """Search OpenAlex for academic papers."""
    client = _get_openalex_client()
    try:
        results = client.search(query, max_results=max_results, date_from=date_from, date_to=date_to)
        return json.dumps({"query": query, "count": len(results), "results": results}, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# TOOL LIST HELPER
# =============================================================================

def get_all_tool_names() -> list[str]:
    """Get names of all tools registered by this module."""
    return [
        # Bash
        "bash_exec",
        # File
        "file_read", "file_edit", "file_create",
        # Exa
        "exa_search", "exa_fetch_full",
        # Gmail
        "gmail_list_from_date", "gmail_list_recent", "gmail_list_unread",
        "gmail_get_email", "gmail_send_message",
        "gmail_reply_to", "gmail_search_emails",
        "gmail_create_draft", "gmail_draft_reply",
        # Calendar
        "calendar_list_on_date", "calendar_create_event", "calendar_delete_event",
        # Notes
        "notes_search", "notes_get", "notes_list", "notes_add",
        "notes_delete", "notes_read", "notes_edit",
        # Account
        "account_get_current", "account_list", "account_set_current",
        # Browser
        "browser_session_status", "browser_session_open", "browser_session_close",
        "browser_goto", "browser_get_url", "browser_snapshot_controls",
        "browser_read_text", "browser_click", "browser_fill", "browser_type",
        "browser_press", "browser_select", "browser_back",
        # Literature Search
        "literature_search", "literature_fulltext",
        "arxiv_search", "arxiv_get", "arxiv_fulltext",
        "pubmed_search", "pubmed_get", "pubmed_fulltext", "pubmed_related",
        "extract_url",
        "openalex_search",
    ]


# =============================================================================
# MESH TOOLS
# =============================================================================

def _load_mesh_config():
    """Load and parse the mesh.yaml configuration file."""
    import yaml
    config_path = Path(__file__).parent.parent / "mesh.yaml"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return yaml.safe_load(f)


@tool(
    name="mesh_list",
    description="List all nodes configured in the mesh network, including their types, tools, and LLM backends.",
    parameters=[],
)
def mesh_list() -> str:
    """List all configured nodes in the mesh."""
    config = _load_mesh_config()
    if config is None:
        return "Error: mesh.yaml not found"

    nodes = config.get("nodes", {})
    if not nodes:
        return "No nodes configured in mesh."

    lines = ["## Mesh Nodes\n"]

    for node_id, node_config in nodes.items():
        parts = node_id.split(":")
        node_type = parts[0]  # "user" or "agent"
        node_name = parts[1] if len(parts) > 1 else node_id

        if node_type == "user":
            lines.append(f"### {node_id}")
            lines.append(f"- Type: user")
            lines.append("")
        else:
            lines.append(f"### {node_id}")
            lines.append(f"- Type: {node_type}")
            if node_config:
                backend = node_config.get("llm_backend", "default")
                model = node_config.get("llm_model", "")
                tools = node_config.get("tools", [])
                prompt_file = node_config.get("system_prompt_file", "")

                lines.append(f"- Backend: {backend}" + (f" ({model})" if model else ""))
                if prompt_file:
                    lines.append(f"- Role: {prompt_file.replace('.md', '')}")
                if tools:
                    lines.append(f"- Tools: {', '.join(tools[:5])}" + ("..." if len(tools) > 5 else ""))
            lines.append("")

    return "\n".join(lines)


# =============================================================================
# AGENT SHUTDOWN (remote control)
# =============================================================================

@tool(
    name="agent_shutdown",
    description=(
        "Remotely shut down an agent running on any host in the mesh. "
        "Use this to stop agents that are running on other machines when you "
        "cannot SSH to them directly. Requires the mesh auth token for validation."
    ),
    parameters=[
        ToolParameter(
            name="target",
            type="string",
            description=(
                "The agent node ID to shut down. Format: 'agent:{type}:{nickname}' "
                "(e.g., 'agent:assistant:alice', 'agent:coder:tron')."
            ),
            required=True,
        ),
        ToolParameter(
            name="reason",
            type="string",
            description="Optional reason for the shutdown (for logging).",
            required=False,
        ),
    ],
)
def agent_shutdown_tool(target: str, reason: str = "") -> str:
    """
    Placeholder handler - actual execution is handled by AgentNode._execute_agent_shutdown().

    This tool definition exists so it appears in the tool registry and prompts.
    The agent intercepts agent_shutdown calls and sends a control message.
    """
    return "Error: agent_shutdown should be handled by the agent, not executed directly"


# =============================================================================
# MESH STATUS (live agent dashboard)
# =============================================================================

@tool(
    name="mesh_status",
    description=(
        "Show live status of all agents in the mesh. "
        "Returns a dashboard with each agent's state (IDLE/BUSY), "
        "context token count, history utilization, memory stats, and uptime. "
        "Uses heartbeat data cached at the router — no round-trip to each agent."
    ),
    parameters=[],
)
def mesh_status_tool() -> str:
    """Placeholder — actual execution in AgentNode._execute_mesh_status()."""
    return "Error: mesh_status should be handled by the agent, not executed directly"


# =============================================================================
# AGENT STATUS (detailed diagnostics for one agent)
# =============================================================================

@tool(
    name="agent_status",
    description=(
        "Get detailed diagnostic status of any agent in the mesh. "
        "Returns router state, history stats, memory metrics, and health checks. "
        "Use target='self' for self-diagnosis, or a node ID like "
        "'agent:assistant:alice' for remote queries."
    ),
    parameters=[
        ToolParameter(
            name="target",
            type="string",
            description=(
                "Node ID to query, or 'self' for self-diagnosis. "
                "Format: 'agent:{type}:{nickname}' (e.g., 'agent:assistant:alice')."
            ),
            required=True,
        ),
        ToolParameter(
            name="section",
            type="string",
            description=(
                "Filter to one section: identity, llm, router, "
                "history, memory, context_health. Omit for all sections."
            ),
            required=False,
        ),
    ],
)
def agent_status_tool(target: str, section: str = None) -> str:
    """Placeholder — actual execution in AgentNode._execute_agent_status()."""
    return "Error: agent_status should be handled by the agent, not executed directly"


# =============================================================================
# SEND MESSAGE (required for routing responses)
# =============================================================================

@tool(
    name="send_message",
    description=(
        "Send a message to a user or channel. This is the ONLY way to deliver messages - "
        "plain text responses are not automatically routed. You MUST call this tool to "
        "communicate with users or other nodes."
    ),
    parameters=[
        ToolParameter(
            name="to",
            type="string",
            description=(
                "The recipient node ID. Use 'user:{name}' for users (e.g., 'user:yourname'), "
                "'agent:{type}:{name}' for agents, or 'channel:{name}' for channels."
            ),
            required=True,
        ),
        ToolParameter(
            name="content",
            type="string",
            description="The message content to send.",
            required=True,
        ),
        ToolParameter(
            name="attachments",
            type="array",
            description=(
                "Optional list of attachment objects returned by attach_file. "
                "Each object must include at least the attachment id."
            ),
            required=False,
        ),
    ],
)
def send_message_tool(to: str, content: str, attachments: list[dict] | None = None) -> str:
    """
    Placeholder handler - actual execution is handled by AgentNode._execute_send_message().

    This tool definition exists so it appears in the tool registry and prompts.
    The agent intercepts send_message calls and routes them specially.
    """
    # This should never be called directly - the agent handles it specially
    return "Error: send_message should be handled by the agent, not executed directly"


@tool(
    name="attach_file",
    description=(
        "Upload a local file to the mesh attachment store. Returns an attachment "
        "object that can be passed to send_message attachments."
    ),
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Absolute path to the file to attach.",
            required=True,
        ),
    ],
)
def attach_file_tool(path: str) -> str:
    """Placeholder handler - actual execution is handled by AgentNode."""
    return "Error: attach_file should be handled by the agent, not executed directly"


# =============================================================================
# CHANNEL TOOLS (query channel membership)
# =============================================================================

@tool(
    name="channel_list",
    description=(
        "List all channels you are a member of. Returns channel names, descriptions, "
        "member counts, and your membership status."
    ),
    parameters=[],
)
def channel_list_tool() -> str:
    """
    Placeholder handler - actual execution is handled by AgentNode._execute_channel_list().

    This tool definition exists so it appears in the tool registry and prompts.
    The agent intercepts channel_list calls and queries the router.
    """
    return "Error: channel_list should be handled by the agent, not executed directly"


@tool(
    name="channel_members",
    description=(
        "List all members of a specific channel. Returns member node IDs and "
        "their online/offline status. You must be a member of the channel to query it."
    ),
    parameters=[
        ToolParameter(
            name="channel_name",
            type="string",
            description=(
                "The channel name to query (without 'channel:' prefix). "
                "For example, use 'general' not 'channel:general'."
            ),
            required=True,
        ),
    ],
)
def channel_members_tool(channel_name: str) -> str:
    """
    Placeholder handler - actual execution is handled by AgentNode._execute_channel_members().

    This tool definition exists so it appears in the tool registry and prompts.
    The agent intercepts channel_members calls and queries the router.
    """
    return "Error: channel_members should be handled by the agent, not executed directly"


# =============================================================================
# PLAID (BANKING) TOOLS
# =============================================================================

_plaid_client = None


def _get_plaid_client(user_id: str = "default"):
    """Get or create PlaidClient instance."""
    global _plaid_client
    if _plaid_client is None:
        from .clients.plaid_client import PlaidClient
        _plaid_client = PlaidClient(user_id=user_id)
    return _plaid_client


@tool(
    name="plaid_link_start",
    description=(
        "Generate a Plaid Link URL for connecting a new bank account. "
        "Returns a URL that the user should open in their browser to authenticate "
        "with their bank. After authentication, Plaid will redirect to the callback URL."
    ),
    parameters=[],
)
def plaid_link_start() -> str:
    """Generate a Plaid Link token and return instructions."""
    client = _get_plaid_client()
    if not client.is_available():
        return "Error: Plaid not configured. Add credentials to ~/.config/mesh/plaid.yaml"

    result = client.get_link_token()
    if "error" in result:
        return f"Error: {result['error']}"

    # Build the hosted Link URL
    link_token = result["link_token"]
    redirect_uri = client.config.get("redirect_uri", "")

    return json.dumps({
        "status": "link_token_created",
        "link_token": link_token,
        "expiration": result["expiration"],
        "instructions": (
            "Open the Plaid Link URL in a browser to connect your bank account. "
            f"After authentication, you'll be redirected to: {redirect_uri}"
        ),
        "link_url": f"https://cdn.plaid.com/link/v2/stable/link.html?token={link_token}",
    }, indent=2)


@tool(
    name="plaid_link_status",
    description=(
        "Check which bank institutions are currently linked. "
        "Shows institution names and when they were connected."
    ),
    parameters=[],
)
def plaid_link_status() -> str:
    """List linked institutions."""
    client = _get_plaid_client()
    if not client.is_available():
        return "Error: Plaid not configured"

    institutions = client.list_linked_institutions()

    if not institutions:
        return "No bank accounts linked. Use plaid_link_start to connect one."

    lines = ["Linked Institutions:"]
    for inst in institutions:
        lines.append(f"  - {inst['institution_name']} ({inst['institution_id']})")
        lines.append(f"    Connected: {inst['created_at']}")

    return "\n".join(lines)


@tool(
    name="plaid_accounts",
    description=(
        "List all bank accounts and their current balances. "
        "Shows account names, types, and balance information."
    ),
    parameters=[
        ToolParameter(
            name="institution_id",
            type="string",
            description="Filter to specific institution (optional)",
            required=False,
        ),
    ],
)
def plaid_accounts(institution_id: str = None) -> str:
    """Get accounts and balances."""
    client = _get_plaid_client()
    if not client.is_available():
        return "Error: Plaid not configured"

    balances = client.get_balances(institution_id)

    if not balances:
        return "No accounts found. Link a bank first with plaid_link_start."

    if balances and "error" in balances[0]:
        return f"Error: {balances[0]['error']}"

    lines = ["Accounts:"]
    for acc in balances:
        name = acc.get("name", "Unknown")
        inst = acc.get("institution_name", "")
        acc_type = f"{acc.get('type', '')} / {acc.get('subtype', '')}"
        mask = acc.get("mask", "")
        current = acc.get("current")
        available = acc.get("available")
        currency = acc.get("currency", "USD")

        lines.append(f"\n  {name} (***{mask}) - {inst}")
        lines.append(f"    Type: {acc_type}")
        if current is not None:
            lines.append(f"    Current: {currency} {current:,.2f}")
        if available is not None:
            lines.append(f"    Available: {currency} {available:,.2f}")

    return "\n".join(lines)


@tool(
    name="plaid_sync",
    description=(
        "Sync latest transactions from all linked banks. "
        "This fetches new transactions and updates the local cache. "
        "Run this before querying transactions to get the latest data."
    ),
    parameters=[
        ToolParameter(
            name="institution_id",
            type="string",
            description="Sync only this institution (optional)",
            required=False,
        ),
    ],
)
def plaid_sync(institution_id: str = None) -> str:
    """Sync transactions from Plaid."""
    client = _get_plaid_client()
    if not client.is_available():
        return "Error: Plaid not configured"

    result = client.sync_transactions(institution_id)

    if "error" in result:
        return f"Error: {result['error']}"

    return json.dumps({
        "status": "sync_complete",
        "transactions_added": result["added"],
        "transactions_modified": result["modified"],
        "transactions_removed": result["removed"],
        "institutions_synced": result["institutions_synced"],
    }, indent=2)


@tool(
    name="plaid_transactions",
    description=(
        "Query transactions from the local cache. "
        "Supports date range filtering and account filtering. "
        "Run plaid_sync first to ensure you have the latest data."
    ),
    parameters=[
        ToolParameter(
            name="start_date",
            type="string",
            description="Start date (YYYY-MM-DD), defaults to 30 days ago",
            required=False,
        ),
        ToolParameter(
            name="end_date",
            type="string",
            description="End date (YYYY-MM-DD), defaults to today",
            required=False,
        ),
        ToolParameter(
            name="account_id",
            type="string",
            description="Filter to specific account ID (optional)",
            required=False,
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="Max transactions to return (default 100)",
            required=False,
            default=100,
        ),
    ],
)
def plaid_transactions(
    start_date: str = None,
    end_date: str = None,
    account_id: str = None,
    limit: int = 100,
) -> str:
    """Query cached transactions."""
    client = _get_plaid_client()
    if not client.is_available():
        return "Error: Plaid not configured"

    txns = client.get_transactions(
        start_date=start_date,
        end_date=end_date,
        account_id=account_id,
        limit=int(limit) if limit else 100,
    )

    if not txns:
        return "No transactions found. Try running plaid_sync first."

    # Format for display
    lines = [f"Transactions ({len(txns)} results):"]
    for txn in txns:
        date = txn.get("date", "")
        name = txn.get("merchant_name") or txn.get("name", "Unknown")
        amount = txn.get("amount", 0)
        currency = txn.get("currency", "USD")
        category = txn.get("category", "")
        pending = " (pending)" if txn.get("pending") else ""

        # Plaid amounts: positive = money out, negative = money in
        sign = "-" if amount > 0 else "+"
        amount_abs = abs(amount)

        lines.append(f"  {date}  {sign}{currency} {amount_abs:,.2f}  {name}{pending}")
        if category:
            lines.append(f"           Category: {category}")

    return "\n".join(lines)


@tool(
    name="plaid_unlink",
    description=(
        "Unlink a bank institution. This revokes access to that bank "
        "but keeps previously downloaded transactions in the cache."
    ),
    parameters=[
        ToolParameter(
            name="institution_id",
            type="string",
            description="The institution ID to unlink (use plaid_link_status to see IDs)",
            required=True,
        ),
    ],
    requires_confirmation=True,
)
def plaid_unlink(institution_id: str) -> str:
    """Unlink an institution."""
    client = _get_plaid_client()
    if not client.is_available():
        return "Error: Plaid not configured"

    result = client.unlink_institution(institution_id)

    if "error" in result:
        return f"Error: {result['error']}"

    return f"Successfully unlinked {institution_id}"


# =============================================================================
# SYNTHETIC API QUOTA
# =============================================================================

@tool(
    name="synthetic_quota",
    description=(
        "Check Synthetic.ai API quota usage. Returns current request count, "
        "limit, and when the quota resets. Use this before running benchmarks "
        "or when hitting rate limits."
    ),
    parameters=[],
)
def synthetic_quota() -> str:
    """Check Synthetic API quota."""
    import httpx

    api_key = os.environ.get("SYNTHETIC_API_KEY", "")
    if not api_key:
        return "Error: SYNTHETIC_API_KEY not set"

    try:
        r = httpx.get(
            "https://api.synthetic.new/v2/quotas",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return f"Error checking quota: {e}"

    sub = data.get("subscription", {})
    used = sub.get("requests", 0)
    limit = sub.get("limit", 0)
    renews = sub.get("renewsAt", "unknown")
    pct = (used / limit * 100) if limit > 0 else 0

    lines = [
        f"Subscription: {used}/{limit} requests ({pct:.0f}% used)",
        f"Resets at: {renews}",
    ]

    # Tool call discounts
    tc = data.get("toolCallDiscounts", {})
    if tc:
        tc_used = tc.get("requests", 0)
        tc_limit = tc.get("limit", 0)
        lines.append(f"Tool call discounts: {tc_used}/{tc_limit}")

    if pct >= 90:
        lines.append("WARNING: Quota nearly exhausted!")

    return "\n".join(lines)


# =============================================================================
# CLAUDE CODE USAGE
# =============================================================================

@tool(
    name="claude_code_usage",
    description=(
        "Check Claude Code Max subscription usage via OAuth. "
        "Shows utilization percentages and reset times for 5-hour, "
        "7-day, and per-model windows. Reads OAuth credentials "
        "from ~/.claude/.credentials.json."
    ),
    parameters=[],
)
def claude_code_usage() -> str:
    """Check Claude Code Max subscription usage."""
    import httpx
    import json as _json
    import time as _time
    from pathlib import Path

    creds_path = Path.home() / ".claude" / ".credentials.json"
    if not creds_path.exists():
        return "Error: Claude Code credentials not found at ~/.claude/.credentials.json"

    try:
        creds = _json.loads(creds_path.read_text())
    except Exception as e:
        return f"Error: Failed to parse credentials file: {e}"

    oauth = creds.get("claudeAiOauth")
    if not oauth or not isinstance(oauth, dict):
        return "Error: No claudeAiOauth section in credentials file"

    access_token = oauth.get("accessToken", "")
    refresh_token = oauth.get("refreshToken", "")
    expires_at_ms = oauth.get("expiresAt", 0)

    if not access_token:
        return "Error: No accessToken in credentials"

    # Refresh token if expired (10-minute buffer) and refresh_token available
    now_ms = int(_time.time() * 1000)
    token_refreshed = False
    if expires_at_ms > 0 and now_ms > expires_at_ms - 600_000 and refresh_token:
        try:
            r = httpx.post(
                "https://api.anthropic.com/v1/oauth/token",
                json={"grant_type": "refresh_token", "refresh_token": refresh_token},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if r.status_code == 200:
                token_data = r.json()
                access_token = token_data.get("access_token", access_token)
                new_refresh = token_data.get("refresh_token", refresh_token)
                new_expires_in = token_data.get("expires_in", 3600)
                new_expires_at = int(_time.time() * 1000) + (new_expires_in * 1000)

                # Try to write back updated tokens
                try:
                    oauth["accessToken"] = access_token
                    oauth["refreshToken"] = new_refresh
                    oauth["expiresAt"] = new_expires_at
                    creds["claudeAiOauth"] = oauth

                    # Preserve file permissions by writing to temp then renaming
                    import stat
                    tmp_path = creds_path.with_suffix(".tmp")
                    tmp_path.write_text(_json.dumps(creds, indent=2))
                    tmp_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600
                    tmp_path.rename(creds_path)
                    token_refreshed = True
                except Exception:
                    # Write-back failed, still use the refreshed token in-memory
                    token_refreshed = False
        except Exception:
            pass  # Refresh failed, try with existing token

    # Fetch usage with 429 retry
    data = None
    for attempt in range(3):
        try:
            r = httpx.get(
                "https://api.anthropic.com/api/oauth/usage",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "anthropic-beta": "oauth-2025-04-20",
                    "Content-Type": "application/json",
                    "User-Agent": "claude-code/2.1.69",
                },
                timeout=15,
            )
            if r.status_code == 401:
                return (
                    "Error: Token expired or invalid (HTTP 401). "
                    "Re-authenticate via Claude Code to refresh credentials."
                )
            if r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", 30 * (attempt + 1)))
                if attempt < 2:
                    _time.sleep(min(retry_after, 60))
                    continue
                return "Error: Rate limited by Anthropic API (HTTP 429). Try again later."
            r.raise_for_status()
            data = r.json()
            break
        except httpx.HTTPStatusError as e:
            return f"Error fetching usage: HTTP {e.response.status_code}"
        except Exception as e:
            if attempt == 2:
                return f"Error fetching usage: {e}"
            _time.sleep(5 * (attempt + 1))

    if data is None:
        return "Error: No data after retries"

    # Format output
    from datetime import datetime, timezone

    def _fmt_window(name: str, info: dict) -> str:
        util = info.get("utilization", 0)  # Already a percentage (0-100)
        resets_at = info.get("resets_at", "")
        reset_str = ""
        if resets_at:
            try:
                reset_dt = datetime.fromisoformat(resets_at.replace("Z", "+00:00"))
                now_dt = datetime.now(timezone.utc)
                delta = reset_dt - now_dt
                total_secs = int(delta.total_seconds())
                if total_secs > 0:
                    days = total_secs // 86400
                    hours = (total_secs % 86400) // 3600
                    mins = (total_secs % 3600) // 60
                    parts = []
                    if days:
                        parts.append(f"{days}d")
                    if hours:
                        parts.append(f"{hours}h")
                    if mins:
                        parts.append(f"{mins}m")
                    reset_str = f" (resets in {' '.join(parts)})"
                else:
                    reset_str = " (resetting now)"
            except Exception:
                reset_str = f" (resets: {resets_at})"
        return f"  {name}: {util:.1f}% used{reset_str}"

    lines = ["Claude Code Max Subscription Usage:\n"]

    window_names = {
        "five_hour": "Five Hour",
        "seven_day": "Seven Day",
        "seven_day_opus": "Seven Day (Opus)",
        "seven_day_sonnet": "Seven Day (Sonnet)",
        "seven_day_oauth_apps": "Seven Day (OAuth Apps)",
        "seven_day_cowork": "Seven Day (Cowork)",
    }

    for key, name in window_names.items():
        info = data.get(key)
        if info and info.get("utilization") is not None:
            lines.append(_fmt_window(name, info))

    # Extra usage (different format)
    extra = data.get("extra_usage")
    if extra and extra.get("is_enabled"):
        used = extra.get("used_credits", 0) or 0
        limit = extra.get("monthly_limit", 0) or 0
        util = extra.get("utilization")
        if limit > 0:
            lines.append(f"  Extra Usage: ${used:.2f} / ${limit:.2f}")
        elif util is not None:
            lines.append(f"  Extra Usage: {util:.1f}% used")

    if token_refreshed:
        lines.append("\n(Token was refreshed and saved to credentials file)")

    return "\n".join(lines)


# =============================================================================
# MEMORY TOOLS
# =============================================================================
# All memory tools use the module-level _memory_system singleton,
# set by AgentNode during init.


@tool(
    name="remember",
    description=(
        "Retrieve deeper details of a memory entry by ID. "
        "Returns the reflection (Tier 2). Set full=true to also include "
        "the tool call trace (Tier 3). Memory IDs are visible in the "
        "<memory> block in the system prompt."
    ),
    parameters=[
        ToolParameter(
            name="id",
            type="string",
            description="The memory entry ID to retrieve.",
            required=True,
        ),
        ToolParameter(
            name="full",
            type="boolean",
            description="If true, include the full trace in addition to the reflection.",
            required=False,
            default=False,
        ),
    ],
)
def remember(id: str, full: bool = False) -> str:
    """Retrieve deeper tiers of a memory entry."""
    if _memory_system is None:
        return "Error: Memory system not initialized."
    result = _memory_system.remember(id, full=full)
    if result is None:
        return f"No memory entry found with ID '{id}'."
    return result


@tool(
    name="memory_list",
    description=(
        "List all memory entries. Shows ID, date, tags, outcome, weight, "
        "and summary for each entry. Use the tag parameter to filter by exact tag match."
    ),
    parameters=[
        ToolParameter(
            name="tag",
            type="string",
            description="Filter entries to only those containing this exact tag.",
            required=False,
        ),
    ],
)
def memory_list(tag: str | None = None) -> str:
    """List all memory entries, optionally filtered by tag."""
    if _memory_system is None:
        return "Error: Memory system not initialized."
    entries = _memory_system.list_entries()
    if tag:
        entries = [e for e in entries if tag in e.tags]
    if not entries:
        if tag:
            return f"No memory entries with tag '{tag}'."
        return "No memory entries."
    active_count = sum(1 for e in entries if _memory_system.is_active(e.id))
    lines = []
    for e in entries:
        tags_str = ", ".join(e.tags) if e.tags else "-"
        date_str = e.created_at.strftime("%Y-%m-%d %H:%M")
        status = "[active]" if _memory_system.is_active(e.id) else "[pool]"
        lines.append(
            f"**{e.id}** {status} | {date_str} | {e.outcome} | w={e.weight:.3f} | "
            f"tags=[{tags_str}]\n  {e.summary}"
        )
    header = f"{len(entries)} entries"
    if tag:
        header += f" matching tag '{tag}'"
    header += f" ({active_count} active, {len(entries) - active_count} pool-only):"
    return header + "\n\n" + "\n\n".join(lines)


@tool(
    name="memory_get",
    description=(
        "Get full details of a memory entry: summary, reflection, trace, "
        "and metadata."
    ),
    parameters=[
        ToolParameter(
            name="id",
            type="string",
            description="The memory entry ID.",
            required=True,
        ),
    ],
)
def memory_get(id: str) -> str:
    """Get all three tiers of a memory entry."""
    if _memory_system is None:
        return "Error: Memory system not initialized."
    entry = _memory_system.get_entry(id)
    if entry is None:
        return f"No memory entry found with ID '{id}'."
    tags_str = ", ".join(entry.tags) if entry.tags else "-"
    date_str = entry.created_at.strftime("%Y-%m-%d %H:%M:%S")
    project_str = entry.project or "(none)"
    retrieval_key_str = entry.retrieval_key or "(none — pre-v3 entry)"

    max_chars = getattr(_memory_system, "_payload_max_chars", 6000)

    reflection = entry.reflection or "(none)"
    trace = entry.trace or "(none)"

    out = (
        f"**ID**: {entry.id}\n"
        f"**Date**: {date_str}\n"
        f"**Project**: {project_str}\n"
        f"**Outcome**: {entry.outcome}\n"
        f"**Weight**: {entry.weight:.4f}\n"
        f"**Tags**: {tags_str}\n"
        f"**Retrieval key**: {retrieval_key_str}\n\n"
        f"## Summary\n{entry.summary}\n\n"
        f"## Reflection\n{reflection}\n\n"
        f"## Trace\n{trace}"
    )
    if len(out) > max_chars:
        out = out[:max_chars] + f"\n\n[truncated: {len(out) - max_chars} more chars]"
    return out


@tool(
    name="memory_delete",
    description="Delete a memory entry by ID and recompute diversity weights.",
    parameters=[
        ToolParameter(
            name="id",
            type="string",
            description="The memory entry ID to delete.",
            required=True,
        ),
    ],
)
async def memory_delete(id: str) -> str:
    """Delete a memory entry."""
    if _memory_system is None:
        return "Error: Memory system not initialized."
    was_active = _memory_system.is_active(id)
    deleted = await _memory_system.delete_entry(id)
    if not deleted:
        return f"No memory entry found with ID '{id}'."
    if was_active:
        return f"Deleted memory entry '{id}' (was in active set). Active set reselected."
    return f"Deleted memory entry '{id}' (was pool-only)."


@tool(
    name="memory_add",
    description=(
        "Manually add a memory entry. The entry is always stored in the "
        "memory pool. Whether it enters the active set (shown in router "
        "prompt) depends on the diversity selection."
    ),
    parameters=[
        ToolParameter(
            name="summary",
            type="string",
            description="One-paragraph summary (Tier 1, always visible in context).",
            required=True,
        ),
        ToolParameter(
            name="reflection",
            type="string",
            description="Deeper reflection text (Tier 2, returned by remember tool).",
            required=False,
            default="",
        ),
        ToolParameter(
            name="trace",
            type="string",
            description="Tool call trace or additional detail (Tier 3).",
            required=False,
            default="",
        ),
        ToolParameter(
            name="tags",
            type="string",
            description="Comma-separated tags (e.g. 'nginx,benchmark,mesh-routing').",
            required=False,
            default="",
        ),
        ToolParameter(
            name="outcome",
            type="string",
            description="Outcome label: 'success', 'partial', or 'failure'.",
            required=False,
            default="success",
        ),
    ],
)
async def memory_add(
    summary: str,
    reflection: str = "",
    trace: str = "",
    tags: str = "",
    outcome: str = "success",
) -> str:
    """Manually add a memory entry."""
    if _memory_system is None:
        return "Error: Memory system not initialized."
    if outcome not in ("success", "partial", "failure"):
        return f"Invalid outcome '{outcome}'. Must be success, partial, or failure."
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    entry, accepted = await _memory_system.add_entry(
        summary=summary,
        reflection=reflection,
        trace=trace,
        tags=tag_list,
        outcome=outcome,
    )
    if accepted:
        return f"Added memory entry '{entry.id}' to pool and active set (tags={tag_list}, outcome={outcome})."
    return f"Added memory entry '{entry.id}' to pool (tags={tag_list}, outcome={outcome}). Not in active set — didn't improve diversity over current selection."


@tool(
    name="memory_search",
    description=(
        "Search the agent's memory pool. Supports three modes: "
        "'hybrid' (default) combines embedding similarity with lexical "
        "full-text search via reciprocal-rank fusion; 'embedding' uses "
        "cosine similarity only; 'lexical' uses FTS5/BM25 only. "
        "Use this when the user references prior work the TOC didn't "
        "surface — pronouns, 'have we…?', past sessions, cross-project "
        "queries. Returns top-k entries. "
        "Searches ALL projects by default; pass project=<name> to "
        "scope to a single project."
    ),
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="What you want to recall from memory.",
            required=True,
        ),
        ToolParameter(
            name="k",
            type="integer",
            description="How many results to return (default 5).",
            required=False,
            default=5,
        ),
        ToolParameter(
            name="mode",
            type="string",
            description="Search mode: 'hybrid' (default), 'embedding', or 'lexical'.",
            required=False,
            default="hybrid",
        ),
        ToolParameter(
            name="project",
            type="string",
            description="Project to scope the search to. Omit (or pass empty "
                        "string) to search all projects — the default.",
            required=False,
        ),
        ToolParameter(
            name="tag",
            type="string",
            description="Filter results to only entries containing this exact tag.",
            required=False,
        ),
    ],
)
async def memory_search(
    query: str,
    k: int = 5,
    mode: str | None = None,
    project: str | None = None,
    tag: str | None = None,
) -> str:
    """Search agent memory pool with embedding, lexical, or hybrid mode."""
    if mode is None:
        mode = _memory_search_mode
    if _memory_system is None:
        return "Error: Memory system not initialized."
    try:
        if hasattr(_memory_system, "search_block"):
            block = await _memory_system.search_block(
                query, k=k, project=project, tag=tag, mode=mode,
            )
        else:
            block = await _memory_system.render_block_for_query(query, k=k, tag=tag)
        return block or "No relevant memories found."
    except Exception as e:
        return f"Error during memory search: {e}"


# =============================================================================
# Memory editing tools (interactive correction — exempt from "once formed,
# never deleted" for corrections at the moment of realization)
# =============================================================================


@tool(
    name="memory_edit",
    description=(
        "Edit a minted memory entry in place by ID. The ID stays stable so "
        "digest citations referencing it do not dangle. Re-embeds changed "
        "fields so search stays consistent. Use this for corrections — "
        "edit beats delete when the core fact is salvageable."
    ),
    parameters=[
        ToolParameter(
            name="id",
            type="string",
            description="The memory entry ID to edit.",
            required=True,
        ),
        ToolParameter(
            name="summary",
            type="string",
            description="New summary text (Tier 1). Omit to keep current.",
            required=False,
        ),
        ToolParameter(
            name="reflection",
            type="string",
            description="New reflection text (Tier 2). Omit to keep current.",
            required=False,
        ),
        ToolParameter(
            name="retrieval_key",
            type="string",
            description="New retrieval key. Omit to keep current.",
            required=False,
        ),
        ToolParameter(
            name="tags",
            type="string",
            description="New comma-separated tags (replaces all existing tags). Omit to keep current.",
            required=False,
        ),
        ToolParameter(
            name="outcome",
            type="string",
            description="New outcome: 'success', 'partial', or 'failure'. Omit to keep current.",
            required=False,
        ),
    ],
)
async def memory_edit(
    id: str,
    summary: str | None = None,
    reflection: str | None = None,
    retrieval_key: str | None = None,
    tags: str | None = None,
    outcome: str | None = None,
) -> str:
    """Edit a memory entry in place, keeping the ID stable."""
    if _memory_system is None:
        return "Error: Memory system not initialized."
    if outcome is not None and outcome not in ("success", "partial", "failure"):
        return f"Invalid outcome '{outcome}'. Must be success, partial, or failure."

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags is not None else None

    if not hasattr(_memory_system, "edit_entry"):
        return "Error: memory editing requires MemorySystemV2."

    old = _memory_system.get_entry(id)
    if old is None:
        return f"No memory entry found with ID '{id}'."
    before = old.summary[:200]

    result = await _memory_system.edit_entry(
        id,
        summary=summary,
        reflection=reflection,
        retrieval_key=retrieval_key,
        tags=tag_list,
        outcome=outcome,
    )

    import logging
    logging.getLogger("mesh.memory.audit").info(
        "memory_edit id=%s before=%r after_summary=%r",
        id, before, (summary or "(unchanged)")[:200],
    )
    return result


# =============================================================================
# Standing digest tools (interactive correction of the published digest)
# =============================================================================


def _resolve_digest_path() -> str | None:
    """Return the absolute path to the agent's standing digest, or None."""
    if _memory_system is None:
        return None
    config = getattr(_memory_system, "_config", None) or getattr(_memory_system, "config", None)
    if config is None:
        return None
    raw = getattr(config, "standing_digest_path", "") or ""
    if not raw:
        return None
    return os.path.expanduser(raw)


@tool(
    name="digest_get",
    description=(
        "Read the agent's standing digest. Returns the full markdown text "
        "of the published digest file."
    ),
    parameters=[],
)
def digest_get() -> str:
    """Read the standing digest."""
    path = _resolve_digest_path()
    if not path:
        return "Error: no standing_digest_path configured for this agent."
    try:
        with open(path) as f:
            content = f.read()
    except FileNotFoundError:
        return f"Error: digest file not found at {path}."
    except OSError as e:
        return f"Error reading digest: {e}"
    if not content.strip():
        return "(digest is empty)"
    return content


@tool(
    name="digest_edit",
    description=(
        "Edit the agent's standing digest in place. Exact string replacement "
        "like map_edit / file_edit. Use this to correct inaccuracies — "
        "the next fold reads the corrected digest as its baseline."
    ),
    parameters=[
        ToolParameter(
            name="old_text",
            type="string",
            description="Exact text to find in the digest (must match uniquely unless replace_all=true).",
            required=True,
        ),
        ToolParameter(
            name="new_text",
            type="string",
            description="Replacement text.",
            required=True,
        ),
        ToolParameter(
            name="replace_all",
            type="boolean",
            description="Replace all occurrences (default false — requires unique match).",
            required=False,
        ),
    ],
)
def digest_edit(old_text: str, new_text: str, replace_all: bool = False) -> str:
    """Exact string replacement in the standing digest."""
    path = _resolve_digest_path()
    if not path:
        return "Error: no standing_digest_path configured for this agent."
    try:
        content = Path(path).read_text()
    except FileNotFoundError:
        return f"Error: digest file not found at {path}."
    except OSError as e:
        return f"Error reading digest: {e}"

    count = content.count(old_text)
    if count == 0:
        return "Error: old_text not found in digest."
    if not replace_all and count > 1:
        return (
            f"Error: old_text matches {count} locations in digest — "
            f"provide a more specific string or set replace_all=true."
        )

    if replace_all:
        new_content = content.replace(old_text, new_text)
    else:
        new_content = content.replace(old_text, new_text, 1)

    import fcntl
    try:
        with open(path, "r+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.seek(0)
            f.write(new_content)
            f.truncate()
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError as e:
        return f"Error writing digest: {e}"

    n_replaced = count if replace_all else 1
    import logging
    logging.getLogger("mesh.memory.audit").info(
        "digest_edit old=%r new=%r n=%d path=%s",
        old_text[:100], new_text[:100], n_replaced, path,
    )
    return f"Digest updated successfully ({n_replaced} replacement{'s' if n_replaced > 1 else ''})."


# =============================================================================
# Personality tools
# =============================================================================


@tool(
    name="personality_get",
    description=(
        "Get your current personality text. Returns the personality block "
        "that is injected into your system prompt."
    ),
    parameters=[],
)
def personality_get() -> str:
    """Get the agent's current personality."""
    if _memory_system is None:
        return "Error: Memory system not initialized."
    text = _memory_system.get_personality()
    if not text:
        return "No personality set."
    return text


@tool(
    name="personality_set",
    description=(
        "Set your personality text. This replaces your current personality "
        "entirely. The new personality takes effect on the next LLM call. "
        "Use personality_get first to see your current personality."
    ),
    parameters=[
        ToolParameter(
            name="content",
            type="string",
            description="The new personality text.",
            required=True,
        ),
    ],
)
async def personality_set(content: str) -> str:
    """Set the agent's personality."""
    if _memory_system is None:
        return "Error: Memory system not initialized."
    _memory_system.set_personality(content)
    return f"Personality updated ({len(content)} chars)."


# =============================================================================
# PROJECT MAP TOOLS (Memory v2)
# =============================================================================
# These tools are available when memory_version >= 2. They operate on
# project maps — living markdown documents that represent the agent's
# structural understanding of a project.


@tool(
    name="map_list",
    description=(
        "List all project maps. Shows project name, last updated, and "
        "whether each map is currently active."
    ),
    parameters=[],
)
async def map_list() -> str:
    """List all project maps."""
    if _memory_system is None:
        return "Error: Memory system not initialized."
    if not hasattr(_memory_system, 'list_maps'):
        return "Error: Project maps require memory_version: 2."
    maps = await _memory_system.list_maps()
    if not maps:
        return "No project maps."
    lines = []
    for m in maps:
        active_str = " [ACTIVE]" if m.get("is_active") else ""
        lines.append(
            f"**{m['project_name']}**{active_str} | "
            f"updated: {m['updated_at'][:16]}"
        )
    return f"{len(maps)} project maps:\n\n" + "\n".join(lines)


@tool(
    name="map_get",
    description="Read the full content of a project map.",
    parameters=[
        ToolParameter(
            name="project_name",
            type="string",
            description="The project name (e.g., 'hello-world', 'mesh-system').",
            required=True,
        ),
    ],
)
async def map_get(project_name: str) -> str:
    """Read a project map."""
    if _memory_system is None:
        return "Error: Memory system not initialized."
    if not hasattr(_memory_system, 'get_map'):
        return "Error: Project maps require memory_version: 2."
    content = await _memory_system.get_map(project_name)
    if content is None:
        return f"No map found for project '{project_name}'."
    return content


@tool(
    name="map_edit",
    description=(
        "Line-edit a project map. Find the exact text that's wrong and "
        "provide the replacement. Like file_edit but for project maps. "
        "Use this when the user corrects, clarifies, or refines understanding."
    ),
    parameters=[
        ToolParameter(
            name="project_name",
            type="string",
            description="The project name (must match an existing map).",
            required=True,
        ),
        ToolParameter(
            name="old_text",
            type="string",
            description="Exact text to find in the map (must match uniquely).",
            required=True,
        ),
        ToolParameter(
            name="new_text",
            type="string",
            description="Replacement text.",
            required=True,
        ),
        ToolParameter(
            name="replace_all",
            type="boolean",
            description="Replace all occurrences (default false — requires unique match).",
            required=False,
        ),
    ],
)
async def map_edit(
    project_name: str, old_text: str, new_text: str,
    replace_all: bool = False,
) -> str:
    """Exact string replacement in a project map."""
    if _memory_system is None:
        return "Error: Memory system not initialized."
    if not hasattr(_memory_system, 'apply_map_edit'):
        return "Error: Project maps require memory_version: 2."
    return await _memory_system.apply_map_edit(
        project_name, old_text, new_text, replace_all=replace_all,
    )


@tool(
    name="map_create",
    description="Create a new project map with the given content.",
    parameters=[
        ToolParameter(
            name="project_name",
            type="string",
            description="The project name (e.g., 'my-project').",
            required=True,
        ),
        ToolParameter(
            name="content",
            type="string",
            description="The full map content (markdown).",
            required=True,
        ),
    ],
)
async def map_create(project_name: str, content: str) -> str:
    """Create a new project map."""
    if _memory_system is None:
        return "Error: Memory system not initialized."
    if not hasattr(_memory_system, 'create_map'):
        return "Error: Project maps require memory_version: 2."
    # Check if map already exists
    existing = await _memory_system.get_map(project_name)
    if existing is not None:
        return (
            f"Map '{project_name}' already exists. "
            f"Use map_edit to modify it, or set_project_context with reset=true."
        )
    ok = await _memory_system.create_map(project_name, content)
    if not ok:
        return f"Error: failed to create map '{project_name}' — no project_dir available. Use set_project_context first."
    return f"Map '{project_name}' created ({len(content)} chars)."


@tool(
    name="set_project_context",
    description=(
        "Initialize or load a project context. Sets the active project "
        "for this agent. If no map exists, runs an exhaustive scan of the "
        "project directory to build one. Set reset=true to force a fresh "
        "scan even if a map already exists."
    ),
    parameters=[
        ToolParameter(
            name="project_dir",
            type="string",
            description="Full path to the project directory.",
            required=True,
        ),
        ToolParameter(
            name="reset",
            type="boolean",
            description="If true, discard existing map and re-scan from scratch.",
            required=False,
            default=False,
        ),
    ],
)
async def set_project_context(project_dir: str, reset: bool = False) -> str:
    """Initialize or load a project context."""
    if _memory_system is None:
        return "Error: Memory system not initialized."
    if not hasattr(_memory_system, 'set_project_context'):
        return "Error: Project maps require memory_version: 2."
    return await _memory_system.set_project_context(project_dir, reset=reset)


@tool(
    name="map_review",
    description=(
        "Deep review of the active project map against the current filesystem state. "
        "Scans the project directory, compares every claim in the map against what's "
        "actually on disk and fixes discrepancies using interactive exploration. "
        "Use this when the map may have drifted from reality. "
        "project_dir is optional if the project was previously set via set_project_context."
    ),
    parameters=[
        ToolParameter(
            name="project_dir",
            type="string",
            description="Full path to the project directory. Optional if already set via set_project_context.",
            required=False,
        ),
    ],
)
async def map_review(project_dir: str = "") -> str:
    """Review and reconcile the active project map against the filesystem."""
    if _memory_system is None:
        return "Error: Memory system not initialized."
    if not hasattr(_memory_system, 'review_active_map'):
        return "Error: Map review requires memory_version: 2."

    result = await _memory_system.review_active_map(project_dir or None)
    return result["summary"]


# =============================================================================
# Router Worker Tools
# =============================================================================
# These tools are intercepted by AgentNode._execute_all_tools and routed
# through RouterV2's per-instance _worker_tool_handlers dict BEFORE the
# global ToolRegistry. Handlers here are None — the real execution path
# is in router_v2.py's _tool_worker_launch / _tool_worker_status.
# Registered with None handlers so they appear in tool_help / mesh-tool.


@tool(
    name="worker_launch",
    description=(
        "Launch a worker to execute a task autonomously. The worker receives "
        "full conversation context and can use all tools including file edits "
        "and bash. Returns immediately — the router continues its loop while "
        "the worker runs. Use this for tasks requiring sustained autonomous "
        "work: code changes, multi-step investigations, deployments."
    ),
    parameters=[
        ToolParameter(
            name="task",
            type="string",
            description=(
                "Rich task description for the worker. Be specific about what "
                "needs to be done, what files/contexts are relevant, and what "
                "success looks like."
            ),
            required=True,
        ),
    ],
)
async def worker_launch(task: str) -> str:  # pragma: no cover
    """Placeholder — executed by RouterV2._tool_worker_launch."""
    raise NotImplementedError("worker_launch is handled by RouterV2")


@tool(
    name="worker_status",
    description=(
        "Check progress of the currently running worker. There is only one "
        "worker at a time — do NOT pass a worker_id (no such parameter "
        "exists). The ONLY accepted parameter is max_lines (integer, "
        "default 100). Returns metadata (elapsed time, task description, "
        "tool call count, state) plus the worker's activity transcript. "
        "Set max_lines=0 for the full unbounded transcript."
    ),
    parameters=[
        ToolParameter(
            name="max_lines",
            type="integer",
            description=(
                "Maximum activity lines to return. Default 100. Set to 0 "
                "for the full unbounded transcript."
            ),
            required=False,
            default=100,
        ),
    ],
)
async def worker_status(max_lines: int = 100) -> str:  # pragma: no cover
    """Placeholder — executed by RouterV2._tool_worker_status."""
    raise NotImplementedError("worker_status is handled by RouterV2")


@tool(
    name="worker_stop",
    description=(
        "Stop the currently running worker (self-cancellation). Call this when "
        "you have completed your task or determined that you cannot make further "
        "progress. Sets a cooperative cancellation flag — the worker loop exits "
        "cleanly on the next iteration check."
    ),
    parameters=[
        ToolParameter(
            name="reason",
            type="string",
            description="Why the worker is stopping (logged for diagnostics).",
            required=False,
            default="Worker self-stop",
        ),
    ],
)
async def worker_stop(reason: str = "Worker self-stop") -> str:  # pragma: no cover
    """Placeholder — executed by AgentNode._execute_special_tool."""
    raise NotImplementedError("worker_stop is handled by AgentNode")


# =============================================================================
# Conversation Todo Tools
# =============================================================================
# These tools are executed by AgentNode._execute_special_tool so they can default
# conversation_id from the triggering message and route through the router broker.


@tool(
    name="todo_list",
    description="List todo items for the current conversation, or for an explicit conversation_id.",
    parameters=[
        ToolParameter(name="conversation_id", type="string", description="Conversation ID. Defaults to the triggering conversation.", required=False),
        ToolParameter(name="include_done", type="boolean", description="Include done and cancelled items. Default true.", required=False, default=True),
        ToolParameter(name="limit", type="integer", description="Maximum items to return.", required=False, default=100),
    ],
)
async def todo_list(conversation_id: str = None, include_done: bool = True, limit: int = 100) -> str:  # pragma: no cover
    raise NotImplementedError("todo_list is handled by AgentNode")


@tool(
    name="todo_add",
    description="Add a todo item to the current conversation, optionally under a custom section.",
    parameters=[
        ToolParameter(name="text", type="string", description="Todo item text.", required=True),
        ToolParameter(name="conversation_id", type="string", description="Conversation ID. Defaults to the triggering conversation.", required=False),
        ToolParameter(name="section", type="string", description="Optional custom section label, e.g. today or medium-term.", required=False),
        ToolParameter(name="priority", type="integer", description="Optional priority. Default 0.", required=False, default=0),
        ToolParameter(name="position", type="integer", description="Optional sparse display position. Defaults to append.", required=False),
    ],
)
async def todo_add(text: str, conversation_id: str = None, section: str = None, priority: int = 0, position: int = None) -> str:  # pragma: no cover
    raise NotImplementedError("todo_add is handled by AgentNode")


@tool(
    name="todo_update",
    description="Update a todo item text, status, section, priority, or position.",
    parameters=[
        ToolParameter(name="todo_id", type="string", description="Stable todo ID.", required=True),
        ToolParameter(name="text", type="string", description="Replacement todo text.", required=False),
        ToolParameter(name="section", type="string", description="Replacement section label. Use an empty string to clear.", required=False),
        ToolParameter(name="status", type="string", description="open, in_progress, done, or cancelled.", required=False),
        ToolParameter(name="priority", type="integer", description="New priority.", required=False),
        ToolParameter(name="position", type="integer", description="New sparse display position.", required=False),
        ToolParameter(name="expected_version", type="integer", description="Optional optimistic-concurrency version.", required=False),
        ToolParameter(name="conversation_id", type="string", description="Conversation ID. Defaults to the triggering conversation.", required=False),
    ],
)
async def todo_update(todo_id: str, text: str = None, section: str = None, status: str = None, priority: int = None, position: int = None, expected_version: int = None, conversation_id: str = None) -> str:  # pragma: no cover
    raise NotImplementedError("todo_update is handled by AgentNode")


@tool(
    name="todo_toggle",
    description="Mark a todo done or reopen it.",
    parameters=[
        ToolParameter(name="todo_id", type="string", description="Stable todo ID.", required=True),
        ToolParameter(name="done", type="boolean", description="True marks done; false reopens to open.", required=False, default=True),
        ToolParameter(name="expected_version", type="integer", description="Optional optimistic-concurrency version.", required=False),
        ToolParameter(name="conversation_id", type="string", description="Conversation ID. Defaults to the triggering conversation.", required=False),
    ],
)
async def todo_toggle(todo_id: str, done: bool = True, expected_version: int = None, conversation_id: str = None) -> str:  # pragma: no cover
    raise NotImplementedError("todo_toggle is handled by AgentNode")


@tool(
    name="todo_remove",
    description="Soft-delete a todo item from the current conversation.",
    parameters=[
        ToolParameter(name="todo_id", type="string", description="Stable todo ID.", required=True),
        ToolParameter(name="expected_version", type="integer", description="Optional optimistic-concurrency version.", required=False),
        ToolParameter(name="conversation_id", type="string", description="Conversation ID. Defaults to the triggering conversation.", required=False),
    ],
)
async def todo_remove(todo_id: str, expected_version: int = None, conversation_id: str = None) -> str:  # pragma: no cover
    raise NotImplementedError("todo_remove is handled by AgentNode")


@tool(
    name="todo_reorder",
    description="Replace todo display ordering for a conversation. Positions become dense in supplied order.",
    parameters=[
        ToolParameter(name="ordered_ids", type="array", description="Todo IDs in desired display order.", required=True),
        ToolParameter(name="conversation_id", type="string", description="Conversation ID. Defaults to the triggering conversation.", required=False),
    ],
)
async def todo_reorder(ordered_ids: list, conversation_id: str = None) -> str:  # pragma: no cover
    raise NotImplementedError("todo_reorder is handled by AgentNode")


@tool(
    name="todo_set_section_order",
    description="Set the custom display order for todo sections in a conversation.",
    parameters=[
        ToolParameter(name="section_order", type="array", description="Section labels in desired display order. Empty list clears custom order.", required=True),
        ToolParameter(name="conversation_id", type="string", description="Conversation ID. Defaults to the triggering conversation.", required=False),
    ],
)
async def todo_set_section_order(section_order: list, conversation_id: str = None) -> str:  # pragma: no cover
    raise NotImplementedError("todo_set_section_order is handled by AgentNode")


# =============================================================================
# Scratchpad Read Tool (agent read-only access)
# =============================================================================

@tool(
    name="scratchpad_read",
    description=(
        "Read the scratchpad notes for the current conversation. "
        "Scratchpads are per-conversation user notes that persist across "
        "sessions. Read-only — agents cannot write to scratchpads."
    ),
    parameters=[
        ToolParameter(
            name="conversation_id",
            type="string",
            description="The conversation ID to read the scratchpad for.",
            required=True,
        ),
    ],
)
async def scratchpad_read(conversation_id: str) -> str:
    """Read scratchpad content for a conversation."""
    import json as _json
    from .storage import MessageStore
    from pathlib import Path

    from .paths import real_home
    db_path = real_home() / ".mesh" / "router.db"
    if not db_path.exists():
        return _json.dumps({"error": "Router database not found"})

    store = MessageStore(str(db_path))
    note = store.get_scratchpad(conversation_id)
    if note is None:
        return _json.dumps({"conversation_id": conversation_id, "content": "", "exists": False})
    return _json.dumps({
        "conversation_id": conversation_id,
        "content": note["content"],
        "updated_at": note["updated_at"],
        "updated_by": note["updated_by"],
        "exists": True,
    })


# =============================================================================
# History Search Tool (read-only, lossless recall over raw conversation history)
# =============================================================================

def _resolve_history_file() -> "Path | None":
    """Resolve the calling agent's history file.

    Identity comes from MESH_NODE_ID (set by AgentNode in its own process
    and inherited by tool subprocesses), matching how mesh-tool identifies
    the calling agent. Paths come from mesh.paths.HISTORY_DIR — no
    hardcoded home directories.
    """
    from .paths import HISTORY_DIR

    node_id = os.environ.get("MESH_NODE_ID", "")
    if not node_id:
        return None
    # AgentNode persists to agent-{nickname}.json (nickname = last segment)
    nickname = node_id.split(":")[-1]
    candidate = HISTORY_DIR / f"agent-{nickname}.json"
    if candidate.exists():
        return candidate
    # Fallback: base Node default path uses the full node id
    fallback = HISTORY_DIR / f"{node_id.replace(':', '-')}.json"
    if fallback.exists():
        return fallback
    return candidate  # Return primary path so the error message names it


def _iter_history_entries(path: "Path"):
    """Yield raw history entry dicts from a JSONL (or legacy JSON array) file."""
    with open(path, "r") as f:
        first_char = f.read(1)
        if not first_char:
            return
        f.seek(0)
        if first_char == "[":
            for entry in json.load(f):
                yield entry
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # Skip corrupt lines, same as Node.load_history


def _history_snippet(content: str, needle: str, width: int = 300) -> str:
    """Trim content to a ~width-char window centered on the first match."""
    flat = " ".join(content.split())  # Collapse whitespace/newlines
    if len(flat) <= width:
        return flat
    idx = flat.lower().find(needle.lower())
    if idx < 0:
        idx = 0
    start = max(0, idx - width // 3)
    end = min(len(flat), start + width)
    start = max(0, end - width)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(flat) else ""
    return f"{prefix}{flat[start:end]}{suffix}"


@tool(
    name="history_search",
    description=(
        "Full-text search over your own raw conversation history — the "
        "durable, lossless record of every message you've sent or received. "
        "Use this when memory_search comes up empty or you need exact "
        "wording, timestamps, or details from past sessions that memory "
        "summaries may have pruned. Case-insensitive keyword match (all "
        "words must appear); returns newest matches first as compact "
        "snippets with timestamp and from/to."
    ),
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Keywords to search for (case-insensitive; all words must appear in a message).",
            required=True,
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="Maximum matches to return (default 10).",
            required=False,
            default=10,
        ),
        ToolParameter(
            name="role",
            type="string",
            description="Filter by sender: 'user', 'agent', 'incoming', 'outgoing', or a full node id prefix (e.g. 'agent:sysadmin:bob').",
            required=False,
        ),
        ToolParameter(
            name="date_from",
            type="string",
            description="Only messages on/after this ISO date or datetime (e.g. '2026-02-01').",
            required=False,
        ),
        ToolParameter(
            name="date_to",
            type="string",
            description="Only messages on/before this ISO date or datetime (inclusive).",
            required=False,
        ),
    ],
    requires_confirmation=False,
)
def history_search(
    query: str,
    limit: int = 10,
    role: str = None,
    date_from: str = None,
    date_to: str = None,
) -> str:
    """Deterministic keyword search over the calling agent's raw history file."""
    history_file = _resolve_history_file()
    if history_file is None:
        return (
            "Error: cannot determine calling agent — MESH_NODE_ID is not set. "
            "history_search searches the calling agent's own history."
        )
    if not history_file.exists():
        return f"Error: no history file found at {history_file}."

    tokens = [t.lower() for t in query.split() if t.strip()]
    if not tokens:
        return "Error: empty query."
    if limit is None or limit < 1:
        limit = 10

    # Inclusive date_to: a bare date must include the whole day.
    date_to_eff = None
    if date_to:
        date_to_eff = date_to if "T" in date_to else date_to + "T~"  # '~' > any time char

    role_norm = role.strip().lower() if role else None

    total = 0
    scanned = 0
    hits: list[str] = []  # Built newest-first below
    matches: list[dict] = []

    try:
        for entry in _iter_history_entries(history_file):
            scanned += 1
            msg = entry.get("message") or {}
            content = msg.get("content")
            if not isinstance(content, str):
                continue

            ts = msg.get("timestamp") or ""
            if date_from and ts and ts < date_from:
                continue
            if date_to_eff and ts and ts > date_to_eff:
                continue

            if role_norm:
                direction = (entry.get("direction") or "").lower()
                from_node = (msg.get("from_node") or "").lower()
                if role_norm in ("incoming", "outgoing"):
                    if direction != role_norm:
                        continue
                elif not from_node.startswith(role_norm):
                    continue

            lowered = content.lower()
            if not all(t in lowered for t in tokens):
                continue

            total += 1
            matches.append(
                {
                    "ts": ts,
                    "from": msg.get("from_node") or "?",
                    "to": msg.get("to_node") or "?",
                    "direction": entry.get("direction") or "?",
                    "snippet": _history_snippet(content, tokens[0]),
                }
            )
    except OSError as e:
        return f"Error reading history file {history_file}: {e}"

    if not matches:
        filters = []
        if role:
            filters.append(f"role={role}")
        if date_from:
            filters.append(f"from={date_from}")
        if date_to:
            filters.append(f"to={date_to}")
        filter_str = f" ({', '.join(filters)})" if filters else ""
        return (
            f"No matches for '{query}'{filter_str} in {history_file.name} "
            f"({scanned} entries scanned)."
        )

    # Newest first (file is chronological; sort by timestamp to be safe)
    matches.sort(key=lambda m: m["ts"], reverse=True)
    shown = matches[:limit]
    for m in shown:
        ts_short = m["ts"][:19] if m["ts"] else "unknown-time"
        hits.append(f"[{ts_short} | {m['from']} → {m['to']} | {m['direction']}]\n{m['snippet']}")

    header = (
        f"{total} match{'es' if total != 1 else ''} for '{query}' in "
        f"{history_file.name} ({scanned} entries scanned), "
        f"showing {len(shown)} newest first:"
    )
    return header + "\n\n" + "\n\n".join(hits)


# =============================================================================
# CANVAS LMS
# =============================================================================

def _get_canvas_client():
    """Lazy-init Canvas client singleton."""
    from .clients.canvas_client import CanvasClient
    return CanvasClient()


@tool(
    name="canvas_list_students",
    description="List students enrolled in a Canvas course.",
    parameters=[
        ToolParameter(name="course_id", type="integer", description="Course ID (uses active course if omitted)", required=False),
        ToolParameter(name="limit", type="integer", description="Max students to return (default 200)", required=False),
    ],
)
def canvas_list_students(course_id: int = None, limit: int = None) -> str:
    try:
        client = _get_canvas_client()
        kwargs: dict[str, Any] = {}
        if course_id is not None:
            kwargs["course_id"] = course_id
        if limit is not None:
            kwargs["limit"] = limit
        result = client.list_students(**kwargs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="canvas_list_assignments",
    description="List assignments in a Canvas course.",
    parameters=[
        ToolParameter(name="course_id", type="integer", description="Course ID (uses active course if omitted)", required=False),
        ToolParameter(name="bucket", type="string", description="Filter: past, overdue, undated, ungraded, unsubmitted, upcoming, future", required=False),
        ToolParameter(name="limit", type="integer", description="Max assignments to return (default 50)", required=False),
    ],
)
def canvas_list_assignments(course_id: int = None, bucket: str = None, limit: int = None) -> str:
    try:
        client = _get_canvas_client()
        kwargs: dict[str, Any] = {}
        if course_id is not None:
            kwargs["course_id"] = course_id
        if bucket is not None:
            kwargs["bucket"] = bucket
        if limit is not None:
            kwargs["limit"] = limit
        result = client.list_assignments(**kwargs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="canvas_list_submissions",
    description="List submissions for an assignment in a Canvas course.",
    parameters=[
        ToolParameter(name="assignment_id", type="integer", description="Assignment ID", required=True),
        ToolParameter(name="course_id", type="integer", description="Course ID (uses active course if omitted)", required=False),
        ToolParameter(name="include", type="string", description="Extra data to include (e.g. 'submission_comments,rubric_assessment')", required=False),
        ToolParameter(name="limit", type="integer", description="Max submissions to return (default 200)", required=False),
    ],
)
def canvas_list_submissions(assignment_id: int, course_id: int = None, include: str = None, limit: int = None) -> str:
    try:
        client = _get_canvas_client()
        kwargs: dict[str, Any] = {"assignment_id": assignment_id}
        if course_id is not None:
            kwargs["course_id"] = course_id
        if include is not None:
            kwargs["include"] = include
        if limit is not None:
            kwargs["limit"] = limit
        result = client.list_submissions(**kwargs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="canvas_get_grades",
    description="Get grades (enrollments with scores) for a Canvas course. Optionally filter by student.",
    parameters=[
        ToolParameter(name="course_id", type="integer", description="Course ID (uses active course if omitted)", required=False),
        ToolParameter(name="student_id", type="integer", description="Filter to a specific student", required=False),
        ToolParameter(name="limit", type="integer", description="Max enrollments to return (default 200)", required=False),
    ],
)
def canvas_get_grades(course_id: int = None, student_id: int = None, limit: int = None) -> str:
    try:
        client = _get_canvas_client()
        kwargs: dict[str, Any] = {}
        if course_id is not None:
            kwargs["course_id"] = course_id
        if student_id is not None:
            kwargs["student_id"] = student_id
        if limit is not None:
            kwargs["limit"] = limit
        result = client.get_grades(**kwargs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="canvas_list_announcements",
    description="List announcements in a Canvas course.",
    parameters=[
        ToolParameter(name="course_id", type="integer", description="Course ID (uses active course if omitted)", required=False),
        ToolParameter(name="limit", type="integer", description="Max announcements to return (default 20)", required=False),
    ],
)
def canvas_list_announcements(course_id: int = None, limit: int = None) -> str:
    try:
        client = _get_canvas_client()
        kwargs: dict[str, Any] = {}
        if course_id is not None:
            kwargs["course_id"] = course_id
        if limit is not None:
            kwargs["limit"] = limit
        result = client.list_announcements(**kwargs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="canvas_list_modules",
    description="List modules in a Canvas course.",
    parameters=[
        ToolParameter(name="course_id", type="integer", description="Course ID (uses active course if omitted)", required=False),
        ToolParameter(name="limit", type="integer", description="Max modules to return (default 50)", required=False),
    ],
)
def canvas_list_modules(course_id: int = None, limit: int = None) -> str:
    try:
        client = _get_canvas_client()
        kwargs: dict[str, Any] = {}
        if course_id is not None:
            kwargs["course_id"] = course_id
        if limit is not None:
            kwargs["limit"] = limit
        result = client.list_modules(**kwargs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="canvas_list_pages",
    description="List pages in a Canvas course.",
    parameters=[
        ToolParameter(name="course_id", type="integer", description="Course ID (uses active course if omitted)", required=False),
        ToolParameter(name="sort", type="string", description="Sort order: title, created_at, updated_at", required=False),
        ToolParameter(name="published", type="boolean", description="Filter by published status", required=False),
        ToolParameter(name="limit", type="integer", description="Max pages to return (default 50)", required=False),
    ],
)
def canvas_list_pages(course_id: int = None, sort: str = None, published: bool = None, limit: int = None) -> str:
    try:
        client = _get_canvas_client()
        kwargs: dict[str, Any] = {}
        if course_id is not None:
            kwargs["course_id"] = course_id
        if sort is not None:
            kwargs["sort"] = sort
        if published is not None:
            kwargs["published"] = published
        if limit is not None:
            kwargs["limit"] = limit
        result = client.list_pages(**kwargs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="canvas_grade_submission",
    description="Grade a student's submission. Requires confirmation.",
    parameters=[
        ToolParameter(name="assignment_id", type="integer", description="Assignment ID", required=True),
        ToolParameter(name="student_id", type="integer", description="Student's Canvas user ID", required=True),
        ToolParameter(name="grade", type="string", description="Grade value (e.g. '95', 'A', 'pass')", required=True),
        ToolParameter(name="course_id", type="integer", description="Course ID (uses active course if omitted)", required=False),
        ToolParameter(name="comment", type="string", description="Grading comment visible to student", required=False),
    ],
    requires_confirmation=True,
)
def canvas_grade_submission(assignment_id: int, student_id: int, grade: str, course_id: int = None, comment: str = None) -> str:
    try:
        client = _get_canvas_client()
        kwargs: dict[str, Any] = {
            "assignment_id": assignment_id,
            "student_id": student_id,
            "grade": grade,
        }
        if course_id is not None:
            kwargs["course_id"] = course_id
        if comment is not None:
            kwargs["comment"] = comment
        result = client.grade_submission(**kwargs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="canvas_post_announcement",
    description="Post an announcement to a Canvas course. Requires confirmation.",
    parameters=[
        ToolParameter(name="title", type="string", description="Announcement title", required=True),
        ToolParameter(name="message", type="string", description="Announcement body (HTML supported)", required=True),
        ToolParameter(name="course_id", type="integer", description="Course ID (uses active course if omitted)", required=False),
        ToolParameter(name="delayed_post_at", type="string", description="Schedule for later (ISO 8601 datetime)", required=False),
    ],
    requires_confirmation=True,
)
def canvas_post_announcement(title: str, message: str, course_id: int = None, delayed_post_at: str = None) -> str:
    try:
        client = _get_canvas_client()
        kwargs: dict[str, Any] = {"title": title, "message": message}
        if course_id is not None:
            kwargs["course_id"] = course_id
        if delayed_post_at is not None:
            kwargs["delayed_post_at"] = delayed_post_at
        result = client.post_announcement(**kwargs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="canvas_auth_status",
    description="Check Canvas API authentication status and active course.",
    parameters=[],
)
def canvas_auth_status() -> str:
    try:
        client = _get_canvas_client()
        if not client.is_available():
            return json.dumps({"authenticated": False, "error": "No access token configured"})
        user = client.get_self()
        active_course = client.get_active_course()
        return json.dumps({
            "authenticated": True,
            "user_id": user.get("id"),
            "user_name": user.get("name"),
            "base_url": client.base_url,
            "active_course_id": active_course,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"authenticated": False, "error": str(e)})


@tool(
    name="canvas_list_courses",
    description="List Canvas courses for the authenticated user.",
    parameters=[
        ToolParameter(name="enrollment_state", type="string", description="Filter: active, completed, invited (default active)", required=False),
        ToolParameter(name="limit", type="integer", description="Max courses to return (default 50)", required=False),
    ],
)
def canvas_list_courses(enrollment_state: str = None, limit: int = None) -> str:
    try:
        client = _get_canvas_client()
        kwargs: dict[str, Any] = {}
        if enrollment_state is not None:
            kwargs["enrollment_state"] = enrollment_state
        if limit is not None:
            kwargs["limit"] = limit
        result = client.list_courses(**kwargs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="canvas_list_quizzes",
    description="List quizzes in a Canvas course.",
    parameters=[
        ToolParameter(name="course_id", type="integer", description="Course ID (uses active course if omitted)", required=False),
        ToolParameter(name="search_term", type="string", description="Filter quizzes by title", required=False),
        ToolParameter(name="limit", type="integer", description="Max quizzes to return (default 50)", required=False),
    ],
)
def canvas_list_quizzes(course_id: int = None, search_term: str = None, limit: int = None) -> str:
    try:
        client = _get_canvas_client()
        kwargs: dict[str, Any] = {}
        if course_id is not None:
            kwargs["course_id"] = course_id
        if search_term is not None:
            kwargs["search_term"] = search_term
        if limit is not None:
            kwargs["limit"] = limit
        result = client.list_quizzes(**kwargs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="canvas_create_assignment",
    description="Create an assignment in a Canvas course. Requires confirmation.",
    parameters=[
        ToolParameter(name="name", type="string", description="Assignment name", required=True),
        ToolParameter(name="course_id", type="integer", description="Course ID (uses active course if omitted)", required=False),
        ToolParameter(name="due_at", type="string", description="Due date (ISO 8601 datetime)", required=False),
        ToolParameter(name="points_possible", type="number", description="Maximum points", required=False),
        ToolParameter(name="description", type="string", description="Assignment description (HTML supported)", required=False),
        ToolParameter(name="submission_types", type="string", description="Comma-separated: online_upload,online_text_entry,online_url,etc.", required=False),
        ToolParameter(name="published", type="boolean", description="Publish immediately (default false)", required=False),
    ],
    requires_confirmation=True,
)
def canvas_create_assignment(
    name: str, course_id: int = None, due_at: str = None,
    points_possible: float = None, description: str = None,
    submission_types: str = None, published: bool = None,
) -> str:
    try:
        client = _get_canvas_client()
        kwargs: dict[str, Any] = {"name": name}
        if course_id is not None:
            kwargs["course_id"] = course_id
        if due_at is not None:
            kwargs["due_at"] = due_at
        if points_possible is not None:
            kwargs["points_possible"] = points_possible
        if description is not None:
            kwargs["description"] = description
        if submission_types is not None:
            kwargs["submission_types"] = [s.strip() for s in submission_types.split(",")]
        if published is not None:
            kwargs["published"] = published
        result = client.create_assignment(**kwargs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="canvas_create_module",
    description="Create a module in a Canvas course. Requires confirmation.",
    parameters=[
        ToolParameter(name="name", type="string", description="Module name", required=True),
        ToolParameter(name="course_id", type="integer", description="Course ID (uses active course if omitted)", required=False),
        ToolParameter(name="position", type="integer", description="Position in the module list", required=False),
    ],
    requires_confirmation=True,
)
def canvas_create_module(name: str, course_id: int = None, position: int = None) -> str:
    try:
        client = _get_canvas_client()
        kwargs: dict[str, Any] = {"name": name}
        if course_id is not None:
            kwargs["course_id"] = course_id
        if position is not None:
            kwargs["position"] = position
        result = client.create_module(**kwargs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="canvas_upload_file",
    description="Upload a file to a Canvas course. Requires confirmation.",
    parameters=[
        ToolParameter(name="local_path", type="string", description="Absolute path to the file to upload", required=True),
        ToolParameter(name="course_id", type="integer", description="Course ID (uses active course if omitted)", required=False),
        ToolParameter(name="folder_path", type="string", description="Canvas folder path (default /)", required=False),
        ToolParameter(name="name", type="string", description="Override filename in Canvas", required=False),
    ],
    requires_confirmation=True,
)
def canvas_upload_file(local_path: str, course_id: int = None, folder_path: str = None, name: str = None) -> str:
    try:
        client = _get_canvas_client()
        kwargs: dict[str, Any] = {"local_path": local_path}
        if course_id is not None:
            kwargs["course_id"] = course_id
        if folder_path is not None:
            kwargs["folder_path"] = folder_path
        if name is not None:
            kwargs["name"] = name
        result = client.upload_file(**kwargs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="canvas_get_page",
    description="Get a Canvas page by URL slug or ID. Returns page body HTML.",
    parameters=[
        ToolParameter(name="page_url", type="string", description="Page URL slug or numeric ID", required=True),
        ToolParameter(name="course_id", type="integer", description="Course ID (uses active course if omitted)", required=False),
    ],
)
def canvas_get_page(page_url: str, course_id: int = None) -> str:
    try:
        client = _get_canvas_client()
        kwargs: dict[str, Any] = {"page_url": page_url}
        if course_id is not None:
            kwargs["course_id"] = course_id
        result = client.get_page(**kwargs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="canvas_update_page",
    description="Update a Canvas page (title, body HTML, published status). Requires confirmation.",
    parameters=[
        ToolParameter(name="page_url", type="string", description="Page URL slug or numeric ID", required=True),
        ToolParameter(name="course_id", type="integer", description="Course ID (uses active course if omitted)", required=False),
        ToolParameter(name="title", type="string", description="New page title", required=False),
        ToolParameter(name="body", type="string", description="New page body (HTML)", required=False),
        ToolParameter(name="published", type="boolean", description="Published status", required=False),
    ],
    requires_confirmation=True,
)
def canvas_update_page(page_url: str, course_id: int = None, title: str = None, body: str = None, published: bool = None) -> str:
    try:
        client = _get_canvas_client()
        update_kwargs: dict[str, Any] = {}
        if title is not None:
            update_kwargs["title"] = title
        if body is not None:
            update_kwargs["body"] = body
        if published is not None:
            update_kwargs["published"] = published
        if not update_kwargs:
            return json.dumps({"error": "No update fields provided. Specify at least one of: title, body, published"})
        kwargs: dict[str, Any] = {"page_url": page_url, **update_kwargs}
        if course_id is not None:
            kwargs["course_id"] = course_id
        result = client.update_page(**kwargs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="canvas_get_analytics",
    description="Get course analytics (page views, participation, assignment stats).",
    parameters=[
        ToolParameter(name="course_id", type="integer", description="Course ID (uses active course if omitted)", required=False),
        ToolParameter(name="analytics_type", type="string", description="Type: activity, assignments, student_summaries (default activity)", required=False),
    ],
)
def canvas_get_analytics(course_id: int = None, analytics_type: str = None) -> str:
    try:
        client = _get_canvas_client()
        kwargs: dict[str, Any] = {}
        if course_id is not None:
            kwargs["course_id"] = course_id
        if analytics_type is not None:
            kwargs["analytics_type"] = analytics_type
        result = client.get_analytics(**kwargs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="canvas_list_module_items",
    description="List items inside a Canvas module.",
    parameters=[
        ToolParameter(name="module_id", type="integer", description="Module ID", required=True),
        ToolParameter(name="course_id", type="integer", description="Course ID (uses active course if omitted)", required=False),
        ToolParameter(name="limit", type="integer", description="Max items to return (default 50)", required=False),
    ],
)
def canvas_list_module_items(module_id: int, course_id: int = None, limit: int = None) -> str:
    try:
        client = _get_canvas_client()
        kwargs: dict[str, Any] = {"module_id": module_id}
        if course_id is not None:
            kwargs["course_id"] = course_id
        if limit is not None:
            kwargs["limit"] = limit
        result = client.list_module_items(**kwargs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="canvas_get_student",
    description="Get per-student detail: enrollment status, total activity time, last login.",
    parameters=[
        ToolParameter(name="user_id", type="integer", description="Student's Canvas user ID", required=True),
        ToolParameter(name="course_id", type="integer", description="Course ID (uses active course if omitted)", required=False),
    ],
)
def canvas_get_student(user_id: int, course_id: int = None) -> str:
    try:
        client = _get_canvas_client()
        kwargs: dict[str, Any] = {"user_id": user_id}
        if course_id is not None:
            kwargs["course_id"] = course_id
        result = client.get_student(**kwargs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="canvas_create_page",
    description="Create a new Canvas page. Requires confirmation.",
    parameters=[
        ToolParameter(name="title", type="string", description="Page title", required=True),
        ToolParameter(name="body", type="string", description="Page body (HTML)", required=True),
        ToolParameter(name="course_id", type="integer", description="Course ID (uses active course if omitted)", required=False),
        ToolParameter(name="published", type="boolean", description="Publish immediately (default false)", required=False),
    ],
    requires_confirmation=True,
)
def canvas_create_page(title: str, body: str, course_id: int = None, published: bool = None) -> str:
    try:
        client = _get_canvas_client()
        kwargs: dict[str, Any] = {"title": title, "body": body}
        if course_id is not None:
            kwargs["course_id"] = course_id
        if published is not None:
            kwargs["published"] = published
        result = client.create_page(**kwargs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# CC INTERACTIVE SESSION TOOLS (router-only, gated by cc_interactive_tools)
# =============================================================================
# These are schema-only stubs — actual execution is handled by RouterV2's
# per-instance tool handlers (_tool_cc_start_session, etc.).

@tool(
    name="cc_start_session",
    description=(
        "Start an interactive Claude Code session in a tmux window. "
        "Only one session at a time per agent. A background monitor watches the "
        "session and will notify you when Claude Code finishes (❯ prompt appears). "
        "Claude starts with --dangerously-skip-permissions. "
        "Preferred usage: provide initial_input to start and send the task in one call. "
        "If task is omitted, it defaults to the first 200 chars of initial_input."
    ),
    parameters=[
        ToolParameter(
            name="model",
            type="string",
            description="Claude model to use (default: opus)",
            required=False,
        ),
        ToolParameter(
            name="working_directory",
            type="string",
            description="Working directory for the Claude session (default: home directory)",
            required=False,
        ),
        ToolParameter(
            name="task",
            type="string",
            description=(
                "Clear, scoped description of the task for the monitor. "
                "If omitted and initial_input is provided, defaults to the "
                "first 200 characters of initial_input."
            ),
            required=False,
        ),
        ToolParameter(
            name="initial_input",
            type="string",
            description=(
                "Task text to send to Claude Code immediately after session creation. "
                "This is the PREFERRED way to start a session — it combines "
                "cc_start_session + cc_send_input into a single call, avoiding the "
                "risk of the router loop exiting before input is sent."
            ),
            required=False,
        ),
    ],
)
def cc_start_session(model: str = "", working_directory: str = "", task: str = "", initial_input: str = "") -> str:
    """Stub — routed to RouterV2 handler."""
    return json.dumps({"error": "cc_start_session must be executed via router handler"})


@tool(
    name="cc_get_screen",
    description=(
        "Capture the current screen content of the active Claude Code tmux session. "
        "Returns the visible text — use this to see what Claude is doing, whether it's "
        "idle (showing ❯ prompt), working, or waiting for input."
    ),
    parameters=[
        ToolParameter(
            name="lines",
            type="integer",
            description="Number of scrollback lines to capture (default: 200)",
            required=False,
        ),
    ],
)
def cc_get_screen(lines: int = 200) -> str:
    """Stub — routed to RouterV2 handler."""
    return json.dumps({"error": "cc_get_screen must be executed via router handler"})


@tool(
    name="cc_send_input",
    description=(
        "Send text input to the active Claude Code tmux session. "
        "Uses tmux send-keys for all input (chunked for long text). "
        "Set press_enter=false to type without submitting. "
        "For new sessions, prefer cc_start_session(initial_input=...) instead."
    ),
    parameters=[
        ToolParameter(
            name="text",
            type="string",
            description="The text to send to Claude Code",
            required=True,
        ),
        ToolParameter(
            name="press_enter",
            type="boolean",
            description="Press Enter after sending text (default: true)",
            required=False,
        ),
    ],
)
def cc_send_input(text: str, press_enter: bool = True) -> str:
    """Stub — routed to RouterV2 handler."""
    return json.dumps({"error": "cc_send_input must be executed via router handler"})


@tool(
    name="cc_stop_session",
    description=(
        "Stop the active Claude Code tmux session and clean up. "
        "When stopping a session that has not yet completed its task "
        "(e.g., it drifted off-task or entered a degenerate loop), "
        "you MUST provide a rationale explaining the observed drift. "
        "If the session has active child processes (background jobs), "
        "the tool will REFUSE to kill it unless force=true. This prevents "
        "accidentally killing long-running pipelines or builds."
    ),
    parameters=[
        ToolParameter(
            name="rationale",
            type="string",
            description=(
                "Required when stopping a session before task completion. "
                "Explain what drift or degenerate behavior was observed "
                "and why recovery is unlikely."
            ),
            required=False,
        ),
        ToolParameter(
            name="force",
            type="boolean",
            description=(
                "Set to true to kill the session even if it has active child "
                "processes (pipelines, builds, benchmarks). Without force, "
                "the tool refuses to kill a session with running children."
            ),
            required=False,
        ),
    ],
)
def cc_stop_session(rationale: str = "", force: bool = False) -> str:
    """Stub — routed to RouterV2 handler."""
    return json.dumps({"error": "cc_stop_session must be executed via router handler"})


# =============================================================================
# NATIVE HARNESS SESSION TOOLS (router-only, gated by harness_session_tools)
# =============================================================================
# Schema-only stubs — execution is handled by RouterV2's per-instance handlers
# (HarnessSessionManager._tool_harness_*). The native equivalent of the CC
# interactive session tools: a persistent harness subprocess driven over pipes
# instead of a scraped tmux pane.

@tool(
    name="harness_start_session",
    description=(
        "Start a native interactive harness session — a persistent worker "
        "subprocess that edits files, runs commands, and works across many turns "
        "on the agent's configured session backend (e.g. a local model at zero "
        "marginal cost). Spawns the worker AND sends the task in one call. A "
        "background event pump streams its activity and notifies you when it "
        "yields or finishes. Only one session at a time; a new task cold-starts "
        "a fresh worker."
    ),
    parameters=[
        ToolParameter(name="task", type="string",
                      description="Clear, scoped description of the task to execute.",
                      required=True),
        ToolParameter(name="working_directory", type="string",
                      description="Working directory for the session (default: home directory).",
                      required=False),
        ToolParameter(name="max_iters", type="integer",
                      description="Max loop iterations before yielding to you (default: 100).",
                      required=False),
        ToolParameter(name="budget", type="integer",
                      description="Token soft limit for the session context (default: backend config).",
                      required=False),
        ToolParameter(name="checkpoint_interval", type="integer",
                      description="Yield a checkpoint every N iterations for your review (0 = free-running).",
                      required=False),
    ],
)
def harness_start_session(task: str = "", working_directory: str = "", max_iters: int = 0,
                          budget: int = 0, checkpoint_interval: int = 0) -> str:
    """Stub — routed to RouterV2 handler."""
    return json.dumps({"error": "harness_start_session must be executed via router handler"})


@tool(
    name="harness_send_input",
    description=(
        "Send a command to the active harness session. Lossless; applied at the "
        "next iteration boundary. kind: 'steer' (a correction), 'task' (new work "
        "item), 'continue' (resume after a checkpoint, optional nudge), 'reset' "
        "(clear history and seed with content after context exhaustion), 'abort' "
        "(stop the worker)."
    ),
    parameters=[
        ToolParameter(name="content", type="string",
                      description="The message/instruction to send (may be empty for a bare continue).",
                      required=False),
        ToolParameter(name="kind", type="string",
                      description="One of: steer, task, continue, reset, abort (default: steer).",
                      required=False),
    ],
)
def harness_send_input(content: str = "", kind: str = "steer") -> str:
    """Stub — routed to RouterV2 handler."""
    return json.dumps({"error": "harness_send_input must be executed via router handler"})


@tool(
    name="harness_get_status",
    description=(
        "Get a structured status digest of the active harness session: loop "
        "state, iteration, recent tool calls, files touched, and token totals. "
        "Use only when troubleshooting — the event pump notifies you of "
        "lifecycle events automatically, so you do not need to poll."
    ),
    parameters=[],
)
def harness_get_status() -> str:
    """Stub — routed to RouterV2 handler."""
    return json.dumps({"error": "harness_get_status must be executed via router handler"})


@tool(
    name="harness_stop_session",
    description=(
        "Stop the active harness session (abort command → SIGTERM → SIGKILL on "
        "the process group, so child processes are cleaned up). Provide a "
        "rationale when stopping before task completion (drift or degenerate loop)."
    ),
    parameters=[
        ToolParameter(name="rationale", type="string",
                      description="Why the session is being stopped (required if the task is incomplete).",
                      required=False),
        ToolParameter(name="force", type="boolean",
                      description="Skip the graceful abort and signal the process group immediately.",
                      required=False),
    ],
)
def harness_stop_session(rationale: str = "", force: bool = False) -> str:
    """Stub — routed to RouterV2 handler."""
    return json.dumps({"error": "harness_stop_session must be executed via router handler"})
