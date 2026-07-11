"""
Tool providers for compiled pipeline agent loops.

The `_MESH_TOOL_SCHEMAS` table is manually curated. When mesh-tool
signatures change, update both the schema dict and the arg-building logic in
`dispatch()`. See TOOL_SCHEMAS.md for the maintenance procedure.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pipeline_tools import TOOL_DEFINITIONS, TOOL_HANDLERS, _resolve_home


MAX_RESULT_CHARS = 200_000
SHELL_HEAD_LINES = 200
SHELL_TAIL_LINES = 50


def _openai_tool(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _exec_shell(
    command: str,
    timeout: int = 120,
    cwd: str | None = None,
    env: dict | None = None,
) -> str:
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
        out = result.stdout
        if result.stderr:
            out += "\n[stderr]\n" + result.stderr
        if result.returncode != 0:
            out += f"\n[exit code: {result.returncode}]"
    except subprocess.TimeoutExpired:
        out = f"[command timed out after {timeout}s]"
    except Exception as e:
        out = f"[error: {e}]"

    lines = out.split("\n")
    if len(lines) > SHELL_HEAD_LINES + SHELL_TAIL_LINES + 5:
        head = "\n".join(lines[:SHELL_HEAD_LINES])
        tail = "\n".join(lines[-SHELL_TAIL_LINES:])
        omitted = len(lines) - SHELL_HEAD_LINES - SHELL_TAIL_LINES
        out = f"{head}\n\n[... {omitted} lines omitted ...]\n\n{tail}"

    if len(out) > MAX_RESULT_CHARS:
        out = out[:MAX_RESULT_CHARS] + f"\n[truncated at {MAX_RESULT_CHARS} chars]"
    return out


def _list_dir(
    path: str | None = None,
    dir_path: str | None = None,
    depth: int = 1,
    offset: int = 0,
    limit: int = 200,
) -> str:
    root = Path(_resolve_home(path or dir_path or "."))
    if not root.exists():
        return f"[error: path does not exist: {root}]"
    if not root.is_dir():
        return f"[error: path is not a directory: {root}]"

    depth = max(1, min(int(depth or 1), 5))
    offset = max(0, int(offset or 0))
    limit = max(1, min(int(limit or 200), 1000))

    entries: list[tuple[str, bool]] = []

    def walk(current: Path, current_depth: int, prefix: str = "") -> None:
        if current_depth > depth:
            return
        try:
            children = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as e:
            entries.append((f"{prefix}[error reading {current}: {e}]", False))
            return
        for child in children:
            rel = child.relative_to(root)
            is_dir = child.is_dir()
            entries.append((f"{rel}{'/' if is_dir else ''}", is_dir))
            if is_dir and current_depth < depth:
                walk(child, current_depth + 1, prefix)

    walk(root, 1)
    sliced = entries[offset:offset + limit]
    lines = [f"Absolute path: {root.resolve()}"]
    lines.extend(name for name, _ in sliced)
    if offset + limit < len(entries):
        lines.append(f"[... {len(entries) - offset - limit} entries omitted ...]")
    return "\n".join(lines)


def _resolve_local_path(path: str, cwd: str | None = None) -> Path:
    candidate = Path(_resolve_home(path))
    if not candidate.is_absolute() and cwd:
        candidate = Path(_resolve_home(cwd)) / candidate
    resolved = candidate.resolve()
    if not resolved.exists() and cwd:
        path_str = str(resolved)
        cwd_str = str(Path(cwd).resolve())
        idx = path_str.find(cwd_str, len(cwd_str))
        if idx > 0:
            fixed = Path(path_str[idx:])
            if fixed.exists():
                return fixed
    return resolved


class ToolProvider(ABC):
    """Source of tool definitions and execution for pipeline steps."""

    name: str

    @abstractmethod
    def tool_names(self) -> set[str]:
        """Return all tool names this provider can handle."""

    @abstractmethod
    def definitions_for(self, tool_names: list[str]) -> list[dict]:
        """Return OpenAI-format tool definitions for requested tool names."""

    @abstractmethod
    async def dispatch(self, tool_name: str, args: dict) -> str:
        """Execute a tool call and return the result as text."""


class StandaloneToolProvider(ToolProvider):
    """Provider for standalone research tools plus local shell/file helpers."""

    name = "standalone"

    _EXTRA_DEFINITIONS = [
        _openai_tool(
            "shell",
            "Execute a shell command and return stdout/stderr.",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds."},
                },
                "required": ["command"],
            },
        ),
        _openai_tool(
            "bash_exec",
            "Execute a bash command and return stdout/stderr. Alias for shell.",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command to execute."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds."},
                },
                "required": ["command"],
            },
        ),
        _openai_tool(
            "get_working_directory",
            "Return the current working directory used for local shell commands.",
            {"type": "object", "properties": {}, "required": []},
        ),
        _openai_tool(
            "set_working_directory",
            "Set the current working directory used for local shell commands.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path."},
                },
                "required": ["path"],
            },
        ),
        _openai_tool(
            "list_dir",
            "List files and subdirectories under a local directory.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path."},
                    "dir_path": {"type": "string", "description": "Directory path alias."},
                    "depth": {"type": "integer", "description": "Recursion depth, default 1."},
                    "offset": {"type": "integer", "description": "Entry offset for pagination."},
                    "limit": {"type": "integer", "description": "Maximum entries to return."},
                },
            },
        ),
        _openai_tool(
            "file_edit",
            "Perform exact string replacement in a local file.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "old_string": {"type": "string", "description": "Exact text to replace."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences."},
                },
                "required": ["path", "old_string", "new_string"],
            },
        ),
        _openai_tool(
            "file_create",
            "Create a new local file. Fails if the file already exists.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path for the new file."},
                    "content": {"type": "string", "description": "File content."},
                },
                "required": ["path", "content"],
            },
        ),
        _openai_tool(
            "file_write",
            "Create or overwrite a local file.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to write."},
                    "content": {"type": "string", "description": "File content."},
                },
                "required": ["path", "content"],
            },
        ),
        _openai_tool(
            "file_diff",
            "Apply a unified diff patch to a local file.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to patch."},
                    "diff": {"type": "string", "description": "Unified diff content."},
                    "fuzz": {"type": "integer", "description": "Patch fuzz factor."},
                },
                "required": ["path", "diff"],
            },
        ),
    ]

    def __init__(
        self,
        cwd: str | None = None,
        env: dict | None = None,
    ):
        self._cwd = cwd
        self._env = env
        self._definitions = TOOL_DEFINITIONS + self._EXTRA_DEFINITIONS
        self._definition_by_name = {
            item["function"]["name"]: item for item in self._definitions
        }

    def tool_names(self) -> set[str]:
        return set(self._definition_by_name)

    def definitions_for(self, tool_names: list[str]) -> list[dict]:
        return [
            self._definition_by_name[name]
            for name in tool_names
            if name in self._definition_by_name
        ]

    async def dispatch(self, tool_name: str, args: dict) -> str:
        if tool_name in {"shell", "bash_exec"}:
            return _exec_shell(
                args["command"],
                timeout=int(args.get("timeout", 120)),
                cwd=self._cwd,
                env=self._env,
            )
        if tool_name == "get_working_directory":
            return self._cwd or os.getcwd()
        if tool_name == "set_working_directory":
            path = _resolve_local_path(args["path"], self._cwd)
            if not path.exists():
                return f"Error: directory does not exist: {path}"
            if not path.is_dir():
                return f"Error: path is not a directory: {path}"
            self._cwd = str(path)
            return f"Working directory set to {self._cwd}"
        if tool_name == "list_dir":
            return _list_dir(
                path=args.get("path"),
                dir_path=args.get("dir_path"),
                depth=int(args.get("depth", 1)),
                offset=int(args.get("offset", 0)),
                limit=int(args.get("limit", 200)),
            )
        if tool_name == "file_edit":
            path = _resolve_local_path(args["path"], self._cwd)
            try:
                content = path.read_text(encoding="utf-8")
            except Exception as e:
                return f"Error reading file: {e}"
            old_string = args["old_string"]
            count = content.count(old_string)
            if count == 0:
                return "Error: old_string not found in file."
            replace_all = bool(args.get("replace_all", False))
            if count > 1 and not replace_all:
                return (
                    f"Error: old_string found {count} times. Use replace_all=true "
                    "or provide more context."
                )
            new_content = content.replace(
                old_string,
                args["new_string"],
                count if replace_all else 1,
            )
            try:
                path.write_text(new_content, encoding="utf-8")
            except Exception as e:
                return f"Error writing file: {e}"
            return f"Successfully replaced {count if replace_all else 1} occurrence(s) in {path}"
        if tool_name == "file_create":
            path = _resolve_local_path(args["path"], self._cwd)
            if path.exists():
                return f"Error: File already exists: {path}"
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(args["content"], encoding="utf-8")
            except Exception as e:
                return f"Error creating file: {e}"
            return f"Successfully created {path}"
        if tool_name == "file_write":
            path = _resolve_local_path(args["path"], self._cwd)
            existed = path.exists()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(args["content"], encoding="utf-8")
            except Exception as e:
                return f"Error writing file: {e}"
            action = "overwrote" if existed else "created"
            return f"Successfully {action} {path} ({len(args['content'])} bytes)"
        if tool_name == "file_diff":
            path = _resolve_local_path(args["path"], self._cwd)
            if not path.exists():
                return f"Error: File does not exist: {path}"
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
                tmp.write(args["diff"])
                tmp_path = tmp.name
            try:
                proc = subprocess.run(
                    [
                        "patch",
                        "--batch",
                        f"--fuzz={int(args.get('fuzz', 1))}",
                        str(path),
                        tmp_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=self._cwd,
                )
            except Exception as e:
                return f"Error applying patch: {e}"
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            out = proc.stdout
            if proc.stderr:
                out += "\n[stderr]\n" + proc.stderr
            if proc.returncode != 0:
                out += f"\n[exit code: {proc.returncode}]"
            return out.strip() or f"Patched {path}"

        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return f"Unknown standalone tool: {tool_name}"
        try:
            result = handler(args)
        except Exception as e:
            result = {"error": f"{tool_name} raised: {e}"}
        if isinstance(result, str):
            out = result
        else:
            out = json.dumps(result, ensure_ascii=False, indent=2)
        if len(out) > MAX_RESULT_CHARS:
            out = out[:MAX_RESULT_CHARS] + f"\n[truncated at {MAX_RESULT_CHARS} chars]"
        return out


class MeshToolProvider(ToolProvider):
    """Provider that calls mesh tools through the mesh-tool CLI."""

    name = "mesh"

    _MESH_TOOL_SCHEMAS: dict[str, dict] = {
        "memory_search": {
            "name": "memory_search",
            "description": "Semantic search over the agent memory pool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "k": {"type": "integer", "description": "Number of entries to return."},
                    "project": {"type": "string", "description": "Optional project filter."},
                    "tag": {"type": "string", "description": "Optional exact tag filter."},
                },
                "required": ["query"],
            },
        },
        "history_search": {
            "name": "history_search",
            "description": (
                "Full-text keyword search over the agent's raw conversation "
                "history (lossless recall; newest matches first)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords (case-insensitive; all must appear)."},
                    "limit": {"type": "integer", "description": "Maximum matches to return (default 10)."},
                    "role": {"type": "string", "description": "Filter: 'user', 'agent', 'incoming', 'outgoing', or node id prefix."},
                    "date_from": {"type": "string", "description": "Only messages on/after this ISO date."},
                    "date_to": {"type": "string", "description": "Only messages on/before this ISO date (inclusive)."},
                },
                "required": ["query"],
            },
        },
        "memory_get": {
            "name": "memory_get",
            "description": "Fetch a full memory entry by ID.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string", "description": "Memory ID."}},
                "required": ["id"],
            },
        },
        "memory_list": {
            "name": "memory_list",
            "description": "List memory entries, optionally filtered by tag.",
            "parameters": {
                "type": "object",
                "properties": {"tag": {"type": "string", "description": "Optional tag."}},
            },
        },
        "todo_list": {
            "name": "todo_list",
            "description": "List todo items for the triggering conversation or an explicit conversation_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "conversation_id": {"type": "string", "description": "Optional conversation ID."},
                    "include_done": {"type": "boolean", "description": "Include done/cancelled items."},
                    "limit": {"type": "integer", "description": "Maximum items to return."},
                },
            },
        },
        "todo_add": {
            "name": "todo_add",
            "description": "Add a todo item to the current conversation, optionally under a custom section.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Todo text."},
                    "conversation_id": {"type": "string", "description": "Optional conversation ID."},
                    "section": {"type": "string", "description": "Optional custom section label."},
                    "priority": {"type": "integer", "description": "Optional priority."},
                    "position": {"type": "integer", "description": "Optional sparse position."},
                },
                "required": ["text"],
            },
        },
        "todo_update": {
            "name": "todo_update",
            "description": "Update a todo item text, status, section, priority, or position.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todo_id": {"type": "string", "description": "Stable todo ID."},
                    "text": {"type": "string", "description": "Replacement text."},
                    "section": {"type": "string", "description": "Replacement section label; empty clears it."},
                    "status": {"type": "string", "description": "open, in_progress, done, or cancelled."},
                    "priority": {"type": "integer", "description": "New priority."},
                    "position": {"type": "integer", "description": "New sparse position."},
                    "expected_version": {"type": "integer", "description": "Optional expected version."},
                    "conversation_id": {"type": "string", "description": "Optional conversation ID."},
                },
                "required": ["todo_id"],
            },
        },
        "todo_toggle": {
            "name": "todo_toggle",
            "description": "Mark a todo done or reopen it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todo_id": {"type": "string", "description": "Stable todo ID."},
                    "done": {"type": "boolean", "description": "True marks done; false reopens."},
                    "expected_version": {"type": "integer", "description": "Optional expected version."},
                    "conversation_id": {"type": "string", "description": "Optional conversation ID."},
                },
                "required": ["todo_id"],
            },
        },
        "todo_remove": {
            "name": "todo_remove",
            "description": "Soft-delete a todo item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todo_id": {"type": "string", "description": "Stable todo ID."},
                    "expected_version": {"type": "integer", "description": "Optional expected version."},
                    "conversation_id": {"type": "string", "description": "Optional conversation ID."},
                },
                "required": ["todo_id"],
            },
        },
        "todo_reorder": {
            "name": "todo_reorder",
            "description": "Replace todo display ordering for a conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ordered_ids": {"type": "array", "description": "Todo IDs in display order."},
                    "conversation_id": {"type": "string", "description": "Optional conversation ID."},
                },
                "required": ["ordered_ids"],
            },
        },
        "todo_set_section_order": {
            "name": "todo_set_section_order",
            "description": "Set the custom display order for todo sections in a conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section_order": {"type": "array", "description": "Section labels in display order."},
                    "conversation_id": {"type": "string", "description": "Optional conversation ID."},
                },
                "required": ["section_order"],
            },
        },
        "notes_search": {
            "name": "notes_search",
            "description": "Full-text search notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "db": {"type": "string", "description": "Notes DB, e.g. work or personal."},
                    "limit": {"type": "integer", "description": "Maximum results."},
                    "date_from": {"type": "string", "description": "Start date filter."},
                    "date_to": {"type": "string", "description": "End date filter."},
                },
                "required": ["query", "db"],
            },
        },
        "notes_get": {
            "name": "notes_get",
            "description": "Get a note by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "Note ID."},
                    "db": {"type": "string", "description": "Notes DB."},
                },
                "required": ["id", "db"],
            },
        },
        "notes_list": {
            "name": "notes_list",
            "description": "List notes by recency, date, tag, or source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "db": {"type": "string", "description": "Notes DB."},
                    "recent": {"type": "integer", "description": "Number of recent notes."},
                    "date": {"type": "string", "description": "Date filter."},
                    "tag": {"type": "string", "description": "Tag filter."},
                    "source": {"type": "string", "description": "Source filter."},
                },
                "required": ["db"],
            },
        },
        "notes_read": {
            "name": "notes_read",
            "description": "Read a note with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "Note ID."},
                    "db": {"type": "string", "description": "Notes DB."},
                    "start_line": {"type": "integer", "description": "Starting line."},
                    "end_line": {"type": "integer", "description": "Ending line."},
                },
                "required": ["id", "db"],
            },
        },
        "gmail_search_emails": {
            "name": "gmail_search_emails",
            "description": "Search Gmail using Gmail query syntax.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Gmail query."},
                    "limit": {"type": "integer", "description": "Maximum results."},
                    "account": {"type": "string", "description": "Gmail account."},
                },
                "required": ["query"],
            },
        },
        "gmail_get_email": {
            "name": "gmail_get_email",
            "description": "Fetch the full content of a Gmail message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string", "description": "Gmail message ID."},
                    "account": {"type": "string", "description": "Gmail account."},
                },
                "required": ["message_id"],
            },
        },
        "gmail_list_from_date": {
            "name": "gmail_list_from_date",
            "description": "List Gmail messages received on a specific date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date YYYY-MM-DD."},
                    "account": {"type": "string", "description": "Gmail account."},
                },
                "required": ["date"],
            },
        },
        "calendar_list_on_date": {
            "name": "calendar_list_on_date",
            "description": "List calendar events on a specific date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date YYYY-MM-DD."},
                    "timezone": {"type": "string", "description": "Timezone."},
                    "account": {"type": "string", "description": "Optional account, e.g. work or personal."},
                },
                "required": ["date"],
            },
        },
        "current_time": {
            "name": "current_time",
            "description": "Return the current date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        "mesh_status": {
            "name": "mesh_status",
            "description": "Show live status of all agents in the mesh.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        "agent_status": {
            "name": "agent_status",
            "description": "Get detailed diagnostic status for an agent in the mesh.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Agent node ID, e.g. agent:sysadmin:bob, or self.",
                    },
                    "section": {
                        "type": "string",
                        "description": "Optional diagnostic section to return.",
                    },
                },
                "required": ["target"],
            },
        },
        "file_read": {
            "name": "file_read",
            "description": "Read a file with line numbers through mesh-tool if available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path."},
                    "start_line": {"type": "integer", "description": "Starting line."},
                    "num_lines": {"type": "integer", "description": "Number of lines."},
                    "end_line": {"type": "integer", "description": "Ending line."},
                },
                "required": ["path", "start_line"],
            },
        },
        "list_dir": {
            "name": "list_dir",
            "description": "List a directory through mesh-tool if available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {"type": "string", "description": "Directory path."},
                    "path": {"type": "string", "description": "Directory path alias."},
                    "depth": {"type": "integer", "description": "Recursion depth."},
                    "offset": {"type": "integer", "description": "Pagination offset."},
                    "limit": {"type": "integer", "description": "Maximum entries."},
                },
                "required": ["dir_path"],
            },
        },
        "exa_search": {
            "name": "exa_search",
            "description": "Search the web using Exa.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "num_results": {"type": "integer", "description": "Number of results."},
                },
                "required": ["query"],
            },
        },
        "exa_fetch_full": {
            "name": "exa_fetch_full",
            "description": "Fetch full page content from a URL using Exa.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "URL to fetch."}},
                "required": ["url"],
            },
        },
        "extract_url": {
            "name": "extract_url",
            "description": "Extract text from a URL, PDF, or HTML page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to extract."},
                    "max_tokens": {"type": "integer", "description": "Maximum tokens."},
                    "max_pages": {"type": "integer", "description": "Maximum PDF pages."},
                },
                "required": ["url"],
            },
        },
        "arxiv_search": {
            "name": "arxiv_search",
            "description": "Search arXiv.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {"type": "integer", "description": "Maximum results."},
                    "search_field": {"type": "string", "description": "Search field."},
                },
                "required": ["query"],
            },
        },
        "arxiv_get": {
            "name": "arxiv_get",
            "description": "Fetch arXiv metadata by ID.",
            "parameters": {
                "type": "object",
                "properties": {"arxiv_id": {"type": "string", "description": "arXiv ID."}},
                "required": ["arxiv_id"],
            },
        },
        "arxiv_fulltext": {
            "name": "arxiv_fulltext",
            "description": "Download and extract arXiv PDF full text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "arXiv ID."},
                    "max_pages": {"type": "integer", "description": "Maximum pages."},
                },
                "required": ["arxiv_id"],
            },
        },
        "literature_search": {
            "name": "literature_search",
            "description": "Search academic literature across sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {"type": "integer", "description": "Maximum results."},
                    "sources": {"type": "string", "description": "Optional source list."},
                },
                "required": ["query"],
            },
        },
        "s2_search": {
            "name": "s2_search",
            "description": "Search Semantic Scholar for academic papers. Returns metadata, abstract, citation count, and external IDs (arXiv, DOI, PubMed). Faster and more reliable than arXiv search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string."},
                    "limit": {"type": "integer", "description": "Max results (default 10, max 100)."},
                    "year": {"type": "string", "description": "Year filter, e.g. '2023', '2020-2024', '2023-'."},
                    "fields_of_study": {"type": "string", "description": "Filter by field: Computer Science, Medicine, Physics, etc."},
                },
                "required": ["query"],
            },
        },
        "s2_get": {
            "name": "s2_get",
            "description": "Get detailed paper metadata from Semantic Scholar, including full reference list and citing papers. Accepts S2 paper ID, DOI (DOI:10.xxx), arXiv ID (arXiv:1706.03762), or PMID (PMID:123).",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "S2 paper ID or external ID with prefix (DOI:, arXiv:, PMID:)."},
                },
                "required": ["paper_id"],
            },
        },
        "s2_citations": {
            "name": "s2_citations",
            "description": "Get papers that cite a given paper. Accepts S2 paper ID, DOI (DOI:10.xxx), arXiv ID (arXiv:1706.03762), or PMID (PMID:123).",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "S2 paper ID or external ID with prefix (DOI:, arXiv:, PMID:)."},
                    "limit": {"type": "integer", "description": "Max citing papers to return (default 50)."},
                },
                "required": ["paper_id"],
            },
        },
        "s2_references": {
            "name": "s2_references",
            "description": "Get papers referenced by a given paper. Accepts S2 paper ID, DOI (DOI:10.xxx), arXiv ID (arXiv:1706.03762), or PMID (PMID:123).",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "S2 paper ID or external ID with prefix (DOI:, arXiv:, PMID:)."},
                    "limit": {"type": "integer", "description": "Max referenced papers to return (default 50)."},
                },
                "required": ["paper_id"],
            },
        },
        "literature_fulltext": {
            "name": "literature_fulltext",
            "description": "Fetch full text for a paper by identifier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "arXiv ID."},
                    "pmid": {"type": "string", "description": "PubMed ID."},
                    "doi": {"type": "string", "description": "DOI."},
                    "max_tokens": {"type": "integer", "description": "Maximum tokens."},
                    "max_pages": {"type": "integer", "description": "Maximum pages."},
                },
            },
        },
        "pubmed_search": {
            "name": "pubmed_search",
            "description": "Search PubMed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {"type": "integer", "description": "Maximum results."},
                    "sort": {"type": "string", "description": "Sort order."},
                    "min_date": {"type": "string", "description": "Minimum date."},
                    "max_date": {"type": "string", "description": "Maximum date."},
                },
                "required": ["query"],
            },
        },
        "pubmed_get": {
            "name": "pubmed_get",
            "description": "Fetch PubMed metadata by PMID.",
            "parameters": {
                "type": "object",
                "properties": {"pmid": {"type": "string", "description": "PubMed ID."}},
                "required": ["pmid"],
            },
        },
        "pubmed_fulltext": {
            "name": "pubmed_fulltext",
            "description": "Fetch PubMed Central open-access full text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pmid": {"type": "string", "description": "PubMed ID."},
                    "pmcid": {"type": "string", "description": "PMC ID."},
                },
            },
        },
        # ── Standard router tools (expanded set) ───────────────────────────
        "bash_exec": {
            "name": "bash_exec",
            "description": "Execute a bash command and return stdout/stderr/exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds."},
                },
                "required": ["command"],
            },
        },
        "mesh_list": {
            "name": "mesh_list",
            "description": "List all nodes configured in the mesh network.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        "channel_list": {
            "name": "channel_list",
            "description": "List all channels the agent is a member of.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        "channel_members": {
            "name": "channel_members",
            "description": "List members of a specific channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_name": {"type": "string", "description": "Channel name without prefix."},
                },
                "required": ["channel_name"],
            },
        },
        "worker_launch": {
            "name": "worker_launch",
            "description": "Dispatch a worker task with a detailed task description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Detailed task description."},
                },
                "required": ["task"],
            },
        },
        "worker_status": {
            "name": "worker_status",
            "description": "Check status of running workers.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        "send_message": {
            "name": "send_message",
            "description": "Send a message to a user or channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Destination (user:x, channel:x, agent:x)."},
                    "content": {"type": "string", "description": "Message content."},
                    "in_reply_to": {"type": "string", "description": "Optional message ID being replied to."},
                },
                "required": ["to", "content"],
            },
        },
        "schedule_wake": {
            "name": "schedule_wake",
            "description": "Schedule a future wake-up with a prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "wake_time": {"type": "string", "description": "ISO time, relative ('in 30 min'), or natural ('5pm')."},
                    "prompt": {"type": "string", "description": "Prompt to fire at wake time."},
                    "recurrence": {"type": "string", "description": "Optional recurrence: daily, weekly, weekdays, hourly."},
                },
                "required": ["wake_time", "prompt"],
            },
        },
        "schedule_list": {
            "name": "schedule_list",
            "description": "List pending scheduled wakes.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        "schedule_cancel": {
            "name": "schedule_cancel",
            "description": "Cancel a scheduled wake by ID.",
            "parameters": {
                "type": "object",
                "properties": {"wake_id": {"type": "string", "description": "Wake ID to cancel."}},
                "required": ["wake_id"],
            },
        },
        "notes_add": {
            "name": "notes_add",
            "description": "Create a new note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "body": {"type": "string", "description": "Note body."},
                    "db": {"type": "string", "description": "Notes DB (work, personal)."},
                    "title": {"type": "string", "description": "Optional title."},
                    "tags": {"type": "array", "description": "Optional tags."},
                },
                "required": ["body", "db"],
            },
        },
        "notes_edit": {
            "name": "notes_edit",
            "description": "Edit a note by exact string replacement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "Note ID."},
                    "db": {"type": "string", "description": "Notes DB."},
                    "old_string": {"type": "string", "description": "Text to replace."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                },
                "required": ["id", "db", "old_string", "new_string"],
            },
        },
        "notes_delete": {
            "name": "notes_delete",
            "description": "Delete a note by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "Note ID."},
                    "db": {"type": "string", "description": "Notes DB."},
                },
                "required": ["id", "db"],
            },
        },
        "memory_add": {
            "name": "memory_add",
            "description": "Add a memory entry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Memory summary."},
                    "reflection": {"type": "string", "description": "Optional reflection."},
                    "tags": {"type": "string", "description": "Optional tags."},
                    "outcome": {"type": "string", "description": "Optional outcome."},
                },
                "required": ["summary"],
            },
        },
        "remember": {
            "name": "remember",
            "description": "Retrieve deeper details of a memory entry by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Memory ID."},
                    "full": {"type": "boolean", "description": "Include full trace."},
                },
                "required": ["id"],
            },
        },
        "map_list": {
            "name": "map_list",
            "description": "List all project maps.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        "map_get": {
            "name": "map_get",
            "description": "Read the full content of a project map.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string", "description": "Project name."},
                },
                "required": ["project_name"],
            },
        },
        "map_edit": {
            "name": "map_edit",
            "description": "Line-edit a project map.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string", "description": "Project name."},
                    "old_text": {"type": "string", "description": "Text to find."},
                    "new_text": {"type": "string", "description": "Replacement text."},
                },
                "required": ["project_name", "old_text", "new_text"],
            },
        },
        "map_create": {
            "name": "map_create",
            "description": "Create a new project map.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string", "description": "Project name."},
                    "content": {"type": "string", "description": "Initial map content."},
                },
                "required": ["project_name", "content"],
            },
        },
        "set_project_context": {
            "name": "set_project_context",
            "description": "Set the active project context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string", "description": "Project directory."},
                    "reset": {"type": "boolean", "description": "Force fresh scan."},
                },
                "required": ["project_dir"],
            },
        },
        "set_working_directory": {
            "name": "set_working_directory",
            "description": "Set working directory for bash commands.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory path."}},
                "required": ["path"],
            },
        },
        "gmail_send_message": {
            "name": "gmail_send_message",
            "description": "Send an email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient."},
                    "subject": {"type": "string", "description": "Subject."},
                    "body": {"type": "string", "description": "Body."},
                    "cc": {"type": "string", "description": "Optional CC."},
                },
                "required": ["to", "subject", "body"],
            },
        },
        "gmail_reply_to": {
            "name": "gmail_reply_to",
            "description": "Reply to an existing email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string", "description": "Message ID."},
                    "body": {"type": "string", "description": "Reply body."},
                    "cc": {"type": "string", "description": "Optional CC."},
                },
                "required": ["message_id", "body"],
            },
        },
        "gmail_create_draft": {
            "name": "gmail_create_draft",
            "description": "Create a Gmail draft (does NOT send). Lands in Drafts for the user to edit and send.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient."},
                    "subject": {"type": "string", "description": "Subject."},
                    "body": {"type": "string", "description": "Body."},
                    "cc": {"type": "string", "description": "Optional CC."},
                },
                "required": ["to", "subject", "body"],
            },
        },
        "gmail_draft_reply": {
            "name": "gmail_draft_reply",
            "description": "Create a reply draft to an existing email (does NOT send). Threads under the original; lands in Drafts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string", "description": "Message ID to draft a reply to."},
                    "body": {"type": "string", "description": "Reply body."},
                    "cc": {"type": "string", "description": "Optional CC."},
                },
                "required": ["message_id", "body"],
            },
        },
        "calendar_create_event": {
            "name": "calendar_create_event",
            "description": "Create a calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Event summary."},
                    "start": {"type": "string", "description": "Start time."},
                    "end": {"type": "string", "description": "End time."},
                    "description": {"type": "string", "description": "Optional description."},
                    "location": {"type": "string", "description": "Optional location."},
                },
                "required": ["summary", "start", "end"],
            },
        },
        "calendar_delete_event": {
            "name": "calendar_delete_event",
            "description": "Delete a calendar event.",
            "parameters": {
                "type": "object",
                "properties": {"event_id": {"type": "string", "description": "Event ID."}},
                "required": ["event_id"],
            },
        },
        "plaid_accounts": {
            "name": "plaid_accounts",
            "description": "List bank accounts and balances.",
            "parameters": {
                "type": "object",
                "properties": {
                    "institution_id": {"type": "string", "description": "Optional institution filter."},
                },
            },
        },
        "plaid_transactions": {
            "name": "plaid_transactions",
            "description": "Query transactions from local cache.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Start date YYYY-MM-DD."},
                    "end_date": {"type": "string", "description": "End date YYYY-MM-DD."},
                    "limit": {"type": "integer", "description": "Row limit."},
                },
            },
        },
        "account_get_current": {
            "name": "account_get_current",
            "description": "Get current account context.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        "account_list": {
            "name": "account_list",
            "description": "List available accounts.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        "account_set_current": {
            "name": "account_set_current",
            "description": "Switch to a different account context.",
            "parameters": {
                "type": "object",
                "properties": {"account": {"type": "string", "description": "Account name."}},
                "required": ["account"],
            },
        },
        "synthetic_quota": {
            "name": "synthetic_quota",
            "description": "Check Synthetic.ai API quota usage.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        "claude_code_usage": {
            "name": "claude_code_usage",
            "description": "Check Claude Code Max usage.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        "tool_help": {
            "name": "tool_help",
            "description": "Get help for a specific tool, or list all tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string", "description": "Tool name (omit to list all)."},
                },
            },
        },
    }

    def __init__(
        self,
        account: str = "work",
        mesh_tool_cmd: str = "mesh-tool",
        mesh_repo: str = os.environ.get("MESH_HOME", str(Path(__file__).resolve().parent.parent.parent)),
    ):
        self._account = account
        self._mesh_tool_cmd = mesh_tool_cmd
        self._mesh_repo = mesh_repo

    def tool_names(self) -> set[str]:
        return set(self._MESH_TOOL_SCHEMAS)

    def definitions_for(self, tool_names: list[str]) -> list[dict]:
        definitions = []
        for name in tool_names:
            schema = self._MESH_TOOL_SCHEMAS.get(name)
            if schema:
                definitions.append({"type": "function", "function": schema})
        return definitions

    async def dispatch(self, tool_name: str, args: dict) -> str:
        if tool_name not in self._MESH_TOOL_SCHEMAS:
            return f"Unknown mesh tool: {tool_name}"

        cmd = [self._mesh_tool_cmd, tool_name]
        saw_account = False
        for key, value in args.items():
            if value is None:
                continue
            if key == "account":
                saw_account = True
            flag = f"--{key}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            elif isinstance(value, (list, dict)):
                cmd.extend([flag, json.dumps(value)])
            else:
                cmd.extend([flag, str(value)])

        if tool_name.startswith("gmail") and not saw_account:
            cmd.extend(["--account", self._account])
        if tool_name.startswith("notes_") and "db" not in args:
            cmd.extend(["--db", "work"])

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            if self._mesh_repo not in existing_pythonpath.split(os.pathsep):
                env["PYTHONPATH"] = self._mesh_repo + os.pathsep + existing_pythonpath
        else:
            env["PYTHONPATH"] = self._mesh_repo

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await proc.communicate()
        out = stdout.decode(errors="replace").strip()
        err = stderr.decode(errors="replace").strip()
        if proc.returncode != 0:
            detail = err or out or "no error text"
            return (
                f"Error: mesh-tool {tool_name} failed "
                f"(exit {proc.returncode}): {detail[:1000]}"
            )
        if len(out) > MAX_RESULT_CHARS:
            out = out[:MAX_RESULT_CHARS] + f"\n[truncated at {MAX_RESULT_CHARS} chars]"
        return out
