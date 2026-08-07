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

import asyncio
from collections import deque
import contextlib
import contextvars
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import pwd
import re
import signal
import shutil
import subprocess
import sys
import time
from typing import Any, Optional
import uuid

from .tools import tool, ToolParameter, get_registry

logger = logging.getLogger(__name__)

# =============================================================================
# Client singletons (lazy initialization)
# =============================================================================

_bash_tools = None
_bash_working_directory = None  # Working directory for bash commands
_memory_system = None  # MemorySystem singleton, set by AgentNode on init
_memory_search_mode = "hybrid"  # default search mode, set from agent config
_standing_digest_path = ""  # digest path, set by AgentNode on init
_exa_client = None
_tool_host = None
_browser_client = None
_scholar_client = None

# Ephemeral process-supervisor state.  This intentionally belongs to the

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


# ─────────────────────────────────────────────────────────────────────
# Phase 2B — isolation context for module-level tool implementations
# ─────────────────────────────────────────────────────────────────────
#
# Tool implementations in this module are free functions invoked by the
# registry with only their declared arguments, so there is no call-site to
# thread a policy through.  AgentNode installs the active policy here instead.
#
# Two layers, deliberately:
#   * a process global, matching the existing ``configure_sandbox`` pattern,
#     for the ordinary one-agent-per-process deployment;
#   * a ContextVar override that wins when set, so a test process (or future
#     multi-agent host) can run two policies without leaking one into the
#     other.  This is the concurrency risk the plan flags for this module.
#
# Both default to ``None`` = "no policy" = today's exact behaviour.

_isolation_policy = None            # type: ignore[var-annotated]
_isolation_state_paths = None       # type: ignore[var-annotated]
_ISOLATION_CTX_UNSET = object()
_ISOLATION_CTX: "contextvars.ContextVar[object]" = contextvars.ContextVar(
    "mesh_tool_isolation", default=_ISOLATION_CTX_UNSET
)


def configure_isolation(policy=None, state_paths=None) -> None:
    """Install the active isolation policy for module-level tool functions.

    Called by :class:`AgentNode` during initialization.  Passing ``None``
    (the default) clears the policy, which restores the legacy fast path — the
    reset matters because a test process constructs many agents in sequence and
    an enabled policy must not survive into the next one.
    """
    global _isolation_policy, _isolation_state_paths, _bash_working_directory, _bash_tools
    previous_policy = _isolation_policy
    _isolation_policy = policy if (policy is not None and policy.enabled) else None
    _isolation_state_paths = state_paths if _isolation_policy is not None else None
    # A cwd selected before isolation was installed must not survive as the
    # shell's implicit starting point.  The disabled path deliberately leaves
    # it untouched.
    if _isolation_policy is not None:
        _bash_working_directory = str(_isolation_policy.workspace)
    elif previous_policy is not None:
        _bash_working_directory = None
    # Cached BashTools captures sandbox, network, scratch, and policy settings
    # at construction time.  It must never outlive a policy transition.
    _bash_tools = None


def current_isolation():
    """Return ``(policy, state_paths)`` for this call, or ``(None, None)``.

    One ContextVar read plus one global read on the disabled path; no
    filesystem work and no allocation.
    """
    scoped = _ISOLATION_CTX.get()
    if scoped is not _ISOLATION_CTX_UNSET:
        return scoped
    return _isolation_policy, _isolation_state_paths


def _current_worker_scope():
    """The frozen scope to hand a worker subprocess launched from this call.

    Derived from the active policy, or recovered from the environment when
    this process is itself an isolated child (a nested PEV phase), so a scope
    can be narrowed by the parent but never lost on the way down.
    """
    from .isolation import WorkerIsolationScope

    policy, _ = current_isolation()
    scope = WorkerIsolationScope.from_policy(policy)
    if scope.enabled:
        return scope
    return WorkerIsolationScope.from_env()


@contextlib.contextmanager
def isolation_context(policy=None, state_paths=None):
    """Temporarily scope the isolation policy to the current context."""
    active = (
        (policy, state_paths)
        if (policy is not None and policy.enabled)
        else (None, None)
    )
    token = _ISOLATION_CTX.set(active)
    try:
        yield
    finally:
        _ISOLATION_CTX.reset(token)


def _scratch_dir(policy, state_paths) -> "Path | None":
    """The agent's private scratch directory, created on demand."""
    if policy is None:
        return None
    scratch = (
        state_paths.tmp_dir if state_paths is not None else policy.scratch_dir
    )
    scratch = Path(scratch)
    try:
        scratch.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    # The directory is writable by the isolated agent.  If it was replaced
    # with a symlink to a host path, it must stop being the scratch exception.
    if not policy.contains(scratch):
        return None
    return scratch


def _enforce_isolation_path(resolved: "Path", original: str, require_write: bool) -> None:
    """Apply the isolation boundary to an already-resolved path.

    No-op when no policy is installed, which is every agent today.  When a
    policy *is* installed this is the single decision point behind both
    ``_validate_path`` and ``_resolve_path`` so the two cannot drift.

    Raises:
        PermissionError: escape, protected state, or read-only violation.
    """
    policy, state_paths = current_isolation()
    if policy is None:
        return

    from .isolation import is_path_contained

    scratch = _scratch_dir(policy, state_paths)
    if scratch is not None and is_path_contained(resolved, (scratch,)):
        # Scratch is inside a workspace root and inside state_root; allow it
        # before the protected-state check rather than after.
        if require_write and not policy.writable:
            raise PermissionError(
                f"Cannot write to '{original}': isolation filesystem_mode is "
                f"'{policy.filesystem_mode.value}'."
            )
        return

    if not policy.contains(resolved):
        op = "write to" if require_write else "access"
        roots = ", ".join(str(p) for p in policy.workspaces)
        raise PermissionError(
            f"Cannot {op} '{original}': resolves to {resolved}, which is outside "
            f"this agent's isolation boundary. Allowed roots: {roots}"
        )

    if policy.is_protected_state(resolved):
        raise PermissionError(
            f"Cannot access '{original}': {resolved} is protected agent state. "
            f"Use the typed memory/digest/history tools instead of raw file access."
        )

    if require_write and not policy.writable:
        raise PermissionError(
            f"Cannot write to '{original}': isolation filesystem_mode is "
            f"'{policy.filesystem_mode.value}'."
        )


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

    # Isolation boundary is checked first and independently of the legacy
    # sandbox flag: an isolated agent is bounded whether or not `sandboxed`
    # was ever set. No-op when no policy is installed.
    _enforce_isolation_path(resolved, path, require_write)

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

    # Also allow /tmp — but only for an unisolated agent. Under an enabled
    # policy the host tmp is a shared, world-writable escape hatch; the
    # private scratch directory above replaces it.
    if _isolation_allows_host_tmp():
        try:
            resolved.relative_to(Path("/tmp").resolve())
            return str(resolved)
        except ValueError:
            pass

    if require_write:
        raise PermissionError(
            f"Path '{path}' is not in allowed directories. "
            f"Allowed: {_allowed_dirs + _host_tmp_suffix()}"
        )
    else:
        # For read operations, we're more permissive but still log
        # Actually, let's be consistent - sandbox means sandbox
        raise PermissionError(
            f"Path '{path}' is not in allowed directories. "
            f"Allowed: {_allowed_dirs + _host_tmp_suffix()}"
        )


def _isolation_allows_host_tmp() -> bool:
    """Whether the implicit host ``/tmp`` allowance still applies."""
    policy, _ = current_isolation()
    return policy is None


def _host_tmp_suffix() -> list[str]:
    """The ``/tmp`` entry in sandbox error messages, when it still applies."""
    return ["/tmp"] if _isolation_allows_host_tmp() else []


def _validated_attachment_paths(
    attachments: list | None,
    tool_name: str,
) -> list | None:
    """Canonicalize file-bearing attachment arguments under isolation.

    Gmail opens these paths in its client layer, outside the ordinary raw-file
    helpers.  Returning the original object on the disabled fast path preserves
    the legacy call exactly; enabled callers receive a shallow copy whose paths
    are the canonical paths that were actually authorized.
    """
    policy, _ = current_isolation()
    if policy is None or attachments is None:
        return attachments
    if not isinstance(attachments, list):
        raise PermissionError(f"{tool_name} attachments must be a list")

    validated: list = []
    for index, item in enumerate(attachments):
        if not isinstance(item, dict) or not str(item.get("path") or "").strip():
            raise PermissionError(
                f"{tool_name} attachment {index} must contain a non-empty path"
            )
        original = str(item["path"])
        from .paths import resolve_path as _rp

        resolved = Path(_rp(original)).resolve()
        _enforce_isolation_path(resolved, original, require_write=False)
        copied = dict(item)
        copied["path"] = str(resolved)
        validated.append(copied)
    return validated


def _get_bash_tools():
    """Get or create BashTools instance."""
    global _bash_tools
    if _bash_tools is None:
        from .clients.bash_tools import BashTools
        policy, state_paths = current_isolation()
        # Phase 4: an enabled policy always runs its shell inside bwrap. The
        # legacy `_sandboxed` flag stays the only switch for unisolated
        # agents, so nothing changes for the fleet running with isolation off.
        _bash_tools = BashTools(
            user_confirm=False,  # No CLI confirmation in mesh context
            timeout_sec=30.0,
            max_output_chars=100000,
            sandboxed=_sandboxed or policy is not None,
            allowed_dirs=_allowed_dirs,
            allow_network=_allow_network,
            isolation_policy=policy,
            workdir=(str(policy.workspace) if policy is not None else None),
            scratch_dir=str(_scratch_dir(policy, state_paths) or "") or None,
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
    """Set the working directory for bash commands.

    Under an enabled isolation policy the target must resolve inside a
    declared workspace root: otherwise this tool is a one-call escape, since
    every later relative path in ``_resolve_path`` is joined against it.
    Symlinks are resolved before the check, so a link inside the workspace
    pointing out of it is rejected too.
    """
    global _bash_working_directory
    from .paths import resolve_path as _rp
    expanded = _rp(path)

    if not os.path.isdir(expanded):
        return json.dumps({"error": f"Directory does not exist: {path}"})

    try:
        _enforce_isolation_path(Path(expanded).resolve(), path, require_write=False)
    except PermissionError as exc:
        return json.dumps({"error": str(exc)})

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


# Local-Qwen simple-code executor. This deliberately uses the clean-room
# harness rather than a one-shot completion: callers delegate an atomic code
# subtask and Qwen gets its own bounded inspect → edit → test ReAct loop. Its
# model, endpoint, prompt, toolset, and thinking controls come from the named
# backend in backends.yaml rather than a second hard-coded configuration.
_MESH_QWEN_BACKEND = "mesh-harness-qwen"
_MESH_QWEN_TIMEOUT_SECS = 3600
_MESH_QWEN_MAX_ITERS = 500
_MESH_QWEN_MAX_TOKENS = 16_384
_MESH_QWEN_SYNTHESIS_GRACE_SECS = 120
_MESH_QWEN_STDOUT_TAIL_BYTES = 2 * 1024 * 1024
_MESH_QWEN_STDERR_TAIL_BYTES = 256 * 1024
_MESH_QWEN_RETURNED_EVENT_TAIL_CHARS = 64 * 1024


def _mesh_qwen_result_from_jsonl(stdout: str) -> tuple[str, dict[str, Any], list[str]]:
    """Extract terminal or latest partial output and fatal diagnostics."""
    final_text = ""
    usage: dict[str, Any] = {}
    errors: list[str] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type")
        data = event.get("data") or {}
        if event_type == "thread.finished":
            terminal_text = str(data.get("final_text") or "")
            if terminal_text:
                final_text = terminal_text
            usage = data.get("usage") or {}
        elif event_type == "assistant.message":
            assistant_text = str(data.get("text") or "")
            if assistant_text:
                final_text = assistant_text
        elif event_type == "usage" and isinstance(data, dict):
            usage = data
        elif event_type == "error" and data.get("fatal", True):
            errors.append(str(data.get("message") or "unknown harness error"))
    return final_text, usage, errors


def _mesh_qwen_event_tail(stdout: str) -> str:
    """Return a bounded recent JSONL tail for hard-timeout recovery."""
    tail = "\n".join(stdout.splitlines()[-30:])
    if len(tail) > _MESH_QWEN_RETURNED_EVENT_TAIL_CHARS:
        tail = tail[-_MESH_QWEN_RETURNED_EVENT_TAIL_CHARS:]
    return tail


def _mesh_qwen_work_deadline(timeout: int) -> tuple[int, int]:
    """Reserve synthesis time inside the caller's absolute timeout."""
    grace = min(
        _MESH_QWEN_SYNTHESIS_GRACE_SECS,
        max(5, timeout // 10),
    )
    return max(1, timeout - grace), grace


async def _collect_capped_stream(
    stream: asyncio.StreamReader | None,
    *,
    max_bytes: int,
) -> bytes:
    """Drain a subprocess stream while retaining only its newest bytes."""
    if stream is None:
        return b""
    chunks: deque[bytes] = deque()
    retained = 0
    while True:
        chunk = await stream.read(64 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
        retained += len(chunk)
        while chunks and retained > max_bytes:
            excess = retained - max_bytes
            first = chunks[0]
            if len(first) <= excess:
                retained -= len(chunks.popleft())
            else:
                chunks[0] = first[excess:]
                retained -= excess
    return b"".join(chunks)


async def _feed_process_stdin(
    writer: asyncio.StreamWriter | None,
    payload: bytes,
) -> None:
    """Write one prompt and close stdin without buffering child output."""
    if writer is None:
        return
    try:
        writer.write(payload)
        await writer.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        writer.close()
        wait_closed = getattr(writer, "wait_closed", None)
        if wait_closed is not None:
            try:
                await wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass


async def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    """Kill the dedicated harness process group and reap the child."""
    if process.returncode is not None:
        return
    pid = getattr(process, "pid", None)
    if isinstance(pid, int) and pid > 0:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            process.kill()
    else:
        process.kill()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except asyncio.TimeoutError:
        logger.error("Timed out reaping mesh_qwen process after SIGKILL")


def _mesh_codex_result_from_jsonl(stdout: str) -> tuple[str, dict[str, Any], list[str]]:
    """Extract a terminal response from Codex's ``exec --json`` event stream."""
    text_blocks: list[str] = []
    usage: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "cached_input_tokens": 0,
    }
    errors: list[str] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type")
        if event_type in ("turn.completed", "turn_complete"):
            event_usage = event.get("usage") or {}
            if isinstance(event_usage, dict):
                for name in usage:
                    usage[name] += int(event_usage.get(name) or 0)
        elif event_type == "agent_message":
            message = event.get("message")
            if isinstance(message, str) and message.strip():
                text_blocks.append(message.strip())
        elif event_type in ("item.completed", "item_completed"):
            item = event.get("item") or {}
            if isinstance(item, dict) and item.get("type") in (None, "agent_message"):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    text_blocks.append(text.strip())
        elif event_type == "error":
            detail = event.get("message") or event.get("error") or "unknown Codex error"
            errors.append(str(detail))
    return "\n\n".join(text_blocks), usage, errors


async def _run_mesh_qwen_codex(
    *,
    task: str,
    cwd: str,
    delegated_task: str,
    llm_config: Any,
    timeout: int,
) -> str:
    """Run a configured Codex backend as a PEV phase worker.

    Codex already owns its native ReAct loop, so wrapping it in
    ``mesh.harness exec`` would be both redundant and unsupported.  This keeps
    the same JSON envelope as mesh_qwen while pinning Codex to the PEV cwd.
    """
    from .isolation import (
        build_codex_isolation_args,
        codex_home_dir,
        strip_codex_broadening_args,
    )

    # Phase 3: prefer the scope the caller pinned onto the backend config, and
    # fall back to the module-level policy this process was configured with so
    # a nested PEV phase cannot silently drop it.  Both are disabled today.
    configured_scope = getattr(llm_config, "isolation_scope", None)
    inherited_scope = _current_worker_scope()
    if inherited_scope.enabled:
        if configured_scope is not None and configured_scope.enabled:
            if configured_scope != inherited_scope:
                return json.dumps({
                    "status": "error",
                    "task": task,
                    "error": (
                        "configured Codex scope differs from the inherited "
                        "worker isolation scope"
                    ),
                })
        scope = inherited_scope
    else:
        scope = configured_scope or inherited_scope
    if scope.enabled:
        from .isolation import assert_cwd_in_scope

        try:
            cwd = assert_cwd_in_scope(scope, cwd)
        except PermissionError as exc:
            return json.dumps({"status": "error", "task": task, "error": str(exc)})

    codex_binary = llm_config.codex_binary or shutil.which("codex") or "codex"
    command = [
        codex_binary,
        "exec",
        "-",
        "-m",
        llm_config.model,
        "--ephemeral",
        "--json",
        "-C",
        cwd,
        "--skip-git-repo-check",
    ]
    extra_args = list(llm_config.codex_extra_args or [])
    if scope.enabled:
        # Same builder as LLMClient._complete_codex so the two Codex launch
        # paths cannot drift apart.
        command.extend(build_codex_isolation_args(scope, extra_args))
        extra_args = strip_codex_broadening_args(extra_args)
    elif not any(arg == "--sandbox" or arg.startswith("--sandbox=") for arg in extra_args):
        command.append("--dangerously-bypass-approvals-and-sandbox")
    if llm_config.cc_effort:
        command.extend(["-c", f'model_reasoning_effort="{llm_config.cc_effort}"'])
    command.extend(extra_args)

    env = os.environ.copy()
    real_home = pwd.getpwuid(os.getuid()).pw_dir
    env["HOME"] = (
        codex_home_dir(scope, create=True) if scope.enabled else real_home
    )
    repo_root = Path(__file__).resolve().parent.parent
    env["PATH"] = (
        f"{repo_root}:{real_home}/.local/share/node-v22/bin:{real_home}/.local/bin:"
        f"{env.get('PATH', '')}"
    )
    # Keep the worker launch path consistent with LLMClient._complete_codex:
    # a configured Codex provider may require endpoint/auth overrides.
    if llm_config.codex_env:
        env.update(llm_config.codex_env)
    if scope.enabled:
        env["HOME"] = codex_home_dir(scope, create=True)
        env.pop("CODEX_HOME", None)
    env.update(scope.to_env())
    started = time.monotonic()
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(delegated_task.encode("utf-8")), timeout=timeout
        )
    except asyncio.TimeoutError:
        if "process" in locals() and process.returncode is None:
            process.kill()
            await process.wait()
        return json.dumps({
            "status": "error",
            "task": task,
            "cwd": cwd,
            "error": f"Codex phase exceeded {timeout} seconds",
        })
    except OSError as exc:
        return json.dumps({"status": "error", "error": f"failed to start Codex: {exc}"})

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    final_text, usage, errors = _mesh_codex_result_from_jsonl(stdout)
    if process.returncode != 0 and not errors:
        errors.append(f"Codex exited {process.returncode}")
    if not final_text and not errors:
        errors.append("Codex returned no final report")
    return json.dumps({
        "status": "ok" if process.returncode == 0 and final_text and not errors else "error",
        "task": task,
        "cwd": cwd,
        "exit_code": process.returncode,
        "elapsed_secs": round(time.monotonic() - started, 3),
        "final_text": final_text,
        "usage": usage,
        "errors": errors,
        "stderr_tail": "\n".join(stderr.splitlines()[-30:]),
    }, ensure_ascii=False)


@tool(
    name="mesh_qwen",
    description=(
        "Execute one subtask through a bounded harness ReAct loop. Launches a subprocess "
        "through the configured backend (mesh-harness, codex, or direct API) to inspect, "
        "edit, and test files in cwd. Accepts one task and returns a structured result. "
        "Useful for delegating narrow subtasks from orchestrators like pev_harness or "
        "recursive_harness."
    ),
    parameters=[
        ToolParameter(
            name="task", type="string", required=True,
            description=(
                "Self-contained subtask with concrete success criteria. Avoid broad "
                "architecture tasks; decompose first then pass narrow, verifiable work."
            ),
        ),
        ToolParameter(
            name="cwd", type="string", required=True,
            description="Absolute working directory containing the files Qwen may inspect and edit.",
        ),
        ToolParameter(
            name="max_iters", type="integer", required=False,
            default=_MESH_QWEN_MAX_ITERS,
            description="Maximum ReAct iterations (default controlled by the backend's config cap).",
        ),
        ToolParameter(
            name="max_tokens", type="integer", required=False, default=16384,
            description="Maximum completion tokens per turn (default controlled by the backend's config cap).",
        ),
        ToolParameter(
            name="timeout_secs", type="integer", required=False,
            default=_MESH_QWEN_TIMEOUT_SECS,
            description="Wall-clock execution limit in seconds (default controlled by the backend's config cap).",
        ),
    ],
)
async def mesh_qwen(
    task: str,
    cwd: str,
    max_iters: int = _MESH_QWEN_MAX_ITERS,
    max_tokens: int = _MESH_QWEN_MAX_TOKENS,
    timeout_secs: int = _MESH_QWEN_TIMEOUT_SECS,
    _delegated_prompt: str | None = None,
    _include_trace: bool = False,
    _backend_name: str | None = None,
    _tools: str | None = None,
    _require_write: bool = True,
    _node_id: str = "mesh-qwen-subworker",
    _max_iters_override: int | None = None,
    _max_tokens_override: int | None = None,
    _timeout_override: int | None = None,
    _thinking_budget_override: int | None = None,
) -> str:
    """Run one subtask through a configured harness ReAct loop.

    This is the general-purpose harness subprocess launcher. It spans mesh-harness
    exec, codex exec, and direct API backends based on backend config.  Private
    arguments let internal orchestrators (pev_harness, recursive_harness) reuse
    the same subprocess machinery with an alternative backend, narrower tool
    surface, or relaxed resource caps.
    """
    if not isinstance(task, str) or not task.strip():
        return json.dumps({"status": "error", "error": "task must be a non-empty string"})
    if not isinstance(cwd, str) or not cwd.strip():
        return json.dumps({"status": "error", "error": "cwd must be an absolute directory path"})
    backend_name = (_backend_name or _MESH_QWEN_BACKEND).strip()
    if not backend_name:
        return json.dumps({"status": "error", "error": "backend name must be non-empty"})
    try:
        resolved_cwd = _validate_path(cwd, require_write=_require_write)
    except (OSError, PermissionError) as exc:
        return json.dumps({"status": "error", "error": f"invalid cwd: {exc}"})
    if not os.path.isabs(cwd) or not os.path.isdir(resolved_cwd):
        return json.dumps({"status": "error", "error": "cwd must be an existing absolute directory"})
    try:
        iterations = int(max_iters)
        output_tokens = int(max_tokens)
        timeout = int(timeout_secs)
    except (TypeError, ValueError):
        return json.dumps({"status": "error", "error": "max_iters, max_tokens, and timeout_secs must be integers"})
    _max_iters = _max_iters_override if _max_iters_override is not None else _MESH_QWEN_MAX_ITERS
    _max_tokens = _max_tokens_override if _max_tokens_override is not None else _MESH_QWEN_MAX_TOKENS
    _max_timeout = _timeout_override if _timeout_override is not None else _MESH_QWEN_TIMEOUT_SECS
    if not 1 <= iterations <= _max_iters:
        return json.dumps({"status": "error", "error": f"max_iters must be between 1 and {_max_iters}"})
    if not 256 <= output_tokens <= _max_tokens:
        return json.dumps({"status": "error", "error": f"max_tokens must be between 256 and {_max_tokens}"})
    if not 30 <= timeout <= _max_timeout:
        return json.dumps({"status": "error", "error": f"timeout_secs must be between 30 and {_max_timeout}"})
    if _thinking_budget_override is not None:
        if isinstance(_thinking_budget_override, bool):
            return json.dumps({
                "status": "error",
                "error": "thinking budget must be a positive integer",
            })
        try:
            explicit_thinking_budget = int(_thinking_budget_override)
        except (TypeError, ValueError):
            return json.dumps({
                "status": "error",
                "error": "thinking budget must be a positive integer",
            })
        if not 1 <= explicit_thinking_budget <= 1_000_000:
            return json.dumps({
                "status": "error",
                "error": "thinking budget must be between 1 and 1000000",
            })
    else:
        explicit_thinking_budget = None
    from .config import backend_config_to_llm_config, load_llm_backends

    repo_root = Path(__file__).resolve().parent.parent
    try:
        backend_config = load_llm_backends(repo_root / "mesh.yaml").get(backend_name)
        if backend_config is None:
            return json.dumps({
                "status": "error",
                "error": f"{backend_name} backend is not configured",
            })
        llm_config = backend_config_to_llm_config(backend_config)
    except (OSError, ValueError) as exc:
        return json.dumps({"status": "error", "error": f"failed to load {backend_name} backend: {exc}"})

    delegated_task = _delegated_prompt or f"""You are the local executor for one atomic simple-code subtask delegated by a parent worker.

Work only inside {resolved_cwd}. Inspect the relevant files, make the smallest complete change, and run focused verification. Do not decompose this task further, dispatch another worker, use network or agent-local mesh tools, or broaden the requested scope. Finish with a concise report naming files changed and commands/tests run. The parent worker owns cross-subtask integration and will independently review your result.

## Atomic subtask
{task.strip()}
"""

    inherited_scope = _current_worker_scope()

    if backend_config.backend_type == "mesh-harness":
        harness_python = llm_config.harness_python or sys.executable or "python3"
        harness_backend = llm_config.harness_backend or "openai"
        harness_base_url = llm_config.harness_base_url
        harness_api_key = llm_config.harness_api_key
        harness_tools = llm_config.harness_tools
        harness_toolset = llm_config.harness_toolset
        harness_soft_limit = llm_config.harness_soft_limit
    elif backend_config.backend_type in {"openai", "anthropic", "claude-code"}:
        # A direct backend can still drive the clean-room ReAct loop. This is
        # useful for independent verification without requiring a duplicate
        # mesh-harness wrapper entry in backends.yaml.
        harness_python = sys.executable or "python3"
        harness_backend = backend_config.backend_type
        harness_base_url = llm_config.base_url
        harness_api_key = (
            llm_config.anthropic_api_key
            if backend_config.backend_type == "anthropic"
            else llm_config.api_key
        )
        harness_tools = ""
        harness_toolset = "harness"
        harness_soft_limit = 0
    elif backend_config.backend_type == "codex":
        return await _run_mesh_qwen_codex(
            task=task.strip(),
            cwd=resolved_cwd,
            delegated_task=delegated_task,
            llm_config=llm_config,
            timeout=timeout,
        )
    else:
        return json.dumps({
            "status": "error",
            "error": (
                f"{backend_name} uses unsupported ReAct backend type "
                f"{backend_config.backend_type!r}"
            ),
        })

    if inherited_scope.enabled and harness_backend in {
        "claude-code", "claude-interactive", "zai"
    }:
        return json.dumps({
            "status": "error",
            "error": (
                "isolated mesh_qwen workers cannot use unsupported "
                f"backend {harness_backend!r}"
            ),
        })

    thinking_budget = (
        explicit_thinking_budget
        if explicit_thinking_budget is not None
        else llm_config.thinking_budget
    )
    logger.info(
        "mesh_qwen backend=%s resolved thinking_budget=%s",
        backend_name,
        thinking_budget,
    )
    system_prompt_file = llm_config.harness_system_prompt_file
    if system_prompt_file:
        system_prompt_path = Path(system_prompt_file)
        if not system_prompt_path.is_absolute():
            system_prompt_path = repo_root / system_prompt_path
    else:
        system_prompt_path = Path(__file__).with_name("harness") / "system_prompt.md"
    if not system_prompt_path.is_file():
        return json.dumps({"status": "error", "error": "mesh harness system prompt is unavailable"})

    command = [
        harness_python,
        "-m", "mesh.harness", "exec",
        "--backend", harness_backend,
        "--model", llm_config.model,
        "--prompt", "-",
    ]
    if llm_config.cc_effort:
        command.extend(["--effort", llm_config.cc_effort])
    if isinstance(thinking_budget, int) and thinking_budget > 0:
        command.extend(["--thinking-budget", str(thinking_budget)])
    if harness_base_url:
        command.extend(["--base-url", harness_base_url])
    if harness_api_key:
        command.extend(["--api-key", harness_api_key])
    if backend_config.backend_type == "claude-code" and llm_config.cc_binary:
        command.extend(["--cc-binary", llm_config.cc_binary])
    if _tools is not None:
        command.extend(["--tools", _tools])
    elif harness_tools:
        command.extend(["--tools", harness_tools])
    elif harness_toolset:
        command.extend(["--toolset", harness_toolset])
    command.extend(["--system-prompt-file", str(system_prompt_path)])
    if llm_config.harness_agent_socket:
        command.extend(["--agent-socket", llm_config.harness_agent_socket])
    if harness_soft_limit:
        command.extend(["--soft-limit", str(harness_soft_limit)])
    work_deadline_secs, synthesis_grace_secs = _mesh_qwen_work_deadline(timeout)
    command.extend([
        "--deadline-secs", str(work_deadline_secs),
        "--max-iters", str(iterations),
        "--max-tokens", str(output_tokens),
        "--node-id", _node_id,
        "--cwd", resolved_cwd,
    ])
    started = time.monotonic()
    # ``mesh.harness`` is imported by module name before its ``--cwd`` task
    # directory is applied.  A delegated task may legitimately run outside
    # the repository (for example an isolated dark-fold workspace), so launch
    # the child from the repository root and let ``--cwd`` select its actual
    # tool workspace after imports succeed.
    harness_launch_dir = str(repo_root)
    timed_out = False
    stdout_bytes = b""
    stderr_bytes = b""
    # Phase 3: hand the child harness its scope.  A disabled scope produces an
    # empty fragment, so ``harness_env`` stays ``None`` and the subprocess
    # inherits this process's environment exactly as it does today.
    harness_scope = inherited_scope
    scope_env = harness_scope.to_env()
    harness_env: dict[str, str] | None = None
    if scope_env:
        harness_env = os.environ.copy()
        harness_env.update(scope_env)
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            cwd=harness_launch_dir,
            **({"env": harness_env} if harness_env is not None else {}),
        )
        stdout_task = asyncio.create_task(
            _collect_capped_stream(
                process.stdout,
                max_bytes=_MESH_QWEN_STDOUT_TAIL_BYTES,
            )
        )
        stderr_task = asyncio.create_task(
            _collect_capped_stream(
                process.stderr,
                max_bytes=_MESH_QWEN_STDERR_TAIL_BYTES,
            )
        )

        async def _feed_and_wait() -> int:
            await _feed_process_stdin(
                process.stdin,
                delegated_task.encode("utf-8"),
            )
            return await process.wait()

        await asyncio.wait_for(_feed_and_wait(), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
        if "process" in locals() and process.returncode is None:
            await _kill_process_group(process)
    except OSError as exc:
        return json.dumps({"status": "error", "error": f"failed to start mesh_qwen: {exc}"})
    finally:
        if "stdout_task" in locals():
            stdout_bytes, stderr_bytes = await asyncio.gather(
                stdout_task,
                stderr_task,
            )

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    final_text, usage, fatal_errors = _mesh_qwen_result_from_jsonl(stdout)
    if timed_out:
        result = {
            "status": "error",
            "error": (
                f"mesh_qwen reached its absolute {timeout}-second timeout; "
                f"the child had {work_deadline_secs} seconds for normal work "
                f"and {synthesis_grace_secs} seconds reserved for synthesis, "
                "but synthesis did not complete"
            ),
            "task": task.strip(),
            "cwd": resolved_cwd,
            "exit_code": process.returncode,
            "elapsed_secs": round(time.monotonic() - started, 3),
            "final_text": final_text,
            "usage": usage,
            "errors": fatal_errors,
            "stderr_tail": "\n".join(stderr.splitlines()[-30:]),
        }
        event_tail = _mesh_qwen_event_tail(stdout)
        if event_tail:
            result["event_tail"] = event_tail
        if _include_trace:
            result["trace"] = stdout
        return json.dumps(result, ensure_ascii=False)

    result: dict[str, Any] = {
        "status": "ok" if process.returncode == 0 and final_text and not fatal_errors else "error",
        "task": task.strip(),
        "cwd": resolved_cwd,
        "exit_code": process.returncode,
        "elapsed_secs": round(time.monotonic() - started, 3),
        "final_text": final_text,
        "usage": usage,
        "stderr_tail": "\n".join(stderr.splitlines()[-30:]),
    }
    if fatal_errors:
        result["errors"] = fatal_errors
    if not final_text and not fatal_errors:
        result["errors"] = ["mesh harness returned no final report"]
    if _include_trace:
        result["trace"] = stdout
    return json.dumps(result, ensure_ascii=False)


@tool(
    name="recursive_harness",
    description=(
        "Execute a potentially multi-phase task through bounded recursive decomposition. "
        "The harness classifies cognitive phases, delegates each atomic phase to local "
        "Qwen's ReAct executor, independently diagnoses outcomes, and replans after "
        "partial or failed phases. The interface remains task-in/result-out."
    ),
    parameters=[
        ToolParameter(
            name="task", type="string", required=True,
            description="Natural-language task with concrete success criteria.",
        ),
        ToolParameter(
            name="cwd", type="string", required=True,
            description="Absolute working directory for all delegated phase executions.",
        ),
    ],
)
async def recursive_harness(task: str, cwd: str) -> str:
    """Run the drop-in recursive decomposition wrapper around mesh_qwen."""
    if not isinstance(task, str) or not task.strip():
        return "[status: failed]\n\nTask was not attempted.\n\n## Issues\ntask must be a non-empty string"
    if not isinstance(cwd, str) or not cwd.strip() or not os.path.isabs(cwd):
        return (
            "[status: failed]\n\nTask was not attempted.\n\n"
            "## Issues\ncwd must be an existing absolute directory"
        )
    try:
        resolved_cwd = _validate_path(cwd, require_write=True)
    except (OSError, PermissionError) as exc:
        return (
            "[status: failed]\n\nTask was not attempted.\n\n"
            f"## Issues\ninvalid cwd: {exc}"
        )
    if not os.path.isdir(resolved_cwd):
        return (
            "[status: failed]\n\nTask was not attempted.\n\n"
            "## Issues\ncwd must be an existing absolute directory"
        )

    from .recursive_harness import run

    return await run(task.strip(), resolved_cwd)


# =============================================================================
# STYLE FILTER — synchronous mesh tool
# =============================================================================

@tool(
    name="style_filter",
    description=(
        "Restyle draft text following the house style guide. "
        "Give the draft text (and optional tone/audience directions) and "
        "receive the same content rewritten in house style. For emails, "
        "announcements, and short prose the router has already drafted."
    ),
    parameters=[
        ToolParameter(
            name="text", type="string", required=True,
            description="The draft text to restyle.",
        ),
        ToolParameter(
            name="directions", type="string", required=False,
            description="Optional audience, tone, or format constraints to apply while restyling.",
        ),
        ToolParameter(
            name="max_length", type="integer", required=False,
            description="Optional maximum word count.",
        ),
    ],
)
async def style_filter(
    text: str,
    directions: str | None = None,
    max_length: int | None = None,
) -> str:
    """Restyle supplied draft text without task-prompt or tool dependencies."""
    from mesh.config import MeshConfig, backend_config_to_llm_config

    if not isinstance(text, str) or not text.strip():
        return "Error: style_filter requires non-empty text."

    repo_root = Path(__file__).resolve().parent.parent
    try:
        mesh_config = MeshConfig.load(repo_root / "mesh.yaml")
        backend_config = mesh_config.llm_backends.get("default")
        if backend_config is None:
            return "Error: style_filter default backend is not configured."
        style_guide = (repo_root / "mesh" / "prompts" / "writing_style.md").read_text(
            encoding="utf-8"
        )
    except (OSError, ValueError) as exc:
        return f"Error: style_filter configuration is invalid: {exc}"

    resolved_budget = backend_config_to_llm_config(backend_config).thinking_budget
    backend_model = str(
        getattr(backend_config, "default_model", "") or ""
    ).lower()
    if resolved_budget is None and "qwen" in backend_model:
        return (
            "Error: Qwen-backed style_filter requires a positive "
            "thinking_budget."
        )

    system_prompt = (
        "You are a style filter. Rewrite the provided draft text to follow the "
        "house style guide. Preserve all content, facts, and meaning; change only "
        "the style — clarity, flow, sentence construction, banned words and punctuation. "
        "Never invent content and never remove substance. Return only the restyled text."
    )
    system_prompt = f"{system_prompt}\n\n<style-guide>\n{style_guide}\n</style-guide>"

    length_note = f"\n\nMaximum length: {max_length} words." if max_length else ""
    directions_block = (
        f"## Style directions\n{directions.strip()}"
        if isinstance(directions, str) and directions.strip()
        else ""
    )
    delegated_prompt = "\n\n".join(
        part
        for part in (
            system_prompt,
            directions_block,
            f"## Draft\n{text.strip()}{length_note}",
        )
        if part
    )

    result = await mesh_qwen(
        task=text.strip(),
        cwd=str(Path.cwd()),
        _delegated_prompt=delegated_prompt,
        _backend_name="default",
        _timeout_override=300,
        timeout_secs=300,
        _thinking_budget_override=resolved_budget,
        _tools="",
        _require_write=False,
    )

    try:
        parsed = json.loads(result)
        if parsed.get("status") == "ok":
            return parsed.get("final_text") or parsed.get("output", "")
        return f"Error: style_filter returned status {parsed.get('status')}: {parsed.get('error', result)}"
    except json.JSONDecodeError:
        return f"Error: style_filter returned non-JSON response: {result[:500]}"


def _configured_task_prompt_bundle(task_type: str):
    """Load one task type's shared prompt bundle from the active mesh config."""
    from mesh.config import MeshConfig, TaskPromptConfig
    from mesh.task_prompts import resolve_task_prompt_bundle

    repo_root = Path(__file__).resolve().parent.parent
    config = MeshConfig.load(repo_root / "mesh.yaml")
    requested_node = os.environ.get("MESH_NODE_ID", "").strip()
    node_configs = []
    if requested_node and requested_node in config.nodes:
        node_configs.append(config.nodes[requested_node])
    node_configs.extend(
        node
        for node_id, node in config.nodes.items()
        if node_id != requested_node
    )
    for node_config in node_configs:
        definition = node_config.worker_task_types.get(task_type)
        if not isinstance(definition, dict):
            continue
        prompts = definition.get("prompts")
        if isinstance(prompts, TaskPromptConfig):
            return (
                resolve_task_prompt_bundle(prompts, repo_root),
                config,
            )
    raise ValueError(
        f"worker task type {task_type!r} has no configured prompt bundle"
    )


@tool(
    name="math_thinking",
    description=(
        "Check an argument, fill a proof gap, formulate a lemma, or solve a "
        "bounded mathematical reasoning task synchronously. Returns one "
        "tool-less mathematical response. Dispatch a math-thinking worker "
        "instead when files, computation, literature, or verification are needed."
    ),
    parameters=[
        ToolParameter(
            name="prompt",
            type="string",
            required=True,
            description="The mathematical statement, argument, proof gap, or lemma request.",
        ),
        ToolParameter(
            name="thinking_budget",
            type="integer",
            required=False,
            description="Optional positive reasoning-token budget; configured default is 16384.",
        ),
    ],
)
async def math_thinking(
    prompt: str,
    thinking_budget: int | None = None,
) -> str:
    """Run the shared math prompt once with no tools or filesystem mutation."""
    from mesh.config import backend_config_to_llm_config
    from mesh.task_prompts import compose_task_instructions

    if not isinstance(prompt, str) or not prompt.strip():
        return "Error: math_thinking requires a non-empty prompt."
    if thinking_budget is not None:
        if isinstance(thinking_budget, bool):
            return "Error: thinking_budget must be a positive integer."
        try:
            thinking_budget = int(thinking_budget)
        except (TypeError, ValueError):
            return "Error: thinking_budget must be a positive integer."
        if not 1 <= thinking_budget <= 1_000_000:
            return "Error: thinking_budget must be between 1 and 1000000."

    try:
        bundle, mesh_config = _configured_task_prompt_bundle("math-thinking")
    except (OSError, ValueError) as exc:
        return f"Error: math_thinking configuration is invalid: {exc}"
    if not bundle.sync_backend:
        return "Error: math-thinking prompts.sync_backend is not configured."
    backend_config = mesh_config.llm_backends.get(bundle.sync_backend)
    if backend_config is None:
        return (
            "Error: math-thinking sync backend "
            f"{bundle.sync_backend!r} is not configured."
        )

    resolved_budget = thinking_budget or bundle.thinking_budget
    if resolved_budget is None:
        resolved_budget = backend_config_to_llm_config(
            backend_config
        ).thinking_budget
    backend_model = str(
        getattr(backend_config, "default_model", "") or ""
    ).lower()
    if resolved_budget is None and "qwen" in backend_model:
        return (
            "Error: Qwen-backed math_thinking requires a positive "
            "thinking_budget."
        )

    domain_instructions = compose_task_instructions(
        base=bundle.base_instructions,
        plan=bundle.plan_instructions,
        execute=bundle.execute_instructions,
        sync=bundle.sync_instructions,
    )
    delegated_prompt = "\n\n".join(
        part
        for part in (
            bundle.worker_system_prompt,
            domain_instructions,
            f"## Mathematical task\n{prompt.strip()}",
            (
                "Return only the complete mathematical response. Do not call "
                "tools, refer to worker phases, or emit a report envelope."
            ),
        )
        if part
    )
    result = await mesh_qwen(
        task=prompt.strip(),
        cwd=str(Path.cwd()),
        _delegated_prompt=delegated_prompt,
        _backend_name=bundle.sync_backend,
        _tools="",
        _require_write=False,
        _thinking_budget_override=resolved_budget,
        _timeout_override=300,
        timeout_secs=300,
        _node_id="math-thinking-sync",
    )
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return f"Error: math_thinking returned non-JSON response: {result[:500]}"
    if payload.get("status") != "ok":
        detail = payload.get("errors") or payload.get("error") or result
        if isinstance(detail, list):
            detail = "; ".join(str(item) for item in detail)
        return f"Error: math_thinking failed: {detail}"
    final_text = payload.get("final_text") or payload.get("output")
    if not isinstance(final_text, str) or not final_text.strip():
        return "Error: math_thinking returned an empty response."
    return final_text


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

    # Isolation boundary — independent of the legacy `sandboxed` flag and a
    # no-op when no policy is installed (every live agent today).
    _enforce_isolation_path(resolved, path, require_write)

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

        # Also allow /tmp — only while unisolated; see _validate_path.
        if not allowed and _isolation_allows_host_tmp():
            try:
                resolved.relative_to(Path("/tmp").resolve())
                allowed = True
            except ValueError:
                pass

        if not allowed:
            op = "write to" if require_write else "access"
            raise PermissionError(
                f"Cannot {op} '{path}': not in allowed directories. "
                f"Allowed: {_allowed_dirs + _host_tmp_suffix()}"
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
    name="get_context",
    description=(
        "Read a window of lines around a target line in a file, with line numbers. "
        "Useful for inspecting the neighborhood of a specific line without reading "
        "the entire file. Returns lines from (line - radius) to (line + radius)."
    ),
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Absolute or relative path to the file",
            required=True,
        ),
        ToolParameter(
            name="line",
            type="integer",
            description="The 1-indexed line number to center the window on",
            required=True,
        ),
        ToolParameter(
            name="radius",
            type="integer",
            description="Number of lines to include above and below the target line (default 20)",
            required=False,
            default=20,
        ),
    ],
)
def get_context(path: str, line: int, radius: int = 20) -> str:
    """Read a window of lines around a target line, with line numbers."""
    try:
        path = _resolve_path(path, require_write=False)
    except PermissionError as e:
        return f"Error: {e}"
    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw_lines = f.readlines()
    except Exception as e:
        return f"Error reading file: {e}"
    lines = [ln.rstrip('\n') for ln in raw_lines]
    n = len(lines)
    start = max(1, line - radius)
    end = min(n, line + radius)
    if start > n:
        return f"Error: line {line} out of range (file has {n} lines)"
    chunk = lines[start - 1:end]
    if not chunk:
        return f"Error: line {line} out of range (file has {n} lines)"
    result = "\n".join(f"{i}\t{txt}" for i, txt in enumerate(chunk, start=start))
    return f"Lines {start}-{end} of {n} in {path}:\n\n{result}"


@tool(
    name="count_words",
    description="Count the number of whitespace-separated words in a text block.",
    parameters=[
        ToolParameter(
            name="text",
            type="string",
            description="The text to count words in",
            required=True,
        ),
    ],
)
def count_words(text: str) -> str:
    """Return word count for a text block."""
    return str(len(text.split()))


@tool(
    name="write_lines",
    description=(
        "Replace a range of original lines in a file with new content. "
        "Line numbers are 1-indexed, inclusive on both ends. The line numbers "
        "refer to the file as it was before this call; if you need to make "
        "multiple edits, work in reverse order (higher lines first) so that "
        "earlier line numbers remain valid."
    ),
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Absolute or relative path to the file to modify",
            required=True,
        ),
        ToolParameter(
            name="start_line",
            type="integer",
            description="1-indexed starting line (inclusive)",
            required=True,
        ),
        ToolParameter(
            name="end_line",
            type="integer",
            description="1-indexed ending line (inclusive)",
            required=True,
        ),
        ToolParameter(
            name="content",
            type="string",
            description="The replacement text (newline-separated lines)",
            required=True,
        ),
    ],
)
def write_lines(path: str, start_line: int, end_line: int, content: str) -> str:
    """Replace original lines [start, end] with content."""
    try:
        path = _resolve_path(path, require_write=True)
    except PermissionError as e:
        return f"Error: {e}"
    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    try:
        original_raw = open(path, "r", encoding="utf-8", errors="replace").read()
    except Exception as e:
        return f"Error reading file: {e}"
    had_trailing_newline = original_raw.endswith('\n')
    raw_lines = original_raw.splitlines(True)
    lines = [ln.rstrip('\n') for ln in raw_lines]
    n = len(lines)
    if not (1 <= start_line <= end_line <= n):
        return f"Error: line range {start_line}-{end_line} out of bounds (document has {n} lines)"
    replacement = content.splitlines()
    lines[start_line - 1:end_line] = replacement
    try:
        output = "\n".join(lines)
        if had_trailing_newline:
            output += "\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(output)
    except Exception as e:
        return f"Error writing file: {e}"
    return f"Replaced original lines {start_line}-{end_line} ({end_line - start_line + 1} lines -> {len(replacement)} lines) in {path}"


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
# DIRECTORY LISTING TOOL
# =============================================================================


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
    path = _resolve_path(dir_path)
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    return _list_tree(path, depth=depth, offset=offset, limit=limit)


# =============================================================================
# GREP (SEARCH) TOOL
# =============================================================================

_GREP_MAX_RESULTS = 100


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
    search_path = _resolve_path(path)
    if not os.path.isabs(search_path):
        search_path = os.path.join(os.getcwd(), search_path)

    if not os.path.exists(search_path):
        return f"Error: Path not found: {search_path}"

    cmd = ["grep", "-rn", "-E", "--color=never"]
    if include:
        cmd.extend(["--include", include])
    cmd.extend(["--", pattern, search_path])

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
    if total > _GREP_MAX_RESULTS:
        lines = lines[:_GREP_MAX_RESULTS]
        output = "\n".join(lines)
        output += f"\n\n({total} total matches, showing first {_GREP_MAX_RESULTS}. Narrow your search with a more specific pattern or --include.)"
    else:
        output = "\n".join(lines)
        output += f"\n\n({total} matches)"

    return output


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
    attachment_dir = None
    policy, state_paths = current_isolation()
    if policy is not None:
        scratch = _scratch_dir(policy, state_paths)
        if scratch is None:
            return "Error: isolated Gmail attachment directory is unavailable"
        attachment_dir = str(scratch / "gmail_attachments")

    host = _get_tool_host()
    if host is None:
        return "Error: Gmail not configured"

    gmail = host.gmail_client()
    if not gmail.ready:
        return "Error: Gmail client not initialized"

    result = gmail.get_email(message_id, attachment_dir=attachment_dir)
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
    try:
        attachments = _validated_attachment_paths(attachments, "gmail_send_message")
    except PermissionError as exc:
        return f"Error: {exc}"
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
    try:
        attachments = _validated_attachment_paths(attachments, "gmail_reply_to")
    except PermissionError as exc:
        return f"Error: {exc}"
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
    try:
        attachments = _validated_attachment_paths(attachments, "gmail_create_draft")
    except PermissionError as exc:
        return f"Error: {exc}"
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
    try:
        attachments = _validated_attachment_paths(attachments, "gmail_draft_reply")
    except PermissionError as exc:
        return f"Error: {exc}"
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
    description=(
        "List calendar events on a specific date. Returned start/end times are "
        "normalized to the requested timezone."
    ),
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


@tool(
    name="autonomous_controller_run",
    description=(
        "Run one explicitly authorized recursive autonomous-controller pilot on "
        "the owning live agent. This is one-shot and never schedules a wake."
    ),
    parameters=[
        ToolParameter(
            name="smoke",
            type="string",
            description=(
                "Authorized smoke name: research_resegmentation on Tron or "
                "alice_personal_assistant on Alice."
            ),
            required=True,
        ),
        ToolParameter(
            name="dry_run",
            type="boolean",
            description=(
                "Run live read-only gathering and planning, but record each leaf "
                "worker that would be dispatched instead of executing it."
            ),
            required=False,
            default=False,
        ),
    ],
)
async def autonomous_controller_run(smoke: str, dry_run: bool = False) -> str:
    """Placeholder; the owning AgentNode supplies the live runtime binding."""
    return json.dumps(
        {
            "status": "error",
            "message": (
                "autonomous_controller_run must be routed to its owning live agent"
            ),
            "smoke": smoke,
            "dry_run": dry_run,
        }
    )


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
        # Local delegated executor and academic writing
        "mesh_qwen", "recursive_harness", "style_filter", "math_thinking",
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
        # Devices
        "boox_upload",
        # Security
        "security_scan",
    ]


# =============================================================================
# DEVICE TOOLS
# =============================================================================

@tool(
    name="boox_upload",
    description=(
        "Upload a file to a BOOX e-reader over its WiFi transfer interface. "
        "Defaults to the local BOOX device at 192.168.50.226:8083."
    ),
    parameters=[
        ToolParameter(
            name="filepath",
            type="string",
            description="Path to the local file to upload.",
            required=True,
        ),
        ToolParameter(
            name="host",
            type="string",
            description="BOOX host or URL.",
            required=False,
            default="192.168.50.226",
        ),
        ToolParameter(
            name="port",
            type="integer",
            description="BOOX HTTP port.",
            required=False,
            default=8083,
        ),
    ],
)
def boox_upload(filepath: str, host: str = "192.168.50.226", port: int = 8083) -> str:
    """Upload a local file to a BOOX WiFi transfer server."""
    if not _allow_network:
        return "Error: network access is disabled for this agent"

    try:
        resolved = _validate_path(filepath)
        from tools.boox_upload import BooxUploadError, upload_file

        result = upload_file(resolved, host=host, port=int(port))
        return json.dumps(result, ensure_ascii=False, default=str)
    except BooxUploadError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error uploading to BOOX: {e}"


# =============================================================================
# SECURITY TOOLS
# =============================================================================

@tool(
    name="security_scan",
    description=(
        "Use the local Antares vulnerability-localization model to rank source "
        "files for manual review against one or more CWE classes. Results are "
        "candidates, not confirmed vulnerabilities."
    ),
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Absolute or working-directory-relative codebase root to scan.",
            required=True,
        ),
        ToolParameter(
            name="cwe",
            type="string",
            description=(
                "Optional comma-separated CWE IDs; defaults to CWE-89, CWE-79, CWE-78, "
                "CWE-22, CWE-502, and CWE-94."
            ),
            required=False,
            default="CWE-89,CWE-79,CWE-78,CWE-22,CWE-502,CWE-94",
        ),
        ToolParameter(
            name="model",
            type="string",
            description="Optional served model name (default: fdtn-ai/antares-1b).",
            required=False,
            default="fdtn-ai/antares-1b",
        ),
        ToolParameter(
            name="api",
            type="string",
            description="Optional Antares OpenAI-compatible API base URL.",
            required=False,
            default="http://127.0.0.1:8003/v1",
        ),
    ],
)
def security_scan(
    path: str,
    cwe: str = "CWE-89,CWE-79,CWE-78,CWE-22,CWE-502,CWE-94",
    model: str = "fdtn-ai/antares-1b",
    api: str = "http://127.0.0.1:8003/v1",
) -> str:
    """Run a bounded local CWE localization pass against a codebase."""
    if not _allow_network:
        return "Error: network access is disabled for this agent"
    try:
        resolved = _validate_path(path)
        from tools.security_scan import SecurityScanError, scan_repository

        cwes = tuple(part.strip() for part in cwe.split(",") if part.strip())
        result = scan_repository(resolved, cwes=cwes, model=model, api=api)
        return json.dumps(result, ensure_ascii=False, default=str)
    except SecurityScanError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error running security scan: {e}"


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
                "The recipient node ID. Use 'user:{name}' for users (e.g., 'user:operator'), "
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
    name="send_report",
    description=(
        "Submit your final report to the parent agent. Call this exactly once "
        "when your task is complete. The report content will be delivered to the "
        "agent for synthesis — it will NOT be sent directly to the user."
    ),
    parameters=[
        ToolParameter(
            name="content",
            type="string",
            description="Your final report content.",
            required=True,
        ),
    ],
)
def send_report_tool(content: str) -> str:
    """Placeholder — actual execution handled by AgentNode worker path."""
    return "Error: send_report should be handled by the agent, not executed directly"


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


def _get_plaid_client(user_id: str = "owner"):
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
    # System prompts hand agents ``[m_xxxx]`` digest/essay references and tell
    # them to fetch with memory_get, but the store keys on the bare hex ID.
    # Accept the citation surface; anything unrecognized falls through
    # unchanged so it still reports as a plain not-found.
    from .memory.ids import try_normalize_memory_id

    id = try_normalize_memory_id(id) or id
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
# Entity-link correction (in-process authority only in increment 2a)
# =============================================================================


@tool(
    name="entity_link_correct",
    description=(
        "Transactionally correct one memory's entity links, optionally editing "
        "the memory source fields in the same commit. New entity keys are "
        "generated by the registry service. This increment requires a real "
        "in-process trigger message and fails closed over MCP/socket calls."
    ),
    parameters=[
        ToolParameter(
            name="memory_id",
            type="string",
            description=(
                "Minted memory ID whose source or entity links need correction. "
                "Copy the [m_<id>] handle exactly as it was shown to you — the "
                "m_ prefix and brackets are accepted directly. Never retype or "
                "hand-transcribe the hex digits: a mistyped ID is rejected, not "
                "repaired."
            ),
            required=True,
        ),
        ToolParameter(
            name="reason",
            type="string",
            description="Why this correction is necessary.",
            required=True,
        ),
        ToolParameter(
            name="remove_entity_key",
            type="string",
            description="Existing link to remove. Omit to preserve all current links.",
            required=False,
        ),
        ToolParameter(
            name="add_entity_key",
            type="string",
            description="Existing non-retired registry key to add.",
            required=False,
        ),
        ToolParameter(
            name="new_entity_type",
            type="string",
            description="New entity type: person, project, event, or group.",
            required=False,
        ),
        ToolParameter(
            name="new_display_name",
            type="string",
            description="Display name for a new service-keyed entity.",
            required=False,
        ),
        ToolParameter(
            name="new_identity_note",
            type="string",
            description="Disambiguating note for a new entity.",
            required=False,
        ),
        ToolParameter(
            name="aliases",
            type="array",
            description="Additional display aliases for a new entity.",
            required=False,
        ),
        ToolParameter(
            name="naming_surface",
            type="string",
            description=(
                "Exact verbatim name surface from the triggering user message; "
                "required when creating a new entity."
            ),
            required=False,
        ),
        ToolParameter(
            name="memory_patch",
            type="object",
            description=(
                "Optional memory_edit patch using summary, reflection, "
                "retrieval_key, tags, and outcome."
            ),
            required=False,
        ),
    ],
)
async def entity_link_correct(
    memory_id: str,
    reason: str,
    remove_entity_key: str | None = None,
    add_entity_key: str | None = None,
    new_entity_type: str | None = None,
    new_display_name: str | None = None,
    new_identity_note: str | None = None,
    aliases: list[str] | None = None,
    naming_surface: str | None = None,
    memory_patch: dict | None = None,
) -> str:
    """Fail-closed registry handler; AgentNode supplies trusted context in-process."""
    del (
        memory_id,
        reason,
        remove_entity_key,
        add_entity_key,
        new_entity_type,
        new_display_name,
        new_identity_note,
        aliases,
        naming_surface,
        memory_patch,
    )
    return (
        "Error: entity_link_correct requires an in-process execution context; "
        "MCP, worker socket, Codex, Claude Code, harness, and mesh-tool "
        "subprocess calls are not supported in increment 2a."
    )


# =============================================================================
# Entity/group self-curation mutations (docs/plans/entity-self-curation.md §3)
#
# Every one of these is a fail-closed stub.  The real implementations live in
# ``AgentNode`` behind a ``CurationExecutionContext`` the model cannot forge:
# authority metadata (``source_message_id``, ``actor_node``, ``origin``,
# activation, status, ``replacement_key``) is context- or service-owned and is
# never a model argument.  Reaching this function body at all means no live
# curation scope was registered, so the only correct behavior is refusal.
# =============================================================================


_CURATION_FAIL_CLOSED = (
    "Error: {name} requires a live self-curation execution scope; it is only "
    "executable inside an internal curation turn. MCP, worker socket, Codex, "
    "Claude Code, harness, and mesh-tool subprocess calls are rejected."
)


@tool(
    name="entity_create",
    description=(
        "Create a new registry entity during a self-curation turn. The registry "
        "generates the key and applies the ordinary activation gate; a retired "
        "entity's alias is rejected so a merged-away entity cannot be silently "
        "recreated. Use entity_group_create for groups."
    ),
    parameters=[
        ToolParameter(
            name="entity_type",
            type="string",
            description="Entity type: person, project, or event. 'group' is rejected.",
            required=True,
        ),
        ToolParameter(
            name="display_name",
            type="string",
            description="Canonical display name for the new entity.",
            required=True,
        ),
        ToolParameter(
            name="reason",
            type="string",
            description="Why this entity needs to exist.",
            required=True,
        ),
        ToolParameter(
            name="identity_note",
            type="string",
            description="Short disambiguating note (who or what this is).",
            required=False,
        ),
        ToolParameter(
            name="aliases",
            type="array",
            description="Additional display aliases for the new entity.",
            required=False,
        ),
    ],
)
async def entity_create(
    entity_type: str,
    display_name: str,
    reason: str,
    identity_note: str | None = None,
    aliases: list[str] | None = None,
) -> str:
    """Fail-closed registry handler; AgentNode supplies trusted context in-process."""
    del entity_type, display_name, reason, identity_note, aliases
    return _CURATION_FAIL_CLOSED.format(name="entity_create")


@tool(
    name="entity_merge",
    description=(
        "Merge one entity into another: relink memories, copy aliases, rewrite "
        "group rows, then retire the loser with replacement_key set to the "
        "winner. Memories and the loser's historical dossier are preserved."
    ),
    parameters=[
        ToolParameter(
            name="loser_key",
            type="string",
            description="Registry key of the duplicate that is retired.",
            required=True,
        ),
        ToolParameter(
            name="winner_key",
            type="string",
            description="Registry key that survives and absorbs the loser's links.",
            required=True,
        ),
        ToolParameter(
            name="reason",
            type="string",
            description="Evidence that these two keys are the same entity.",
            required=True,
        ),
    ],
)
async def entity_merge(loser_key: str, winner_key: str, reason: str) -> str:
    """Fail-closed registry handler; AgentNode supplies trusted context in-process."""
    del loser_key, winner_key, reason
    return _CURATION_FAIL_CLOSED.format(name="entity_merge")


@tool(
    name="entity_edit",
    description=(
        "Edit one registry entity: update_details, add_alias, remove_alias, or "
        "retire. Retirement goes through the service's own retire path; it does "
        "not set replacement_key (use entity_merge for that)."
    ),
    parameters=[
        ToolParameter(
            name="entity_key",
            type="string",
            description="Registry key to edit.",
            required=True,
        ),
        ToolParameter(
            name="operation",
            type="string",
            description=(
                "One of: update_details, add_alias, remove_alias, retire."
            ),
            required=True,
        ),
        ToolParameter(
            name="reason",
            type="string",
            description="Why this edit is necessary.",
            required=True,
        ),
        ToolParameter(
            name="display_name",
            type="string",
            description="New display name (update_details only).",
            required=False,
        ),
        ToolParameter(
            name="identity_note",
            type="string",
            description="New identity note (update_details only).",
            required=False,
        ),
        ToolParameter(
            name="alias",
            type="string",
            description="Alias to add or remove (add_alias / remove_alias only).",
            required=False,
        ),
    ],
)
async def entity_edit(
    entity_key: str,
    operation: str,
    reason: str,
    display_name: str | None = None,
    identity_note: str | None = None,
    alias: str | None = None,
) -> str:
    """Fail-closed registry handler; AgentNode supplies trusted context in-process."""
    del entity_key, operation, reason, display_name, identity_note, alias
    return _CURATION_FAIL_CLOSED.format(name="entity_edit")


@tool(
    name="entity_backfill",
    description=(
        "Queue bounded self-curation backfill over older uncurated memories. "
        "Walks your memories oldest-first, slices the uncurated run into "
        "fixed-size batches, and queues each one as an ordinary internal "
        "curation turn. The walk stops at the first already-curated memory, so "
        "nothing is curated twice and the live-curated era is never crossed. "
        "Returns immediately with what was queued; the turns run afterwards, "
        "one at a time, whenever the router is free. Call it again later to "
        "continue where this run's ceiling stopped."
    ),
    parameters=[
        ToolParameter(
            name="max_batches",
            type="integer",
            description=(
                "Optional ceiling on slices queued by this call. Defaults to "
                "entity_self_curation_backfill_max_batches; a larger value is "
                "clamped to it."
            ),
            required=False,
        ),
    ],
)
async def entity_backfill(max_batches: int | None = None) -> str:
    """Fail-closed registry handler; AgentNode owns the router curation queue."""
    del max_batches
    return (
        "Error: entity_backfill requires an in-process AgentNode execution "
        "context with a live curation queue; MCP, worker socket, and "
        "unauthenticated subprocess calls are rejected."
    )


@tool(
    name="entity_group_create",
    description=(
        "Create a pending entity group with an explicit purpose. Activation is "
        "deterministic and not this tool's decision: a group activates only "
        "with two or more active members plus bridge evidence across the "
        "configured number of distinct formation windows."
    ),
    parameters=[
        ToolParameter(
            name="display_name",
            type="string",
            description="Name of the group.",
            required=True,
        ),
        ToolParameter(
            name="purpose",
            type="string",
            description=(
                "Non-empty statement of what this group is for; stored as the "
                "group's identity note and required for activation."
            ),
            required=True,
        ),
        ToolParameter(
            name="reason",
            type="string",
            description="Why this group needs to exist.",
            required=True,
        ),
        ToolParameter(
            name="aliases",
            type="array",
            description="Additional display aliases for the group.",
            required=False,
        ),
    ],
)
async def entity_group_create(
    display_name: str,
    purpose: str,
    reason: str,
    aliases: list[str] | None = None,
) -> str:
    """Fail-closed registry handler; AgentNode supplies trusted context in-process."""
    del display_name, purpose, reason, aliases
    return _CURATION_FAIL_CLOSED.format(name="entity_group_create")


@tool(
    name="entity_group_member_add",
    description=(
        "Add one member entity to a group. Membership lives in the registry, "
        "not in dossier prose; this is the only way to change a roster. "
        "Re-evaluates the deterministic activation gate on return."
    ),
    parameters=[
        ToolParameter(
            name="group_key",
            type="string",
            description="Registry key of the group.",
            required=True,
        ),
        ToolParameter(
            name="member_key",
            type="string",
            description="Registry key of the member entity to add.",
            required=True,
        ),
        ToolParameter(
            name="reason",
            type="string",
            description="Evidence that this entity belongs to this group.",
            required=True,
        ),
        ToolParameter(
            name="role",
            type="string",
            description="Optional role of this member within the group.",
            required=False,
        ),
    ],
)
async def entity_group_member_add(
    group_key: str,
    member_key: str,
    reason: str,
    role: str | None = None,
) -> str:
    """Fail-closed registry handler; AgentNode supplies trusted context in-process."""
    del group_key, member_key, reason, role
    return _CURATION_FAIL_CLOSED.format(name="entity_group_member_add")


@tool(
    name="entity_group_member_remove",
    description=(
        "Remove one member entity from a group. Reconciles that group's "
        "protected roster block before returning."
    ),
    parameters=[
        ToolParameter(
            name="group_key",
            type="string",
            description="Registry key of the group.",
            required=True,
        ),
        ToolParameter(
            name="member_key",
            type="string",
            description="Registry key of the member entity to remove.",
            required=True,
        ),
        ToolParameter(
            name="reason",
            type="string",
            description="Why this entity no longer belongs to this group.",
            required=True,
        ),
    ],
)
async def entity_group_member_remove(
    group_key: str, member_key: str, reason: str
) -> str:
    """Fail-closed registry handler; AgentNode supplies trusted context in-process."""
    del group_key, member_key, reason
    return _CURATION_FAIL_CLOSED.format(name="entity_group_member_remove")


@tool(
    name="token_count",
    description=(
        "Measure a candidate body with the same tokenizer the budget gate uses. "
        "Call this before writing a dossier or digest edit: over-ceiling writes "
        "are refused, never truncated, so measuring first saves a round trip."
    ),
    parameters=[
        ToolParameter(
            name="text",
            type="string",
            description="The text to measure.",
            required=True,
        ),
    ],
)
async def token_count(text: str) -> str:
    """Measure text with ``mesh.llm.estimate_tokens`` (§3.5).

    Unlike the mutation tools this is read-only and safe outside a curation
    scope, so it has a real implementation here rather than a fail-closed stub.
    """
    from .llm import estimate_tokens, _encoder

    if not isinstance(text, str):
        return "Error: token_count requires a string 'text' argument."
    return json.dumps(
        {
            "tokens": estimate_tokens(text),
            "chars": len(text),
            "tokenizer": "tiktoken" if _encoder is not None else "word-heuristic",
        },
        sort_keys=True,
    )


# =============================================================================
# Standing digest tools (interactive correction of the published digest)
# =============================================================================


def _resolve_digest_path() -> str | None:
    """Return the absolute path to the agent's standing digest, or None.

    ``MemorySystemV2`` exposes neither ``_config`` nor ``config``, so the
    config-walk below returns None for every V2 agent.  AgentNode publishes the
    path directly on init; prefer that and keep the walk as the V1 fallback.
    """
    if _standing_digest_path:
        return os.path.expanduser(_standing_digest_path)
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
    from .digest_io import read_digest

    try:
        content = read_digest(path)
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
    from .digest_io import edit_digest

    # The read, match check and write all happen inside one exclusive lock —
    # a concurrent editor (curation turn, fold driver) must not be able to
    # slip between our snapshot and our write and lose one of the two edits.
    ok, error, n_replaced = edit_digest(path, old_text, new_text, replace_all)
    if not ok:
        return error

    import logging
    logging.getLogger("mesh.memory.audit").info(
        "digest_edit old=%r new=%r n=%d path=%s",
        old_text[:100], new_text[:100], n_replaced, path,
    )
    return f"Digest updated successfully ({n_replaced} replacement{'s' if n_replaced > 1 else ''})."


# =============================================================================
# Autonomous agent mode: project dossier, session reports, budget ledger
#
# Narrow, artifact-scoped tools in the digest_get/digest_edit mould. The path
# is always derived from the project entity key, never supplied by the model,
# so these grant no general filesystem authority. All logic lives in
# mesh/project_dossier.py; these are the registry surface.
# =============================================================================


@tool(
    name="dossier_read",
    description=(
        "Read a project dossier — the durable state artifact for one project "
        "(Identity, Goals, Tasks, Timeline, Narrative, Standing decisions, "
        "Open threads). The path is derived from the project entity key."
    ),
    parameters=[
        ToolParameter(
            name="entity_key",
            type="string",
            description="Project entity key, e.g. 'project:mesh-infra'.",
            required=True,
        ),
    ],
)
def dossier_read(entity_key: str, state_paths=None) -> str:
    """Read the project dossier."""
    from .project_dossier import DossierError, read_dossier

    try:
        return read_dossier(entity_key, state_paths)
    except DossierError as e:
        return f"Error: {e}"


@tool(
    name="dossier_edit",
    description=(
        "Edit a project dossier in place by exact string replacement, like "
        "digest_edit. The edit is validated against the seven-section dossier "
        "constitution and the token ceiling before it lands; a refused edit "
        "leaves the dossier byte-identical."
    ),
    parameters=[
        ToolParameter(
            name="entity_key",
            type="string",
            description="Project entity key, e.g. 'project:mesh-infra'.",
            required=True,
        ),
        ToolParameter(
            name="old_text",
            type="string",
            description="Exact text to find (must match uniquely unless replace_all=true).",
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
def dossier_edit(
    entity_key: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
    state_paths=None,
) -> str:
    """Exact string replacement in a project dossier."""
    from .project_dossier import DossierError, edit_dossier

    try:
        result = edit_dossier(
            entity_key,
            old_text,
            new_text,
            replace_all=bool(replace_all),
            state_paths=state_paths,
        )
    except DossierError as e:
        return f"Error: {e}"
    logging.getLogger("mesh.memory.audit").info(
        "dossier_edit key=%s old=%r new=%r n=%d path=%s",
        entity_key, old_text[:100], new_text[:100], result.replacements, result.path,
    )
    plural = "s" if result.replacements > 1 else ""
    return (
        f"Dossier updated ({result.replacements} replacement{plural}). "
        f"Now {result.tokens} tokens of {result.token_budget} budget."
    )


@tool(
    name="dossier_write_report",
    description=(
        "Create an immutable autonomous session report and link it from the "
        "project dossier's Timeline. Create-once: identical content is an "
        "idempotent success, different content for an existing path is "
        "refused. Corrections are written as a new report, never an overwrite."
    ),
    parameters=[
        ToolParameter(
            name="entity_key",
            type="string",
            description="Project entity key, e.g. 'project:mesh-infra'.",
            required=True,
        ),
        ToolParameter(
            name="date",
            type="string",
            description="Session date as YYYY-MM-DD.",
            required=True,
        ),
        ToolParameter(
            name="seq",
            type="integer",
            description="Sequence number of the session within that date (1-based).",
            required=True,
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Full markdown body of the session report.",
            required=True,
        ),
    ],
)
def dossier_write_report(
    entity_key: str, date: str, seq: int, content: str, state_paths=None
) -> str:
    """Write an immutable session report and link it from the Timeline."""
    from .project_dossier import DossierError, write_report

    try:
        path = write_report(entity_key, date, seq, content, state_paths)
    except DossierError as e:
        return f"Error: {e}"
    return f"Session report written to {path} and linked from the dossier Timeline."


@tool(
    name="dossier_check_budget",
    description=(
        "Read the autonomous worker budget for a project without spending it. "
        "Returns JSON with remaining, used, limit, and resets_at."
    ),
    parameters=[
        ToolParameter(
            name="entity_key",
            type="string",
            description="Project entity key, e.g. 'project:mesh-infra'.",
            required=True,
        ),
    ],
)
def dossier_check_budget(entity_key: str, state_paths=None) -> str:
    """Report remaining worker admissions for a project."""
    from .project_dossier import DossierError, check_budget

    try:
        return json.dumps(check_budget(entity_key, state_paths), sort_keys=True)
    except DossierError as e:
        return f"Error: {e}"


@tool(
    name="dossier_spend_budget",
    description=(
        "Charge worker admissions against a project's autonomous budget. An "
        "autonomous controller must NOT call this around a dispatch: the "
        "router already charges exactly one admission when it admits the "
        "worker, so calling it here charges the same worker twice. Use "
        "dossier_check_budget to read headroom instead. Refuses without "
        "mutating the ledger when the spend would exceed the limit; do not "
        "retry with different wording after a refusal."
    ),
    parameters=[
        ToolParameter(
            name="entity_key",
            type="string",
            description="Project entity key, e.g. 'project:mesh-infra'.",
            required=True,
        ),
        ToolParameter(
            name="count",
            type="integer",
            description="Number of admissions to charge (default 1).",
            required=False,
        ),
    ],
)
def dossier_spend_budget(entity_key: str, count: int = 1, state_paths=None) -> str:
    """Charge worker admissions against the project budget."""
    from .project_dossier import BudgetExhausted, DossierError, spend_budget

    try:
        return json.dumps(spend_budget(entity_key, count, state_paths), sort_keys=True)
    except BudgetExhausted as e:
        return json.dumps(
            {"status": "autonomous_budget_exhausted", "detail": str(e)}, sort_keys=True
        )
    except DossierError as e:
        return f"Error: {e}"


# =============================================================================
# Interpretive essay tools (standing-digest essay layer)
#
# essay_edit: exact string replacement on essay body text. Available to
#   agents (via router tool set) and mesh-tool CLI.
# essay_get / essay_list: pull-based retrieval tools (Phase 4). Exposed to
#   live agents when essays_retrieval_enabled=true in NodeConfig. Added to
#   the agent's enabled_tools list dynamically in agent_node.py (not in the
#   static mesh.yaml allowlist) so the gate can be flipped per-agent.
# All three are available via mesh-tool CLI for testing and ops.
# =============================================================================


def _get_essay_store():
    """Return the MemoryStore backing the memory system, or None."""
    if _memory_system is None:
        return None
    return getattr(_memory_system, "_store", None)


@tool(
    name="essay_get",
    description=(
        "Read an interpretive essay by entity key. Returns the full markdown "
        "body plus metadata (title, citations, cross-references, patch count). "
        "Entity keys follow the pattern 'person:name', 'project:name', or "
        "'event:description'."
    ),
    parameters=[
        ToolParameter(
            name="key",
            type="string",
            description="Entity key (e.g., 'person:kaylee', 'project:novelty-pipeline').",
            required=True,
        ),
    ],
)
def essay_get(key: str) -> str:
    """Read an interpretive essay by entity key."""
    store = _get_essay_store()
    if store is None:
        return "Error: Memory system not initialized."
    essay = store.get_essay(key)
    if essay is None:
        return f"No essay found for key '{key}'."
    lines = [
        f"# {essay['title'] or key}",
        "",
        essay["body"],
        "",
        "---",
        f"**Entity key:** {essay['entity_key']}",
        f"**Patch count:** {essay['patch_count']}",
        f"**Updated:** {essay['updated_at'][:16]}",
        f"**Created:** {essay['created_at'][:16]}",
    ]
    if essay["citations"]:
        lines.append(f"**Citations:** {', '.join(essay['citations'])}")
    if essay["cross_refs"]:
        lines.append(f"**Cross-references:** {', '.join(essay['cross_refs'])}")
    return "\n".join(lines)


@tool(
    name="essay_list",
    description=(
        "List all interpretive essays. Shows entity keys, titles, patch "
        "counts, and last-updated timestamps."
    ),
    parameters=[],
)
def essay_list() -> str:
    """List all interpretive essays."""
    store = _get_essay_store()
    if store is None:
        return "Error: Memory system not initialized."
    essays = store.list_essays()
    if not essays:
        return "No essays."
    lines = []
    for e in essays:
        title_str = f" — {e['title']}" if e["title"] else ""
        lines.append(
            f"**{e['entity_key']}**{title_str} | "
            f"patches: {e['patch_count']} | "
            f"updated: {e['updated_at'][:16]}"
        )
    return f"{len(essays)} essay{'s' if len(essays) != 1 else ''}:\n\n" + "\n".join(lines)


@tool(
    name="essay_edit",
    description=(
        "Edit an interpretive essay in place. Exact string replacement "
        "like digest_edit / file_edit. If the entity key does not exist, "
        "creates a new essay with new_text as the body (old_text is ignored "
        "for creation). Use this to maintain, correct, or extend essays. "
        "Optionally updates citations and cross_refs atomically with the edit."
    ),
    parameters=[
        ToolParameter(
            name="key",
            type="string",
            description="Entity key (e.g., 'person:kaylee', 'project:novelty-pipeline').",
            required=True,
        ),
        ToolParameter(
            name="old_text",
            type="string",
            description=(
                "Exact text to find in the essay body (must match uniquely "
                "unless replace_all=true). For new essays, this is ignored."
            ),
            required=True,
        ),
        ToolParameter(
            name="new_text",
            type="string",
            description="Replacement text (or full body for new essays).",
            required=True,
        ),
        ToolParameter(
            name="title",
            type="string",
            description="Essay title (optional; set on create, updated on edit if provided).",
            required=False,
        ),
        ToolParameter(
            name="replace_all",
            type="boolean",
            description="Replace all occurrences (default false — requires unique match).",
            required=False,
        ),
        ToolParameter(
            name="citations",
            type="string",
            description=(
                "JSON array of memory IDs this essay cites (e.g., "
                "'[\"m_abc123\", \"m_def456\"]'). Replaces existing citations."
            ),
            required=False,
        ),
        ToolParameter(
            name="cross_refs",
            type="string",
            description=(
                "JSON array of entity keys this essay cross-references (e.g., "
                "'[\"project:fishing\", \"person:owner\"]'). Replaces existing cross-refs."
            ),
            required=False,
        ),
    ],
)
def essay_edit(
    key: str,
    old_text: str,
    new_text: str,
    title: str = "",
    replace_all: bool = False,
    citations: str = "",
    cross_refs: str = "",
) -> str:
    """Edit or create an interpretive essay."""
    import json as _json
    import logging

    store = _get_essay_store()
    if store is None:
        return "Error: Memory system not initialized."

    # Parse citations/cross_refs JSON upfront so we fail early on bad input.
    parsed_citations = None
    parsed_cross_refs = None
    if citations:
        try:
            parsed_citations = _json.loads(citations)
            if not isinstance(parsed_citations, list):
                return "Error: citations must be a JSON array of strings."
        except _json.JSONDecodeError as e:
            return f"Error: invalid citations JSON — {e}"
    if cross_refs:
        try:
            parsed_cross_refs = _json.loads(cross_refs)
            if not isinstance(parsed_cross_refs, list):
                return "Error: cross_refs must be a JSON array of strings."
        except _json.JSONDecodeError as e:
            return f"Error: invalid cross_refs JSON — {e}"

    existing = store.get_essay(key)
    if existing is None:
        store.create_essay(
            key, body=new_text, title=title,
            citations=parsed_citations, cross_refs=parsed_cross_refs,
        )
        logging.getLogger("mesh.memory.audit").info(
            "essay_create key=%r title=%r body_len=%d",
            key, title, len(new_text),
        )
        return f"Essay '{key}' created ({len(new_text)} chars)."

    # Validate the text replacement FIRST — fail early, mutate nothing.
    body = existing["body"]
    count = body.count(old_text)
    if count == 0:
        return "Error: old_text not found in essay body."
    if not replace_all and count > 1:
        return (
            f"Error: old_text matches {count} locations — "
            f"provide a more specific string or set replace_all=true."
        )

    # Replacement is valid — compute the new body.
    if replace_all:
        new_body = body.replace(old_text, new_text)
    else:
        new_body = body.replace(old_text, new_text, 1)
    n_replaced = count if replace_all else 1

    # Apply title + body + citations + cross_refs atomically, gated on the
    # revision we read above.  The read → compute → write window is now
    # genuinely concurrent (curation turns no longer serialise behind message
    # turns), so an unconditional UPDATE would silently drop whichever patch
    # landed in between.
    ok, conflict = store.update_essay_if_revision(
        key,
        int(existing.get("patch_count") or 0),
        body=new_body,
        title=title if title else None,
        citations=parsed_citations,
        cross_refs=parsed_cross_refs,
    )
    if not ok:
        return f"Error: {conflict}"

    logging.getLogger("mesh.memory.audit").info(
        "essay_edit key=%r old=%r new=%r replace_all=%s",
        key, old_text[:100], new_text[:100], replace_all,
    )
    return f"Essay updated ({n_replaced} replacement{'s' if n_replaced > 1 else ''})."


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
# global ToolRegistry. The functions here only define schemas; the real
# execution path is in RouterV2's bound handlers. They are intentionally not
# included in get_all_tool_names(), because invoking them through mesh-tool
# would have no RouterV2 instance or worker slot to own.


@tool(
    name="solicitation_scout",
    description=(
        "Search for grant solicitations matching a PI's CV and research "
        "program. Launches a three-phase pipeline: (1) extracts the PI's "
        "capability profile, (2) searches broadly for currently open "
        "solicitations, and (3) reads promising solicitations deeply and "
        "ranks them by fit. For a new run, provide cv and research_threads. "
        "To continue an existing run, provide run_dir instead; completed "
        "phases are skipped. Set dry_run=true to inspect the planned phases "
        "without claiming the worker slot (dry-run and resume are mutually "
        "exclusive). A real run occupies the worker slot; use worker_status "
        "for phase progress. On completion it returns ranked opportunities "
        "with fit scores, deadlines, budget ranges, eligibility, and 'why "
        "this PI' arguments."
    ),
    parameters=[
        ToolParameter(
            name="cv",
            type="string",
            description=(
                "Path to the PI's CV file. Accepted source formats are PDF, "
                "plain text, Markdown, and LaTeX. The file must already exist."
            ),
            required=False,
        ),
        ToolParameter(
            name="research_threads",
            type="string",
            description=(
                "Path to a Markdown or text file describing the PI's active "
                "research threads. The file must already exist."
            ),
            required=False,
        ),
        ToolParameter(
            name="pi_papers",
            type="string",
            description=(
                "Optional comma-separated paper identifiers, DOI/arXiv values, "
                "URLs, titles, or local file paths."
            ),
            required=False,
        ),
        ToolParameter(
            name="project_name",
            type="string",
            description=(
                "Optional short, filesystem-safe label for this search and "
                "its run directory."
            ),
            required=False,
        ),
        ToolParameter(
            name="run_dir",
            type="string",
            description=(
                "Path to an existing solicitation-scout run directory to "
                "resume. Its run-config.json is reused and completed phases "
                "are skipped. Mutually exclusive with dry_run."
            ),
            required=False,
        ),
        ToolParameter(
            name="dry_run",
            type="boolean",
            description=(
                "Print the planned three-phase commands and prompts without "
                "creating a run or claiming the worker slot. Mutually "
                "exclusive with run_dir."
            ),
            required=False,
            default=False,
        ),
    ],
)
async def solicitation_scout(
    cv: str = "",
    research_threads: str = "",
    pi_papers: str = "",
    project_name: str = "",
    run_dir: str = "",
    dry_run: bool = False,
) -> str:  # pragma: no cover
    """Placeholder — executed by RouterV2._tool_solicitation_scout."""
    raise NotImplementedError("solicitation_scout is handled by RouterV2")


@tool(
    name="skill_draft",
    description=(
        "Launch a governed drafting worker after completing a procedure the "
        "user expects to repeat. The worker receives the recent episode, any "
        "named authoritative source files, the card schema, and a worked "
        "example. It may create only a validated status=proposed card under "
        "the calling agent's .proposals directory. It cannot activate a card "
        "or modify index.yaml. The drafting worker occupies the normal worker "
        "slot and reports the proposal for human review."
    ),
    parameters=[
        ToolParameter(
            name="task_summary",
            type="string",
            description=(
                "What recurring procedure the card should capture, including "
                "the just-completed task and any scope the user specified."
            ),
            required=True,
        ),
        ToolParameter(
            name="source_files",
            type="array",
            description=(
                "Optional absolute paths to authoritative local runbooks, "
                "configuration, tests, or implementation files. Their contents "
                "and SHA-256 fingerprints are included in the drafting handoff."
            ),
            required=False,
            default=[],
        ),
        ToolParameter(
            name="trace_path",
            type="string",
            description=(
                "Optional absolute path to a specific older completed-worker "
                "trace. When omitted, the latest completed non-drafting worker "
                "trace for the calling agent is selected automatically."
            ),
            required=False,
            default="",
        ),
    ],
)
async def skill_draft(
    task_summary: str,
    source_files: list[str] | None = None,
    trace_path: str = "",
) -> str:  # pragma: no cover
    """Placeholder — executed by RouterV2._tool_skill_draft."""
    raise NotImplementedError("skill_draft is handled by RouterV2")


@tool(
    name="worker_launch",
    description=(
        "Launch a worker to execute a task autonomously. The worker does NOT "
        "see the conversation — it receives the task text you write here plus "
        "its standing digest, and nothing else — and can use all tools "
        "including file edits and bash. Returns immediately: the router "
        "continues its loop while the worker runs. Use this for tasks "
        "requiring sustained autonomous work: code changes, multi-step "
        "investigations, deployments."
    ),
    parameters=[
        ToolParameter(
            name="task",
            type="string",
            description=(
                "The worker's entire input. It cannot see this conversation, "
                "so write it self-contained: what needs to be done, the "
                "relevant file paths, any findings or constraints already "
                "established, and what success looks like. Resolve every "
                "pronoun — a task reading 'fix it' gives the worker nothing."
            ),
            required=True,
        ),
        ToolParameter(
            name="task_type",
            type="string",
            description=(
                "REQUIRED configured task type (for example simple-code, "
                "moderate-code, complex-code, writing, research, audit, or "
                "plan). The type resolves to a backend and workflow through "
                "mesh.yaml. A dispatch without a task_type is refused and no "
                "worker runs."
            ),
            required=True,
        ),
        ToolParameter(
            name="backend",
            type="string",
            description=(
                "HARD OVERRIDE ONLY: populate this only when the user "
                "explicitly named a backend in the current instruction, and "
                "copy that name verbatim. Never choose a backend directly. "
                "Still requires a task_type, and is refused on a task type "
                "with a configured Plan-Execute-Verify workflow."
            ),
            required=False,
        ),
        ToolParameter(
            name="reason",
            type="string",
            description=(
                "Concise one-line rationale for the selected task_type. "
                "Omitting it falls back to the configured default with a "
                "warning, so always supply one."
            ),
            required=False,
        ),
    ],
)
async def worker_launch(
    task: str,
    task_type: str = "",
    backend: str = "",
    reason: str = "",
) -> str:  # pragma: no cover
    """Placeholder — executed by RouterV2._tool_worker_launch."""
    raise NotImplementedError("worker_launch is handled by RouterV2")


@tool(
    name="worker_list",
    description=(
        "List every fixed worker slot, including empty slots, with the current "
        "slot-table revision, worker IDs, lifecycle states, elapsed time, and "
        "bounded task/activity previews."
    ),
    parameters=[],
)
async def worker_list() -> str:  # pragma: no cover
    """Placeholder — executed by RouterV2._tool_worker_list."""
    raise NotImplementedError("worker_list is handled by RouterV2")


@tool(
    name="worker_status",
    description=(
        "Check worker progress. Pass worker_id for a detailed transcript. "
        "Without worker_id, a single active worker is detailed; zero or "
        "multiple active workers return the compact all-slot view."
    ),
    parameters=[
        ToolParameter(
            name="worker_id",
            type="string",
            description="Exact worker ID to inspect.",
            required=False,
        ),
        ToolParameter(
            name="max_lines",
            type="integer",
            description=(
                "Maximum activity lines to return. Default 100; values are "
                "clamped to the safe range 1-500."
            ),
            required=False,
            default=100,
        ),
    ],
)
async def worker_status(
    worker_id: str | None = None,
    max_lines: int = 100,
) -> str:  # pragma: no cover
    """Placeholder — executed by RouterV2._tool_worker_status."""
    raise NotImplementedError("worker_status is handled by RouterV2")


@tool(
    name="worker_cancel",
    description=(
        "Cancel one router-owned worker by exact worker_id, or explicitly "
        "cancel every active worker with cancel_all=true. If multiple workers "
        "run and neither target is supplied, returns the slot list and does "
        "not cancel anything."
    ),
    parameters=[
        ToolParameter(
            name="worker_id",
            type="string",
            description="Exact worker ID to cancel.",
            required=False,
        ),
        ToolParameter(
            name="cancel_all",
            type="boolean",
            description="Explicitly cancel every active worker.",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="reason",
            type="string",
            description="Optional cancellation reason for diagnostics.",
            required=False,
            default="",
        ),
    ],
)
async def worker_cancel(
    worker_id: str | None = None,
    cancel_all: bool = False,
    reason: str = "",
) -> str:  # pragma: no cover
    """Placeholder — executed by RouterV2._tool_worker_cancel."""
    raise NotImplementedError("worker_cancel is handled by RouterV2")


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
    name="conversation_notes_get",
    description=(
        "Get pinned notes for the current conversation, or for an explicit "
        "conversation_id. These notes usually point to worklog or operations files."
    ),
    parameters=[
        ToolParameter(name="conversation_id", type="string", description="Conversation ID. Defaults to the triggering conversation.", required=False),
    ],
)
async def conversation_notes_get(conversation_id: str = None) -> str:  # pragma: no cover
    raise NotImplementedError("conversation_notes_get is handled by AgentNode")


@tool(
    name="conversation_notes_set",
    description=(
        "Set pinned notes for the current conversation. Keep this short: file "
        "paths plus one-line descriptions, not the file contents."
    ),
    parameters=[
        ToolParameter(name="content", type="string", description="Pinned note content. Use concise pointers to relevant files.", required=True),
        ToolParameter(name="conversation_id", type="string", description="Conversation ID. Defaults to the triggering conversation.", required=False),
    ],
)
async def conversation_notes_set(content: str, conversation_id: str = None) -> str:  # pragma: no cover
    raise NotImplementedError("conversation_notes_set is handled by AgentNode")


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

def _history_dir() -> "Path":
    """The history root for this call.

    Scoped to ``StatePaths.history_dir`` under an enabled policy, so an
    isolated agent reads the history inside its own boundary rather than the
    shared ``~/.mesh/history`` directory that holds every other agent's file.
    Falls back to the global constant when no policy is installed.
    """
    policy, state_paths = current_isolation()
    if policy is not None and state_paths is not None:
        return Path(state_paths.history_dir)
    if policy is not None:
        return Path(policy.state_root) / "history"
    from .paths import HISTORY_DIR

    return HISTORY_DIR


def _resolve_history_file(node_id: str | None = None) -> "Path | None":
    """Resolve the calling agent's history file.

    Identity comes from MESH_NODE_ID (set by AgentNode in its own process
    and inherited by tool subprocesses), matching how mesh-tool identifies
    the calling agent. Paths come from the scoped history root — no
    hardcoded home directories.

    ``node_id`` is accepted only so callers can be checked against the
    calling identity; under an enabled policy a request for another agent's
    history raises :class:`PermissionError` rather than reading it.
    """
    caller = os.environ.get("MESH_NODE_ID", "")
    requested = node_id or caller
    if not requested:
        return None

    policy, _ = current_isolation()
    if policy is not None and requested != caller:
        raise PermissionError(
            f"Cannot read history for '{requested}': an isolated agent may only "
            f"search its own history ({caller or 'unknown caller'})."
        )

    history_dir = _history_dir()
    # AgentNode persists to agent-{nickname}.json (nickname = last segment)
    nickname = requested.split(":")[-1]
    candidate = history_dir / f"agent-{nickname}.json"
    if candidate.exists():
        return candidate
    # Fallback: base Node default path uses the full node id
    fallback = history_dir / f"{requested.replace(':', '-')}.json"
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
    try:
        history_file = _resolve_history_file()
    except PermissionError as exc:
        return f"Error: {exc}"
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
        policy, _ = current_isolation()
        if policy is not None:
            from .paths import resolve_path as _rp

            resolved = Path(_rp(local_path)).resolve()
            _enforce_isolation_path(resolved, local_path, require_write=False)
            local_path = str(resolved)
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
