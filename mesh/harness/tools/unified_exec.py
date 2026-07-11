"""
Shell tool — persistent shell sessions with optional PTY support.

Supports two modes:
1. One-shot (default): each call spawns a fresh bash subprocess. The working
   directory persists across calls via `cd` interception.
2. Persistent session: when `session_id` is provided, the command runs inside
   a long-lived bash process. Shell variables, environment, working directory,
   and background jobs survive between calls. Use `write_stdin` to feed input
   to interactive programs (read prompts, REPLs, etc.).

Long output is automatically truncated (head + tail with elision marker).
"""

from __future__ import annotations

import atexit
import logging
import os
import pty
import re
import secrets
import select
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ...tools import tool, ToolParameter

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120
DEFAULT_YIELD_TIME_MS = 5000
HEAD_LINES = 200
TAIL_LINES = 50
ELIDE_THRESHOLD = HEAD_LINES + TAIL_LINES + 20
OUTPUT_BUFFER_MAX = 512 * 1024  # 512 KB max buffered output per session

# Sentinel prefix echoed after each command so we know it finished.
# A per-command random nonce is appended at runtime to prevent false
# positives when user output contains the sentinel string.
_SENTINEL_PREFIX = "__MESH_CMD_DONE_8f3a_"


# ---------------------------------------------------------------------------
# Output truncation (shared between one-shot and persistent modes)
# ---------------------------------------------------------------------------

def _smart_truncate(output: str, max_lines: int = ELIDE_THRESHOLD) -> str:
    """Truncate long output keeping head + tail with elision marker."""
    lines = output.split("\n")
    if len(lines) <= max_lines:
        return output
    head = lines[:HEAD_LINES]
    tail = lines[-TAIL_LINES:]
    elided = len(lines) - HEAD_LINES - TAIL_LINES
    return "\n".join(head) + f"\n\n[... {elided} lines elided ...]\n\n" + "\n".join(tail)


# ---------------------------------------------------------------------------
# One-shot execution (unchanged from Phase 2)
# ---------------------------------------------------------------------------

def _run_command(command: str, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Execute a command in a subprocess, preserving working directory."""
    cwd = os.getcwd()
    env = os.environ.copy()
    env["TERM"] = "dumb"
    env["NO_COLOR"] = "1"

    try:
        proc = subprocess.Popen(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=env,
            preexec_fn=os.setsid,
        )
        try:
            stdout, _ = proc.communicate(timeout=timeout)
            output = stdout.decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            output = proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else ""
            output += f"\n\n[Command timed out after {timeout}s]"
            return {"exit_code": -1, "output": _smart_truncate(output), "timed_out": True}

        return {
            "exit_code": proc.returncode,
            "output": _smart_truncate(output),
            "timed_out": False,
        }
    except Exception as e:
        return {"exit_code": -1, "output": f"Error: {e}", "timed_out": False}


# ---------------------------------------------------------------------------
# Persistent shell session
# ---------------------------------------------------------------------------

@dataclass
class ShellSession:
    """A persistent bash shell session."""
    session_id: str
    pid: int
    # For PTY mode: master_fd is the PTY master. stdin_pipe is None.
    # For pipe mode: master_fd is None. stdin_pipe and stdout_pipe are set.
    master_fd: int | None = None
    stdin_pipe: Any = None  # subprocess stdin PIPE
    stdout_pipe: Any = None  # subprocess stdout PIPE (fd)
    proc: subprocess.Popen | None = None
    tty: bool = False
    cwd: str = ""
    last_used_at: float = 0.0

    @property
    def alive(self) -> bool:
        if self.proc is None:
            return False
        return self.proc.poll() is None

    @property
    def read_fd(self) -> int:
        """The file descriptor to read output from."""
        if self.master_fd is not None:
            return self.master_fd
        if self.stdout_pipe is not None:
            return self.stdout_pipe
        raise RuntimeError("No readable fd for session")

    def write(self, data: str) -> None:
        """Write data to the shell's stdin."""
        encoded = data.encode("utf-8")
        if self.master_fd is not None:
            os.write(self.master_fd, encoded)
        elif self.stdin_pipe is not None:
            self.stdin_pipe.write(encoded)
            self.stdin_pipe.flush()
        else:
            raise RuntimeError("No writable fd for session")

    def drain(self, timeout_ms: int = DEFAULT_YIELD_TIME_MS) -> str:
        """Read available output, waiting up to timeout_ms for data."""
        deadline = time.monotonic() + timeout_ms / 1000.0
        collected = bytearray()
        fd = self.read_fd

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                ready, _, _ = select.select([fd], [], [], min(remaining, 0.1))
            except (ValueError, OSError):
                break
            if ready:
                try:
                    chunk = os.read(fd, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                collected.extend(chunk)
                if len(collected) > OUTPUT_BUFFER_MAX:
                    break
            # No data and we've waited at least a short time — check if
            # we've seen enough silence to return early.
            elif collected and (deadline - time.monotonic()) > 0.05:
                # Got data earlier but nothing new for 100ms — likely done.
                time.sleep(0.05)
                try:
                    ready2, _, _ = select.select([fd], [], [], 0.05)
                except (ValueError, OSError):
                    break
                if not ready2:
                    break

        text = collected.decode("utf-8", errors="replace")
        self.last_used_at = time.monotonic()
        return text

    def terminate(self) -> None:
        """Terminate the shell process."""
        if self.proc and self.alive:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=2)
                except Exception:
                    pass
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None


# Module-level session storage
_sessions: dict[str, ShellSession] = {}
_sessions_lock = threading.Lock()


def _create_session(session_id: str, tty: bool = False) -> ShellSession:
    """Create a new persistent shell session."""
    env = os.environ.copy()
    env["TERM"] = "dumb" if not tty else "xterm-256color"
    env["NO_COLOR"] = "1"
    env["PS1"] = ""  # suppress prompt to reduce noise
    cwd = os.getcwd()

    if tty:
        master_fd, slave_fd = pty.openpty()
        proc = subprocess.Popen(
            ["bash", "--norc", "--noprofile", "-i"],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=cwd,
            env=env,
            preexec_fn=os.setsid,
        )
        os.close(slave_fd)
        session = ShellSession(
            session_id=session_id,
            pid=proc.pid,
            master_fd=master_fd,
            proc=proc,
            tty=True,
            cwd=cwd,
            last_used_at=time.monotonic(),
        )
    else:
        proc = subprocess.Popen(
            ["bash", "--norc", "--noprofile"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=env,
            preexec_fn=os.setsid,
        )
        session = ShellSession(
            session_id=session_id,
            pid=proc.pid,
            stdin_pipe=proc.stdin,
            stdout_pipe=proc.stdout.fileno() if proc.stdout else None,
            proc=proc,
            tty=False,
            cwd=cwd,
            last_used_at=time.monotonic(),
        )

    # Make stdout non-blocking for pipe mode
    if not tty and proc.stdout:
        import fcntl
        flags = fcntl.fcntl(proc.stdout.fileno(), fcntl.F_GETFL)
        fcntl.fcntl(proc.stdout.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)

    # Also make master_fd non-blocking for PTY mode
    if tty and master_fd is not None:
        import fcntl
        flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    with _sessions_lock:
        _sessions[session_id] = session

    # Drain any startup output (bash banner, etc.)
    session.drain(timeout_ms=500)

    logger.debug("Created persistent shell session %s (pid=%d, tty=%s)", session_id, proc.pid, tty)
    return session


def _get_session(session_id: str) -> ShellSession | None:
    with _sessions_lock:
        return _sessions.get(session_id)


def _destroy_session(session_id: str) -> None:
    with _sessions_lock:
        session = _sessions.pop(session_id, None)
    if session:
        session.terminate()
        logger.debug("Destroyed shell session %s", session_id)


def _cleanup_all_sessions() -> None:
    """Terminate all persistent sessions. Registered with atexit."""
    with _sessions_lock:
        ids = list(_sessions.keys())
    for sid in ids:
        _destroy_session(sid)


atexit.register(_cleanup_all_sessions)


def _sigterm_handler(signum: int, frame: Any) -> None:
    """Clean up sessions on SIGTERM so benchmark kills don't orphan bash processes."""
    _cleanup_all_sessions()
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, _sigterm_handler)


def _exec_in_session(
    session: ShellSession,
    command: str,
    yield_time_ms: int = DEFAULT_YIELD_TIME_MS,
) -> str:
    """Execute a command in an existing persistent session."""
    if not session.alive:
        return "[Session terminated — process exited]"

    # Per-command random nonce prevents false positives when user output
    # happens to contain the sentinel prefix.
    nonce = secrets.token_hex(4)
    sentinel = f"{_SENTINEL_PREFIX}{nonce}__"
    sentinel_pattern = re.compile(re.escape(sentinel) + r"\r?\n?")

    # Send the command followed by a sentinel echo so we can detect completion.
    full_cmd = f"{command}\necho {sentinel}\n"
    session.write(full_cmd)

    # Drain output, waiting for the sentinel or timeout
    deadline = time.monotonic() + yield_time_ms / 1000.0
    collected = []
    while time.monotonic() < deadline:
        remaining_ms = int((deadline - time.monotonic()) * 1000)
        if remaining_ms <= 0:
            break
        chunk = session.drain(timeout_ms=min(remaining_ms, 500))
        if chunk:
            collected.append(chunk)
            full = "".join(collected)
            if sentinel in full:
                result = sentinel_pattern.sub("", full)
                return _smart_truncate(result.strip())
        else:
            if not session.alive:
                break

    result = "".join(collected)
    result = sentinel_pattern.sub("", result)
    if time.monotonic() >= deadline and result:
        result += f"\n\n[Output truncated after {yield_time_ms}ms — session still active, use session_id to continue]"
    return _smart_truncate(result.strip()) if result.strip() else "(no output yet — session still active)"


def _write_stdin_to_session(
    session: ShellSession,
    chars: str,
    yield_time_ms: int = DEFAULT_YIELD_TIME_MS,
) -> str:
    """Write raw input to a session's stdin and return output."""
    if not session.alive:
        return "[Session terminated — process exited]"

    session.write(chars)
    output = session.drain(timeout_ms=yield_time_ms)
    return _smart_truncate(output.strip()) if output.strip() else "(no output)"


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

@tool(
    name="shell",
    description=(
        "Execute a shell command. Supports two modes:\n\n"
        "**One-shot mode** (default): runs the command in a fresh bash subprocess. "
        "The working directory persists between calls (use cd freely). "
        "Long output is automatically truncated.\n\n"
        "**Persistent session mode**: provide `session_id` to run commands in a "
        "long-lived bash process. Shell variables, environment, cd state, and "
        "background jobs survive between calls. A new session_id auto-creates "
        "the session. Use `write_stdin` to feed input to interactive programs "
        "(read prompts, REPLs).\n\n"
        "Set `tty=true` when a command needs a terminal (e.g., interactive "
        "programs, curses UIs, programs that check isatty).\n\n"
        "Use `yield_time_ms` to control how long to wait for output before "
        "returning. For long-running commands, use a short yield time to get "
        "partial output, then call again with the same session_id to continue."
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
            description="Timeout in seconds for one-shot mode (default 120). Ignored in session mode.",
            required=False,
            default=120,
        ),
        ToolParameter(
            name="session_id",
            type="string",
            description=(
                "Run the command in a named persistent session. "
                "If the session doesn't exist, it is created automatically. "
                "Omit for one-shot mode."
            ),
            required=False,
        ),
        ToolParameter(
            name="write_stdin",
            type="string",
            description=(
                "Write raw characters to an existing session's stdin "
                "instead of executing a command. Requires session_id. "
                "Use this to answer interactive prompts or feed input to REPLs."
            ),
            required=False,
        ),
        ToolParameter(
            name="yield_time_ms",
            type="integer",
            description=(
                "How long to wait for output in session mode (default 5000ms). "
                "For long-running commands, use a shorter value to get partial output."
            ),
            required=False,
            default=5000,
        ),
        ToolParameter(
            name="tty",
            type="boolean",
            description=(
                "Run in a PTY (pseudo-terminal). Use for commands that need "
                "a terminal (e.g., REPLs, ncurses, programs checking isatty). "
                "Default false. Only applies when creating a new session."
            ),
            required=False,
            default=False,
        ),
    ],
)
def shell(
    command: str,
    timeout: float = DEFAULT_TIMEOUT,
    session_id: str | None = None,
    write_stdin: str | None = None,
    yield_time_ms: int = DEFAULT_YIELD_TIME_MS,
    tty: bool = False,
) -> str:
    """Execute a shell command with optional persistent session support."""
    # -----------------------------------------------------------------------
    # write_stdin mode: feed input to an existing session
    # -----------------------------------------------------------------------
    if write_stdin is not None:
        if session_id is None:
            return "Error: write_stdin requires session_id"
        session = _get_session(session_id)
        if session is None:
            return f"Error: no session with id '{session_id}'"
        return _write_stdin_to_session(session, write_stdin, yield_time_ms)

    # -----------------------------------------------------------------------
    # Persistent session mode
    # -----------------------------------------------------------------------
    if session_id is not None:
        session = _get_session(session_id)
        if session is None:
            session = _create_session(session_id, tty=tty)
        if not session.alive:
            _destroy_session(session_id)
            session = _create_session(session_id, tty=tty)
        return _exec_in_session(session, command, yield_time_ms)

    # -----------------------------------------------------------------------
    # One-shot mode (original behavior, backward compatible)
    # -----------------------------------------------------------------------
    from ...paths import real_home, resolve_path as _resolve_home
    stripped = command.strip()
    if stripped == "cd":
        target = str(real_home())
        os.chdir(target)
        return f"Changed directory to {target}"
    cd_match = re.match(r'^cd\s+(\S+)\s*$', stripped)
    if cd_match:
        target = cd_match.group(1).strip("'\"")
        target = _resolve_home(target)
        if not os.path.isabs(target):
            target = os.path.join(os.getcwd(), target)
        target = os.path.realpath(target)
        if os.path.isdir(target):
            os.chdir(target)
            return f"Changed directory to {target}"
        else:
            return f"Error: Not a directory: {target}"

    result = _run_command(command, timeout=timeout)

    parts = []
    if result["output"]:
        parts.append(result["output"])
    if result["exit_code"] != 0:
        parts.append(f"\n[Exit code: {result['exit_code']}]")
    return "\n".join(parts) if parts else "(no output)"
