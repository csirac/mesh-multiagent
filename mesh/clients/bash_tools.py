import json
import subprocess
import shutil
import os
from pathlib import Path
from typing import Any, Dict, Tuple, List
import shlex
import select
import time
from .bash_persistent_experiment import PersistentBashSession, MARKER_PREFIX


_EXIT1_NO_STDERR_NOTE = (
    "Exit code 1 with no stderr typically means 'no match' (grep), "
    "'false' (test/diff), or 'no differences' — this is the shell's exit "
    "code, not a tool failure."
)


def _annotate_exit_code(result: Dict[str, Any]) -> Dict[str, Any]:
    """Add a clarifying note when an exit code is ambiguous.

    A returncode of 1 with empty/whitespace-only stderr is almost always a
    shell-level "no match"/"false" signal (grep, test, diff), not a tool
    error. Annotating it keeps the LLM from misreading it as a broken tool
    and looping on the identical command.
    """
    if (
        result.get("returncode") == 1
        and not (result.get("stderr") or "").strip()
    ):
        result["note"] = _EXIT1_NO_STDERR_NOTE
    return result


def _check_bwrap_available() -> bool:
    """Check if bubblewrap (bwrap) is installed."""
    return shutil.which("bwrap") is not None


def _check_bwrap_functional() -> bool:
    """
    Check if bwrap can actually create sandboxes.

    Ubuntu 24.04 enables apparmor_restrict_unprivileged_userns by default,
    which blocks unprivileged user namespace creation. This prevents bwrap
    from working even if the binary is installed.

    To enable bwrap on Ubuntu 24.04+, run:
        sudo ./scripts/setup_bwrap_apparmor.sh
    """
    if not _check_bwrap_available():
        return False
    try:
        # Use symlinks for /bin etc. on modern Ubuntu
        result = subprocess.run(
            [
                "bwrap",
                "--ro-bind", "/usr", "/usr",
                "--ro-bind", "/etc", "/etc",
                "--symlink", "usr/bin", "/bin",
                "--symlink", "usr/lib", "/lib",
                "--symlink", "usr/lib64", "/lib64",
                "--symlink", "usr/sbin", "/sbin",
                "--dev", "/dev",
                "--proc", "/proc",
                "--", "/usr/bin/echo", "sandbox_test"
            ],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0 and "sandbox_test" in result.stdout
    except Exception:
        return False


class BashTools:
    def __init__(
        self,
        *,
        user_confirm: bool = True,
        workdir: str | None = None,
        timeout_sec: float = 10.0,
        max_output_chars: int = 2000000,
        use_persistent: bool = True,
        sandboxed: bool = False,
        allowed_dirs: List[str] | None = None,
        allow_network: bool = True,
    ):
        self.user_confirm = user_confirm
        self.workdir = workdir
        self.timeout_sec = timeout_sec
        self.max_output_chars = max_output_chars
#        self.use_persistent = use_persistent
        self.use_persistent = False
        self._session: PersistentBashSession | None = None

        # Sandbox settings
        self.sandboxed = sandboxed
        self.allowed_dirs = allowed_dirs or []
        self.allow_network = allow_network

        # Validate sandbox is functional if enabled
        if self.sandboxed:
            if not _check_bwrap_available():
                raise RuntimeError(
                    "Sandbox mode requires bubblewrap (bwrap). "
                    "Install with: sudo apt install bubblewrap"
                )
            if not _check_bwrap_functional():
                raise RuntimeError(
                    "bwrap is installed but cannot create sandboxes. "
                    "On Ubuntu 24.04+, run: sudo ./scripts/setup_bwrap_apparmor.sh"
                )

    def _get_session(self) -> PersistentBashSession:
        if self._session is None or not self._session.is_alive():
            self._session = PersistentBashSession(workdir=self.workdir)
        return self._session

    def _build_sandbox_command(self, cmd: str) -> List[str]:
        """
        Build a bwrap command that sandboxes the given shell command.

        The sandbox:
        - Mounts /usr, /etc, /run read-only (system)
        - Creates symlinks for /bin, /lib, /lib64, /sbin (Ubuntu 24.04 structure)
        - Mounts allowed_dirs read-write
        - Mounts /tmp read-write
        - Optionally blocks network access
        """
        bwrap_args = ["bwrap"]

        # System directories (read-only)
        for sysdir in ["/usr", "/etc", "/run"]:
            if os.path.exists(sysdir):
                bwrap_args.extend(["--ro-bind", sysdir, sysdir])

        # Create symlinks for /bin, /lib, etc. (Ubuntu 24.04 uses symlinks to /usr/*)
        bwrap_args.extend(["--symlink", "usr/bin", "/bin"])
        bwrap_args.extend(["--symlink", "usr/lib", "/lib"])
        bwrap_args.extend(["--symlink", "usr/lib64", "/lib64"])
        bwrap_args.extend(["--symlink", "usr/sbin", "/sbin"])

        # Device and proc
        bwrap_args.extend(["--dev", "/dev"])
        bwrap_args.extend(["--proc", "/proc"])

        # Tmpfs for home (so it exists but is empty by default)
        from ..paths import resolve_path as _rp
        home = str(_rp("~"))
        bwrap_args.extend(["--tmpfs", home])

        # Read-only mounts for common config dirs
        for config_dir in [
            _rp("~/.gitconfig"),
            _rp("~/.config"),
            _rp("~/.local/share"),
            _rp("~/.local/bin"),
        ]:
            if os.path.exists(config_dir):
                bwrap_args.extend(["--ro-bind", config_dir, config_dir])

        # Allowed directories (read-write)
        for allowed_dir in self.allowed_dirs:
            expanded = _rp(allowed_dir)
            resolved = str(Path(expanded).resolve())
            if os.path.exists(resolved):
                bwrap_args.extend(["--bind", resolved, resolved])

        # /tmp is always writable
        bwrap_args.extend(["--bind", "/tmp", "/tmp"])

        # Network isolation (optional)
        if not self.allow_network:
            bwrap_args.append("--unshare-net")

        # Set working directory
        if self.workdir:
            bwrap_args.extend(["--chdir", self.workdir])

        # The command to run
        bwrap_args.extend(["--", "/bin/bash", "-c", cmd])

        return bwrap_args

    def _run_sandboxed_command(self, cmd: str) -> Dict[str, Any]:
        """Run a command inside the bwrap sandbox."""
        bwrap_cmd = self._build_sandbox_command(cmd)

        proc = subprocess.Popen(
            bwrap_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,  # Prevent backgrounded children from inheriting pipe FDs
        )

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        stdout_len = 0
        stderr_len = 0
        truncated = False
        timed_out = False

        max_chars = self.max_output_chars
        timeout_sec = self.timeout_sec
        start = time.time()

        stdout_done = proc.stdout is None
        stderr_done = proc.stderr is None
        proc_exited_at: float | None = None  # Track when process exited
        PIPE_GRACE_SEC = 5.0  # Max seconds to wait for pipe EOF after process exits

        def _store(chunk: str, which: str) -> None:
            nonlocal stdout_len, stderr_len, truncated
            if truncated or not chunk:
                return
            current_total = stdout_len + stderr_len
            remaining = max_chars - current_total
            if remaining <= 0:
                truncated = True
                return
            piece = chunk[:remaining]
            if which == "stdout":
                stdout_chunks.append(piece)
                stdout_len += len(piece)
            else:
                stderr_chunks.append(piece)
                stderr_len += len(piece)
            if len(chunk) > len(piece):
                truncated = True

        try:
            while True:
                if timeout_sec is not None and timeout_sec > 0:
                    elapsed = time.time() - start
                    if elapsed > timeout_sec and proc.poll() is None:
                        proc.kill()
                        timed_out = True
                        try:
                            out, err = proc.communicate(timeout=1)
                        except subprocess.TimeoutExpired:
                            out, err = "", ""
                        if out:
                            _store(out, "stdout")
                        if err:
                            _store(err, "stderr")
                        stdout_done = True
                        stderr_done = True
                        break

                reads = []
                if not stdout_done and proc.stdout is not None:
                    reads.append(proc.stdout)
                if not stderr_done and proc.stderr is not None:
                    reads.append(proc.stderr)

                if not reads:
                    break

                # Grace period: if process has exited but pipes are still open
                # (e.g., backgrounded child inherited FDs), don't wait forever.
                if proc.poll() is not None:
                    if proc_exited_at is None:
                        proc_exited_at = time.time()
                    elif time.time() - proc_exited_at > PIPE_GRACE_SEC:
                        break

                rlist, _, _ = select.select(reads, [], [], 0.1)

                if not rlist:
                    if proc.poll() is not None and stdout_done and stderr_done:
                        break
                    continue

                for r in rlist:
                    line = r.readline()
                    if not line:
                        if r is proc.stdout:
                            stdout_done = True
                        else:
                            stderr_done = True
                        continue

                    if r is proc.stdout:
                        _store(line, "stdout")
                    else:
                        _store(line, "stderr")

                if proc.poll() is not None and stdout_done and stderr_done:
                    break

        finally:
            try:
                proc.wait(timeout=1)
            except Exception:
                pass
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.stderr is not None:
                proc.stderr.close()

        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)

        if stdout_len + stderr_len > max_chars:
            total = (stdout + "\n[STDERR]\n" + stderr)[:max_chars]
            stdout = total
            stderr = ""
            truncated = True

        if timed_out:
            return {
                "stdout": stdout,
                "stderr": "Command timed out.",
                "returncode": None,
                "timeout": True,
                "truncated": truncated,
            }

        return _annotate_exit_code({
            "stdout": stdout,
            "stderr": stderr,
            "returncode": proc.returncode,
            "timeout": False,
            "truncated": truncated,
        })

    def _needs_confirmation(self, cmd: str) -> bool:
        """
        Decide whether a shell command should prompt the user for confirmation.

        Heuristics:
          - Always confirm for rn-add / rn-edit / rn-delete (note mutations).
          - Confirm for obviously dangerous commands (rm, chmod, chown, etc.).
          - Confirm for obvious redirections (>) that could overwrite files.
        """
        cmd = (cmd or "").strip()
        if not cmd:
            return False

        try:
            tokens = shlex.split(cmd, posix=True)
        except ValueError:
            # Fallback: crude split if shlex fails
            tokens = cmd.split()

        if not tokens:
            return False

        prog = tokens[0]

        # 1. Note-editing commands: always confirm
        if prog in {"rn-add", "rn-edit", "rn-delete"}:
            return True

        # 2. Obvious dangerous commands
        dangerous_programs = {
            "rm",
            "mv",
            "chmod",
            "chown",
            "truncate",
            "dd",
            "mkfs",
            "wipefs",
            "mount",
            "umount",
            "systemctl",
            "kill",
            "pkill",
            "reboot",
            "shutdown",
            "userdel",
            "groupdel",
            "sudo",  # be paranoid
        }
        if prog in dangerous_programs:
            return True

        # 3. Dangerous patterns anywhere in the command string
        lower_cmd = cmd.lower()

        # rm -rf, rm -r, or obvious variants
        if "rm -rf" in lower_cmd or "rm -r " in lower_cmd:
            return True

        # Redirection that could overwrite files (conservative: any ">")
        if ">" in cmd:
            return True

        # Obvious "pipe to rm" pattern
        if "xargs rm" in lower_cmd:
            return True

        # Shell function / fork bomb style nonsense (very rough)
        if ":(){:|:&};:" in cmd:
            return True

        return False

    def _confirm(self, prompt: str) -> tuple[bool, str]:
        if not self.user_confirm:
            return True, ""
        try:
            ans = input(prompt).strip().lower()
        except EOFError:
            return False, ""
        return ans in ("y", "yes"), ans

    def _run_command(self, cmd: str) -> Dict[str, Any]:
        """
        Run a one-off command in a fresh bash shell.

        This is the stateless fallback behavior.
        """
        proc = subprocess.Popen(
            ["/bin/bash", "-lc", cmd],
            cwd=self.workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
            start_new_session=True,  # Prevent backgrounded children from inheriting pipe FDs
        )

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        stdout_len = 0
        stderr_len = 0
        truncated = False
        timed_out = False

        max_chars = self.max_output_chars
        timeout_sec = self.timeout_sec
        start = time.time()

        stdout_done = proc.stdout is None
        stderr_done = proc.stderr is None
        proc_exited_at: float | None = None  # Track when bash itself exited
        PIPE_GRACE_SEC = 5.0  # Max seconds to wait for pipe EOF after bash exits

        def _store(chunk: str, which: str) -> None:
            nonlocal stdout_len, stderr_len, truncated
            if truncated or not chunk:
                return
            current_total = stdout_len + stderr_len
            remaining = max_chars - current_total
            if remaining <= 0:
                truncated = True
                return
            piece = chunk[:remaining]
            if which == "stdout":
                stdout_chunks.append(piece)
                stdout_len += len(piece)
            else:
                stderr_chunks.append(piece)
                stderr_len += len(piece)
            if len(chunk) > len(piece):
                truncated = True

        try:
            while True:
                # Timeout check
                if timeout_sec is not None and timeout_sec > 0:
                    elapsed = time.time() - start
                    if elapsed > timeout_sec and proc.poll() is None:
                        proc.kill()
                        timed_out = True
                        try:
                            out, err = proc.communicate(timeout=1)
                        except subprocess.TimeoutExpired:
                            out, err = "", ""
                        if out:
                            _store(out, "stdout")
                        if err:
                            _store(err, "stderr")
                        stdout_done = True
                        stderr_done = True
                        break

                reads = []
                if not stdout_done and proc.stdout is not None:
                    reads.append(proc.stdout)
                if not stderr_done and proc.stderr is not None:
                    reads.append(proc.stderr)

                if not reads:
                    # both streams have hit EOF
                    break

                # Grace period: if bash has exited but pipes are still open
                # (e.g., backgrounded child inherited FDs), don't wait forever.
                if proc.poll() is not None:
                    if proc_exited_at is None:
                        proc_exited_at = time.time()
                    elif time.time() - proc_exited_at > PIPE_GRACE_SEC:
                        # Bash exited but pipes still open — orphaned FDs from
                        # backgrounded children (nohup ... &). Break out.
                        break

                rlist, _, _ = select.select(reads, [], [], 0.1)

                if not rlist:
                    if proc.poll() is not None and stdout_done and stderr_done:
                        break
                    continue

                for r in rlist:
                    line = r.readline()
                    if not line:
                        # EOF on this stream
                        if r is proc.stdout:
                            stdout_done = True
                        else:
                            stderr_done = True
                        continue

                    if r is proc.stdout:
                        _store(line, "stdout")
                    else:
                        _store(line, "stderr")

                if proc.poll() is not None and stdout_done and stderr_done:
                    break

        finally:
            try:
                proc.wait(timeout=1)
            except Exception:
                pass

            if proc.stdout is not None:
                proc.stdout.close()
            if proc.stderr is not None:
                proc.stderr.close()

        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)

        if stdout_len + stderr_len > max_chars:
            total = (stdout + "\n[STDERR]\n" + stderr)[:max_chars]
            stdout = total
            stderr = ""
            truncated = True

        if timed_out:
            return {
                "stdout": stdout,
                "stderr": "Command timed out.",
                "returncode": None,
                "timeout": True,
                "truncated": truncated,
            }

        return _annotate_exit_code({
            "stdout": stdout,
            "stderr": stderr,
            "returncode": proc.returncode,
            "timeout": False,
            "truncated": truncated,
        })

    def dispatch_tool_call(self, tool_call: Any) -> Tuple[Dict[str, str], bool]:
        # Normalize tool_call access (same pattern as your other tools)
        try:
            name = tool_call.function.name
            raw_args = tool_call.function.arguments
            call_id = tool_call.id
        except AttributeError:
            try:
                name = tool_call["function"]["name"]
                raw_args = tool_call["function"]["arguments"]
                call_id = tool_call.get("id", "")
            except Exception:
                return ({
                    "role": "tool",
                    "tool_call_id": "",
                    "content": "Malformed tool call: missing fields."
                }, True)

        try:
            args = json.loads(raw_args) if raw_args else {}
        except Exception as e:
            return ({
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"Invalid JSON in tool arguments: {e} args: {raw_args}"
            }, True)

        is_error = False
        output: Any

        if name == "bash_exec":
            command = (args.get("command") or "").strip()
            timeout = args.get("timeout")
            if timeout:
                timeout_val = float(timeout)
                # If timeout > 1000, assume it's milliseconds (common API convention)
                # and convert to seconds. 1000+ seconds (16+ min) would be unusual.
                if timeout_val > 1000:
                    timeout_val = timeout_val / 1000.0
                self.timeout_sec = timeout_val
            else:
                self.timeout_sec = float(15.0)

            if not command:
                output = "Missing required parameter: command"
                is_error = True
            else:
                # Confirmation is currently disabled per user request; ok=True.
                ok = True

                if not ok:
                    output = "Command not executed (cancelled by user)."
                else:
                    if self.sandboxed:
                        # Use bwrap sandbox
                        result = self._run_sandboxed_command(command)
                        output = result
                    elif self.use_persistent:
                        sess = self._get_session()
                        result = sess.run(
                            command,
                            timeout_sec=self.timeout_sec,
                            max_output_chars=self.max_output_chars,
                        )
                        output = _annotate_exit_code(result)
                    else:
                        result = self._run_command(command)
                        output = result

        else:
            output = f"Unknown tool: {name}"
            is_error = True

        if not isinstance(output, str):
            try:
                content = json.dumps(output, ensure_ascii=False)
            except TypeError:
                content = str(output)
        else:
            content = output

        return ({
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,
        }, is_error)
