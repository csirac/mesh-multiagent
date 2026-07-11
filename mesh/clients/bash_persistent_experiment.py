#!/usr/bin/env python3
"""Prototype for a persistent bash session using pipes (no pty).

This does NOT affect the TUI tools; it's a standalone experiment.

We keep a single /bin/bash process alive, send it wrapped commands via stdin,
embed a unique marker with the exit status, and read stdout/stderr until we
see that marker.
"""

import json
import subprocess
import select
import time
from typing import Optional, Dict, Any

MARKER_PREFIX = "__ALAN_MARK__"


class PersistentBashSession:
    def __init__(self, workdir: Optional[str] = None):
        self.workdir = workdir
        self.proc: Optional[subprocess.Popen[str]] = None
        self._start_bash()

    def _start_bash(self) -> None:
        self.proc = subprocess.Popen(
            ["/bin/bash"],
            cwd=self.workdir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        if self.proc.stdin is None or self.proc.stdout is None or self.proc.stderr is None:
            raise RuntimeError("Failed to start bash with pipes")

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def run(self, cmd: str, timeout_sec: float = 10.0, max_output_chars: int = 4000) -> Dict[str, Any]:
        if not self.is_alive() or self.proc is None or self.proc.stdin is None:
            return {
                "stdout": "",
                "stderr": "Session not alive",
                "returncode": None,
                "timeout": False,
                "truncated": False,
            }

        # Build a wrapped command that records exit status and prints a marker
        # line on stdout.
        wrapped_cmd = (
            f"{cmd}\n"
            "status=$?\n"
            "printf '__ALAN_MARK__%s\\n' \"$status\"\n"
        )

        self.proc.stdin.write(wrapped_cmd)
        self.proc.stdin.flush()

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        stdout_len = 0
        stderr_len = 0
        truncated = False
        timed_out = False
        exit_code: Optional[int] = None

        max_chars = max_output_chars
        start = time.time()

        stdout_done_for_this = False
        stderr_done_for_this = False
        marker_seen = False

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
                if timeout_sec and time.time() - start > timeout_sec:
                    # If we never saw our marker, treat as a real timeout.
                    if exit_code is None:
                        timed_out = True
                    break

                reads = []
                if not stdout_done_for_this and self.proc.stdout is not None:
                    reads.append(self.proc.stdout)
                if not stderr_done_for_this and self.proc.stderr is not None:
                    reads.append(self.proc.stderr)

                if not reads:
                    # No more streams to read; stop.
                    break

                rlist, _, _ = select.select(reads, [], [], 0.1)
                if not rlist:
                    # check if process died
                    if self.proc.poll() is not None:
                        break
                    continue

                for r in rlist:
                    line = r.readline()
                    if not line:
                        if r is self.proc.stdout:
                            stdout_done_for_this = True
                        else:
                            stderr_done_for_this = True
                        continue

                    if r is self.proc.stdout and line.startswith(MARKER_PREFIX):
                        # Marker line: parse exit code, but do NOT print or store it.
                        tail = line[len(MARKER_PREFIX):].strip()
                        try:
                            exit_code = int(tail)
                        except ValueError:
                            exit_code = None
                        stdout_done_for_this = True
                        marker_seen = True
                        # We have our marker; no need to read further for this command.
                        break

                    # Echo non-marker lines to console for debugging
                    print(line, end="")

                    if r is self.proc.stdout:
                        _store(line, "stdout")
                    else:
                        _store(line, "stderr")

                if marker_seen:
                    break

        finally:
            if self.proc is not None and self.proc.poll() is not None:
                # Process died; clean up
                try:
                    self.proc.wait(timeout=1)
                except Exception:
                    pass

        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)

        return {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": exit_code,
            "timeout": timed_out,
            "truncated": truncated,
        }


def main() -> None:
    print("Starting persistent bash session prototype (pipes)...")
    sess = PersistentBashSession(workdir=None)
    while True:
        try:
            cmd = input("\n[experiment]$ ")
        except EOFError:
            break
        cmd = cmd.strip()
        if not cmd:
            continue
        if cmd in {"exit", "quit"}:
            break
        res = sess.run(cmd, timeout_sec=10.0, max_output_chars=4000)
        print("\nRESULT:", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
