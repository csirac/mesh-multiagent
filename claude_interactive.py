#!/usr/bin/env python3
"""
claude_interactive.py — Standalone Claude Code interactive session wrapper.

Wraps Claude Code's interactive mode via tmux, driven by an LLM state machine.
Prompt on stdin, response on stdout. All config via CLI args or env vars.

The driver is a STATE MACHINE, not a thinker. It injects prompts verbatim,
watches for completion, handles interrupts mechanically, and extracts output.
It does NOT reason about the task — Claude does the thinking.

NO MESH IMPORTS. Mesh shells out to this the same way it shells out to 'claude -p'.
"""

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_TIMEOUT = 2
EXIT_SESSION_CRASH = 3
EXIT_DRIVER_ERROR = 4

log = logging.getLogger("claude_interactive")


class ClaudeInteractive:
    """Manages a Claude Code interactive session via tmux."""

    def __init__(self, *, model, effort, driver_url, driver_model,
                 permission_mode, timeout, session_name, working_dir,
                 cc_binary, poll_interval, keep_session, log_file=None):
        self.model = model
        self.effort = effort
        self.driver_url = driver_url
        self.driver_model = driver_model
        self.permission_mode = permission_mode
        self.timeout = timeout
        self.session_name = session_name or f"ci-{uuid.uuid4().hex[:8]}"
        self.working_dir = working_dir or os.getcwd()
        self.cc_binary = cc_binary
        self.poll_interval = poll_interval
        self.keep_session = keep_session
        self.log_file = log_file

    # ── tmux primitives ──────────────────────────────────────────────

    def _tmux(self, *args, timeout=10):
        r = subprocess.run(
            ["tmux"] + list(args),
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout, r.returncode

    def _capture(self, lines=1000):
        out, _ = self._tmux("capture-pane", "-t", self.session_name,
                            "-p", f"-S-{lines}")
        return out

    def _send_keys(self, text):
        """Send literal text (no key-name interpretation)."""
        subprocess.run(
            ["tmux", "send-keys", "-t", self.session_name, "-l", "--", text],
            timeout=10,
        )

    def _send_raw(self, *keys):
        """Send raw key names (Enter, Down, C-c, etc.)."""
        for k in keys:
            subprocess.run(
                ["tmux", "send-keys", "-t", self.session_name, k],
                timeout=5,
            )

    def _session_alive(self):
        _, rc = self._tmux("has-session", "-t", self.session_name)
        return rc == 0

    # ── session lifecycle ────────────────────────────────────────────

    def _start_session(self):
        subprocess.run(
            ["tmux", "kill-session", "-t", self.session_name],
            capture_output=True, timeout=5,
        )
        time.sleep(0.5)

        cmd_parts = [self.cc_binary, "--dangerously-skip-permissions",
                     "--model", self.model, "--effort", self.effort]
        if self.permission_mode == "restricted":
            cmd_parts += ["--allowedTools", "Read,Glob,Grep"]
        claude_cmd = " ".join(cmd_parts)

        shell_cmd = f"cd {shlex.quote(self.working_dir)} && {claude_cmd}"
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", self.session_name,
             "-x", "200", "-y", "50", shell_cmd],
            timeout=10,
        )
        log.debug("tmux session %s created", self.session_name)
        time.sleep(3)

        self._handle_confirmation()
        self._wait_idle(max_wait=60)
        log.info("Session %s ready", self.session_name)

    def _handle_confirmation(self):
        screen = self._capture()
        if "No, exit" in screen and "Yes, I accept" in screen:
            log.debug("Confirmation dialog detected — accepting")
            self._send_raw("Down")
            time.sleep(0.5)
            self._send_raw("Enter")
            time.sleep(5)
            screen = self._capture()

        if self._detect_trust_prompt(screen):
            log.debug("Trust prompt detected — confirming")
            self._send_raw("Enter")
            time.sleep(3)
            screen = self._capture()
            if self._detect_trust_prompt(screen):
                self._send_raw("Down")
                time.sleep(0.3)
                self._send_raw("Enter")
                time.sleep(3)
        else:
            log.debug("No confirmation/trust dialog")

    def _detect_trust_prompt(self, screen):
        lower = screen.lower()
        trust_signals = ("trust", "safe", "do you want to proceed",
                         "is this folder", "is this directory",
                         "approve this directory", "allow access")
        return any(s in lower for s in trust_signals) and "❯" not in screen

    def _wait_idle(self, max_wait=60):
        """Wait for Claude's idle prompt via screen stabilization."""
        prev = ""
        stable = 0
        start = time.time()
        while time.time() - start < max_wait:
            time.sleep(self.poll_interval)
            screen = self._capture()
            if screen == prev and screen.strip():
                stable += 1
                if stable >= 2:
                    log.debug("Idle after %.1fs", time.time() - start)
                    return screen
            else:
                stable = 0
            prev = screen
        raise RuntimeError(f"Timeout ({max_wait}s) waiting for idle prompt")

    # ── screen stabilization ─────────────────────────────────────────

    def _wait_stable(self, max_wait=10):
        """Wait for screen to stop changing. Returns when stable or max_wait elapsed."""
        prev = self._capture()
        start = time.time()
        while time.time() - start < max_wait:
            time.sleep(self.poll_interval)
            screen = self._capture()
            if screen == prev and screen.strip():
                log.debug("Screen stabilized after %.1fs", time.time() - start)
                return screen
            prev = screen
        log.warning("Screen didn't stabilize within %ds — proceeding", max_wait)
        return self._capture()

    # ── prompt injection ─────────────────────────────────────────────

    def _inject_prompt(self, prompt):
        """Send a prompt into the Claude session."""
        prompt = prompt.rstrip()

        if len(prompt) < 2000 and "\n" not in prompt:
            self._send_keys(prompt)
            time.sleep(0.2)
        else:
            fd, path = tempfile.mkstemp(prefix="ci_prompt_", suffix=".txt")
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(prompt)
                subprocess.run(
                    ["tmux", "load-buffer", "-b", f"ci_buf_{self.session_name}", path],
                    timeout=5, check=True,
                )
                subprocess.run(
                    ["tmux", "paste-buffer", "-t", self.session_name,
                     "-b", f"ci_buf_{self.session_name}", "-d"],
                    timeout=5, check=True,
                )
            finally:
                os.unlink(path)
            self._wait_stable()

        self._send_raw("Enter")
        log.debug("Prompt injected (%d chars)", len(prompt))

    def _inject_prompt_buffer(self, prompt):
        """Fallback: inject via tmux load-buffer + paste-buffer."""
        prompt = prompt.rstrip()
        fd, path = tempfile.mkstemp(prefix="ci_fallback_", suffix=".txt")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(prompt)
            subprocess.run(
                ["tmux", "load-buffer", "-b", f"ci_fb_{self.session_name}", path],
                timeout=5, check=True,
            )
            subprocess.run(
                ["tmux", "paste-buffer", "-t", self.session_name,
                 "-b", f"ci_fb_{self.session_name}", "-d"],
                timeout=5, check=True,
            )
        finally:
            os.unlink(path)
        self._wait_stable()
        self._send_raw("Enter")
        log.debug("Prompt injected via load-buffer fallback (%d chars)", len(prompt))

    def _verify_injection(self, prompt):
        """Verify that Enter registered — confirm screen transitions to WORKING.
        
        Polls for up to 10 seconds. If screen remains IDLE (❯ prompt still visible),
        resends Enter once. This self-heals the case where a TUI redraw swallows
        the Enter keystroke.
        """
        start = time.time()
        while time.time() - start < 10:
            time.sleep(self.poll_interval)
            screen = self._capture()
            state = self._classify_screen(screen)
            if state == "WORKING":
                log.debug("Enter confirmed — state is WORKING")
                return True
            if state in ("IDLE", "PERMISSION_PROMPT"):
                log.debug("Verification state=%s, waiting...", state)
            else:
                log.debug("Verification state=%s", state)
        # 10s elapsed and still IDLE — resend Enter once
        log.warning("Enter may not have registered — resending")
        self._send_raw("Enter")
        time.sleep(self.poll_interval * 2)
        screen = self._capture()
        if self._classify_screen(screen) == "WORKING":
            log.debug("Enter re-send confirmed — state is WORKING")
            return True
        snippet = prompt[:30].replace("\n", " ")
        if snippet in screen:
            log.debug("Injection verified — prompt text visible on screen")
            return True
        if "●" in screen and self._classify_screen(screen) == "IDLE":
            already_done = self._extract_response_regex(screen)
            if already_done:
                log.debug("Injection verified — response already appeared")
                return True
        return False

    # ── driver LLM ───────────────────────────────────────────────────

    def _call_driver(self, prompt, max_tokens=256):
        """Call the driver LLM (OpenAI-compatible /chat/completions)."""
        url = f"{self.driver_url}/chat/completions"
        body = json.dumps({
            "model": self.driver_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        return content.strip()

    def _call_driver_with_system(self, system, user, max_tokens=256):
        """Call driver LLM with separate system/user roles (resists injection)."""
        url = f"{self.driver_url}/chat/completions"
        body = json.dumps({
            "model": self.driver_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        return content.strip()

    # ── screen classification ────────────────────────────────────────

    def _classify_screen(self, screen):
        """Classify screen state. Returns one of:
        IDLE, WORKING, PERMISSION_PROMPT, QUESTION, ERROR, STUCK."""
        # --- regex fast path ---
        if "No, exit" in screen and "Yes, I accept" in screen:
            return "PERMISSION_PROMPT"

        lines = [l.rstrip() for l in screen.split("\n")]
        non_empty = [l for l in lines if l.strip()]
        if not non_empty:
            return "WORKING"

        # Find the last "significant" line (skip status bar + separators)
        last_sig = ""
        for line in reversed(non_empty):
            s = line.strip()
            if s.startswith("⏵⏵"):   # ⏵⏵  status bar
                continue
            if s and all(c in "─ " for c in s):  # ─── separator
                continue
            last_sig = s
            break

        has_bullet = any("●" in l for l in lines)     # ●
        has_timing = any("✻" in l for l in lines)     # ✻

        if has_bullet and has_timing and last_sig.startswith("❯"):
            return "IDLE"

        # Fresh idle: ❯ prompt visible but no prior ● response or ✻ timing
        if last_sig.startswith("❯") and not has_bullet and not has_timing:
            return "IDLE"

        # Check for error markers
        for l in lines:
            ls = l.strip().lower()
            if ls.startswith("error:") or ls.startswith("fatal:"):
                return "ERROR"

        return "WORKING"

    def _classify_screen_llm(self, screen):
        """LLM-based screen classification (fallback)."""
        try:
            result = self._call_driver_with_system(
                system="You are a terminal screen classifier. Respond with EXACTLY "
                       "ONE of these words: WORKING, IDLE, PERMISSION_PROMPT, "
                       "QUESTION, ERROR. No other text.",
                user="Classify this Claude Code terminal screen state:\n"
                     "- WORKING: Claude is still processing\n"
                     "- IDLE: Claude finished and is waiting for input\n"
                     "- PERMISSION_PROMPT: Claude is asking for permission\n"
                     "- QUESTION: Claude is asking a clarifying question\n"
                     "- ERROR: An error message is displayed\n\n"
                     f"```\n{screen[-3000:]}\n```\n\nState:",
                max_tokens=10,
            )
            label = result.strip().upper()
            for valid in ("IDLE", "WORKING", "PERMISSION_PROMPT",
                          "QUESTION", "ERROR", "STUCK"):
                if valid in label:
                    return valid
            log.warning("LLM returned unrecognised state: %s", label)
            return "WORKING"
        except Exception as exc:
            log.warning("Driver LLM classification failed: %s", exc)
            return "WORKING"

    # ── response extraction ──────────────────────────────────────────

    def _extract_response_regex(self, screen):
        """Fast-path: extract response from ● -prefixed block."""
        lines = screen.split("\n")

        # Find the last ● line (start of most recent response)
        last_bullet = None
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("●"):
                last_bullet = i
                break
        if last_bullet is None:
            return None

        result_lines = [lines[last_bullet].strip()[1:].strip()]

        # Collect forward from ● until hitting ✻, ❯, separator, or status bar
        for i in range(last_bullet + 1, len(lines)):
            s = lines[i].strip()
            if s.startswith("✻") or s.startswith("❯") or s.startswith("⏵⏵"):
                break
            if len(s) > 3 and all(c in "─ " for c in s):
                break
            if re.match(r"^(Read|Wrote|Ran|Edited|Created)\s", s):
                continue
            result_lines.append(s)

        # Trim trailing empty lines
        while result_lines and not result_lines[-1]:
            result_lines.pop()

        return "\n".join(result_lines) if result_lines else None

    def _extract_response_llm(self, screen):
        """LLM-based extraction (complex / multi-block responses)."""
        try:
            return self._call_driver_with_system(
                system="You are a text extractor. Extract ONLY the assistant's "
                       "response text from Claude Code terminal output. "
                       "No prompts, tool summaries, timing, or separators.",
                user="Extract the assistant's MOST RECENT response from this "
                     "Claude Code terminal output.\n"
                     "Response lines are typically prefixed with ● (bullet).\n"
                     "Return ONLY the response text — no prompts (❯), "
                     "no tool summaries (Read/Wrote/Ran), no timing (✻), "
                     "no status bars (⏵⏵), no separators (─).\n"
                     "Preserve markdown, code blocks, and structure.\n\n"
                     f"```\n{screen[-4000:]}\n```\n\nResponse text:",
                max_tokens=8192,
            )
        except Exception as exc:
            log.error("Driver LLM extraction failed: %s", exc)
            return None

    def _extract_response(self, screen):
        """Extract response via regex fast-path, LLM fallback."""
        resp = self._extract_response_regex(screen)
        if resp:
            return resp
        return self._extract_response_llm(screen)

    # ── interrupt handlers ───────────────────────────────────────────

    def _handle_permission(self):
        log.info("Permission prompt — approving")
        self._send_raw("Enter")

    def _handle_stuck(self, prompt):
        log.warning("Stuck detected — sending Ctrl-C, /clear, re-injecting")
        self._send_raw("C-c")
        time.sleep(2)
        self._send_keys("/clear")
        self._send_raw("Enter")
        time.sleep(3)
        self._inject_prompt(prompt)

    def _handle_question(self):
        if self.permission_mode == "agentic":
            log.info("Question detected — answering 'yes, proceed'")
            self._send_keys("yes, proceed")
            self._send_raw("Enter")
        else:
            log.info("Question detected in restricted mode — declining")
            self._send_keys("no, skip this")
            self._send_raw("Enter")

    # ── constraint enforcement ───────────────────────────────────────

    def _check_constraint_violation(self, screen):
        """In restricted mode, check if Claude used write tools."""
        if self.permission_mode != "restricted":
            return False
        for marker in ("Wrote ", "Edited ", "Created ", "Ran bash"):
            if marker in screen:
                log.error("Constraint violation: '%s' in restricted mode", marker)
                return True
        return False

    # ── main poll loop ───────────────────────────────────────────────

    def run(self, prompt):
        """Run a prompt through the interactive session.

        Returns (response_text, exit_code).
        """
        # Start or reuse session
        if self._session_alive():
            log.info("Reusing existing session %s", self.session_name)
        else:
            self._start_session()

        # Start output capture via tmux pipe-pane
        if self.log_file:
            self._tmux("pipe-pane", "-t", self.session_name,
                       f"cat >> {shlex.quote(self.log_file)}")
            log.debug("pipe-pane started → %s", self.log_file)

        try:
            # Inject with verification
            self._inject_prompt(prompt)
            if not self._verify_injection(prompt):
                log.warning("Prompt injection not verified — retrying")
                time.sleep(0.5)
                self._inject_prompt(prompt)
                if not self._verify_injection(prompt):
                    log.warning("Retry failed — falling back to load-buffer")
                    self._inject_prompt_buffer(prompt)
                    if not self._verify_injection(prompt):
                        log.warning("Load-buffer fallback unverified — proceeding anyway")

            prev_screen = ""
            stable_count = 0
            stuck_retries = 0
            start = time.time()

            while time.time() - start < self.timeout:
                time.sleep(self.poll_interval)

                if not self._session_alive():
                    log.error("Session %s died", self.session_name)
                    return None, EXIT_SESSION_CRASH

                screen = self._capture()

                # Constraint enforcement
                if self._check_constraint_violation(screen):
                    self._send_raw("C-c")
                    return None, EXIT_ERROR

                if screen == prev_screen and screen.strip():
                    stable_count += 1
                else:
                    stable_count = 0
                    prev_screen = screen
                    continue

                prev_screen = screen

                if stable_count < 2:
                    continue

                # Screen is stable — classify
                state = self._classify_screen(screen)
                log.debug("Stable screen, state=%s (%.0fs elapsed)",
                          state, time.time() - start)

                if state == "IDLE":
                    resp = self._extract_response(screen)
                    if resp:
                        return resp, EXIT_SUCCESS
                    # regex + LLM both failed to extract — try LLM classification
                    state = self._classify_screen_llm(screen)
                    if state == "IDLE":
                        resp = self._extract_response_llm(screen)
                        return resp or "", EXIT_SUCCESS
                    # Not actually idle, keep waiting
                    stable_count = 0
                    continue

                if state == "PERMISSION_PROMPT":
                    self._handle_permission()
                    stable_count = 0
                    continue

                if state == "QUESTION":
                    self._handle_question()
                    stable_count = 0
                    continue

                if state == "ERROR":
                    log.error("Error detected on screen")
                    return None, EXIT_ERROR

                if state == "STUCK" or (state == "WORKING" and stable_count >= 5):
                    if stuck_retries < 1:
                        self._handle_stuck(prompt)
                        stuck_retries += 1
                        stable_count = 0
                        continue
                    else:
                        log.error("Stuck and retry exhausted")
                        return None, EXIT_ERROR

                # WORKING but stable — screen may have stabilised mid-tool
                # Reset stability counter so we don't immediately re-classify
                if stable_count >= 4:
                    # Might be genuinely stuck without matching STUCK heuristic
                    state = self._classify_screen_llm(screen)
                    if state == "IDLE":
                        resp = self._extract_response(screen)
                        return resp or "", EXIT_SUCCESS
                    stable_count = 0

            # Timeout
            log.error("Timeout after %ds", self.timeout)
            final = self._capture()
            log.debug("Final screen:\n%s", final[:2000])
            self._send_raw("C-c")
            return None, EXIT_TIMEOUT

        finally:
            # Stop pipe-pane regardless of exit path
            if self.log_file:
                try:
                    self._tmux("pipe-pane", "-t", self.session_name)
                except Exception:
                    pass

    def cleanup(self):
        if self.log_file:
            try:
                self._tmux("pipe-pane", "-t", self.session_name)
            except Exception:
                pass
        if not self.keep_session and self._session_alive():
            log.info("Cleaning up session %s", self.session_name)
            try:
                self._send_keys("/exit")
                self._send_raw("Enter")
                time.sleep(2)
            except Exception:
                pass
            subprocess.run(
                ["tmux", "kill-session", "-t", self.session_name],
                capture_output=True, timeout=5,
            )


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Standalone Claude Code interactive session wrapper. "
                    "Prompt on stdin, response on stdout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit codes: 0=success, 1=error, 2=timeout, "
               "3=session crash, 4=driver error",
    )
    env = os.environ.get

    p.add_argument("--model",
                   default=env("CLAUDE_INTERACTIVE_MODEL", "claude-opus-4-6"))
    p.add_argument("--effort",
                   default=env("CLAUDE_INTERACTIVE_EFFORT", "xhigh"))
    p.add_argument("--driver-llm-url",
                   default=env("DRIVER_LLM_URL",
                               env("CLAUDE_INTERACTIVE_DRIVER_URL",
                                   "http://localhost:8002/v1")))
    p.add_argument("--driver-llm-model",
                   default=env("CLAUDE_INTERACTIVE_DRIVER_MODEL", "local-27b"))
    p.add_argument("--permission-mode", choices=["agentic", "restricted"],
                   default=env("CLAUDE_INTERACTIVE_PERMISSION_MODE", "agentic"))
    p.add_argument("--timeout", type=int,
                   default=int(env("CLAUDE_INTERACTIVE_TIMEOUT", "10800")))
    p.add_argument("--session-name", default=None)
    p.add_argument("--working-dir", default=None)
    p.add_argument("--cc-binary",
                   default=env("CLAUDE_INTERACTIVE_CC_BINARY", "claude"))
    p.add_argument("--poll-interval", type=int, default=3)
    p.add_argument("--keep-session", action="store_true")
    p.add_argument("--log-file", default=None,
                   help="Write full session transcript to PATH (tmux pipe-pane)")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args(argv)


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    # Verify tmux is available
    if subprocess.run(["tmux", "-V"], capture_output=True).returncode != 0:
        log.error("tmux is not installed or not on PATH")
        sys.exit(EXIT_ERROR)

    # Read prompt from stdin
    if sys.stdin.isatty():
        log.error("No prompt on stdin. Usage: echo 'prompt' | %s [opts]",
                  sys.argv[0])
        sys.exit(EXIT_ERROR)

    prompt = sys.stdin.read().strip()
    if not prompt:
        log.error("Empty prompt on stdin")
        sys.exit(EXIT_ERROR)

    session = ClaudeInteractive(
        model=args.model,
        effort=args.effort,
        driver_url=args.driver_llm_url,
        driver_model=args.driver_llm_model,
        permission_mode=args.permission_mode,
        timeout=args.timeout,
        session_name=args.session_name,
        working_dir=args.working_dir,
        cc_binary=args.cc_binary,
        poll_interval=args.poll_interval,
        keep_session=args.keep_session,
        log_file=args.log_file,
    )

    try:
        response, code = session.run(prompt)
        if code == EXIT_SUCCESS and response:
            sys.stdout.write(response)
            if not response.endswith("\n"):
                sys.stdout.write("\n")
        sys.exit(code)
    except KeyboardInterrupt:
        log.info("Interrupted")
        sys.exit(EXIT_ERROR)
    except Exception as exc:
        log.error("Fatal: %s", exc, exc_info=args.verbose)
        sys.exit(EXIT_ERROR)
    finally:
        session.cleanup()


if __name__ == "__main__":
    main()
