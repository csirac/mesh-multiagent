"""
CC Session Manager: persistent Claude Code session per agent.

Manages the lifecycle of a CC subprocess that uses --resume for
session continuity. Each `send_turn()` call invokes `claude -p`
once, streams JSONL events, and returns when CC finishes.

The session ID is extracted from the first invocation's stream-json
output and persisted to disk for cross-restart survival.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pwd
import re
import shutil
import signal
import sys
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .llm import _build_subprocess_env

if TYPE_CHECKING:
    from .config import CCSessionConfig, LLMConfig
    from .memory.system_v2 import MemorySystemV2

logger = logging.getLogger(__name__)

# Sentinel for _account_key: distinguishes "no argument" from explicit None
_UNSET = object()

# Regex to detect rate-limit messages in CC's plain-text output.
# When CC hits subscription limits before the API starts streaming,
# it outputs the error as plain text instead of a structured rate_limit_event.
_RATE_LIMIT_TEXT_RE = re.compile(
    r"You've hit your limit|"
    r"you have hit your limit|"
    r"rate.limit|"
    r"usage.limit|"
    r"resets?\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+",
    re.IGNORECASE,
)


# ============================================================================
# Stream event dataclass
# ============================================================================

@dataclass
class CCStreamEvent:
    """A single event parsed from CC's stream-json output."""
    type: str  # "text", "tool_use", "tool_result", "rate_limit", "error", "session_id", "done"
    content: str = ""
    tool_name: str = ""
    tool_id: str = ""
    session_id: str = ""
    raw: dict = field(default_factory=dict)


# ============================================================================
# Communication cadence (injected into CC-session system prompts)
# ============================================================================

COMMUNICATION_CADENCE_BLOCK = """\
## Communication Cadence

You communicate with users through the mesh network. Follow this cadence:

**On receiving a task:**
- Send a brief acknowledgment confirming you understood the request and outlining \
your approach (1-3 sentences). Do this BEFORE starting work.

**During long-running work (>2 minutes):**
- Send periodic status updates as you hit milestones — files read, changes made, \
tests run, errors encountered. One or two sentences per update is fine.
- If you hit a blocker or change approach, say so immediately.

**On completion:**
- Send a final report summarizing: what you did, what changed, the outcome \
(success/failure/partial), and any follow-up items.
- Include concrete details: file paths, test results, error messages — not just \
"it's done."

**Routing:**
- Use `send_message` to communicate. Your text output is NOT automatically delivered.
- When replying to a message that was sent to a **channel** (e.g., `to=channel:llm-aa`), \
always `send_message` to that **channel**, not to the individual sender. The channel is \
where the conversation is happening and where other participants expect to see your reply.
- Only send directly to a user (e.g., `to=user:yourname`) when replying to a **direct message** \
(i.e., the trigger message's `to` field is your own node ID, not a channel).

**General:**
- Be concise. Status updates should be 1-3 sentences. Final reports can be longer \
but should lead with the result.
- If a task will take many minutes, front-load the acknowledgment so the user \
knows you're working on it.
"""


# ============================================================================
# CC Session Manager
# ============================================================================

class CCSession:
    """Manages a persistent Claude Code session for one agent."""

    def __init__(
        self,
        nickname: str,
        agent_type: str,
        node_id: str,
        config: 'CCSessionConfig',
        llm_config: 'LLMConfig',
        memory_system: 'MemorySystemV2 | None',
        identity_block: str,
        personality_block: str = "",
        mesh_protocol_block: str = "",
        router_host: str = "127.0.0.1",
        router_port: int = 8765,
        auth_token: str = "",
    ):
        self.nickname = nickname
        self.agent_type = agent_type
        self.node_id = node_id
        self.config = config
        self.llm_config = llm_config
        self._memory_system = memory_system
        self._identity_block = identity_block
        self._personality_block = personality_block
        self._mesh_protocol_block = mesh_protocol_block
        self._router_host = router_host
        self._router_port = router_port
        self._auth_token = auth_token

        # Session state
        self._session_ids: dict[str, str] = {}  # account_key -> session_id
        self._proc: asyncio.subprocess.Process | None = None
        self._turn_count: int = 0

        # Round-robin account rotation (ported from llm.py)
        # Use passwd-based home dir for tilde expansion — $HOME may be
        # overridden to a CC account dir (e.g., ~/.claude-acct5) which
        # would cause os.path.expanduser("~/.claude-acct2") to resolve
        # to ~/.claude-acct5/.claude-acct2 instead of
        # ~/.claude-acct2.
        real_user_home = pwd.getpwuid(os.getuid()).pw_dir
        self._all_homes: list[str | None] = [None] + [
            h.replace("~", real_user_home, 1) if h.startswith("~") else h
            for h in (llm_config.cc_fallback_homes or [])
        ]
        self._real_user_home = real_user_home
        self._cc_home_idx: int = 0
        self._cc_depleted: dict[str | None, float] = {}  # home -> cooldown_expiry
        self._current_home: str | None = None

        # Session persistence paths — expand ~ using real home, not $HOME
        raw_sd = config.session_dir
        if raw_sd.startswith("~"):
            raw_sd = raw_sd.replace("~", real_user_home, 1)
        self._session_dir = Path(raw_sd)
        self._sessions_file = self._session_dir / f"{nickname}.sessions.json"

    # ── Account key ──────────────────────────────────────────────

    def _account_key(self, home: str | None | object = _UNSET) -> str:
        """Stable key for the per-account session map.

        When called with no argument, uses self._current_home.
        When called with explicit None (default account), returns "__default__".
        When called with a path string, returns that path.
        """
        if home is _UNSET:
            h = self._current_home
        else:
            h = home
        return h if h else "__default__"

    @property
    def _current_session_id(self) -> str | None:
        return self._session_ids.get(self._account_key())

    @_current_session_id.setter
    def _current_session_id(self, value: str | None) -> None:
        key = self._account_key()
        if value:
            self._session_ids[key] = value
        else:
            self._session_ids.pop(key, None)

    # ── Lifecycle ────────────────────────────────────────────────

    async def start(self) -> None:
        """Load session IDs from disk (if exists). Does NOT spawn CC yet."""
        self._load_session_ids()
        enabled = sum(1 for h in self._all_homes if not self._is_account_disabled(h))
        logger.info(
            f"[{self.nickname}] CCSession started: "
            f"{len(self._session_ids)} persisted session(s), "
            f"{enabled}/{len(self._all_homes)} account(s) enabled"
        )

    async def stop(self) -> None:
        """Kill CC process if running. Persist session ID."""
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    self._proc.kill()
            except ProcessLookupError:
                pass
        self._save_session_ids()
        logger.info(f"[{self.nickname}] CCSession stopped")

    # ── Account rotation ─────────────────────────────────────────

    @staticmethod
    def _is_account_disabled(home: str | None) -> bool:
        """Check if an account has a .disabled marker file."""
        if home is None:
            # Default account: ~/.claude/.disabled
            real_home = pwd.getpwuid(os.getuid()).pw_dir
            marker = os.path.join(real_home, ".claude", ".disabled")
        else:
            marker = os.path.join(home, ".claude", ".disabled")
        return os.path.exists(marker)

    def _pick_account(self) -> str | None:
        """Pick the next CC account via round-robin, skipping depleted and disabled ones.

        Returns the HOME directory override (None = default account).
        Also sets self._current_home for _build_env().
        """
        now = time.time()
        n = len(self._all_homes)

        for _ in range(n):
            home = self._all_homes[self._cc_home_idx % n]
            self._cc_home_idx = (self._cc_home_idx + 1) % n
            if self._is_account_disabled(home):
                continue
            expiry = self._cc_depleted.get(home, 0)
            if expiry <= now:
                self._cc_depleted.pop(home, None)
                self._current_home = home
                return home

        # All enabled accounts depleted — pick the one that resets soonest
        enabled = [h for h in self._all_homes if not self._is_account_disabled(h)]
        if not enabled:
            logger.error(f"[{self.nickname}] All accounts are disabled — no account available")
            self._current_home = self._all_homes[0]  # last resort
            return self._all_homes[0]
        soonest = min(enabled, key=lambda h: self._cc_depleted.get(h, 0))
        self._current_home = soonest
        return soonest

    def mark_account_depleted(self, home: str | None = None, cooldown: float = 60) -> None:
        """Mark the current (or specified) account as depleted."""
        h = home if home is not None else self._current_home
        self._cc_depleted[h] = time.time() + cooldown
        label = h or "default"
        logger.warning(f"[{self.nickname}] Account {label} marked depleted for {cooldown:.0f}s")

    def clear_account_depleted(self, home: str | None = None) -> None:
        """Clear depletion status on success."""
        h = home if home is not None else self._current_home
        self._cc_depleted.pop(h, None)

    def _ensure_session_propagated(self) -> None:
        """Propagate session ID to the current account if it doesn't have one.

        When round-robin picks a new account that has never been used,
        _current_session_id returns None and _build_cc_command() omits
        --resume, causing CC to start a blank conversation. This method
        fixes the chicken-and-egg problem:

        1. If the current account already has a session ID → no-op.
        2. Otherwise, find a donor account that has one.
        3. Copy the session ID to the current account's entry.
        4. Sync the JSONL + companion files so --resume finds them.
        """
        current_key = self._account_key()
        if current_key in self._session_ids:
            return  # Already has a session ID

        # Find a donor — any account with a session ID
        donor_key: str | None = None
        donor_sid: str | None = None
        for key, sid in self._session_ids.items():
            if sid:
                donor_key = key
                donor_sid = sid
                break

        if not donor_key or not donor_sid:
            return  # No existing session anywhere — first-ever turn

        # Copy session ID to current account
        self._session_ids[current_key] = donor_sid
        self._save_session_ids()

        # Sync the JSONL files from donor to current
        donor_home = None if donor_key == "__default__" else donor_key
        if self._sync_session_to_account(donor_sid, donor_home, self._current_home):
            logger.info(
                f"[{self.nickname}] Propagated session {donor_sid[:12]}... "
                f"from {donor_key} → {current_key}"
            )
        else:
            logger.warning(
                f"[{self.nickname}] Failed to propagate session files "
                f"from {donor_key} → {current_key} — CC may start fresh"
            )

    # ── Turn execution ───────────────────────────────────────────

    async def send_turn(
        self,
        message_text: str,
        catchup_xml: str = "",
    ) -> AsyncGenerator[CCStreamEvent, None]:
        """
        Send a user turn to CC and yield stream events.

        If the selected account is rate-limited, automatically retries
        on the next available account (up to len(all_accounts) attempts).
        """
        max_attempts = len(self._all_homes)

        for attempt in range(max_attempts):
            home = self._pick_account()
            label = home or "default"

            # Propagate session ID to new accounts BEFORE building the
            # CC command — without this, --resume is omitted and CC
            # starts a fresh conversation with no history.
            self._ensure_session_propagated()

            logger.info(
                f"[{self.nickname}] send_turn attempt {attempt + 1}/{max_attempts} "
                f"on account {label}, session={self._current_session_id or 'NEW'}"
            )

            # Build system prompt and MCP config
            system_prompt = await self._build_system_prompt(query=message_text)
            mcp_config_json = self._build_mcp_config()

            # Build the full prompt text
            prompt = message_text
            if catchup_xml:
                prompt = catchup_xml + "\n\n" + prompt

            # Build and run CC command
            cmd = self._build_cc_command(system_prompt, mcp_config_json)
            env = self._build_env()

            logger.debug(f"[{self.nickname}] CC command: {' '.join(cmd[:6])}...")

            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Send prompt via stdin
            if self._proc.stdin:
                self._proc.stdin.write(prompt.encode("utf-8"))
                await self._proc.stdin.drain()
                self._proc.stdin.close()

            # Parse stream events
            hit_rate_limit = False
            rate_limit_cooldown = 60.0
            yielded_text_parts: list[str] = []  # track text we yielded

            async for event in self._parse_stream(self._proc):
                if event.type == "session_id" and event.session_id:
                    self._current_session_id = event.session_id
                    self._save_session_ids()

                if event.type == "rate_limit":
                    hit_rate_limit = True
                    # Extract cooldown from resetsAt
                    resets_at = event.raw.get("resetsAt", "")
                    if resets_at:
                        try:
                            from datetime import datetime, timezone
                            reset_time = datetime.fromisoformat(resets_at.replace("Z", "+00:00"))
                            now = datetime.now(timezone.utc)
                            rate_limit_cooldown = max(
                                (reset_time - now).total_seconds() + 5, 10
                            )
                        except Exception:
                            rate_limit_cooldown = 60.0
                    logger.warning(
                        f"[{self.nickname}] Rate limited on account {label}: "
                        f"type={event.raw.get('rateLimitType', '?')}, "
                        f"cooldown={rate_limit_cooldown:.0f}s"
                    )
                    # Don't yield rate_limit to caller — we'll retry
                    continue

                # Detect rate-limit text in CC's plain-text output.
                # When CC hits subscription limits before the API streams,
                # the error appears as a text/result event instead of
                # a structured rate_limit_event.
                if event.type == "text" and event.content:
                    if _RATE_LIMIT_TEXT_RE.search(event.content):
                        hit_rate_limit = True
                        logger.warning(
                            f"[{self.nickname}] Rate-limit TEXT detected on "
                            f"account {label}: {event.content[:120]!r}"
                        )
                        # Don't yield this to the caller
                        continue

                if event.type == "text" and event.content:
                    yielded_text_parts.append(event.content)

                yield event

            # Wait for process to finish
            await self._proc.wait()
            rc = self._proc.returncode

            # Read and log stderr (captures MCP server errors, import failures, etc.)
            stderr_text = ""
            if self._proc.stderr:
                try:
                    stderr_bytes = await self._proc.stderr.read()
                    if stderr_bytes:
                        stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
                        if stderr_text:
                            # Log first 500 chars to avoid flooding
                            logger.info(
                                f"[{self.nickname}] CC stderr (acct={label}, rc={rc}): "
                                f"{stderr_text[:500]}"
                            )
                except Exception as e:
                    logger.debug(f"[{self.nickname}] Failed to read CC stderr: {e}")

            self._proc = None

            # Post-turn rate-limit detection: check if CC exited with a
            # non-zero code and the only text output looks like a rate-limit
            # message. This catches the case where the rate-limit text was
            # yielded before we could intercept it (e.g., it arrived as the
            # sole result event and was the only output).
            if not hit_rate_limit and rc != 0:
                all_text = " ".join(yielded_text_parts).strip()
                if all_text and _RATE_LIMIT_TEXT_RE.search(all_text):
                    hit_rate_limit = True
                    logger.warning(
                        f"[{self.nickname}] Post-turn rate-limit detected on "
                        f"account {label} (rc={rc}): {all_text[:120]!r}"
                    )
                # Also check stderr for rate-limit indicators
                elif stderr_text and _RATE_LIMIT_TEXT_RE.search(stderr_text):
                    hit_rate_limit = True
                    logger.warning(
                        f"[{self.nickname}] Rate-limit detected in stderr on "
                        f"account {label} (rc={rc}): {stderr_text[:120]!r}"
                    )

            if hit_rate_limit:
                # Kill process if still alive, mark depleted, try next account
                self.mark_account_depleted(home, rate_limit_cooldown)
                logger.info(f"[{self.nickname}] Retrying on next account after rate limit")
                continue

            # Success
            self.clear_account_depleted(home)
            self._turn_count += 1

            # Sync session to other accounts for seamless --resume
            self._sync_session_to_all_accounts()

            if rc != 0:
                logger.warning(f"[{self.nickname}] CC exited with code {rc}")

            return  # Done

        # Exhausted all accounts
        logger.error(f"[{self.nickname}] All {max_attempts} accounts exhausted")
        yield CCStreamEvent(type="error", content="All accounts rate-limited")

    # ── System prompt ────────────────────────────────────────────

    async def _build_system_prompt(self, query: str = "") -> str:
        """
        Build the system prompt for this invocation.

        Includes (per config.system_prompt_includes):
        - identity: agent name, type, node_id
        - personality: personality directives
        - memories: rendered memory blocks
        - rolling_summary: compressed conversation summary (older context)
        - retrieved_context: query-relevant memory entries
        - project_map: active project map (authoritative context on restart)
        - mesh_protocol: how to use MCP tools, message routing
        - communication: cadence guidelines for proactive messaging
        """
        includes = self.config.system_prompt_includes
        sections: list[str] = []

        if "identity" in includes and self._identity_block:
            sections.append(f"## Identity\n{self._identity_block}")

        if "personality" in includes and self._personality_block:
            sections.append(f"## Personality\n{self._personality_block}")

        if "memories" in includes and self._memory_system:
            memory_block = await self._render_memories(query)
            if memory_block:
                sections.append(f"## Memories\n{memory_block}")

        if "rolling_summary" in includes and self._memory_system:
            try:
                from .memory.system_v2 import MemorySystemV2
                if isinstance(self._memory_system, MemorySystemV2):
                    summary_block = await self._memory_system.render_summary_block()
                    if summary_block:
                        sections.append(
                            f"## Conversation Summary\n{summary_block}"
                        )
            except Exception as e:
                logger.warning(f"[{self.nickname}] Rolling summary render failed: {e}")

        if "retrieved_context" in includes and self._memory_system and query:
            try:
                from .memory.system_v2 import MemorySystemV2
                if isinstance(self._memory_system, MemorySystemV2):
                    retrieved_block = await self._memory_system.render_retrieved_context(
                        query=query, budget_tokens=2000,
                    )
                    if retrieved_block:
                        sections.append(
                            f"## Retrieved Context\n"
                            f"<retrieved_context>\n{retrieved_block}\n</retrieved_context>"
                        )
            except Exception as e:
                logger.warning(f"[{self.nickname}] Retrieved context render failed: {e}")

        if "project_map" in includes and self._memory_system:
            try:
                from .memory.system_v2 import MemorySystemV2
                if isinstance(self._memory_system, MemorySystemV2):
                    map_block = await self._memory_system.render_maps_block()
                    if map_block:
                        sections.append(f"## Project Context\n{map_block}")
            except Exception as e:
                logger.warning(f"[{self.nickname}] Project map render failed: {e}")

        if "mesh_protocol" in includes and self._mesh_protocol_block:
            sections.append(f"## Mesh Protocol\n{self._mesh_protocol_block}")

        if "communication" in includes:
            sections.append(COMMUNICATION_CADENCE_BLOCK)

        return "\n\n".join(sections)

    async def _render_memories(self, query: str = "") -> str:
        """Render memory blocks for the system prompt."""
        if not self._memory_system:
            return ""
        try:
            return await self._memory_system.render_block(query=query or None)
        except Exception as e:
            logger.warning(f"[{self.nickname}] Memory render failed: {e}")
            return ""

    # ── CC command building ──────────────────────────────────────

    def _build_mcp_config(self) -> str:
        """Build the --mcp-config JSON for CC to launch the MCP server."""
        config = {
            "mcpServers": {
                "mesh": {
                    "command": sys.executable,
                    "args": [
                        "-m", "mesh.mcp_server",
                        "--router", f"ws://{self._router_host}:{self._router_port}/ws",
                        "--token", self._auth_token,
                        "--node-id", self.node_id,
                    ],
                    "env": {
                        "PYTHONPATH": os.getcwd(),
                    },
                },
            },
        }
        return json.dumps(config)

    def _build_cc_command(
        self,
        system_prompt: str,
        mcp_config_json: str,
    ) -> list[str]:
        """Build the claude CLI command for this invocation."""
        claude_bin = shutil.which("claude") or "claude"
        cmd = [claude_bin, "-p"]

        session_id = self._current_session_id
        if session_id:
            cmd.extend(["--resume", session_id])

        model = self.config.cc_model or self.llm_config.model
        cmd.extend(["--model", model])

        cmd.extend(["--system-prompt", system_prompt])
        cmd.extend(["--output-format", self.config.cc_output_format])
        cmd.append("--verbose")
        cmd.extend(["--mcp-config", mcp_config_json])
        cmd.extend(["--permission-mode", "bypassPermissions"])

        if self.config.cc_max_turns:
            cmd.extend(["--max-turns", str(self.config.cc_max_turns)])

        return cmd

    def _build_env(self) -> dict[str, str]:
        """Build environment for the CC subprocess."""
        env = _build_subprocess_env()
        env.pop("CLAUDE_CODE_ENTRYPOINT", None)
        env["DISABLE_TELEMETRY"] = "1"
        env["CLAUDE_CODE_ENABLE_TELEMETRY"] = "0"
        env["CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY"] = "1"
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

        # ALWAYS set HOME explicitly for the CC subprocess.
        # When running inside another CC session, $HOME may already be
        # overridden to a CC account dir (e.g., ~/.claude-acct3).  If we
        # don't set HOME for the "default" account case, CC inherits the
        # parent's $HOME and stores its session JSONL in the wrong place,
        # breaking --resume on account rotation.
        target_home = self._current_home or self._real_user_home
        env["HOME"] = target_home

        # Ensure ~/.local/bin is on PATH (claude binary lives there)
        real_local_bin = os.path.join(self._real_user_home, ".local", "bin")
        path = env.get("PATH", "")
        if real_local_bin not in path.split(os.pathsep):
            env["PATH"] = real_local_bin + os.pathsep + path

        # Apply CC env overrides from llm_config (e.g., ANTHROPIC_BASE_URL)
        if self.llm_config.cc_env:
            env.update(self.llm_config.cc_env)

        return env

    # ── Session persistence ──────────────────────────────────────

    def _load_session_ids(self) -> None:
        """Load per-account session IDs from disk.

        Handles migration from legacy single-ID file (nickname.id).
        """
        # Try new format first
        if self._sessions_file.exists():
            try:
                data = json.loads(self._sessions_file.read_text())
                if isinstance(data, dict):
                    self._session_ids = data
                    logger.debug(
                        f"[{self.nickname}] Loaded {len(data)} session ID(s) "
                        f"from {self._sessions_file}"
                    )
                    return
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"[{self.nickname}] Failed to load sessions file: {e}")

        # Try legacy single-ID format
        legacy_file = self._session_dir / f"{self.nickname}.id"
        if legacy_file.exists():
            try:
                sid = legacy_file.read_text().strip()
                if sid:
                    self._session_ids["__default__"] = sid
                    # Migrate to new format
                    self._save_session_ids()
                    legacy_file.unlink()
                    logger.info(
                        f"[{self.nickname}] Migrated legacy session ID to new format"
                    )
            except OSError as e:
                logger.warning(f"[{self.nickname}] Failed to load legacy session file: {e}")

    def _save_session_ids(self) -> None:
        """Write per-account session IDs to disk as JSON."""
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._sessions_file.write_text(json.dumps(self._session_ids, indent=2))

    # ── CC project directory helpers ─────────────────────────────

    def _cc_projects_dir(self, home: str | None) -> Path:
        """Return the CC projects directory for a given HOME."""
        base = Path(home) if home else Path(self._real_user_home)
        return base / ".claude" / "projects"

    def _project_slug(self) -> str:
        """Derive the CC project slug from the current working directory.

        CC stores sessions under ~/.claude/projects/{slug}/ where slug
        is the absolute CWD with / replaced by -.
        """
        cwd = os.getcwd()
        # CC uses the CWD with / replaced by - (keeping the leading dash)
        return cwd.replace("/", "-")

    def _sync_session_to_account(
        self,
        session_id: str,
        source_home: str | None,
        target_home: str | None,
    ) -> bool:
        """Copy a CC session's local files from one account to another.

        CC sessions are entirely client-side: a JSONL transcript at
        {projects_dir}/{slug}/{session_id}.jsonl and an optional
        companion directory {session_id}/ (tool results, subagents).

        Copying these files to the target account's project dir lets
        --resume find the session under the new HOME, preserving full
        conversation continuity across account rotation.

        Returns True if sync succeeded.
        """
        slug = self._project_slug()
        src_dir = self._cc_projects_dir(source_home) / slug
        dst_dir = self._cc_projects_dir(target_home) / slug
        jsonl_name = f"{session_id}.jsonl"
        src_jsonl = src_dir / jsonl_name
        dst_jsonl = dst_dir / jsonl_name

        if not src_jsonl.exists():
            logger.warning(
                f"[{self.nickname}] Cannot sync session {session_id[:12]}...: "
                f"source JSONL not found at {src_jsonl}"
            )
            return False

        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src_jsonl), str(dst_jsonl))

            # Copy companion directory if it exists
            src_companion = src_dir / session_id
            dst_companion = dst_dir / session_id
            if src_companion.is_dir():
                if dst_companion.exists():
                    shutil.rmtree(str(dst_companion))
                shutil.copytree(str(src_companion), str(dst_companion))

            source_label = source_home or "default"
            target_label = target_home or "default"
            size_kb = src_jsonl.stat().st_size // 1024
            logger.info(
                f"[{self.nickname}] Synced session {session_id[:12]}... "
                f"from {source_label} → {target_label} ({size_kb}KB)"
            )
            return True
        except Exception as e:
            logger.error(f"[{self.nickname}] Session sync failed: {e}")
            return False

    def _sync_session_to_all_accounts(self) -> None:
        """After a successful turn, sync the session from the current account
        to ALL other accounts.

        This keeps session state consistent so any account can --resume
        with the latest conversation history. We sync to every account
        (not just those with matching session IDs) because
        _ensure_session_propagated may not have run for all accounts yet,
        and we want proactive propagation.
        """
        sid = self._current_session_id
        if not sid:
            return

        current_key = self._account_key()
        synced = 0

        # Build set of all account keys from _all_homes
        all_keys = {self._account_key(h) for h in self._all_homes}

        for key in all_keys:
            if key == current_key:
                continue
            target_home = None if key == "__default__" else key

            # Ensure this account has the session ID mapped
            if self._session_ids.get(key) != sid:
                self._session_ids[key] = sid

            if self._sync_session_to_account(sid, self._current_home, target_home):
                synced += 1

        if synced:
            self._save_session_ids()
            logger.info(f"[{self.nickname}] Post-turn sync: updated {synced} account(s)")

    # ── Stream parsing ───────────────────────────────────────────

    async def _parse_stream(
        self,
        proc: asyncio.subprocess.Process,
    ) -> AsyncGenerator[CCStreamEvent, None]:
        """Parse CC's stream-json output, yielding events in real time."""
        if not proc.stdout:
            return

        call_id_to_name: dict[str, str] = {}
        buffer = b""

        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            buffer += chunk

            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                try:
                    events = self._parse_line(line, call_id_to_name)
                    for event in events:
                        yield event
                except json.JSONDecodeError:
                    logger.debug(f"[{self.nickname}] Non-JSON line: {line[:200]}")
                except Exception as e:
                    logger.warning(f"[{self.nickname}] Stream parse error: {e}")

    def _parse_line(
        self,
        line: str,
        call_id_to_name: dict[str, str],
    ) -> list[CCStreamEvent]:
        """Parse a single JSONL line into CCStreamEvents.

        Returns a list because a single assistant message can contain
        multiple content blocks (text + tool_use, or parallel tool_use).
        """
        data = json.loads(line)
        events: list[CCStreamEvent] = []

        msg_type = data.get("type", "")

        # Session ID extraction (from first message)
        if "session_id" in data:
            events.append(CCStreamEvent(
                type="session_id",
                session_id=data["session_id"],
                raw=data,
            ))

        # Rate limit event
        if msg_type == "rate_limit_event":
            status = data.get("status", "")
            if status == "rejected":
                events.append(CCStreamEvent(
                    type="rate_limit",
                    content=data.get("result", data.get("message", "Rate limited")),
                    raw=data,
                ))
            return events

        # Error messages
        if msg_type == "error":
            error_content = data.get("error", {})
            if isinstance(error_content, dict):
                msg_text = error_content.get("message", str(error_content))
            else:
                msg_text = str(error_content)
            events.append(CCStreamEvent(type="error", content=msg_text, raw=data))
            return events

        # Assistant message with content blocks
        if msg_type == "assistant" and "message" in data:
            message = data["message"]
            content_blocks = message.get("content", [])
            for block in content_blocks:
                block_type = block.get("type", "")
                if block_type == "text":
                    events.append(CCStreamEvent(
                        type="text",
                        content=block.get("text", ""),
                        raw=data,
                    ))
                elif block_type == "tool_use":
                    tool_id = block.get("id", "")
                    tool_name = block.get("name", "")
                    call_id_to_name[tool_id] = tool_name
                    events.append(CCStreamEvent(
                        type="tool_use",
                        content=json.dumps(block.get("input", {})),
                        tool_name=tool_name,
                        tool_id=tool_id,
                        raw=data,
                    ))

        # Tool result
        if msg_type == "tool_result" or (msg_type == "result" and "tool_use_id" in data):
            tool_id = data.get("tool_use_id", "")
            tool_name = call_id_to_name.get(tool_id, data.get("tool_name", "unknown"))
            content = data.get("content", "")
            if isinstance(content, list):
                # Extract text from content blocks
                text_parts = [
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            events.append(CCStreamEvent(
                type="tool_result",
                content=str(content),
                tool_name=tool_name,
                tool_id=tool_id,
                raw=data,
            ))

        # Result message (final output)
        if msg_type == "result":
            result_text = data.get("result", "")
            # Only emit result text if we didn't already emit text from
            # an assistant content block — CC sends the same text in both
            # the assistant message and the result message.
            has_text_already = any(e.type == "text" for e in events)
            if isinstance(result_text, str) and result_text and not has_text_already:
                events.append(CCStreamEvent(
                    type="text",
                    content=result_text,
                    raw=data,
                ))
            # Extract session_id from result if present
            if "session_id" in data and data["session_id"]:
                events.append(CCStreamEvent(
                    type="session_id",
                    session_id=data["session_id"],
                    raw=data,
                ))

        return events

    # ── Properties ───────────────────────────────────────────────

    @property
    def session_id(self) -> str | None:
        """Current session ID for the active account."""
        return self._current_session_id

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None
