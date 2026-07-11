#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Mesh User TUI - Terminal interface for mesh communication.

Features:
- Unified timeline showing messages from all agents
- Target selection: /target <node> sets default, /to <node> for one-off
- Nickname support: address agents by nickname (e.g., @alice instead of @agent:coder:alice)
- Presence notifications: shows when agents join/leave
- Custom markdown + math rendering (via MarkdownRenderer + WezMathRenderer)
- Input history and multiline editing
- Session persistence: history is loaded and persisted by default

Usage:
    python run_user_tui.py [--nickname <nick>] [--config PATH]

Examples:
    python run_user_tui.py --nickname yourname
    python run_user_tui.py -n sarah
    python run_user_tui.py -n yourname --fresh  # Start fresh (no history)
    python run_user_tui.py user:yourname  # Legacy style

By default, sessions resume from previous history. Use --fresh or --no-resume
to start without history.
"""

import argparse
import asyncio
import getpass
import io
import logging
import shutil
import signal
import shlex
import subprocess
import sys
import os
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout

from mesh.config import load_config, NodeConfig
from mesh.node import Node
from mesh.user_node import UserNode, RosterEntry
from mesh.protocol import (
    Message, MessageType, make_confirm_response, get_display_name, build_user_node_id, make_status_request,
    ControlAction, is_channel_address, parse_channel_name, build_channel_address,
    make_channel_create, make_channel_delete, make_channel_join, make_channel_leave,
    make_channel_list, make_channel_members, make_channel_invite, make_history_sync,
    make_scratchpad_get, make_scratchpad_set, make_todo_get, make_todo_mutate,
    parse_node_id,
)
from mesh.wez_math_renderer import WezMathRenderer
from mesh.markdown_renderer import MarkdownRenderer
import json


# =============================================================================
# Tool Activity Formatting (ported from chat-app for real-time tool display)
# =============================================================================

# ANSI codes for tool activity display
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


def format_tool_activity_call(name: str, args: dict, max_width: int = 120) -> str:
    """Format a tool call in Claude Code style: ● tool_name(param: "value", ...)

    Args:
        name: Tool name
        args: Arguments dictionary
        max_width: Maximum width before truncating

    Returns:
        Formatted string with ANSI colors
    """
    if not args:
        return f"{_BOLD}{_CYAN}●{_RESET} {_BOLD}{name}{_RESET}()"

    # Format each parameter
    param_parts = []
    for key, value in args.items():
        if value is None or value == "":
            continue
        if isinstance(value, str):
            # Truncate long strings
            if len(value) > 80:
                value = value[:77] + "..."
            # Escape newlines for single-line display
            value = value.replace("\n", "\\n")
            param_parts.append(f'{key}: "{value}"')
        elif isinstance(value, bool):
            param_parts.append(f'{key}: {str(value).lower()}')
        elif isinstance(value, (int, float)):
            param_parts.append(f'{key}: {value}')
        else:
            # For complex types, use compact JSON
            try:
                val_str = json.dumps(value)
                if len(val_str) > 60:
                    val_str = val_str[:57] + "..."
                param_parts.append(f'{key}: {val_str}')
            except (TypeError, ValueError):
                param_parts.append(f'{key}: ...')

    params_str = ", ".join(param_parts)

    # Truncate if too long
    full_str = f"{name}({params_str})"
    if len(full_str) > max_width:
        full_str = full_str[:max_width - 3] + "..."

    return f"{_BOLD}{_CYAN}●{_RESET} {_BOLD}{full_str}{_RESET}"


def format_tool_activity_result(
    result: str,
    name: str = "",
    success: bool = True,
    max_lines: int = 8,
) -> str:
    """Format a tool result with ⎿ prefix.

    Args:
        result: Tool output content
        name: Tool name (for context)
        success: Whether the tool succeeded
        max_lines: Maximum lines to show before truncating

    Returns:
        Formatted string with ANSI colors
    """
    if not result:
        return f"  {_DIM}⎿  (No output){_RESET}"

    lines = result.strip().splitlines()
    total_lines = len(lines)

    if total_lines == 0:
        return f"  {_DIM}⎿  (No output){_RESET}"

    # Truncate if needed
    truncated = False
    if total_lines > max_lines:
        lines = lines[:max_lines]
        truncated = True

    # Format with ⎿ prefix
    result_lines = []
    color = _RESET if success else _RED
    for i, line in enumerate(lines):
        # Truncate very long lines
        if len(line) > 150:
            line = line[:147] + "..."
        if i == 0:
            result_lines.append(f"  {_DIM}⎿{_RESET}  {color}{line}{_RESET}")
        else:
            result_lines.append(f"     {color}{line}{_RESET}")

    if truncated:
        remaining = total_lines - max_lines
        result_lines.append(f"     {_DIM}… +{remaining} lines{_RESET}")

    return "\n".join(result_lines)


def iso_to_local_time(timestamp: str) -> str:
    """Convert ISO timestamp to local time HH:MM:SS."""
    try:
        # Handle Z suffix (UTC)
        if timestamp.endswith("Z"):
            timestamp = timestamp[:-1] + "+00:00"
        dt = datetime.fromisoformat(timestamp)
        # Convert to local timezone
        local_dt = dt.astimezone()
        return local_dt.strftime("%H:%M:%S")
    except (ValueError, AttributeError):
        # Fallback: extract time portion
        return timestamp.split("T")[1][:8] if "T" in timestamp else timestamp[:8]


def setup_logging(node_id: str, log_dir: str = "logs"):
    """Configure logging to both file and console."""
    safe_name = node_id.replace(":", "-")
    log_path = Path(log_dir) / f"{safe_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # File handler - detailed logs
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)

    # Console handler - less verbose (only warnings/errors)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_format = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console_handler.setFormatter(console_format)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return str(log_path)


logger = logging.getLogger(__name__)


class CommandAutoSuggest(AutoSuggest):
    """Suggest agent/channel names after commands and bare #channel mentions."""

    def __init__(self, tui: "MeshTUI"):
        self.tui = tui

    def get_suggestion(self, buffer, document: Document) -> Suggestion | None:
        text = document.text_before_cursor
        # Commands that take a node/channel argument — accept either agent
        # nicknames or #channel. Auto-suggests both.
        for prefix in ("/v ", "/view ", "/t ", "/target ", "/to "):
            if text.lower().startswith(prefix):
                partial = text[len(prefix):]
                return self._suggest_target(partial)
        # Commands whose argument is exclusively a channel name. Suggest
        # #channel from _known_channels. Strip leading # from partial if
        # the user already typed it so the matching logic stays uniform.
        for prefix in ("/join ", "/leave ", "/members ", "/delete "):
            if text.lower().startswith(prefix):
                partial = text[len(prefix):]
                return self._suggest_channel(partial)
        # /invite <channel> <node> — suggest channel for the first token only.
        if text.lower().startswith("/invite ") or text.lower().startswith("/add "):
            body = text[text.find(" ", 1) + 1:]
            if " " not in body:
                return self._suggest_channel(body)
            return None
        # Bare #token anywhere in the input — channel mention in a message
        # body or command argument. Suggests from known channels.
        if "#" in text:
            idx = text.rfind("#")
            after = text[idx + 1:]
            # Token ends at the cursor; bail if a space split the mention.
            if after and " " not in after:
                return self._suggest_channel(after)
        return None

    def _suggest_target(self, partial: str) -> Suggestion | None:
        if not self.tui.node:
            return None
        partial_lower = partial.lower()
        candidates = []
        # Collect nicknames from roster
        for entry in self.tui.node.get_roster_list():
            candidates.append(entry.nickname)
        # Collect channel names from live cache (preferred) and fall back
        # to recent conversations so completion works before /channels runs.
        chan_candidates = {f"#{c}" for c in self.tui._known_channels} | set(self.tui._known_channels)
        for partner in self.tui._recent_conversations:
            if partner.startswith("channel:"):
                chan_candidates.add(f"#{partner.split(':', 1)[1]}")
        candidates.extend(sorted(chan_candidates))
        # Also add "all" as an option
        candidates.append("all")
        # Find first match
        for name in candidates:
            if name.lower().startswith(partial_lower) and name.lower() != partial_lower:
                return Suggestion(name[len(partial):])
        return None

    def _suggest_channel(self, partial: str) -> Suggestion | None:
        """Suggest a channel name from _known_channels.

        `partial` is the channel-name fragment the user has typed. Leading
        # is stripped if present. Suggestion only appends, so we return the
        suffix needed to complete `partial` to a full channel name.
        """
        if not self.tui._known_channels:
            return None
        # Strip leading # if present so matching is uniform.
        partial_clean = partial[1:] if partial.startswith("#") else partial
        partial_lower = partial_clean.lower()
        for name in sorted(self.tui._known_channels):
            if name.lower().startswith(partial_lower) and name.lower() != partial_lower:
                return Suggestion(name[len(partial_clean):])
        return None


class MeshTUI:
    """Terminal TUI for mesh communication with custom markdown rendering."""

    # ANSI color codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    WHITE = "\033[37m"
    BLUE = "\033[34m"

    def __init__(
        self,
        node_id: str,
        config: NodeConfig,
        nickname: str,
        fresh: bool = False,
        history_file: str | None = None,
        notifications: bool = True,
        quiet_presence: bool = False,
    ):
        self.node_id = node_id
        self.config = config
        self.nickname = nickname
        self.fresh = fresh
        self.history_file = history_file
        self.node: Optional[TUIUserNode] = None

        # Quiet mode - suppress presence notifications
        self._quiet_presence = quiet_presence

        # Desktop notifications
        self._notifications_enabled = notifications and shutil.which("notify-send") is not None

        # Custom markdown + math renderer
        self.wmr = WezMathRenderer(force_text_fallback=False)
        self.mdr = MarkdownRenderer(self.wmr)

        # Message display
        self.message_history: list[tuple[str, str, str]] = []  # (from, to, content)

        # Conversation view filter (None = show all, otherwise filter to this node/channel)
        self.current_view: Optional[str] = None

        # Target selection
        self.default_target: Optional[str] = None

        # Input setup
        self.username = nickname
        hist_path = os.path.expanduser(f"~/.mesh_tui_history_{self.username}")
        self.input_history = FileHistory(hist_path)

        self.prompt_style = Style.from_dict({
            'prompt': 'bold fg:cyan',
        })

        # Key bindings
        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            event.current_buffer.insert_text("\n")

        @kb.add("c-s")
        def _(event):
            event.current_buffer.validate_and_handle()

        @kb.add("c-z")
        def _(event):
            """Toggle draft: save when buffer has text, restore when empty."""
            buff = event.current_buffer
            text = buff.text.strip()
            tui = self  # capture reference
            partner = tui.current_view or tui.default_target
            if not partner:
                print(f"\n{tui.DIM}No active conversation for draft{tui.RESET}")
                return
            if text:
                # Save draft and clear buffer
                tui._drafts[partner] = text
                buff.reset()
                print(f"\n{tui.DIM}Draft saved for {get_display_name(partner)}{tui.RESET}")
                event.app.invalidate()
            else:
                # Restore draft if one exists (clears stored draft)
                draft = tui._drafts.pop(partner, None)
                if draft:
                    buff.set_document(
                        Document(draft, cursor_position=len(draft))
                    )
                    print(f"\n{tui.DIM}Draft restored for {get_display_name(partner)}{tui.RESET}")
                else:
                    print(f"\n{tui.DIM}No draft saved for {get_display_name(partner)}{tui.RESET}")
                event.app.invalidate()

        @kb.add("c-t")
        def _(event):
            """Toggle the per-conversation todo panel."""
            self._toggle_todo_panel_from_keybinding()
            event.app.invalidate()

        @kb.add("c-g")
        def _(event):
            """Show today's calendar once."""
            asyncio.create_task(self._show_calendar())
            event.app.invalidate()

        @kb.add("tab")
        def _(event):
            buff = event.current_buffer
            suggestion = buff.suggestion
            if suggestion:
                buff.insert_text(suggestion.text)
            else:
                buff.insert_text("    ")

        self.session = PromptSession(
            history=self.input_history,
            style=self.prompt_style,
            multiline=True,
            key_bindings=kb,
            auto_suggest=CommandAutoSuggest(self),
            bottom_toolbar=self._todo_bottom_toolbar,
        )

        # Async message queue for incoming messages
        self._incoming_queue: asyncio.Queue[Message] = asyncio.Queue()

        # Unread counts per conversation (node_id -> count)
        self._unread_counts: dict[str, int] = {}

        # Recent conversations (for numbered switching)
        self._recent_conversations: list[str] = []

        # Pending confirmation requests: msg_id -> asyncio.Future[bool]
        self._pending_confirms: dict[str, asyncio.Future[bool]] = {}

        # Per-conversation drafts: partner_node_id -> draft_text
        self._drafts: dict[str, str] = {}

        # Scratchpad sync: per-conversation notes synced via router
        self._scratchpad_dir = os.path.join(os.path.expanduser("~"), ".mesh", "scratchpads")
        os.makedirs(self._scratchpad_dir, exist_ok=True)
        self._scratchpad_base_versions: dict[str, str] = {}  # conv_id -> updated_at

        # Per-conversation todo panel/cache synced via router
        self._todo_dir = os.path.join(os.path.expanduser("~"), ".mesh", "todos")
        os.makedirs(self._todo_dir, exist_ok=True)
        self._todo_cache: dict[str, list[dict]] = {}
        self._todo_panel_visible: dict[str, bool] = {}
        self._todo_ref_map: dict[str, dict[int, str]] = {}
        self._todo_section_order: dict[str, list[str]] = {}

        # Queue for confirmation requests that need user input from main loop
        self._confirm_queue: asyncio.Queue[Message] = asyncio.Queue()

        # Flag to indicate we're in confirmation mode (suppress inline prompts)
        self._in_confirmation: bool = False

        # Buffer for messages received during confirmation
        self._confirm_message_buffer: list[Message] = []

        # Attached agent for real-time tool activity streaming
        # When set, TOOL_ACTIVITY messages from this agent are rendered inline
        self._attached_agent: str | None = None

        # Watch mode: /watch <nickname> polls status every 5s
        self._watch_target: str | None = None
        self._watch_queue: asyncio.Queue[Message] = asyncio.Queue()

        # Reply threading: recent messages per channel/conversation and ref number mapping
        self._recent_messages: dict[str, deque] = {}  # channel_addr -> deque of (msg_id, from_node, preview)
        self._msg_ref_map: dict[int, str] = {}  # ref_number -> msg_id
        self._msg_ref_counter: int = 0  # cycles 1-99

        # Known channel names (without # prefix), cached from router responses
        # and channel events. Used by CommandAutoSuggest for #channel completion.
        self._known_channels: set[str] = set()

    def _next_msg_ref(self) -> int:
        self._msg_ref_counter = (self._msg_ref_counter % 99) + 1
        return self._msg_ref_counter

    def _track_message(self, msg_id: str, from_node: str, content: str, channel_addr: str | None):
        """Track a message in the recent buffer and assign a ref number."""
        ref = self._next_msg_ref()
        self._msg_ref_map[ref] = msg_id
        key = channel_addr or from_node
        if key not in self._recent_messages:
            self._recent_messages[key] = deque(maxlen=50)
        self._recent_messages[key].append((msg_id, from_node, content[:80]))
        return ref

    def _lookup_reply(self, in_reply_to: str) -> str | None:
        """Look up a reply target, returning 'sender #N' or truncated ID."""
        for ref_num, mid in self._msg_ref_map.items():
            if mid == in_reply_to:
                for dq in self._recent_messages.values():
                    for (mid2, from_node, _preview) in dq:
                        if mid2 == in_reply_to:
                            return f"{get_display_name(from_node)} #{ref_num}"
                return f"#{ref_num}"
        return in_reply_to[:12] + "…"

    def print(self, text: str = ""):
        """Print text with color tag support like [cyan], [bold], [dim], etc."""
        print(self.mdr.colorize(text))

    def show_header(self):
        """Display welcome header."""
        self.print()
        self.print(f"{self.CYAN}╔════════════════════════════════════════╗{self.RESET}")
        self.print(f"{self.CYAN}║{self.RESET}  {self.BOLD}{self.CYAN}Mesh TUI{self.RESET} - {self.BOLD}{self.nickname}{self.RESET}                       {self.CYAN}║{self.RESET}")
        self.print(f"{self.CYAN}╚════════════════════════════════════════╝{self.RESET}")
        self.print()
        self.print(f"{self.DIM}Commands:{self.RESET}")
        self.print(f"  {self.CYAN}/list{self.RESET}             - List connected nodes")
        self.print(f"  {self.CYAN}/target <nick>{self.RESET}    - Set default target (by nickname or channel:name)")
        self.print(f"  {self.CYAN}/to <nick> msg{self.RESET}    - Send to specific node (one-off)")
        self.print(f"  {self.CYAN}/status [nick] [n]{self.RESET} - Get agent's recent context")
        self.print(f"  {self.CYAN}/context [n]{self.RESET}      - Show your own recent context")
        self.print(f"  {self.CYAN}/attach <nick>{self.RESET}    - See agent's tool activity in real-time")
        self.print(f"  {self.CYAN}/detach{self.RESET}           - Stop watching tool activity")
        self.print(f"{self.DIM}Channels:{self.RESET}")
        self.print(f"  {self.CYAN}/channels{self.RESET}         - List all channels")
        self.print(f"  {self.CYAN}/create <name>{self.RESET}    - Create a channel")
        self.print(f"  {self.CYAN}/join <name>{self.RESET}      - Join a channel")
        self.print(f"  {self.CYAN}/leave <name>{self.RESET}     - Leave a channel")
        self.print(f"  {self.CYAN}/members <name>{self.RESET}   - List channel members")
        self.print(f"  {self.CYAN}/invite <ch> <node>{self.RESET} - Add a member to channel")
        self.print(f"  {self.CYAN}/delete <name>{self.RESET}    - Delete a channel")
        self.print(f"  {self.CYAN}/quit{self.RESET}             - Disconnect")
        self.print()
        self.print(f"{self.DIM}Tip: Use nicknames (e.g., alice) or channel:name for targets{self.RESET}")
        self.print(f"{self.DIM}Enter to add newline, Ctrl+S to send{self.RESET}")
        self.print()

    def _print_prompt_hint(self):
        """Print a visual hint showing the current prompt context after messages."""
        parts = []
        # Show current view filter
        if self.current_view:
            view_name = get_display_name(self.current_view)
            if self.current_view.startswith("channel:"):
                parts.append(f"#{self.current_view.split(':')[1]}")
            else:
                parts.append(f"@{view_name}")
        # Show target
        if self.default_target:
            parts.append(f"→{get_display_name(self.default_target)}")
        else:
            parts.append("no target")
        draft_ind = "*" if self.default_target and self.default_target in self._drafts else ""
        prompt_hint = f"[{' '.join(parts)}] {self.username}:{draft_ind} "
        sys.stdout.write(f"{self.CYAN}{prompt_hint}{self.RESET}")
        sys.stdout.flush()

    def render_message(self, from_node: str, content: str, timestamp: Optional[str] = None, show_prompt: bool = True, channel: str | None = None, msg_ref: int | None = None, in_reply_to: str | None = None):
        """Render an incoming message with custom markdown + math rendering."""
        # Get display name from node ID (guard against None)
        from_node = from_node or "unknown"
        display_name = get_display_name(from_node)

        # Build timestamp string (at start for consistency with outgoing)
        time_str = f"{self.DIM}{timestamp}{self.RESET} " if timestamp else ""

        # Determine styling based on sender type
        if from_node.startswith("agent:"):
            name_str = f"{self.BOLD}{self.MAGENTA}{display_name}{self.RESET}"
        elif from_node.startswith("user:"):
            name_str = f"{self.BOLD}{self.YELLOW}{display_name}{self.RESET}"
        else:
            name_str = f"{self.BOLD}{self.WHITE}{display_name}{self.RESET}"

        # Add channel indicator if message is from a channel
        channel_str = f" {self.CYAN}#{channel}{self.RESET}" if channel else ""

        # Add ref number if assigned
        ref_str = f" {self.DIM}[#{msg_ref}]{self.RESET}" if msg_ref else ""

        # Add reply indicator if this message is a reply
        reply_str = ""
        if in_reply_to:
            reply_label = self._lookup_reply(in_reply_to)
            reply_str = f" {self.DIM}(replying to {reply_label}){self.RESET}"

        # Build header: timestamp name [#channel] [#ref] (replying to ...):
        header = f"{time_str}{name_str}{channel_str}{ref_str}{reply_str}:"

        # Render content as markdown with math support (capture to check length)
        old_stdout = sys.stdout
        buf = io.StringIO()
        sys.stdout = buf
        try:
            self.mdr.render(content)
        except Exception as e:
            # Fallback to plain text if markdown fails
            logger.warning(f"Markdown render failed: {e}")
            print(content)
        finally:
            sys.stdout = old_stdout
        rendered = buf.getvalue().rstrip("\n")

        # Always print directly - use terminal scrollback for long content
        print(f"\n{header}")
        print(rendered)
        print("-" * shutil.get_terminal_size().columns)

        # Show prompt hint so user knows their input context
        if show_prompt:
            self._print_prompt_hint()

    def render_presence(self, msg: Message, show_prompt: bool = True):
        """Render a presence notification (join/leave)."""
        # Skip if quiet mode is enabled
        if self._quiet_presence:
            return

        content = msg.content if isinstance(msg.content, dict) else {}
        event = content.get("event", "unknown")
        nickname = content.get("nickname", "")
        node_type = content.get("node_type", "unknown")
        channel = content.get("channel", "")  # For channel presence events

        # Format type info
        if node_type == "user":
            type_str = "user"
            color = self.YELLOW
        else:
            type_str = node_type
            color = self.MAGENTA

        # Channel presence events (from router broadcasting join/leave)
        if channel:
            if event == "join":
                print(f"\n{self.GREEN}[+]{self.RESET} {color}{self.BOLD}{nickname}{self.RESET} {self.DIM}joined #{channel}{self.RESET}")
            elif event == "leave":
                print(f"\n{self.RED}[-]{self.RESET} {color}{self.BOLD}{nickname}{self.RESET} {self.DIM}left #{channel}{self.RESET}")
        else:
            # Node presence (connect/disconnect from mesh)
            if event == "join":
                print(f"\n{self.GREEN}[+]{self.RESET} {color}{self.BOLD}{nickname}{self.RESET} {self.DIM}({type_str}) joined{self.RESET}")
            elif event == "leave":
                print(f"\n{self.RED}[-]{self.RESET} {color}{self.BOLD}{nickname}{self.RESET} {self.DIM}({type_str}) left{self.RESET}")

        # Show prompt hint so user knows their input context
        if show_prompt:
            self._print_prompt_hint()

    def render_confirm_request(self, msg: Message) -> None:
        """Render a confirmation request from an agent."""
        content = msg.content if isinstance(msg.content, dict) else {}
        tool_name = content.get("tool_name", "unknown")
        preview = content.get("preview", "")
        from_node = msg.from_node

        print()
        print(f"{self.YELLOW}{'━' * 50}{self.RESET}")
        print(f"{self.BOLD}{self.YELLOW}Confirmation Required{self.RESET} {self.DIM}(from {from_node}){self.RESET}")
        print(f"{self.YELLOW}{'━' * 50}{self.RESET}")
        print()
        print(f"{self.CYAN}Tool:{self.RESET} {self.BOLD}{tool_name}{self.RESET}")
        print()
        print(f"{self.CYAN}Action:{self.RESET}")
        # Indent preview lines
        for line in preview.split("\n"):
            print(f"  {line}")
        print()
        print(f"{self.GREEN}[y]{self.RESET} Confirm  {self.RED}[n]{self.RESET} Reject")
        print(f"{self.YELLOW}{'━' * 50}{self.RESET}")
        print()

    def render_status_response(self, msg: Message) -> None:
        """Render a status response from an agent."""
        content = msg.content if isinstance(msg.content, dict) else {}
        context = content.get("context", [])
        summary = content.get("summary")
        current_activity = content.get("current_activity")
        status_summary = content.get("status_summary", {})
        # System info fields
        hostname = content.get("hostname")
        model = content.get("model")
        backend = content.get("backend")
        working_directory = content.get("working_directory")
        from_node = msg.from_node

        display_name = get_display_name(from_node)
        print()
        print(f"{self.CYAN}{'━' * 60}{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}Status: {display_name}{self.RESET} {self.DIM}({from_node}){self.RESET}")
        print(f"{self.CYAN}{'━' * 60}{self.RESET}")

        # Show live status (heartbeat-lite) if available
        if status_summary:
            print()
            state = status_summary.get("state", "")
            worker_elapsed = status_summary.get("worker_elapsed_s")
            ctx_tokens = status_summary.get("context_tokens", 0)
            hist_turns = status_summary.get("history_turns", 0)
            hist_pct = status_summary.get("history_pct", 0)
            mem_pool = status_summary.get("memory_pool", 0)
            mem_active = status_summary.get("memory_active", 0)
            uptime_s = status_summary.get("uptime_s", 0)

            # State with color
            if state:
                state_upper = state.upper()
                if state == "busy":
                    state_color = self.YELLOW
                    if worker_elapsed is not None:
                        state_str = f"{state_upper} ({int(worker_elapsed)}s)"
                    else:
                        state_str = state_upper
                else:
                    state_color = self.GREEN
                    state_str = state_upper
                print(f"  {state_color}{self.BOLD}{state_str}{self.RESET}", end="")
                if ctx_tokens:
                    print(f"  {self.DIM}{ctx_tokens // 1000}k ctx{self.RESET}", end="")
                print()

            # Detail line: history, memory, uptime
            detail = []
            detail.append(f"hist: {hist_turns} turns ({int(hist_pct)}%)")
            if mem_pool or mem_active:
                detail.append(f"mem: {mem_pool}/{mem_active}")
            if uptime_s:
                hours = int(uptime_s) // 3600
                mins = (int(uptime_s) % 3600) // 60
                if hours > 0:
                    detail.append(f"up: {hours}h{mins}m")
                else:
                    detail.append(f"up: {mins}m")
            print(f"  {self.DIM}{' · '.join(detail)}{self.RESET}")

            # Active project map (memory v2)
            active_map = status_summary.get("active_map")
            if active_map:
                print(f"  {self.DIM}project:{self.RESET} {active_map}")

        # Show full diagnostics if available (rich status view)
        diagnostics = content.get("diagnostics")
        if diagnostics and isinstance(diagnostics, dict) and "error" not in diagnostics:
            self._render_diagnostics(diagnostics)
        else:
            # Fallback to basic system info
            if hostname or model or backend or working_directory:
                print()
                print(f"{self.BOLD}System Info:{self.RESET}")
                if hostname:
                    print(f"  {self.DIM}Host:{self.RESET} {hostname}")
                if model:
                    print(f"  {self.DIM}Model:{self.RESET} {model}")
                if backend:
                    print(f"  {self.DIM}Backend:{self.RESET} {backend}")
                if working_directory:
                    print(f"  {self.DIM}Working Dir:{self.RESET} {working_directory}")

        # Show real-time CC tool activity if present
        if current_activity:
            print()
            print(f"{self.YELLOW}{self.BOLD}🔄 In-Progress Activity:{self.RESET}")
            for line in current_activity.split("\n"):
                print(f"  {self.YELLOW}{line}{self.RESET}")

        if summary:
            print()
            print(f"{self.DIM}[Earlier summary]{self.RESET}")
            # Show first 200 chars of summary
            summary_preview = summary[:200] + "..." if len(summary) > 200 else summary
            print(f"{self.DIM}{summary_preview}{self.RESET}")

        if context:
            print()
            print(f"{self.BOLD}Recent context ({len(context)} entries):{self.RESET}")
            for entry in context:
                from_id = entry.get("from", "?")
                msg_content = entry.get("content", "")
                timestamp = entry.get("timestamp", "")
                entry_type = entry.get("type", "message")

                # Parse timestamp for display (convert to local time)
                ts_display = iso_to_local_time(timestamp)

                # Color and label based on entry type
                if entry_type == "tool_call":
                    # Tool calls: green, different label
                    sender_color = self.GREEN
                    sender_label = f"{display_name} 🔧"
                elif entry_type == "tool_result":
                    # Tool results: dim cyan
                    sender_color = self.CYAN
                    sender_label = "⚙ system"
                elif from_id == from_node:
                    sender_color = self.MAGENTA
                    sender_label = display_name
                else:
                    sender_color = self.YELLOW
                    sender_label = get_display_name(from_id)

                print()
                print(f"{sender_color}{self.BOLD}{sender_label}{self.RESET} {self.DIM}{ts_display}{self.RESET}")

                # Truncate long messages (shorter for tool results)
                max_len = 300 if entry_type == "tool_result" else 500
                if len(msg_content) > max_len:
                    msg_content = msg_content[:max_len] + f"\n{self.DIM}... ({len(msg_content)} chars total){self.RESET}"

                # Indent content
                for line in msg_content.split("\n")[:20]:  # Max 20 lines
                    print(f"  {line}")
                if msg_content.count("\n") > 20:
                    print(f"  {self.DIM}... ({msg_content.count(chr(10))} lines total){self.RESET}")
        else:
            print()
            print(f"{self.DIM}No messages in context{self.RESET}")

        print()
        print(f"{self.CYAN}{'━' * 60}{self.RESET}")
        print()

    def _render_diagnostics(self, diag: dict) -> None:
        """Render full diagnostic report sections from agent_status."""
        # Identity
        if "identity" in diag:
            i = diag["identity"]
            print()
            print(f"{self.BOLD}Identity:{self.RESET}")
            print(f"  {self.DIM}Node:{self.RESET}      {i.get('node_id', '?')}")
            hostname = i.get('hostname', '?')
            pid = i.get('pid', '?')
            print(f"  {self.DIM}Host:{self.RESET}      {hostname} (PID {pid})")
            uptime_s = i.get('uptime_seconds', 0)
            if uptime_s:
                hours = int(uptime_s) // 3600
                mins = (int(uptime_s) % 3600) // 60
                uptime_str = f"{hours}h {mins:02d}m" if hours > 0 else f"{mins}m"
                print(f"  {self.DIM}Uptime:{self.RESET}    {uptime_str}")
            wd = i.get('working_directory')
            if wd:
                print(f"  {self.DIM}Directory:{self.RESET} {wd}")

        # LLM
        if "llm" in diag:
            ll = diag["llm"]
            print()
            print(f"{self.BOLD}LLM:{self.RESET}")
            print(f"  {self.DIM}Worker:{self.RESET}  {ll.get('backend', '?')} / {ll.get('model', '?')}")
            print(f"  {self.DIM}Router:{self.RESET}  {ll.get('router_llm_backend', '?')} / {ll.get('router_llm_model', '?')}")

        # Router
        if "router" in diag:
            r = diag["router"]
            print()
            print(f"{self.BOLD}Router:{self.RESET}")
            state = r.get("state", "?").upper()
            state_color = self.YELLOW if state == "BUSY" else self.GREEN
            print(f"  {self.DIM}State:{self.RESET}   {state_color}{state}{self.RESET}")
            if r.get("worker_active"):
                elapsed = r.get("worker_elapsed_seconds")
                wid = r.get("worker_id", "?")
                elapsed_str = f", {elapsed:.0f}s" if elapsed else ""
                print(f"  {self.DIM}Worker:{self.RESET}  active ({wid}{elapsed_str})")
                snap = r.get("worker_snapshot_turns")
                if snap is not None:
                    print(f"  {self.DIM}Snapshot:{self.RESET} {snap} turns")
            else:
                print(f"  {self.DIM}Worker:{self.RESET}  inactive")
            if r.get("session_stats"):
                ss = r["session_stats"]
                print(f"  {self.DIM}Session:{self.RESET} {ss.get('user_turns', 0)} user turns, {ss.get('tool_calls', 0)} tool calls, {ss.get('total_chars', 0)} chars")

        # History
        if "history" in diag:
            h = diag["history"]
            print()
            print(f"{self.BOLD}History:{self.RESET}")
            if h.get("detail"):
                print(f"  {h['detail']}")
            else:
                turns = h.get("window_turns", 0)
                tokens = h.get("estimated_tokens", 0)
                soft = h.get("soft_limit_tokens", 0)
                hard = h.get("hard_limit_tokens", 0)
                pct = h.get("utilization_pct", 0)
                print(f"  {self.DIM}Window:{self.RESET}  {turns} turns (~{tokens:,} tokens)")
                print(f"  {self.DIM}Limits:{self.RESET}  {soft:,} soft / {hard:,} hard ({pct:.0f}% utilized)")
                summ = "none (rolling window)" if not h.get("summarization_enabled") else "active"
                if h.get("summary_present"):
                    summ = "present"
                print(f"  {self.DIM}Summary:{self.RESET} {summ}")
                oldest = h.get("oldest_turn_timestamp", "?")
                newest = h.get("newest_turn_timestamp", "?")
                if oldest and newest and oldest != "?" and newest != "?":
                    print(f"  {self.DIM}Range:{self.RESET}   {oldest} → {newest}")

        # Memory
        if "memory" in diag:
            m = diag["memory"]
            print()
            print(f"{self.BOLD}Memory:{self.RESET}")
            if not m.get("enabled"):
                print(f"  {self.DIM}{m.get('detail', 'disabled')}{self.RESET}")
            else:
                version = m.get("version", 1)
                print(f"  {self.DIM}Version:{self.RESET} v{version}")
                pool = m.get("pool_size", 0)
                pool_max = m.get("pool_max_entries", "?")
                active = m.get("active_set_size", 0)
                target = m.get("active_set_target", "?")
                print(f"  {self.DIM}Pool:{self.RESET}    {pool} entries (max {pool_max})")
                print(f"  {self.DIM}Active:{self.RESET}  {active} / {target} target")
                # Active project map (v2 only)
                active_proj = m.get("active_project")
                if active_proj:
                    map_chars = m.get("active_map_chars", 0)
                    map_words = map_chars // 5 if map_chars else 0
                    print(f"  {self.DIM}Map:{self.RESET}     {active_proj} ({map_chars:,} chars, ~{map_words:,} words)")
                    map_count = m.get("map_count", 0)
                    if map_count > 1:
                        print(f"  {self.DIM}Maps:{self.RESET}    {map_count} total")
                elif version == 2:
                    print(f"  {self.DIM}Map:{self.RESET}     none")
                ago = m.get("last_reflection_ago_seconds")
                if ago is not None:
                    hours = int(ago) // 3600
                    mins = (int(ago) % 3600) // 60
                    ago_str = f"{hours}h {mins:02d}m ago" if hours > 0 else f"{mins}m ago"
                    print(f"  {self.DIM}Last reflection:{self.RESET} {ago_str}")
                else:
                    print(f"  {self.DIM}Last reflection:{self.RESET} none")

        # Health checks
        if "context_health" in diag:
            ch = diag["context_health"]
            checks = ch.get("checks", [])
            if checks:
                print()
                print(f"{self.BOLD}Health Checks:{self.RESET}")
                for check in checks:
                    ok = check.get("ok", False)
                    icon = f"{self.GREEN}+{self.RESET}" if ok else f"{self.RED}!{self.RESET}"
                    name = check.get("name", "?")
                    detail = check.get("detail", "")
                    print(f"  {icon} {name} {self.DIM}({detail}){self.RESET}")

    def _render_tool_activity(self, msg: Message) -> None:
        """Render a real-time TOOL_ACTIVITY message from an attached agent."""
        content = msg.content if isinstance(msg.content, dict) else {}
        event_type = content.get("event_type", "")
        tool_name = content.get("tool_name", "unknown")
        tool_source = content.get("tool_source", "mesh")  # "cc" or "mesh"
        data = content.get("data", {})

        # Get agent display name
        display_name = get_display_name(msg.from_node)

        if event_type == "tool_call":
            # Format and display tool call
            args = data.get("args", {})
            formatted = format_tool_activity_call(tool_name, args)
            # Show source indicator for mesh tools vs CC tools
            source_tag = f"{self.DIM}[mesh]{self.RESET} " if tool_source == "mesh" else ""
            print(f"\n{self.DIM}{display_name}:{self.RESET} {source_tag}{formatted}")

        elif event_type == "tool_result":
            # Format and display tool result
            result = data.get("result", "")
            success = data.get("success", True)
            formatted = format_tool_activity_result(result, tool_name, success)
            print(formatted)

    def _render_status_update(self, msg: Message) -> None:
        """Render a STATUS message (phase updates) from an attached agent."""
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        display_name = get_display_name(msg.from_node)
        # Show phase updates with a distinct style
        print(f"\n{self.DIM}{display_name}:{self.RESET} {self.CYAN}⟳{self.RESET} {content}")

    def _show_conversation_context(self, partner: str, n: int = 10):
        """Show last N messages from a specific conversation."""
        # Get messages from node history
        if not self.node or not hasattr(self.node, 'history'):
            print(f"{self.DIM}No message history yet.{self.RESET}")
            return

        # Filter messages for this conversation
        is_channel = partner.startswith("channel:")
        filtered = []

        for entry in self.node.history:
            msg = entry.message
            if msg.type != MessageType.MESSAGE:
                continue

            if is_channel:
                # Channel messages: to_node matches the channel
                if msg.to_node == partner:
                    filtered.append(entry)
            else:
                # DM messages: either from or to the partner
                if (msg.from_node == partner and msg.to_node == self.node_id) or \
                   (msg.from_node == self.node_id and msg.to_node == partner):
                    filtered.append(entry)

        if not filtered:
            partner_name = get_display_name(partner)
            print(f"{self.DIM}No messages with {partner_name} yet.{self.RESET}")
            return

        # Show last n
        recent = filtered[-n:]
        partner_name = get_display_name(partner)
        print()
        print(f"{self.CYAN}{'━' * 60}{self.RESET}")
        print(f"{self.BOLD}Recent with {partner_name}:{self.RESET} {self.DIM}(last {len(recent)}){self.RESET}")
        print(f"{self.CYAN}{'━' * 60}{self.RESET}")

        # Build formatted output lines for potential paging
        output_lines = []

        # Only render math for the last ~2 turns (4 messages)
        math_cutoff = max(0, len(recent) - 2)

        for i, entry in enumerate(recent):
            msg = entry.message
            content = msg.content if msg.content else ""
            if not isinstance(content, str):
                content = str(content)

            # Parse timestamp (convert to local time)
            timestamp = msg.timestamp or ""
            ts_display = iso_to_local_time(timestamp) if timestamp else ""

            # Format based on direction (matches _render_outgoing_message style)
            from_node = msg.from_node or "unknown"
            to_node = msg.to_node or "unknown"
            is_channel = partner.startswith("channel:")

            if from_node == self.node_id:
                # Outgoing message: teal/cyan nickname
                if is_channel:
                    chan_name = partner.split(":")[1] if ":" in partner else partner
                    header = f"{self.DIM}{ts_display}{self.RESET} {self.CYAN}{self.BOLD}{self.nickname}{self.RESET} {self.MAGENTA}[#{chan_name}]{self.RESET}"
                else:
                    to_name = get_display_name(to_node)
                    header = f"{self.DIM}{ts_display}{self.RESET} {self.CYAN}{self.BOLD}{self.nickname}{self.RESET} {self.GREEN}→{self.RESET} {to_name}"
            else:
                # Incoming message: magenta sender
                from_name = get_display_name(from_node)
                header = f"{self.DIM}{ts_display}{self.RESET} {self.MAGENTA}{self.BOLD}{from_name}{self.RESET}"

            # Only render math for the last ~2 turns
            self.mdr.render_math = (i >= math_cutoff)

            # Render full message with markdown
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            self.mdr.render(content)
            rendered = sys.stdout.getvalue().rstrip("\n")
            sys.stdout = old_stdout

            lines = rendered.split("\n")
            if len(lines) == 1:
                output_lines.append(f"  {header}: {lines[0]}")
            else:
                output_lines.append(f"  {header}:")
                for line in lines:
                    output_lines.append(f"    {line}")
            output_lines.append("-" * shutil.get_terminal_size().columns)  # Separator between messages

        # Restore math rendering for real-time messages
        self.mdr.render_math = True

        # Remove trailing blank line
        if output_lines and output_lines[-1] == "":
            output_lines.pop()

        # Print all content directly (no pager) to allow terminal scrollback
        for line in output_lines:
            print(line)

        # Echo current context after displaying
        self._print_current_context()
        print()

    def _pager(self, text: str, keep_content: bool = True):
        """Custom pager that renders markdown/math properly.

        Keys:
            j/↓     - scroll down one line
            k/↑     - scroll up one line
            space/d - page down
            b/u     - page up
            g       - go to top
            G       - go to bottom
            q       - quit

        Args:
            text: Content to page through
            keep_content: If True, display last screenful after closing (default True)
        """
        import termios
        import tty
        from io import StringIO

        # Capture output from MarkdownRenderer by redirecting stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            self.mdr.render(text)
            ansi = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        lines = ansi.splitlines()

        if not lines:
            return

        # Get terminal size
        term_size = shutil.get_terminal_size()
        page_height = term_size.lines - 2  # Leave room for status line

        offset = 0
        total_lines = len(lines)

        def display_page():
            # Clear screen and move to top
            sys.stdout.write("\033[2J\033[H")

            # Display visible lines
            visible = lines[offset:offset + page_height]
            for line in visible:
                print(line)

            # Status line at bottom
            pct = int(100 * min(offset + page_height, total_lines) / total_lines) if total_lines > 0 else 100
            status = f"\033[7m lines {offset+1}-{min(offset+page_height, total_lines)} of {total_lines} ({pct}%) | j/k:scroll  space:page  q:quit \033[0m"
            # Move to last line and print status
            sys.stdout.write(f"\033[{term_size.lines};1H{status}")
            sys.stdout.flush()

        def get_key():
            """Read a single keypress, handling escape sequences."""
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                # Handle escape sequences (arrow keys)
                if ch == '\x1b':
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        ch3 = sys.stdin.read(1)
                        if ch3 == 'A':
                            return 'up'
                        elif ch3 == 'B':
                            return 'down'
                        elif ch3 == '5':
                            sys.stdin.read(1)  # consume ~
                            return 'pgup'
                        elif ch3 == '6':
                            sys.stdin.read(1)  # consume ~
                            return 'pgdn'
                    return 'esc'
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        # Main pager loop
        display_page()
        while True:
            key = get_key()

            if key in ('q', '\x03'):  # q or Ctrl+C
                break
            elif key in ('j', 'down'):
                if offset < total_lines - page_height:
                    offset += 1
            elif key in ('k', 'up'):
                if offset > 0:
                    offset -= 1
            elif key in (' ', 'd', 'pgdn'):
                offset = min(offset + page_height // 2, max(0, total_lines - page_height))
            elif key in ('b', 'u', 'pgup'):
                offset = max(0, offset - page_height // 2)
            elif key == 'g':
                offset = 0
            elif key == 'G':
                offset = max(0, total_lines - page_height)

            display_page()

        # Clear and restore normal terminal
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

        if keep_content:
            # Show last screenful of content
            display_lines = term_size.lines - 3
            tail_lines = lines[-display_lines:] if len(lines) > display_lines else lines
            for line in tail_lines:
                print(line)

        # Print context after pager closes
        self._print_current_context()

    def _print_current_context(self):
        """Print current conversation context (after pager, etc.)."""
        if self.current_view:
            view_name = get_display_name(self.current_view)
            if self.current_view.startswith("channel:"):
                print(f"{self.DIM}[#{self.current_view.split(':')[1]}]{self.RESET}")
            else:
                print(f"{self.DIM}[@{view_name}]{self.RESET}")
        else:
            print(f"{self.DIM}[all messages]{self.RESET}")

    def _matches_current_view(self, msg) -> bool:
        """Check if a message matches the current conversation view filter."""
        if self.current_view is None:
            return True  # No filter, show all

        is_channel = self.current_view.startswith("channel:")

        if is_channel:
            # Channel: to_node must be the channel address
            # Router broadcasts with to_node=channel:name, so we check that
            # Also check metadata.channel as a fallback
            if msg.to_node == self.current_view:
                return True
            # Fallback: check metadata (in case router adds it)
            if msg.metadata and msg.metadata.get("channel"):
                chan_name = msg.metadata.get("channel")
                chan_addr = f"channel:{chan_name}" if not chan_name.startswith("channel:") else chan_name
                return chan_addr == self.current_view
            return False
        else:
            # DM: either from or to the view partner
            return (
                (msg.from_node == self.current_view and msg.to_node == self.node_id) or
                (msg.from_node == self.node_id and msg.to_node == self.current_view)
            )

    def _update_recent_conversations(self, partner: str) -> None:
        """Update the recent conversations list, moving partner to front."""
        if partner in self._recent_conversations:
            self._recent_conversations.remove(partner)
        self._recent_conversations.insert(0, partner)
        # Keep only last 20
        self._recent_conversations = self._recent_conversations[:20]

    def _clear_unread(self, partner: str) -> None:
        """Clear unread count for a conversation."""
        if partner in self._unread_counts:
            del self._unread_counts[partner]

    def _send_notification(self, title: str, body: str) -> None:
        """Send a desktop notification if enabled (non-blocking)."""
        if not self._notifications_enabled:
            return
        try:
            # Fire-and-forget — don't block the event loop
            asyncio.get_running_loop().create_task(self._send_notification_async(title, body))
        except RuntimeError:
            pass  # No event loop — skip notification

    @staticmethod
    async def _send_notification_async(title: str, body: str) -> None:
        """Actually send the notification asynchronously."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "notify-send", "-a", "Mesh TUI", title, body,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except Exception as e:
            logger.debug(f"Notification failed: {e}")

    async def _run_watch_mode(self, target: str) -> None:
        """Enter watch mode: poll agent status every 5s, showing In-Progress Activity.

        Press q or Esc to exit.
        """
        import termios
        import tty
        import select

        display_name = get_display_name(target)
        self._watch_target = target

        # Drain any stale status responses from the watch queue
        while not self._watch_queue.empty():
            try:
                self._watch_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Save terminal state and switch to raw mode for keypress detection
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setcbreak(fd)  # cbreak: keypress available immediately, no echo

            print(f"\033[2J\033[H", end="")  # Clear screen
            print(f"{self.CYAN}{'━' * 60}{self.RESET}")
            print(f"{self.BOLD}Watching: {display_name}{self.RESET} {self.DIM}(q/Esc to exit, refreshes every 5s){self.RESET}")
            print(f"{self.CYAN}{'━' * 60}{self.RESET}")
            print(f"\n{self.DIM}Requesting status...{self.RESET}")
            sys.stdout.flush()

            while True:
                # Send status request (0 context messages, just want activity + status_summary)
                status_msg = make_status_request(
                    from_node=self.node_id,
                    to_node=target,
                    num_messages=0,
                )
                await self.node._conn.send(status_msg)

                # Wait for response (up to 10s timeout)
                try:
                    msg = await asyncio.wait_for(self._watch_queue.get(), timeout=10.0)
                    self._render_watch_screen(msg, display_name)
                except asyncio.TimeoutError:
                    print(f"\033[2J\033[H", end="")  # Clear screen
                    print(f"{self.CYAN}{'━' * 60}{self.RESET}")
                    print(f"{self.BOLD}Watching: {display_name}{self.RESET} {self.DIM}(q/Esc to exit){self.RESET}")
                    print(f"{self.CYAN}{'━' * 60}{self.RESET}")
                    print(f"\n{self.YELLOW}No response (agent may be offline){self.RESET}")
                    sys.stdout.flush()

                # Wait 5 seconds, checking for keypress every 100ms
                exit_watch = False
                for _ in range(50):  # 50 × 100ms = 5s
                    await asyncio.sleep(0.1)
                    # Check for keypress (non-blocking)
                    if select.select([sys.stdin], [], [], 0)[0]:
                        ch = sys.stdin.read(1)
                        if ch in ('q', 'Q', '\x1b'):  # q, Q, or Esc
                            exit_watch = True
                            break
                if exit_watch:
                    break

        finally:
            # Restore terminal settings
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            self._watch_target = None

        print(f"\n{self.DIM}Exited watch mode{self.RESET}")

    def _render_watch_screen(self, msg: Message, display_name: str) -> None:
        """Render the watch mode screen with status info and activity."""
        content = msg.content if isinstance(msg.content, dict) else {}
        current_activity = content.get("current_activity")
        status_summary = content.get("status_summary", {})

        # Clear screen and redraw
        print(f"\033[2J\033[H", end="")  # Clear screen

        # Header
        print(f"{self.CYAN}{'━' * 60}{self.RESET}")
        print(f"{self.BOLD}Watching: {display_name}{self.RESET} {self.DIM}(q/Esc to exit){self.RESET}")
        print(f"{self.CYAN}{'━' * 60}{self.RESET}")

        # Status line
        if status_summary:
            state = status_summary.get("state", "")
            worker_elapsed = status_summary.get("worker_elapsed_s")
            ctx_tokens = status_summary.get("context_tokens", 0)

            if state:
                state_upper = state.upper()
                if state == "busy":
                    state_color = self.YELLOW
                    if worker_elapsed is not None:
                        state_str = f"{state_upper} ({int(worker_elapsed)}s)"
                    else:
                        state_str = state_upper
                else:
                    state_color = self.GREEN
                    state_str = state_upper
                print(f"\n  {state_color}{self.BOLD}{state_str}{self.RESET}", end="")
                if ctx_tokens:
                    print(f"  {self.DIM}{ctx_tokens // 1000}k ctx{self.RESET}", end="")
                print()

        # In-Progress Activity
        if current_activity:
            print()
            print(f"{self.YELLOW}{self.BOLD}In-Progress Activity:{self.RESET}")
            for line in current_activity.split("\n"):
                print(f"  {self.YELLOW}{line}{self.RESET}")
        else:
            state = status_summary.get("state", "idle") if status_summary else "idle"
            if state == "idle":
                print(f"\n  {self.DIM}No activity (idle){self.RESET}")
            else:
                print(f"\n  {self.DIM}Waiting for activity...{self.RESET}")

        # Timestamp
        now = datetime.now().strftime("%H:%M:%S")
        print(f"\n{self.DIM}Last updated: {now}{self.RESET}")
        sys.stdout.flush()

    def _show_cc_usage(self) -> None:
        """Show Claude Code plan usage limits and reset times."""
        import json as _json
        from datetime import timezone, timedelta

        creds_path = os.path.expanduser("~/.claude/.credentials.json")
        if not os.path.exists(creds_path):
            print(f"{self.DIM}No Claude Code credentials found at {creds_path}{self.RESET}")
            return

        try:
            with open(creds_path, 'r') as f:
                creds = _json.load(f)
            token = creds.get("claudeAiOauth", {}).get("accessToken")
            if not token:
                print(f"{self.DIM}No OAuth token found in credentials.{self.RESET}")
                return
        except (ValueError, IOError):
            print(f"{self.DIM}Could not read credentials file.{self.RESET}")
            return

        print(f"{self.DIM}Querying Claude Code usage...{self.RESET}")

        try:
            import urllib.request
            req = urllib.request.Request(
                "https://api.anthropic.com/api/oauth/usage",
                headers={
                    "Authorization": f"Bearer {token}",
                    "anthropic-beta": "oauth-2025-04-20",
                    "Content-Type": "application/json",
                    "User-Agent": "claude-code/2.1.69",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                usage = _json.loads(resp.read().decode())
        except Exception as e:
            print(f"{self.DIM}Failed to fetch usage: {e}{self.RESET}")
            return

        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        RED = "\033[31m"

        def fmt_window(name: str, data: dict | None, window_hours: float) -> None:
            if data is None:
                print(f"  {name}: {self.DIM}N/A{self.RESET}")
                return
            util = data.get("utilization", 0)
            reset_str = data.get("resets_at", "")

            color = RED if util >= 80 else YELLOW if util >= 50 else GREEN

            time_remaining = ""
            reset_display = ""
            projection = None

            if reset_str:
                try:
                    reset_dt = datetime.fromisoformat(reset_str.replace('Z', '+00:00'))
                    now = datetime.now(timezone.utc)
                    delta = reset_dt - now
                    total_hours = delta.total_seconds() / 3600

                    if window_hours >= 24 and total_hours >= 24:
                        d, h = int(total_hours // 24), int(total_hours % 24)
                        time_remaining = f"{d}d {h}h"
                    elif total_hours >= 1:
                        h = int(total_hours)
                        m = int((delta.total_seconds() % 3600) // 60)
                        time_remaining = f"{h}h {m}m"
                    elif delta.total_seconds() > 60:
                        time_remaining = f"{int(delta.total_seconds() // 60)}m"
                    else:
                        time_remaining = "soon"

                    local_dt = reset_dt.astimezone()
                    reset_display = local_dt.strftime("%b %d %H:%M %Z")

                    # Projection: when will we hit 100% at current rate?
                    if util > 0 and total_hours > 0:
                        elapsed_hours = window_hours - total_hours
                        if elapsed_hours > 0:
                            rate = util / elapsed_hours
                            remaining = 100.0 - util
                            if rate > 0:
                                hours_to_limit = remaining / rate
                                limit_dt = now + timedelta(hours=hours_to_limit)
                                if limit_dt < reset_dt:
                                    lt = limit_dt.astimezone().strftime("%b %d %H:%M %Z")
                                    if hours_to_limit >= 24:
                                        lts = f"{int(hours_to_limit // 24)}d {int(hours_to_limit % 24)}h"
                                    elif hours_to_limit >= 1:
                                        lts = f"{int(hours_to_limit)}h {int((hours_to_limit % 1) * 60)}m"
                                    else:
                                        lts = f"{int(hours_to_limit * 60)}m"
                                    projection = f"             {RED}At current rate, limit in {lts} ({lt}){self.RESET}"
                except Exception:
                    time_remaining = "?"
                    reset_display = reset_str[:19]

            print(f"  {name}: {color}{util:5.1f}%{self.RESET}  resets in {time_remaining:>8}  ({reset_display})")
            if projection:
                print(projection)

        print()
        print(f"{self.BOLD}Claude Code Usage{self.RESET}")
        print()
        fmt_window("5-hour window", usage.get("five_hour"), 5.0)
        fmt_window("7-day window ", usage.get("seven_day"), 168.0)

        if usage.get("seven_day_opus"):
            fmt_window("7-day Opus   ", usage.get("seven_day_opus"), 168.0)
        if usage.get("seven_day_sonnet"):
            fmt_window("7-day Sonnet ", usage.get("seven_day_sonnet"), 168.0)

        extra = usage.get("extra_usage", {})
        if extra.get("is_enabled"):
            used = extra.get("used_credits", 0)
            limit = extra.get("monthly_limit", "unlimited")
            print()
            print(f"  {self.DIM}Overage enabled: ${used:.2f} used (limit: ${limit}){self.RESET}")

        print()

    def _show_inbox(self, limit: int | None = None) -> None:
        """Show conversations sorted by recent activity with unread counts."""
        if not self.node:
            print(f"{self.DIM}No connection yet.{self.RESET}")
            return

        # Collect conversations from history
        conversations: dict[str, dict] = {}  # partner -> {last_msg, last_time, unread}

        for entry in self.node.history:
            msg = entry.message
            if msg.type != MessageType.MESSAGE:
                continue

            # Determine conversation partner (must match message_receiver logic)
            if entry.direction == "outgoing":
                partner = msg.to_node
            else:
                if msg.to_node and msg.to_node.startswith("channel:"):
                    partner = msg.to_node
                elif msg.metadata and msg.metadata.get("channel"):
                    channel = msg.metadata["channel"]
                    partner = f"channel:{channel}" if not channel.startswith("channel:") else channel
                else:
                    partner = msg.from_node

            if not partner or partner == self.node_id:
                continue

            # Skip channel addresses for DM counting (channel messages have channel in metadata)
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            timestamp = msg.timestamp or ""

            if partner not in conversations:
                conversations[partner] = {
                    "last_msg": content,
                    "last_time": timestamp,
                    "unread": self._unread_counts.get(partner, 0),
                }
            else:
                # Update if this message is more recent
                if timestamp > conversations[partner]["last_time"]:
                    conversations[partner]["last_msg"] = content
                    conversations[partner]["last_time"] = timestamp

        # Also add any partners with unread but no history (edge case)
        for partner, count in self._unread_counts.items():
            if partner not in conversations:
                conversations[partner] = {
                    "last_msg": "",
                    "last_time": "",
                    "unread": count,
                }

        if not conversations:
            print(f"\n{self.DIM}No conversations yet.{self.RESET}")
            return

        # Sort by recent activity (most recent first)
        sorted_convs = sorted(
            conversations.items(),
            key=lambda x: x[1]["last_time"],
            reverse=True
        )

        # Update recent_conversations list (always full list for /1, /2, etc.)
        self._recent_conversations = [p for p, _ in sorted_convs]

        # Apply display limit
        total = len(sorted_convs)
        display_convs = sorted_convs[:limit] if limit else sorted_convs

        print()
        if limit and total > limit:
            print(f"{self.BOLD}Inbox (showing {len(display_convs)}/{total}):{self.RESET}")
        else:
            print(f"{self.BOLD}Inbox ({total} conversations):{self.RESET}")
        print()

        for idx, (partner, info) in enumerate(display_convs, 1):
            display_name = get_display_name(partner)
            unread = info["unread"]
            last_msg = info["last_msg"]
            last_time = info["last_time"]

            # Format timestamp (convert to local time)
            if last_time:
                ts = iso_to_local_time(last_time)[:5]  # HH:MM only
            else:
                ts = ""

            # Truncate message preview
            preview = last_msg[:50] + "..." if len(last_msg) > 50 else last_msg
            preview = preview.replace("\n", " ")

            # Color based on type
            if partner.startswith("channel:"):
                color = self.CYAN
                display_name = f"#{partner.split(':')[1]}" if ':' in partner else partner
            elif partner.startswith("agent:"):
                color = self.MAGENTA
            elif partner.startswith("user:"):
                color = self.YELLOW
            else:
                color = self.WHITE

            # Unread indicator
            if unread > 0:
                unread_str = f" {self.RED}({unread}){self.RESET}"
            else:
                unread_str = ""

            print(f"  {self.DIM}{idx}.{self.RESET} {color}{self.BOLD}{display_name}{self.RESET}{unread_str}")
            if preview:
                print(f"     {self.DIM}{ts} {preview}{self.RESET}")
            print()  # Blank line between conversation entries

        if limit and total > limit:
            print(f"{self.DIM}Use /i <n> for more · /view <name> or /<number> to switch{self.RESET}")
        else:
            print(f"{self.DIM}Use /view <name> or /<number> to switch{self.RESET}")
        print()

    def _outgoing_matches_view(self, to_node: str) -> bool:
        """Check if an outgoing message matches the current view filter."""
        if self.current_view is None:
            return True  # No filter, show all

        is_channel = self.current_view.startswith("channel:")

        if is_channel:
            return to_node == self.current_view
        else:
            return to_node == self.current_view

    def _render_outgoing_message(self, to_node: str, content: str):
        """Render an outgoing message in conversation view format."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if self._outgoing_matches_view(to_node):
            # Render as part of conversation
            to_name = get_display_name(to_node)
            is_channel = to_node.startswith("channel:")

            if is_channel:
                chan_name = to_node.split(":")[1] if ":" in to_node else to_node
                header = f"{self.DIM}{timestamp}{self.RESET} {self.CYAN}{self.BOLD}{self.nickname}{self.RESET} {self.MAGENTA}[#{chan_name}]{self.RESET}"
            else:
                header = f"{self.DIM}{timestamp}{self.RESET} {self.CYAN}{self.BOLD}{self.nickname}{self.RESET} {self.GREEN}→{self.RESET} {to_name}"

            # Render with markdown (capture stdout since mdr.render prints directly)
            old_stdout = sys.stdout
            buf = io.StringIO()
            sys.stdout = buf
            try:
                self.mdr.render(content)
            finally:
                sys.stdout = old_stdout
            rendered = buf.getvalue().rstrip()
            lines = rendered.split("\n") if rendered else [content]
            if len(lines) == 1:
                print(f"{header}: {lines[0]}")
            else:
                print(f"{header}:")
                for line in lines:
                    print(f"  {line}")
            print("-" * shutil.get_terminal_size().columns)
        else:
            # Just show confirmation
            to_name = get_display_name(to_node)
            print(f"{self.DIM}→ Sent to {to_name}{self.RESET}")

    async def get_user_input(self) -> str:
        """Get input from user with prompt showing target and view."""
        def _build_prompt():
            parts = []
            if self.current_view:
                view_name = get_display_name(self.current_view)
                if self.current_view.startswith("channel:"):
                    parts.append(f"#{self.current_view.split(':')[1]}")
                else:
                    parts.append(f"@{view_name}")
            if self.default_target:
                parts.append(f"→{get_display_name(self.default_target)}")
            else:
                parts.append("no target")
            draft_ind = "*" if self.default_target and self.default_target in self._drafts else ""
            return [('class:prompt', f"\n[{' '.join(parts)}] {self.username}:{draft_ind} ")]

        try:
            return await self.session.prompt_async(
                _build_prompt,
                enable_history_search=True,
            )
        except KeyboardInterrupt:
            return ""

    async def handle_command(self, line: str) -> bool:
        """
        Handle a slash command.

        Returns:
            True to continue, False to quit
        """
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else None

        if cmd == "/quit" or cmd == "/q":
            return False

        elif cmd == "/list" or cmd == "/ls":
            # Query router for connected nodes + heartbeat-lite status + cc_usage
            try:
                response = await self.node.request_node_list_with_status_raw(timeout=3.0)
                connected_node_ids = response.get("nodes", [])
                node_status = response.get("status", {})
                node_metadata = response.get("metadata", {})
                cc_usage = response.get("cc_usage", "")
            except Exception as e:
                logger.error(f"Failed to get node list from router: {e}")
                connected_node_ids, node_status, node_metadata, cc_usage = [], {}, {}, ""

            # Filter roster to only show actually connected nodes
            roster_list = self.node.get_roster_list()
            connected_set = set(connected_node_ids)
            active_entries = [e for e in roster_list if e.node_id in connected_set]

            # Clean up stale roster entries (nodes no longer connected)
            stale_entries = [e for e in roster_list if e.node_id not in connected_set]
            for stale in stale_entries:
                self.node._roster.pop(stale.nickname.lower(), None)

            # Display CC usage summary at the top
            if cc_usage:
                print(f"\n{self.DIM}{cc_usage}{self.RESET}")

            if not active_entries:
                print(f"\n{self.YELLOW}No other nodes connected{self.RESET}")
            else:
                print(f"\n{self.BOLD}Connected Nodes:{self.RESET}")
                for entry in active_entries:
                    if entry.node_type == "user":
                        color = self.YELLOW
                        type_str = "user"
                    else:
                        color = self.MAGENTA
                        type_str = entry.node_type
                    # Show unread count if any
                    unread = self._unread_counts.get(entry.node_id, 0)
                    unread_str = f" {self.RED}({unread} unread){self.RESET}" if unread > 0 else ""
                    # Header line: nickname (type)  [unread]
                    print(f"\n  {color}{self.BOLD}{entry.nickname}{self.RESET} {self.DIM}({type_str}){self.RESET}{unread_str}")
                    # Backend + config detail lines from registration metadata
                    meta = node_metadata.get(entry.node_id, {})
                    # Line 1: worker backend · router backend · host
                    line1 = []
                    worker_be = meta.get("llm_backend") or entry.llm_backend
                    worker_model = meta.get("llm_model") or entry.llm_model
                    if worker_be:
                        w = f"{worker_be}/{worker_model}" if worker_model and worker_model != worker_be else worker_be
                        line1.append(f"worker: {self.CYAN}{w}{self.RESET}")
                    router_be = meta.get("router_v2_llm_backend", "")
                    router_model = meta.get("router_v2_llm_model", "")
                    if router_be:
                        r = f"{router_be}/{router_model}" if router_model and router_model != router_be else router_be
                        line1.append(f"router: {self.CYAN}{r}{self.RESET}")
                    host_str = meta.get("hostname") or entry.hostname
                    if host_str:
                        line1.append(f"host: {host_str}")
                    if line1:
                        print(f"    {self.DIM}{' · '.join(line1)}{self.RESET}")
                    # Line 2: session mode (harness-session or CC interactive)
                    session_parts = []
                    hsb = meta.get("harness_session_backend", "")
                    if hsb:
                        session_parts.append(f"harness-session: {self.CYAN}{hsb}{self.RESET}")
                    if meta.get("cc_interactive_tools"):
                        cc_desc = "cc-interactive"
                        cc_model = meta.get("cc_interactive_model", "")
                        cc_binary = meta.get("cc_interactive_binary", "")
                        cc_effort = meta.get("cc_interactive_effort", "")
                        cc_extras = []
                        if cc_model:
                            cc_extras.append(cc_model)
                        if cc_binary:
                            cc_extras.append(cc_binary.rsplit("/", 1)[-1])
                        if cc_effort:
                            cc_extras.append(f"effort={cc_effort}")
                        if cc_extras:
                            cc_desc += f" ({', '.join(cc_extras)})"
                        session_parts.append(f"{self.CYAN}{cc_desc}{self.RESET}")
                    if session_parts:
                        print(f"    {self.DIM}session: {' + '.join(session_parts)}{self.RESET}")
                    # Heartbeat-lite status block
                    status = node_status.get(entry.node_id, {})
                    if status:
                        state = status.get("state", "")
                        ctx_tokens = status.get("context_tokens", 0)
                        worker_elapsed = status.get("worker_elapsed_s")
                        # State line
                        if state:
                            state_upper = state.upper()
                            if state == "busy":
                                state_color = self.YELLOW
                                if worker_elapsed is not None:
                                    state_str = f"{state_upper} ({int(worker_elapsed)}s)"
                                else:
                                    state_str = state_upper
                            else:
                                state_color = self.GREEN
                                state_str = state_upper
                            print(f"    {state_color}{state_str}{self.RESET}", end="")
                            if ctx_tokens:
                                print(f" {self.DIM}· {ctx_tokens // 1000}k ctx{self.RESET}", end="")
                            print()
                        # History + memory line
                        detail2 = []
                        hist_pct = status.get("history_pct")
                        if hist_pct is not None:
                            detail2.append(f"hist: {int(hist_pct)}%")
                        mem_pool = status.get("memory_pool")
                        mem_active = status.get("memory_active")
                        if mem_pool is not None and mem_active is not None:
                            detail2.append(f"mem: {mem_pool}/{mem_active}")
                        uptime = status.get("uptime_s")
                        if uptime is not None:
                            hours = int(uptime) // 3600
                            mins = (int(uptime) % 3600) // 60
                            if hours > 0:
                                detail2.append(f"up: {hours}h{mins}m")
                            else:
                                detail2.append(f"up: {mins}m")
                        if detail2:
                            print(f"    {self.DIM}{' · '.join(detail2)}{self.RESET}")

        elif cmd == "/target" or cmd == "/t":
            if arg:
                # Try to resolve nickname to full node ID
                resolved = self.node.resolve_target(arg)
                if resolved:
                    self.default_target = resolved
                    display = get_display_name(resolved)
                    print(f"{self.GREEN}Default target set to:{self.RESET} {display} ({resolved})")
                    await self._pull_todos()
                else:
                    # Couldn't resolve - show available options
                    print(f"{self.YELLOW}Unknown target '{arg}'. Use /list to see available nodes.{self.RESET}")
            elif self.default_target:
                display = get_display_name(self.default_target)
                print(f"{self.CYAN}Current target:{self.RESET} {display} ({self.default_target})")
            else:
                print(f"{self.YELLOW}No default target set. Use /target <nickname>{self.RESET}")

        elif cmd == "/to":
            if not arg or " " not in arg:
                print(f"{self.YELLOW}Usage: /to <nickname> <message>{self.RESET}")
            else:
                target, content = arg.split(" ", 1)
                # Resolve nickname to full node ID
                resolved = self.node.resolve_target(target)
                if resolved:
                    await self.node.send(resolved, content)
                    # Render outgoing message if it matches current view
                    self._render_outgoing_message(resolved, content)
                else:
                    print(f"{self.YELLOW}Unknown target '{target}'. Use /list to see available nodes.{self.RESET}")

        elif cmd == "/reply":
            if not arg or " " not in arg:
                print(f"{self.YELLOW}Usage: /reply <N> <message>{self.RESET}")
            else:
                ref_str, content = arg.split(" ", 1)
                try:
                    ref_num = int(ref_str)
                except ValueError:
                    print(f"{self.YELLOW}Invalid ref number '{ref_str}'. Use the [#N] shown on messages.{self.RESET}")
                    return True
                msg_id = self._msg_ref_map.get(ref_num)
                if not msg_id:
                    print(f"{self.YELLOW}Message #{ref_num} not found in recent messages.{self.RESET}")
                elif not self.default_target:
                    print(f"{self.YELLOW}No target set. Use /target <node_id> first.{self.RESET}")
                else:
                    await self.node.send(self.default_target, content, in_reply_to=msg_id)
                    reply_label = self._lookup_reply(msg_id)
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    if self._outgoing_matches_view(self.default_target):
                        to_name = get_display_name(self.default_target)
                        is_channel = self.default_target.startswith("channel:")
                        if is_channel:
                            chan_name = self.default_target.split(":")[1] if ":" in self.default_target else self.default_target
                            header = f"{self.DIM}{timestamp}{self.RESET} {self.CYAN}{self.BOLD}{self.nickname}{self.RESET} {self.MAGENTA}[#{chan_name}]{self.RESET} {self.DIM}(replying to {reply_label}){self.RESET}"
                        else:
                            header = f"{self.DIM}{timestamp}{self.RESET} {self.CYAN}{self.BOLD}{self.nickname}{self.RESET} {self.GREEN}→{self.RESET} {to_name} {self.DIM}(replying to {reply_label}){self.RESET}"
                        print(f"{header}: {content}")
                        print("-" * shutil.get_terminal_size().columns)
                    else:
                        print(f"{self.DIM}→ Reply sent to {get_display_name(self.default_target)} (re: {reply_label}){self.RESET}")

        elif cmd == "/clear":
            print("\033[2J\033[H", end="")  # Clear screen
            self.show_header()

        elif cmd == "/status" or cmd == "/s":
            # Request status from target agent
            target = None
            num_messages = 5

            if arg:
                # Parse: /status [target] [num]
                parts = arg.split()
                if parts:
                    # First arg might be a number or a target
                    if parts[0].isdigit():
                        num_messages = int(parts[0])
                    else:
                        target = self.node.resolve_target(parts[0])
                        if not target:
                            print(f"{self.YELLOW}Unknown target '{parts[0]}'. Use /list to see available nodes.{self.RESET}")
                            return True
                    if len(parts) > 1 and parts[1].isdigit():
                        num_messages = int(parts[1])

            if not target:
                target = self.default_target

            if not target:
                print(f"{self.YELLOW}No target set. Use /status <agent> or set a target first.{self.RESET}")
                return True

            # Send status request with full diagnostics
            status_msg = make_status_request(
                from_node=self.node_id,
                to_node=target,
                num_messages=num_messages,
                diagnostics=True,
            )
            await self.node._conn.send(status_msg)
            display = get_display_name(target)
            print(f"{self.DIM}→ Requested status from {display}{self.RESET}")

        elif cmd == "/context" or cmd == "/ctx":
            # Show user's own recent conversation history (full, no truncation)
            # If a view is set, show only that conversation's context
            num_entries = 7
            if arg and arg.isdigit():
                num_entries = int(arg)

            # If we have a current_view set, delegate to _show_conversation_context
            if self.current_view:
                self._show_conversation_context(self.current_view, num_entries)
                return True

            # Get entries from node history (no view filter - show all)
            entries = self.node.history[-num_entries:] if len(self.node.history) >= num_entries else self.node.history

            print()
            print(f"{self.CYAN}{'━' * 60}{self.RESET}")
            print(f"{self.BOLD}{self.CYAN}Your Context{self.RESET} {self.DIM}(last {len(entries)} of {len(self.node.history)} entries){self.RESET}")
            print(f"{self.CYAN}{'━' * 60}{self.RESET}")

            if not entries:
                print()
                print(f"{self.DIM}No messages in history{self.RESET}")
            else:
                for entry in entries:
                    msg = entry.message
                    direction = entry.direction
                    timestamp = msg.timestamp or ""

                    # Skip non-message types
                    if msg.type != MessageType.MESSAGE:
                        continue

                    # Parse timestamp for display (convert to local time)
                    ts_display = iso_to_local_time(timestamp) if timestamp else ""

                    # Color and label based on direction
                    to_node = msg.to_node or "unknown"
                    from_node = msg.from_node or "unknown"
                    if direction == "outgoing":
                        sender_color = self.YELLOW
                        sender_label = f"{self.nickname} → {get_display_name(to_node)}"
                    else:
                        sender_color = self.MAGENTA
                        sender_label = f"{get_display_name(from_node)} → {self.nickname}"

                    print()
                    print(f"{sender_color}{self.BOLD}{sender_label}{self.RESET} {self.DIM}{ts_display}{self.RESET}")

                    # Full content - no truncation
                    content = msg.content if msg.content and isinstance(msg.content, str) else str(msg.content or "")
                    # Indent content
                    for line in content.split("\n"):
                        print(f"  {line}")

            print()
            print(f"{self.CYAN}{'━' * 60}{self.RESET}")
            print()

        elif cmd == "/attach":
            # Attach to an agent to see real-time tool activity
            if not arg:
                if self._attached_agent:
                    display = get_display_name(self._attached_agent)
                    print(f"{self.DIM}Currently attached to: {display}{self.RESET}")
                else:
                    print(f"{self.YELLOW}Usage: /attach <agent_nickname>{self.RESET}")
                    print(f"{self.DIM}Attach to an agent to see tool calls as they happen{self.RESET}")
            else:
                # Resolve nickname to full node ID
                target = self._resolve_target(arg)
                if not target:
                    print(f"{self.RED}Unknown node: {arg}{self.RESET}")
                elif not target.startswith("agent:"):
                    print(f"{self.YELLOW}Can only attach to agents, not users or channels{self.RESET}")
                else:
                    self._attached_agent = target
                    display = get_display_name(target)
                    print(f"{self.GREEN}✓ Attached to {display}{self.RESET}")
                    print(f"{self.DIM}You'll see tool calls and results in real-time. Use /detach to stop.{self.RESET}")

        elif cmd == "/detach":
            # Detach from agent tool activity
            if self._attached_agent:
                display = get_display_name(self._attached_agent)
                self._attached_agent = None
                print(f"{self.GREEN}✓ Detached from {display}{self.RESET}")
            else:
                print(f"{self.DIM}Not attached to any agent{self.RESET}")

        elif cmd == "/channels" or cmd == "/ch":
            # Request channel list from router
            list_msg = make_channel_list(self.node_id)
            await self.node._conn.send(list_msg)
            print(f"{self.DIM}→ Requesting channel list...{self.RESET}")

        elif cmd == "/create":
            if not arg:
                print(f"{self.YELLOW}Usage: /create <channel_name> [description]{self.RESET}")
            else:
                parts = arg.split(maxsplit=1)
                channel_name = parts[0]
                description = parts[1] if len(parts) > 1 else ""
                create_msg = make_channel_create(self.node_id, channel_name, description)
                await self.node._conn.send(create_msg)
                self._known_channels.add(channel_name)
                print(f"{self.DIM}→ Creating channel #{channel_name}...{self.RESET}")

        elif cmd == "/join":
            if not arg:
                print(f"{self.YELLOW}Usage: /join <channel_name>{self.RESET}")
            else:
                channel_name = arg.strip()
                join_msg = make_channel_join(self.node_id, channel_name)
                await self.node._conn.send(join_msg)
                self._known_channels.add(channel_name)
                print(f"{self.DIM}→ Joining channel #{channel_name}...{self.RESET}")

        elif cmd == "/leave":
            if not arg:
                print(f"{self.YELLOW}Usage: /leave <channel_name>{self.RESET}")
            else:
                channel_name = arg.strip()
                leave_msg = make_channel_leave(self.node_id, channel_name)
                await self.node._conn.send(leave_msg)
                print(f"{self.DIM}→ Leaving channel #{channel_name}...{self.RESET}")

        elif cmd == "/members":
            if not arg:
                print(f"{self.YELLOW}Usage: /members <channel_name>{self.RESET}")
            else:
                channel_name = arg.strip()
                members_msg = make_channel_members(self.node_id, channel_name)
                await self.node._conn.send(members_msg)
                print(f"{self.DIM}→ Requesting members of #{channel_name}...{self.RESET}")

        elif cmd == "/delete":
            if not arg:
                print(f"{self.YELLOW}Usage: /delete <channel_name>{self.RESET}")
            else:
                channel_name = arg.strip()
                delete_msg = make_channel_delete(self.node_id, channel_name)
                await self.node._conn.send(delete_msg)
                print(f"{self.DIM}→ Deleting channel #{channel_name}...{self.RESET}")

        elif cmd == "/invite" or cmd == "/add":
            parts = arg.split(maxsplit=1) if arg else []
            if len(parts) < 2:
                print(f"{self.YELLOW}Usage: /invite <channel_name> <node_id>{self.RESET}")
                print(f"{self.DIM}Example: /invite research agent:coder:alice{self.RESET}")
            else:
                channel_name = parts[0].strip()
                node_id = parts[1].strip()
                # Resolve nickname to full node ID if needed
                if node_id.startswith("@"):
                    node_id = node_id[1:]
                if ":" not in node_id:
                    # Try to find in roster by nickname
                    resolved = self.node.resolve_target(node_id)
                    if resolved:
                        node_id = resolved
                invite_msg = make_channel_invite(self.node_id, channel_name, node_id)
                await self.node._conn.send(invite_msg)
                print(f"{self.DIM}→ Inviting {node_id} to #{channel_name}...{self.RESET}")

        elif cmd == "/view" or cmd == "/v":
            if not arg or arg.lower() == "all":
                self.current_view = None
                print(f"{self.DIM}View: showing all messages{self.RESET}")
            else:
                # Check if arg is a channel name (starts with # or is already channel: prefix)
                target = arg
                if arg.startswith("#"):
                    # Convert #channel to channel:channel
                    target = f"channel:{arg[1:]}"
                elif arg.startswith("channel:"):
                    target = arg
                elif ":" not in arg:
                    # Try to resolve as nickname
                    resolved = self.node.resolve_target(arg)
                    if resolved:
                        target = resolved
                    else:
                        # Try as channel name without #
                        target = f"channel:{arg}"

                if ":" in target:
                    self.current_view = target
                    self.default_target = target  # Auto-set target to match view
                    self._clear_unread(target)  # Mark as read
                    view_name = get_display_name(target)
                    print(f"{self.DIM}View: filtering to {view_name} (target set){self.RESET}")
                    self._show_conversation_context(target, 25)
                    await self._pull_todos()
                else:
                    print(f"{self.YELLOW}Could not resolve: {arg}{self.RESET}")

        elif cmd == "/inbox" or cmd == "/i":
            # Show conversations sorted by recent activity with unread counts
            limit = 5  # default
            if arg and arg.isdigit():
                limit = int(arg)
            self._show_inbox(limit=limit)

        elif cmd == "/conversations" or cmd == "/chats":
            # Alias for /inbox
            self._show_inbox()

        elif cmd == "/recent" or cmd == "/r":
            # Switch to the most recent received message's conversation
            # Show only last 2 messages (brief context). Use /v for full history.
            if not self._recent_conversations:
                print(f"{self.YELLOW}No recent conversations.{self.RESET}")
            else:
                partner = self._recent_conversations[0]
                self.current_view = partner
                self.default_target = partner
                self._clear_unread(partner)
                view_name = get_display_name(partner)
                print(f"{self.DIM}View: filtering to {view_name} (target set){self.RESET}")
                self._show_conversation_context(partner, 2)
                await self._pull_todos()

        elif cmd == "/watch" or cmd == "/w":
            if not arg:
                print(f"{self.YELLOW}Usage: /watch <nickname>{self.RESET}")
                return True
            target = self.node.resolve_target(arg.strip())
            if not target:
                print(f"{self.YELLOW}Unknown target '{arg.strip()}'. Use /list to see available nodes.{self.RESET}")
                return True
            await self._run_watch_mode(target)

        elif cmd == "/cc-usage" or cmd == "/usage":
            self._show_cc_usage()

        elif cmd == "/scratch" or cmd == "/sp":
            conv_id = self._get_current_conversation_id()
            if not conv_id:
                print(f"{self.YELLOW}No active conversation. Use /target or /view first.{self.RESET}")
                return True
            if arg:
                self._save_scratchpad_local(conv_id, arg)
                base = self._scratchpad_base_versions.get(conv_id, "")
                msg = make_scratchpad_set(self.node_id, conv_id, arg, base)
                await self.node._conn.send(msg)
                print(f"{self.DIM}Scratchpad saved and synced.{self.RESET}")
            else:
                text = self._load_scratchpad_local(conv_id)
                if text:
                    print(f"\n{self.BOLD}Scratchpad ({get_display_name(conv_id)}):{self.RESET}")
                    print(text)
                    print()
                else:
                    print(f"{self.DIM}Scratchpad empty for {get_display_name(conv_id)}{self.RESET}")

        elif cmd == "/todo" or cmd == "/td":
            await self._handle_todo_command(arg)

        elif cmd == "/calendar" or cmd == "/cal":
            await self._show_calendar()

        elif cmd == "/clear-draft" or cmd == "/cd":
            partner = self.current_view or self.default_target
            if partner and partner in self._drafts:
                del self._drafts[partner]
                print(f"{self.DIM}Draft cleared for {get_display_name(partner)}{self.RESET}")
            elif partner:
                print(f"{self.DIM}No draft saved for {get_display_name(partner)}{self.RESET}")
            else:
                print(f"{self.YELLOW}No active conversation{self.RESET}")

        elif cmd == "/help" or cmd == "/h":
            print()
            print(f"{self.BOLD}Commands:{self.RESET}")
            print(f"  {self.CYAN}/list, /ls{self.RESET}        - List connected nodes")
            print(f"  {self.CYAN}/inbox, /i{self.RESET}        - Show conversations (sorted by recent)")
            print(f"  {self.CYAN}/recent, /r{self.RESET}       - Switch to most recent message's conversation")
            print(f"  {self.CYAN}/1, /2, /3, ...{self.RESET}   - Quick-switch to conversation by inbox position")
            print(f"  {self.CYAN}/target, /t <node>{self.RESET} - Set default target (or show current)")
            print(f"  {self.CYAN}/to <node> msg{self.RESET}    - Send to specific node (one-off)")
            print(f"  {self.CYAN}/status, /s [node] [n]{self.RESET} - Get agent's recent context")
            print(f"  {self.CYAN}/watch, /w <node>{self.RESET}    - Live-stream agent activity (q/Esc to exit)")
            print(f"  {self.CYAN}/context, /ctx [n]{self.RESET} - Show your own recent context")
            print(f"  {self.CYAN}/view, /v <node|all>{self.RESET} - Filter to conversation or show all")
            print(f"  {self.CYAN}/clear{self.RESET}            - Clear screen")
            print()
            print(f"{self.BOLD}Channels:{self.RESET}")
            print(f"  {self.CYAN}/channels, /ch{self.RESET}    - List all channels")
            print(f"  {self.CYAN}/create <name> [desc]{self.RESET} - Create a channel")
            print(f"  {self.CYAN}/join <name>{self.RESET}      - Join a channel")
            print(f"  {self.CYAN}/leave <name>{self.RESET}     - Leave a channel")
            print(f"  {self.CYAN}/members <name>{self.RESET}   - List channel members")
            print(f"  {self.CYAN}/invite <ch> <node>{self.RESET} - Add a member to channel")
            print(f"  {self.CYAN}/delete <name>{self.RESET}    - Delete a channel")
            print()
            print(f"  {self.CYAN}/scratch, /sp [text]{self.RESET} - View or set scratchpad for current conversation")
            print(f"  {self.CYAN}/todo, /td{self.RESET}          - Show/toggle per-conversation todos")
            print(f"  {self.CYAN}/todo add <text> [--section name]{self.RESET} - Add a todo")
            print(f"  {self.CYAN}/todo done|start <n>{self.RESET} - Update todo status by visible number")
            print(f"  {self.CYAN}/calendar, /cal{self.RESET}    - Show today's calendar once")
            print(f"  {self.CYAN}/cc-usage, /usage{self.RESET}  - Show Claude Code plan usage")
            print(f"  {self.CYAN}/quit, /q{self.RESET}         - Disconnect")
            print()
            print(f"{self.BOLD}Aliases:{self.RESET}")
            print(f"  {self.CYAN}/conversations, /chats{self.RESET} - Same as /inbox")
            print()
            print(f"{self.BOLD}Shortcuts:{self.RESET}")
            print(f"  {self.CYAN}Ctrl+S{self.RESET}             - Send message")
            print(f"  {self.CYAN}Ctrl+Z{self.RESET}             - Toggle draft (save with text, restore when empty)")
            print(f"  {self.CYAN}Ctrl+T{self.RESET}             - Toggle todo panel")
            print(f"  {self.CYAN}Ctrl+G{self.RESET}             - Show today's calendar")
            print(f"  {self.CYAN}/clear-draft{self.RESET}       - Clear saved draft for current conversation")
            print()
            quiet_status = "on" if self._quiet_presence else "off"
            notif_status = "enabled" if self._notifications_enabled else "disabled"
            print(f"{self.DIM}Notifications: {notif_status} · Quiet presence: {quiet_status} (use --quiet / -q to suppress both){self.RESET}")
            print()

        else:
            # Check for numbered target switching (/1, /2, etc.)
            if cmd.startswith("/") and cmd[1:].isdigit():
                idx = int(cmd[1:]) - 1  # 1-indexed
                if 0 <= idx < len(self._recent_conversations):
                    partner = self._recent_conversations[idx]
                    self.current_view = partner
                    self.default_target = partner
                    self._clear_unread(partner)
                    view_name = get_display_name(partner)
                    print(f"{self.DIM}View: filtering to {view_name} (target set){self.RESET}")
                    self._show_conversation_context(partner, 10)
                    await self._pull_todos()
                else:
                    print(f"{self.YELLOW}No conversation at position {idx + 1}. Use /inbox to see conversations.{self.RESET}")
            else:
                print(f"{self.YELLOW}Unknown command:{self.RESET} {cmd}. Try /help")

        return True

    async def handle_input(self, user_input: str) -> bool:
        """
        Process user input.

        Returns:
            True to continue, False to quit
        """
        stripped = user_input.strip()

        if not stripped:
            return True

        # Handle commands
        if stripped.startswith("/"):
            return await self.handle_command(stripped)

        # Regular message - send to default target
        if not self.default_target:
            print(f"{self.YELLOW}No target set. Use /target <node_id> or /to <node_id> <message>{self.RESET}")
            return True

        await self.node.send(self.default_target, stripped)
        # Render outgoing message if it matches current view
        self._render_outgoing_message(self.default_target, stripped)

        return True

    async def message_receiver(self):
        """Background task to receive and display incoming messages."""
        while True:
            try:
                msg = await self._incoming_queue.get()

                if msg.type == MessageType.MESSAGE:
                    # If we're in confirmation mode, buffer the message for later
                    if self._in_confirmation:
                        self._confirm_message_buffer.append(msg)
                        continue

                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    # Use message timestamp (converted to local time) or fall back to now
                    timestamp = iso_to_local_time(msg.timestamp) if msg.timestamp else datetime.now().strftime("%H:%M:%S")

                    # Echoed message from another client (e.g., Android)?
                    # Render as outgoing and track under the recipient, not self.
                    if msg.from_node == self.node_id:
                        to_node = msg.to_node or ""
                        if self._outgoing_matches_view(to_node):
                            self._render_outgoing_message(to_node, content)
                        if to_node and to_node != self.node_id:
                            self._update_recent_conversations(to_node)
                        continue

                    # Check if this message was routed through a channel
                    # The router sends channel messages with to_node = "channel:name"
                    channel = None
                    if msg.to_node and msg.to_node.startswith("channel:"):
                        channel = msg.to_node.split(":", 1)[1]
                    elif msg.metadata and msg.metadata.get("channel"):
                        channel = msg.metadata.get("channel")
                    # Normalize channel partner to full address for consistent tracking
                    channel_addr = f"channel:{channel}" if channel else None

                    # Track message and assign ref number for reply threading
                    ref = self._track_message(msg.id, msg.from_node, content, channel_addr)

                    # Check if message matches current view filter
                    if self._matches_current_view(msg):
                        self.render_message(msg.from_node, content, timestamp, channel=channel, msg_ref=ref, in_reply_to=msg.in_reply_to)
                        conv_id = self._get_current_conversation_id()
                        if conv_id and self._todo_panel_visible.get(conv_id, False):
                            self._render_todo_panel(conv_id)
                    else:
                        # Track unread for messages outside current view
                        partner = channel_addr or msg.from_node
                        if partner:
                            self._unread_counts[partner] = self._unread_counts.get(partner, 0) + 1

                        # Show notification for messages outside current view
                        if not self._quiet_presence:
                            from_name = get_display_name(msg.from_node or "unknown")
                            unread = self._unread_counts.get(partner, 1)
                            if channel:
                                print(f"\n{self.DIM}[new message in #{channel} from {from_name}]{self.RESET} {self.YELLOW}({unread} unread){self.RESET}")
                            else:
                                print(f"\n{self.DIM}[new message from {from_name}]{self.RESET} {self.YELLOW}({unread} unread){self.RESET}")
                            self._print_prompt_hint()

                        # Send desktop notification
                        if self._notifications_enabled:
                            from_name = get_display_name(msg.from_node or "unknown")
                            preview = content[:100] + "..." if len(content) > 100 else content
                            preview = preview.replace("\n", " ")
                            if channel:
                                self._send_notification(f"#{channel}", f"{from_name}: {preview}")
                            else:
                                self._send_notification(f"Message from {from_name}", preview)

                    # Update recent conversations
                    partner = channel_addr or msg.from_node
                    if partner and partner != self.node_id:
                        self._update_recent_conversations(partner)

                elif msg.type == MessageType.CONFIRM_REQUEST:
                    # Queue for main loop to handle (can't prompt from background task)
                    self.render_confirm_request(msg)
                    await self._confirm_queue.put(msg)
                    # Hint to user they need to submit current input first
                    print(f"{self.DIM}(Press Ctrl+S to respond to confirmation){self.RESET}")
                    sys.stdout.flush()

                elif msg.type == MessageType.PRESENCE:
                    # If we're in confirmation mode, buffer the presence for later
                    if self._in_confirmation:
                        self._confirm_message_buffer.append(msg)
                        continue

                    # Update default_target/current_view if nickname rejoins with new node_id
                    content = msg.content if isinstance(msg.content, dict) else {}
                    event = content.get("event", "")
                    nickname = content.get("nickname", "").lower()
                    new_node_id = msg.from_node

                    if event == "join" and nickname and new_node_id:
                        # Check if current_view points to same nickname but old node_id
                        if self.current_view and self.current_view != new_node_id:
                            old_parsed = parse_node_id(self.current_view)
                            if old_parsed:
                                old_nick = (old_parsed[2] or old_parsed[1] or "").lower()
                                if old_nick == nickname:
                                    old_id = self.current_view
                                    logger.info(f"Updating current_view: {old_id} -> {new_node_id}")
                                    # Migrate draft to new key
                                    if old_id in self._drafts:
                                        self._drafts[new_node_id] = self._drafts.pop(old_id)
                                    self.current_view = new_node_id

                        # Check if default_target points to same nickname but old node_id
                        if self.default_target and self.default_target != new_node_id:
                            old_parsed = parse_node_id(self.default_target)
                            if old_parsed:
                                old_nick = (old_parsed[2] or old_parsed[1] or "").lower()
                                if old_nick == nickname:
                                    old_id = self.default_target
                                    logger.info(f"Updating default_target: {old_id} -> {new_node_id}")
                                    # Migrate draft to new key
                                    if old_id in self._drafts and new_node_id not in self._drafts:
                                        self._drafts[new_node_id] = self._drafts.pop(old_id)
                                    self.default_target = new_node_id

                    # Display presence notification
                    self.render_presence(msg)

                elif msg.type == MessageType.STATUS_RESPONSE:
                    # If we're in watch mode, route to watch queue
                    if self._watch_target and msg.from_node == self._watch_target:
                        await self._watch_queue.put(msg)
                        continue

                    # If we're in confirmation mode, buffer for later
                    if self._in_confirmation:
                        self._confirm_message_buffer.append(msg)
                        continue

                    self.render_status_response(msg)
                    self._print_prompt_hint()

                elif msg.type == MessageType.TOOL_ACTIVITY:
                    # Real-time tool activity from attached agent
                    if self._attached_agent and msg.from_node == self._attached_agent:
                        self._render_tool_activity(msg)

                elif msg.type == MessageType.STATUS:
                    # Phase/status updates from attached agent
                    if self._attached_agent and msg.from_node == self._attached_agent:
                        self._render_status_update(msg)

                elif msg.type == MessageType.CONTROL:
                    content = msg.content if isinstance(msg.content, dict) else {}
                    action = content.get("action", "unknown")

                    # Handle channel responses
                    if action == ControlAction.ACK.value:
                        status = content.get("status", "")
                        if status in ("channel_created", "channel_deleted", "joined", "left"):
                            channel = content.get("channel", "")
                            print(f"\n{self.GREEN}✓ {status.replace('_', ' ').title()}: #{channel}{self.RESET}")
                            if channel and status != "channel_deleted":
                                self._known_channels.add(channel)
                            elif channel and status == "channel_deleted":
                                self._known_channels.discard(channel)
                            self._print_prompt_hint()
                    elif action == ControlAction.CHANNEL_INVITE.value:
                        status = content.get("status", "")
                        channel = content.get("channel_name", "")
                        node = content.get("node_id", "")
                        if status == "invited":
                            print(f"\n{self.GREEN}✓ Invited {node} to #{channel}{self.RESET}")
                        else:
                            print(f"\n{self.YELLOW}Invite: {status}{self.RESET}")
                        if "error" in content:
                            print(f"\n{self.RED}✗ {content.get('error', 'Unknown error')}{self.RESET}")
                        self._print_prompt_hint()
                    elif action == ControlAction.CHANNEL_LIST.value:
                        channels = content.get("channels", [])
                        for ch in channels:
                            name = ch.get("name", "?")
                            if name and name != "?":
                                self._known_channels.add(name)
                    elif action == ControlAction.CHANNEL_MEMBERS.value:
                        channel = content.get("channel_name", "?")
                        members = content.get("members", [])
                        print(f"\n{self.BOLD}Members of #{channel} ({len(members)}):{self.RESET}")
                        if not members:
                            print(f"  {self.DIM}No members{self.RESET}")
                        else:
                            for entry in members:
                                if isinstance(entry, dict):
                                    member = entry.get("node_id", str(entry))
                                    online = entry.get("online", False)
                                    status = f"{self.GREEN}●{self.RESET}" if online else f"{self.DIM}○{self.RESET}"
                                else:
                                    member = str(entry)
                                    status = ""
                                display = get_display_name(member)
                                if member.startswith("agent:"):
                                    color = self.MAGENTA
                                elif member.startswith("user:"):
                                    color = self.YELLOW
                                else:
                                    color = self.WHITE
                                print(f"  {status} {color}{display}{self.RESET} {self.DIM}({member}){self.RESET}")
                        self._print_prompt_hint()
                    elif action == ControlAction.HISTORY_RESPONSE.value:
                        # Handle history sync response
                        self._handle_history_response(content)
                    elif action == ControlAction.SCRATCHPAD_RESPONSE.value:
                        self._handle_scratchpad_response(content)
                    elif action == ControlAction.TODO_RESPONSE.value:
                        self._handle_todo_response(content)
                    elif action == ControlAction.CALENDAR_RESPONSE.value:
                        pass
                    elif action != "list_nodes":  # Don't show list_nodes responses here
                        print(f"\n{self.DIM}[control:{action}]{self.RESET}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message receiver: {e}")

    async def _handle_confirm_input(self, msg: Message) -> None:
        """
        Handle user input for a confirmation request.

        Called from main loop (not background task) to avoid prompt_toolkit conflicts.
        """
        # Set flag so message_receiver buffers incoming messages
        self._in_confirmation = True

        # Get user response using prompt_toolkit
        try:
            while True:
                response = await self.session.prompt_async(
                    "Confirm? [y/n]: ",
                    multiline=False,
                )
                response = response.strip().lower()
                if response in ("y", "yes"):
                    confirmed = True
                    break
                elif response in ("n", "no"):
                    confirmed = False
                    break
                else:
                    print(f"{self.YELLOW}Please enter 'y' or 'n'{self.RESET}")

        except (EOFError, KeyboardInterrupt):
            confirmed = False
            print(f"\n{self.RED}Cancelled - rejecting{self.RESET}")

        finally:
            # Clear flag
            self._in_confirmation = False

        # Send response
        response_msg = make_confirm_response(
            from_node=self.node_id,
            to_node=msg.from_node,
            in_reply_to=msg.id,
            confirmed=confirmed,
        )
        await self.node._conn.send(response_msg)

        # Show result
        if confirmed:
            print(f"{self.GREEN}Confirmed - executing tool{self.RESET}")
        else:
            print(f"{self.RED}Rejected - tool aborted{self.RESET}")
        print()

        # Flush any buffered messages that arrived during confirmation
        if self._confirm_message_buffer:
            for i, buffered_msg in enumerate(self._confirm_message_buffer):
                is_last = (i == len(self._confirm_message_buffer) - 1)
                if buffered_msg.type == MessageType.MESSAGE:
                    content = buffered_msg.content if isinstance(buffered_msg.content, str) else str(buffered_msg.content)
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    channel = buffered_msg.metadata.get("channel") if buffered_msg.metadata else None
                    self.render_message(buffered_msg.from_node, content, timestamp, show_prompt=is_last, channel=channel)
                elif buffered_msg.type == MessageType.PRESENCE:
                    self.render_presence(buffered_msg, show_prompt=is_last)
                elif buffered_msg.type == MessageType.STATUS_RESPONSE:
                    self.render_status_response(buffered_msg)
                    if is_last:
                        self._print_prompt_hint()
            self._confirm_message_buffer.clear()

    async def _request_history_sync(self) -> None:
        """Request message history sync from the router."""
        if not self.node:
            return

        msg = make_history_sync(
            from_node=self.node_id,
            since=None,  # Always request full history
            limit=500,
        )
        await self.node._conn.send(msg)

    def _handle_history_response(self, content: dict) -> None:
        """Handle HISTORY_RESPONSE from router."""
        from mesh.node import HistoryEntry

        messages = content.get("messages", [])
        read_receipts = content.get("read_receipts", {})

        if not self.node or not messages:
            return

        # Add synced messages to node's history
        existing_ids = {entry.message.id for entry in self.node.history}
        added = 0

        for msg_dict in messages:
            if msg_dict.get("id") in existing_ids:
                continue  # Skip duplicates
            try:
                msg = Message(
                    id=msg_dict.get("id", ""),
                    type=MessageType(msg_dict.get("type", "message")),
                    from_node=msg_dict.get("from_node", ""),
                    to_node=msg_dict.get("to_node", ""),
                    content=msg_dict.get("content", ""),
                    timestamp=msg_dict.get("timestamp", ""),
                    in_reply_to=msg_dict.get("in_reply_to"),
                )
                direction = "outgoing" if msg.from_node == self.node_id else "incoming"
                self.node._history.append(HistoryEntry(message=msg, direction=direction))
                added += 1
            except Exception as e:
                logger.warning(f"Error adding synced message to history: {e}")

        # Sort history by timestamp
        if added > 0:
            self.node._history.sort(key=lambda e: e.message.timestamp)

        # Update roster from history - add any nodes we've communicated with
        self._update_roster_from_history(messages)

    # ── Scratchpad sync helpers ──────────────────────────────────────────

    def _scratchpad_path(self, conv_id: str) -> str:
        safe = conv_id.replace("/", "_").replace(":", "_")
        return os.path.join(self._scratchpad_dir, safe)

    def _save_scratchpad_local(self, conv_id: str, text: str) -> None:
        try:
            with open(self._scratchpad_path(conv_id), "w") as f:
                f.write(text)
        except OSError as e:
            logger.warning(f"Failed to save scratchpad locally: {e}")

    def _load_scratchpad_local(self, conv_id: str) -> str:
        path = self._scratchpad_path(conv_id)
        if not os.path.exists(path):
            return ""
        try:
            with open(path) as f:
                return f.read()
        except OSError:
            return ""

    def _get_current_conversation_id(self) -> str | None:
        partner = self.current_view or self.default_target
        if not partner:
            return None
        from mesh.storage import MessageStore
        return MessageStore.compute_conversation_id(self.node_id, partner)

    def _handle_scratchpad_response(self, content: dict) -> None:
        notes = content.get("notes", {})
        for conv_id, note in notes.items():
            server_ts = note.get("updated_at", "")
            server_text = note.get("content", "")
            local_ts = self._scratchpad_base_versions.get(conv_id, "")
            if server_ts > local_ts:
                self._scratchpad_base_versions[conv_id] = server_ts
                self._save_scratchpad_local(conv_id, server_text)
        accepted = content.get("accepted")
        if accepted is False:
            conv_id = content.get("conversation_id", "")
            server_note = content.get("server_state", {})
            if server_note:
                server_ts = server_note.get("updated_at", "")
                server_text = server_note.get("content", "")
                self._scratchpad_base_versions[conv_id] = server_ts
                self._save_scratchpad_local(conv_id, server_text)
                print(f"\n{self.YELLOW}Scratchpad conflict — accepted server version.{self.RESET}")
                self._print_prompt_hint()
        elif accepted is True:
            conv_id = content.get("conversation_id", "")
            server_state = content.get("server_state", {})
            if server_state:
                self._scratchpad_base_versions[conv_id] = server_state.get("updated_at", "")

    async def _pull_scratchpad(self, conv_id: str | None = None) -> None:
        if not self.node or not self.node._conn:
            return
        conv_ids = [conv_id] if conv_id else None
        msg = make_scratchpad_get(self.node_id, conv_ids)
        await self.node._conn.send(msg)

    # ── Conversation todo sync helpers ───────────────────────────────────

    def _todo_path(self, conv_id: str) -> str:
        safe = conv_id.replace("/", "_").replace(":", "_")
        return os.path.join(self._todo_dir, f"{safe}.json")

    def _save_todos_local(self, conv_id: str) -> None:
        payload = {
            "conversation_id": conv_id,
            "panel_visible": self._todo_panel_visible.get(conv_id, False),
            "section_order": self._todo_section_order.get(conv_id, []),
            "todos": self._todo_cache.get(conv_id, []),
        }
        try:
            with open(self._todo_path(conv_id), "w") as f:
                json.dump(payload, f, indent=2)
        except OSError as e:
            logger.warning(f"Failed to save todos locally: {e}")

    def _load_todos_local(self, conv_id: str) -> None:
        path = self._todo_path(conv_id)
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        todos = payload.get("todos", [])
        if isinstance(todos, list):
            self._todo_cache[conv_id] = todos
        section_order = payload.get("section_order", [])
        if isinstance(section_order, list):
            self._todo_section_order[conv_id] = [
                str(section).strip()
                for section in section_order
                if str(section).strip()
            ]
        self._todo_panel_visible[conv_id] = bool(payload.get("panel_visible", False))

    @staticmethod
    def _todo_section(todo: dict) -> str | None:
        section = todo.get("section")
        if section is None:
            return None
        clean = str(section).strip()
        return clean or None

    def _todo_sort_key(self, todo: dict) -> tuple[int, int, str]:
        status_rank = {"in_progress": 0, "open": 1, "done": 2, "cancelled": 3}
        status = str(todo.get("status", "open"))
        try:
            position = int(todo.get("position", 0))
        except (TypeError, ValueError):
            position = 0
        return (status_rank.get(status, 9), position, str(todo.get("created_at", "")))

    def _todo_section_display_order(self, conv_id: str, todos: list[dict]) -> list[str]:
        present: dict[str, str] = {}
        for todo in todos:
            section = self._todo_section(todo)
            if section is not None:
                present.setdefault(section.casefold(), section)

        ordered: list[str] = []
        seen: set[str] = set()
        for configured in self._todo_section_order.get(conv_id, []):
            key = configured.casefold()
            if key in present and key not in seen:
                ordered.append(present[key])
                seen.add(key)

        remaining = [
            section for key, section in present.items()
            if key not in seen
        ]
        ordered.extend(sorted(remaining, key=str.casefold))
        return ordered

    def _grouped_todos_for_display(self, conv_id: str) -> list[tuple[str | None, list[dict]]]:
        todos = self._todo_cache.get(conv_id, [])
        unsectioned = [
            todo for todo in todos
            if self._todo_section(todo) is None
        ]
        groups: list[tuple[str | None, list[dict]]] = []
        if unsectioned:
            groups.append((None, sorted(unsectioned, key=self._todo_sort_key)))

        for section in self._todo_section_display_order(conv_id, todos):
            section_items = [
                todo for todo in todos
                if self._todo_section(todo) is not None
                and self._todo_section(todo).casefold() == section.casefold()
            ]
            if section_items:
                groups.append((section, sorted(section_items, key=self._todo_sort_key)))
        return groups

    def _sorted_todos_for_display(self, conv_id: str) -> list[dict]:
        return [
            todo
            for _, group in self._grouped_todos_for_display(conv_id)
            for todo in group
        ]

    def _parse_todo_add_args(self, rest: str) -> dict:
        try:
            tokens = shlex.split(rest)
        except ValueError as e:
            raise ValueError(f"invalid todo arguments: {e}") from e

        text_parts: list[str] = []
        section: str | None = None
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token == "--section":
                idx += 1
                if idx >= len(tokens):
                    raise ValueError("--section requires a value")
                section = tokens[idx].strip()
            elif token.startswith("--section="):
                section = token.split("=", 1)[1].strip()
            else:
                text_parts.append(token)
            idx += 1

        text = " ".join(text_parts).strip()
        if not text:
            raise ValueError("todo text is required")
        payload = {"text": text}
        if section is not None:
            payload["section"] = section
        return payload

    def _resolve_todo_ref(self, conv_id: str, ref: str) -> str | None:
        ref = ref.strip()
        if not ref:
            return None
        if ref.isdigit():
            return self._todo_ref_map.get(conv_id, {}).get(int(ref))
        return ref

    def _todo_status_prefix(self, status: str) -> str:
        return {
            "open": "[ ]",
            "in_progress": "[>]",
            "done": "[x]",
            "cancelled": "[-]",
        }.get(status, "[ ]")

    def _render_todo_panel(self, conv_id: str | None = None, force: bool = False) -> None:
        conv_id = conv_id or self._get_current_conversation_id()
        if not conv_id:
            return
        if not force and not self._todo_panel_visible.get(conv_id, False):
            return
        groups = self._grouped_todos_for_display(conv_id)
        total_todos = sum(len(group) for _, group in groups)
        self._todo_ref_map[conv_id] = {}

        title = f"Todos ({get_display_name(conv_id)})"
        print(f"\n{self.CYAN}╭─ {title} {'─' * max(1, 58 - len(title))}{self.RESET}")
        if not total_todos:
            print(f"{self.CYAN}│{self.RESET} {self.DIM}No todos. Use /todo add <text> [--section name].{self.RESET}")
        else:
            idx = 1
            for section_idx, (section, section_todos) in enumerate(groups):
                if idx > 20:
                    break
                if section_idx > 0:
                    # Blank line between section groups
                    print(f"{self.CYAN}│{self.RESET}")
                if section is not None:
                    header = f"── {section} "
                    line = header + "─" * max(1, 58 - len(header))
                    print(
                        f"{self.CYAN}│{self.RESET} "
                        f"{self.CYAN}{line}{self.RESET}"
                    )
                for todo in section_todos:
                    if idx > 20:
                        break
                    todo_id = str(todo.get("id", ""))
                    self._todo_ref_map[conv_id][idx] = todo_id
                    text = str(todo.get("text", "")).replace("\n", " ")
                    if len(text) > 88:
                        text = text[:85] + "..."
                    status = str(todo.get("status", "open"))
                    prefix = self._todo_status_prefix(status)
                    status_label = status.replace("_", "-")
                    print(
                        f"{self.CYAN}│{self.RESET} "
                        f"{self.DIM}{idx:>2}.{self.RESET} {prefix} "
                        f"{text} {self.DIM}({status_label}){self.RESET}"
                    )
                    idx += 1
            if total_todos > 20:
                print(f"{self.CYAN}│{self.RESET} {self.DIM}... {total_todos - 20} more{self.RESET}")
        print(f"{self.CYAN}╰{'─' * 62}{self.RESET}")
        self._print_prompt_hint()

    def _todo_bottom_toolbar(self):
        conv_id = self._get_current_conversation_id() if hasattr(self, "_todo_cache") else None
        if not conv_id:
            return ""
        if conv_id not in self._todo_cache:
            self._load_todos_local(conv_id)
        todos = self._todo_cache.get(conv_id, [])
        if not todos and not self._todo_panel_visible.get(conv_id, False):
            return ""
        counts = {"open": 0, "in_progress": 0, "done": 0, "cancelled": 0}
        for todo in todos:
            status = str(todo.get("status", "open"))
            counts[status] = counts.get(status, 0) + 1
        visible = "shown" if self._todo_panel_visible.get(conv_id, False) else "hidden"
        return (
            f" Todos: {counts.get('open', 0)} open · "
            f"{counts.get('in_progress', 0)} in-progress · "
            f"{counts.get('done', 0)} done · panel {visible} (Ctrl-T, /todo)"
        )

    def _toggle_todo_panel_from_keybinding(self) -> None:
        conv_id = self._get_current_conversation_id()
        if not conv_id:
            print(f"\n{self.YELLOW}No active conversation. Use /target or /view first.{self.RESET}")
            return
        self._load_todos_local(conv_id)
        visible = not self._todo_panel_visible.get(conv_id, False)
        self._todo_panel_visible[conv_id] = visible
        self._save_todos_local(conv_id)
        if visible:
            asyncio.create_task(self._pull_todos(conv_id))
        else:
            print(f"\n{self.DIM}Todo panel hidden for {get_display_name(conv_id)}{self.RESET}")
            self._print_prompt_hint()

    async def _pull_todos(self, conv_id: str | None = None) -> None:
        if not self.node or not self.node._conn:
            return
        conv_id = conv_id or self._get_current_conversation_id()
        if not conv_id:
            return
        self._load_todos_local(conv_id)
        msg = make_todo_get(self.node_id, [conv_id], include_done=True)
        await self.node._conn.send(msg)

    async def _send_todo_mutation(
        self,
        conv_id: str,
        op: str,
        payload: dict,
        expected_version: int | None = None,
    ) -> None:
        if not self.node or not self.node._conn:
            print(f"{self.RED}Not connected to router.{self.RESET}")
            return
        msg = make_todo_mutate(
            self.node_id,
            conv_id,
            op,
            payload=payload,
            expected_version=expected_version,
        )
        await self.node._conn.send(msg)

    def _handle_todo_response(self, content: dict) -> None:
        todos_by_conv = content.get("todos", {})
        section_order_by_conv = content.get("section_order", {})
        changed_conversations: set[str] = set()
        if isinstance(todos_by_conv, dict):
            for conv_id, todos in todos_by_conv.items():
                if isinstance(todos, list):
                    self._todo_cache[conv_id] = todos
                    changed_conversations.add(conv_id)
        if isinstance(section_order_by_conv, dict):
            for conv_id, section_order in section_order_by_conv.items():
                if isinstance(section_order, list):
                    self._todo_section_order[conv_id] = [
                        str(section).strip()
                        for section in section_order
                        if str(section).strip()
                    ]
                    changed_conversations.add(conv_id)
        for conv_id in changed_conversations:
            if conv_id in self._todo_cache:
                self._save_todos_local(conv_id)
        if content.get("accepted") is False:
            error = content.get("error")
            if error:
                print(f"\n{self.YELLOW}Todo update rejected: {error}{self.RESET}")
            else:
                print(f"\n{self.YELLOW}Todo conflict — refreshed from server.{self.RESET}")
        conv_id = content.get("conversation_id") or self._get_current_conversation_id()
        if conv_id and self._todo_panel_visible.get(conv_id, False):
            self._render_todo_panel(conv_id, force=True)
        else:
            self._print_prompt_hint()

    async def _handle_todo_command(self, arg: str) -> None:
        conv_id = self._get_current_conversation_id()
        if not conv_id:
            print(f"{self.YELLOW}No active conversation. Use /target or /view first.{self.RESET}")
            return
        self._load_todos_local(conv_id)

        if not arg:
            self._todo_panel_visible[conv_id] = True
            self._save_todos_local(conv_id)
            await self._pull_todos(conv_id)
            return

        parts = arg.split(maxsplit=1)
        sub = parts[0].strip().lower()
        rest = parts[1] if len(parts) > 1 else ""

        if sub == "panel":
            mode = rest.strip().lower() or "toggle"
            current = self._todo_panel_visible.get(conv_id, False)
            if mode in {"on", "show", "visible"}:
                visible = True
            elif mode in {"off", "hide", "hidden"}:
                visible = False
            elif mode == "toggle":
                visible = not current
            else:
                print(f"{self.YELLOW}Usage: /todo panel [on|off|toggle]{self.RESET}")
                return
            self._todo_panel_visible[conv_id] = visible
            self._save_todos_local(conv_id)
            if visible:
                await self._pull_todos(conv_id)
            else:
                print(f"{self.DIM}Todo panel hidden for {get_display_name(conv_id)}{self.RESET}")
            return

        if sub in {"list", "ls"}:
            await self._pull_todos(conv_id)
            return

        if sub == "add":
            if not rest.strip():
                print(f"{self.YELLOW}Usage: /todo add <text> [--section name]{self.RESET}")
                return
            try:
                payload = self._parse_todo_add_args(rest)
            except ValueError as e:
                print(f"{self.YELLOW}Usage: /todo add <text> [--section name] ({e}){self.RESET}")
                return
            await self._send_todo_mutation(conv_id, "add", payload)
            return

        if sub in {"done", "start", "reopen", "rm", "remove", "delete"}:
            if not rest.strip():
                print(f"{self.YELLOW}Usage: /todo {sub} <n|id>{self.RESET}")
                return
            todo_id = self._resolve_todo_ref(conv_id, rest.split()[0])
            if not todo_id:
                print(f"{self.YELLOW}Unknown todo reference: {rest.split()[0]}{self.RESET}")
                return
            if sub == "done":
                await self._send_todo_mutation(conv_id, "update", {"todo_id": todo_id, "status": "done"})
            elif sub == "start":
                await self._send_todo_mutation(conv_id, "update", {"todo_id": todo_id, "status": "in_progress"})
            elif sub == "reopen":
                await self._send_todo_mutation(conv_id, "update", {"todo_id": todo_id, "status": "open"})
            else:
                await self._send_todo_mutation(conv_id, "remove", {"todo_id": todo_id})
            return

        if sub == "status":
            status_parts = rest.split(maxsplit=1)
            if len(status_parts) != 2 or status_parts[1] not in {"open", "in_progress", "done", "cancelled"}:
                print(f"{self.YELLOW}Usage: /todo status <n|id> open|in_progress|done|cancelled{self.RESET}")
                return
            todo_id = self._resolve_todo_ref(conv_id, status_parts[0])
            if not todo_id:
                print(f"{self.YELLOW}Unknown todo reference: {status_parts[0]}{self.RESET}")
                return
            await self._send_todo_mutation(
                conv_id, "update", {"todo_id": todo_id, "status": status_parts[1]}
            )
            return

        if sub == "edit":
            edit_parts = rest.split(maxsplit=1)
            if len(edit_parts) != 2:
                print(f"{self.YELLOW}Usage: /todo edit <n|id> <text>{self.RESET}")
                return
            todo_id = self._resolve_todo_ref(conv_id, edit_parts[0])
            if not todo_id:
                print(f"{self.YELLOW}Unknown todo reference: {edit_parts[0]}{self.RESET}")
                return
            await self._send_todo_mutation(
                conv_id, "update", {"todo_id": todo_id, "text": edit_parts[1].strip()}
            )
            return

        if sub == "clear-done":
            await self._send_todo_mutation(conv_id, "clear_done", {})
            return

        print(
            f"{self.YELLOW}Usage: /todo [panel|list|add|start|done|reopen|status|edit|rm|clear-done]{self.RESET}"
        )

    def _calendar_account_events(self, account: str, date_str: str) -> list[dict]:
        """Fetch one account's events for the calendar panel."""
        result = subprocess.run(
            [
                "mesh-tool",
                "calendar_list_on_date",
                "--date",
                date_str,
                "--timezone",
                "America/Chicago",
                "--account",
                account,
            ],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        output = (result.stdout or "").strip()
        error = (result.stderr or "").strip()
        if result.returncode != 0:
            raise RuntimeError(error or output or f"mesh-tool exited {result.returncode}")
        if output.startswith("Error:"):
            raise RuntimeError(output)
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"unexpected calendar output: {output[:160]}") from e
        if not isinstance(parsed, list):
            raise RuntimeError(f"unexpected calendar payload: {type(parsed).__name__}")
        stamped = []
        for event in parsed:
            if isinstance(event, dict):
                item = dict(event)
                item["_account"] = account
                stamped.append(item)
        return stamped

    def _fetch_calendar_events_for_date(self, date_str: str) -> tuple[list[dict], list[str]]:
        events: list[dict] = []
        errors: list[str] = []
        for account in ("work", "personal"):
            try:
                events.extend(self._calendar_account_events(account, date_str))
            except Exception as e:
                errors.append(f"{account}: {e}")
        return sorted(events, key=self._calendar_sort_key), errors

    @staticmethod
    def _calendar_sort_key(event: dict) -> tuple[str, str]:
        start = event.get("start") if isinstance(event.get("start"), dict) else {}
        value = start.get("dateTime") or start.get("date") or ""
        return (str(value), str(event.get("summary", "")))

    @staticmethod
    def _calendar_time_label(event: dict) -> str:
        start = event.get("start") if isinstance(event.get("start"), dict) else {}
        if start.get("date") and not start.get("dateTime"):
            return "All day"
        raw = start.get("dateTime")
        if not raw:
            return "Unknown"
        try:
            parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone()
            return parsed.strftime("%I:%M %p").lstrip("0")
        except ValueError:
            return str(raw)

    def _render_calendar_panel(
        self,
        day_label: str,
        events: list[dict],
        errors: list[str],
    ) -> None:
        title = f"Calendar: {day_label}"
        print(f"\n{self.CYAN}╭─ {title} {'─' * max(1, 58 - len(title))}{self.RESET}")
        if not events:
            print(f"{self.CYAN}│{self.RESET} {self.DIM}No events today.{self.RESET}")
        else:
            for event in events[:30]:
                when = self._calendar_time_label(event)
                account = str(event.get("_account", "")).strip()
                summary = str(event.get("summary") or "(untitled)").replace("\n", " ")
                location = str(event.get("location") or "").replace("\n", " ").strip()
                if len(summary) > 70:
                    summary = summary[:67] + "..."
                suffix = f" ({account})" if account else ""
                if location:
                    loc = location[:40] + "..." if len(location) > 43 else location
                    suffix += f" {self.DIM}@ {loc}{self.RESET}"
                print(
                    f"{self.CYAN}│{self.RESET} "
                    f"{when:>8}  {summary}{suffix}"
                )
            if len(events) > 30:
                print(f"{self.CYAN}│{self.RESET} {self.DIM}... {len(events) - 30} more{self.RESET}")
            print(f"{self.CYAN}│{self.RESET} {self.DIM}({len(events)} event{'s' if len(events) != 1 else ''}){self.RESET}")
        for error in errors:
            print(f"{self.CYAN}│{self.RESET} {self.YELLOW}{error}{self.RESET}")
        print(f"{self.CYAN}╰{'─' * 62}{self.RESET}")

    async def _show_calendar(self) -> None:
        today = datetime.now().date()
        date_str = today.isoformat()
        day_label = today.strftime("%a %Y-%m-%d")
        try:
            events, errors = await asyncio.to_thread(
                self._fetch_calendar_events_for_date,
                date_str,
            )
        except Exception as e:
            events, errors = [], [str(e)]
        self._render_calendar_panel(day_label, events, errors)
        self._print_prompt_hint()

    def _update_roster_from_history(self, messages: list[dict]) -> None:
        """Add nodes from message history to roster (marked as offline)."""
        if not self.node:
            return

        seen_nodes: set[str] = set()

        # Collect unique node IDs from synced messages
        for msg_dict in messages:
            from_node = msg_dict.get("from_node", "")
            to_node = msg_dict.get("to_node", "")

            # Add participants (but not channels and not ourselves)
            if from_node and from_node != self.node_id and not is_channel_address(from_node):
                seen_nodes.add(from_node)
            if to_node and to_node != self.node_id and not is_channel_address(to_node):
                seen_nodes.add(to_node)

        # Also scan existing in-memory history
        for entry in self.node.history:
            msg = entry.message
            if msg.from_node and msg.from_node != self.node_id and not is_channel_address(msg.from_node):
                seen_nodes.add(msg.from_node)
            if msg.to_node and msg.to_node != self.node_id and not is_channel_address(msg.to_node):
                seen_nodes.add(msg.to_node)

        # Add nodes to roster if not already present
        for node_id in seen_nodes:
            nick = get_display_name(node_id)
            nick_lower = nick.lower()
            if nick_lower not in self.node._roster:
                # Determine node type from ID prefix
                parsed = parse_node_id(node_id)
                if parsed:
                    node_type = parsed[0]  # First part is the type
                else:
                    node_type = "unknown"

                self.node._roster[nick_lower] = RosterEntry(
                    node_id=node_id,
                    nickname=nick,
                    node_type=node_type,
                    description="",  # No description available from history
                )

    async def run(self):
        """Main TUI loop."""
        # Create node with custom message handler
        persist = not self.fresh or bool(self.history_file)
        self.node = TUIUserNode(
            self.config,
            self._incoming_queue,
            nickname=self.nickname,
            history_file=self.history_file,
            persist=persist,
        )

        # Load history unless starting fresh
        if persist:
            loaded = self.node.load_history()
            if loaded > 0:
                print(f"{self.GREEN}Resumed {loaded} history entries from {self.node.history_file}{self.RESET}")

        # Connect to router
        try:
            await self.node.connect()
        except Exception as e:
            print(f"{self.RED}Failed to connect:{self.RESET} {e}")
            return

        self.show_header()
        if self.config.ws_url:
            print(f"{self.GREEN}Connected as {self.BOLD}{self.nickname}{self.RESET}{self.GREEN} to router at {self.config.ws_url}{self.RESET}")
        else:
            print(f"{self.GREEN}Connected as {self.BOLD}{self.nickname}{self.RESET}{self.GREEN} to router at {self.config.router_host}:{self.config.router_port}{self.RESET}")
        if persist:
            print(f"{self.DIM}Session persistence enabled: {self.node.history_file}{self.RESET}")

        # Request history sync from router (always full sync - no since timestamp)
        await self._request_history_sync()
        await self._pull_scratchpad()
        await self._pull_todos()

        # Start background tasks
        receive_task = asyncio.create_task(self.node.receive_loop())
        display_task = asyncio.create_task(self.message_receiver())

        # Handle shutdown
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def shutdown():
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, shutdown)
            except NotImplementedError:
                pass  # Windows

        # Main input loop
        try:
            with patch_stdout(raw=True):
                while not stop_event.is_set():
                    try:
                        # Check for pending confirmation requests first
                        try:
                            confirm_msg = self._confirm_queue.get_nowait()
                            await self._handle_confirm_input(confirm_msg)
                            continue  # Check for more confirms before regular input
                        except asyncio.QueueEmpty:
                            pass

                        user_input = await self.get_user_input()
                        if not await self.handle_input(user_input):
                            break
                    except EOFError:
                        break
                    except asyncio.CancelledError:
                        break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            print(f"\n{self.BOLD}{self.RED}Disconnecting...{self.RESET}")
            await self.node.disconnect()
            receive_task.cancel()
            display_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
            try:
                await display_task
            except asyncio.CancelledError:
                pass


class TUIUserNode(UserNode):
    """User node that forwards messages to the TUI queue."""

    def __init__(
        self,
        config: NodeConfig,
        queue: asyncio.Queue,
        nickname: str | None = None,
        history_file: str | None = None,
        persist: bool = False,
    ):
        super().__init__(config, nickname=nickname, history_file=history_file, persist=persist)
        self._tui_queue = queue
        # Register a confirmation callback that returns None to defer to TUI's async handling
        self.on_confirm_request(self._defer_confirm_to_tui)

    def _defer_confirm_to_tui(self, msg: Message) -> None:
        """Defer confirmation handling to the TUI's async queue-based flow.

        Returns None to signal that the base class should NOT send an auto-response.
        The TUI will handle prompting the user and sending the response.
        """
        return None

    async def on_message(self, msg: Message) -> None:
        """Forward messages to TUI for display, after UserNode processing."""
        # Let UserNode handle roster updates, etc.
        await super().on_message(msg)
        # The base Node skips adding own-from_node messages to history (to
        # prevent agent self-loops), but for multi-device user sync the TUI
        # needs echoed messages so the inbox stays current.
        if msg.type == MessageType.MESSAGE and msg.from_node == self.node_id:
            from mesh.node import HistoryEntry
            self._history.append(HistoryEntry(message=msg, direction="outgoing"))
            self.schedule_save()
        # Forward all messages to TUI queue
        await self._tui_queue.put(msg)


async def main(
    nickname: str,
    config_path: Optional[str] = None,
    fresh: bool = False,
    history_file: str | None = None,
    notifications: bool = True,
    quiet_presence: bool = False,
    # TLS and auth overrides
    use_tls: bool | None = None,
    router_host: str | None = None,
    router_port: int | None = None,
    auth_token: str | None = None,
    ws_url: str | None = None,
):
    config = load_config(config_path)

    # Build node ID from nickname
    node_id = build_user_node_id(nickname)

    # Get node config or create default
    if node_id in config.nodes:
        node_config = config.nodes[node_id]
    else:
        node_config = NodeConfig(
            id=node_id,
            router_host=config.router.host,
            router_port=config.router.port,
            nickname=nickname,
        )

    # Apply CLI overrides for TLS and auth
    if router_host is not None:
        node_config.router_host = router_host
    if router_port is not None:
        node_config.router_port = router_port
    if use_tls is not None:
        node_config.use_tls = use_tls
    if auth_token is not None:
        node_config.auth_token = auth_token
    if ws_url is not None:
        node_config.ws_url = ws_url

    tui = MeshTUI(node_id, node_config, nickname, fresh=fresh, history_file=history_file, notifications=notifications, quiet_presence=quiet_presence)
    await tui.run()


def cli():
    """Entry point for the mesh-tui command."""
    parser = argparse.ArgumentParser(description="Mesh TUI - Rich terminal interface")
    parser.add_argument("--nickname", "-n", help="Your nickname for display/addressing")
    parser.add_argument("--config", "-c", help="Path to config file")
    parser.add_argument("--log-dir", default="logs", help="Log directory")
    parser.add_argument(
        "--fresh", "--no-resume",
        action="store_true",
        dest="fresh",
        help="Start fresh (don't load or persist history)"
    )
    parser.add_argument(
        "--history-file",
        help="Path to history file (implies persistence)"
    )
    # TLS and auth settings
    parser.add_argument(
        "--tls",
        action="store_true",
        dest="use_tls",
        help="Enable TLS for router connection"
    )
    parser.add_argument(
        "--router-host",
        help="Router hostname (overrides config)"
    )
    parser.add_argument(
        "--router-port",
        type=int,
        help="Router port (overrides config)"
    )
    parser.add_argument(
        "--auth-token",
        help="Auth token for router authentication"
    )
    parser.add_argument(
        "--ws-url",
        help="WebSocket URL for remote connection (e.g., wss://host/mesh/ws)"
    )
    parser.add_argument(
        "--no-notifications",
        action="store_true",
        dest="no_notifications",
        help="Disable desktop notifications"
    )
    parser.add_argument(
        "--quiet-presence", "-qp",
        action="store_true",
        dest="quiet_presence",
        help="Suppress join/leave presence messages"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode: suppress presence messages and desktop notifications"
    )
    # Legacy positional argument for backwards compatibility
    parser.add_argument("node_id", nargs="?", help="Legacy: Node ID (e.g., user:yourname)")
    args = parser.parse_args()

    # Determine nickname
    nickname = args.nickname

    if args.node_id:
        # Legacy mode: parse from node_id
        if args.node_id.startswith("user:"):
            nickname = args.node_id.split(":")[1]
        else:
            nickname = args.node_id

    if not nickname:
        # Default to system username
        nickname = getpass.getuser()

    node_id = build_user_node_id(nickname)

    log_file = setup_logging(node_id, args.log_dir)
    logger.info(f"Starting TUI for {node_id} (nickname: {nickname}), logging to {log_file}")

    auth_token = args.auth_token
    if not auth_token:
        auth_token = getpass.getpass("Mesh auth token: ")
        if not auth_token:
            print("Error: auth token is required", file=sys.stderr)
            sys.exit(1)

    # Quiet mode implies both quiet_presence and no_notifications
    quiet_presence = args.quiet_presence or args.quiet
    no_notifications = args.no_notifications or args.quiet

    asyncio.run(main(
        nickname,
        args.config,
        fresh=args.fresh,
        history_file=args.history_file,
        notifications=not no_notifications,
        quiet_presence=quiet_presence,
        use_tls=args.use_tls if args.use_tls else None,
        router_host=args.router_host,
        router_port=args.router_port,
        auth_token=auth_token,
        ws_url=args.ws_url,
    ))


if __name__ == "__main__":
    cli()
