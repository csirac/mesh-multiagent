"""
User node - A passive node that receives messages without auto-reply.

Unlike AgentNode which processes messages through LLM and auto-replies,
UserNode simply receives messages and stores them in history, allowing
manual responses through the UI.

Handles CONFIRM_REQUEST messages from agents that need user approval
for sensitive tool operations (e.g., sending emails, creating events).

Node ID format: user:{nickname}
  - nickname: A unique, human-friendly name for addressing
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Awaitable

from .node import Node
from .protocol import Message, MessageType, make_confirm_response, build_user_node_id, parse_node_id
from .config import NodeConfig

logger = logging.getLogger(__name__)


@dataclass
class RosterEntry:
    """An entry in the user's roster of connected nodes."""
    node_id: str
    nickname: str
    node_type: str  # "user" or agent type like "coder"
    description: str = ""  # Optional description (e.g., project name)
    llm_backend: str = ""  # LLM backend (e.g., "openai", "anthropic", "claude-code")
    llm_model: str = ""  # LLM model (e.g., "gpt-5.1", "opus")
    hostname: str = ""  # Hostname where the node is running


class UserNode(Node):
    """
    User-controlled node that doesn't auto-reply.

    This is the base class for user-facing nodes. It receives messages
    and stores them in history, but doesn't automatically respond.
    Responses are sent manually through the UI.

    Supports confirmation requests from agents: when an agent calls a
    sensitive tool, it sends a CONFIRM_REQUEST. The user node notifies
    registered callbacks, which can display a prompt and send a
    CONFIRM_RESPONSE back.
    """

    def __init__(
        self,
        config: NodeConfig,
        nickname: str | None = None,
        history_file: Path | str | None = None,
        persist: bool = False,
    ):
        # Build user node ID from nickname
        self._nickname = nickname or config.nickname
        if self._nickname:
            config.id = build_user_node_id(self._nickname)

        super().__init__(config, history_file=history_file, persist=persist)

        # Store nickname for later access
        self.nickname = self._nickname

        # Callbacks for message notification
        self._on_message_callbacks: list[Callable[[Message], Awaitable[None] | None]] = []
        # Callbacks for confirmation requests (separate from regular messages)
        self._on_confirm_callbacks: list[Callable[[Message], Awaitable[bool] | bool]] = []
        # Callbacks for presence notifications
        self._on_presence_callbacks: list[Callable[[Message], Awaitable[None] | None]] = []

        # Roster of connected nodes (nickname -> RosterEntry)
        self._roster: dict[str, RosterEntry] = {}

    def on_message_received(
        self,
        callback: Callable[[Message], Awaitable[None] | None],
    ) -> None:
        """Register a callback to be notified when messages arrive."""
        self._on_message_callbacks.append(callback)

    def on_confirm_request(
        self,
        callback: Callable[[Message], Awaitable[bool] | bool],
    ) -> None:
        """
        Register a callback to handle confirmation requests.

        The callback receives the CONFIRM_REQUEST message and should return
        True if the user confirms, False otherwise. If the callback returns
        a value, the node automatically sends a CONFIRM_RESPONSE.
        """
        self._on_confirm_callbacks.append(callback)

    def on_presence(
        self,
        callback: Callable[[Message], Awaitable[None] | None],
    ) -> None:
        """
        Register a callback for presence notifications (join/leave).

        The callback receives the PRESENCE message with content:
        {"event": "join"|"leave", "nickname": str, "node_type": str}
        """
        self._on_presence_callbacks.append(callback)

    @property
    def roster(self) -> dict[str, RosterEntry]:
        """Get the current roster of connected nodes."""
        return self._roster

    def resolve_target(self, target: str) -> str | None:
        """
        Resolve a nickname to a full node ID.

        If target is already a full node ID (contains ':'), return as-is.
        Otherwise, look up in roster by nickname.

        Returns None if nickname not found.
        """
        if ":" in target:
            return target

        # Check if target matches a nickname in roster
        target_lower = target.lower()
        if target_lower in self._roster:
            return self._roster[target_lower].node_id

        return None

    def get_roster_list(self) -> list[RosterEntry]:
        """Get the roster as a sorted list."""
        return sorted(self._roster.values(), key=lambda e: (e.node_type, e.nickname))

    async def on_message(self, msg: Message) -> None:
        """Handle incoming messages - just notify, don't reply."""
        if msg.type == MessageType.MESSAGE:
            # Notify callbacks
            for callback in self._on_message_callbacks:
                try:
                    result = callback(msg)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Callback error: {e}")

        elif msg.type == MessageType.CONFIRM_REQUEST:
            # Handle confirmation request from agent
            await self._handle_confirm_request(msg)

        elif msg.type == MessageType.PRESENCE:
            # Handle presence notifications (join/leave)
            await self._handle_presence(msg)

        elif msg.type == MessageType.CONTROL:
            # Handle control messages - notify callbacks so clients can process them
            content = msg.content if isinstance(msg.content, dict) else {}
            action = content.get("action", "unknown")
            logger.debug(f"Received control message: {action}")
            # Call message callbacks for control messages too (e.g., history_response)
            for callback in self._on_message_callbacks:
                try:
                    result = callback(msg)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Callback error handling control message: {e}")

    async def _handle_confirm_request(self, msg: Message) -> None:
        """
        Process a confirmation request from an agent.

        Calls registered callbacks to get user's decision, then sends
        a CONFIRM_RESPONSE back to the agent.
        """
        content = msg.content if isinstance(msg.content, dict) else {}
        tool_name = content.get("tool_name", "unknown")
        preview = content.get("preview", "")

        logger.info(f"Received confirmation request for {tool_name} from {msg.from_node}")

        # If no callbacks registered, auto-reject (safe default)
        if not self._on_confirm_callbacks:
            logger.warning("No confirm callbacks registered, auto-rejecting")
            await self._send_confirm_response(msg, confirmed=False)
            return

        # Call each callback until one handles it
        for callback in self._on_confirm_callbacks:
            try:
                result = callback(msg)
                if asyncio.iscoroutine(result):
                    confirmed = await result
                else:
                    confirmed = result

                # If callback returns None, it will handle the response itself (async UI)
                if confirmed is None:
                    logger.debug("Callback returned None, deferring response to async handler")
                    return

                # Send response
                await self._send_confirm_response(msg, confirmed=confirmed)
                return

            except Exception as e:
                logger.error(f"Confirm callback error: {e}")

        # If all callbacks failed, auto-reject
        logger.warning("All confirm callbacks failed, auto-rejecting")
        await self._send_confirm_response(msg, confirmed=False)

    async def _send_confirm_response(self, request_msg: Message, confirmed: bool) -> None:
        """Send a confirmation response back to the requesting agent."""
        response = make_confirm_response(
            from_node=self.node_id,
            to_node=request_msg.from_node,
            in_reply_to=request_msg.id,
            confirmed=confirmed,
        )
        await self._conn.send(response)
        logger.info(f"Sent confirmation response: {confirmed} to {request_msg.from_node}")

    async def _handle_presence(self, msg: Message) -> None:
        """
        Handle a presence notification (join/leave).

        Updates the roster and notifies callbacks.
        """
        content = msg.content if isinstance(msg.content, dict) else {}
        event = content.get("event", "unknown")
        nickname = content.get("nickname", "")
        node_type = content.get("node_type", "unknown")
        description = content.get("description", "")
        llm_backend = content.get("llm_backend", "")
        llm_model = content.get("llm_model", "")
        hostname = content.get("hostname", "")
        from_node = msg.from_node

        logger.info(f"Presence: {nickname} ({node_type}) {event}" + (f" [{description}]" if description else "") + (f" [backend={llm_backend}]" if llm_backend else "") + (f" [host={hostname}]" if hostname else ""))

        # Update roster
        nick_lower = nickname.lower()
        if event == "join":
            self._roster[nick_lower] = RosterEntry(
                node_id=from_node,
                nickname=nickname,
                node_type=node_type,
                description=description,
                llm_backend=llm_backend,
                llm_model=llm_model,
                hostname=hostname,
            )
        elif event == "leave":
            self._roster.pop(nick_lower, None)

        # Notify callbacks
        for callback in self._on_presence_callbacks:
            try:
                result = callback(msg)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Presence callback error: {e}")


class _MockUserNodeImpl(UserNode):
    """
    User node for testing that collects messages for assertions.

    Stores received messages in a list for easy verification in tests.
    Note: Named with underscore prefix to avoid pytest collection.
    """

    def __init__(self, config: NodeConfig):
        super().__init__(config)
        self.received_messages: list[Message] = []
        self._message_event = asyncio.Event()

    async def on_message(self, msg: Message) -> None:
        """Collect messages for testing."""
        await super().on_message(msg)
        if msg.type == MessageType.MESSAGE:
            self.received_messages.append(msg)
            self._message_event.set()

    async def wait_for_message(self, timeout: float = 2.0) -> Message | None:
        """Wait for a message to arrive."""
        try:
            await asyncio.wait_for(self._message_event.wait(), timeout=timeout)
            self._message_event.clear()
            if self.received_messages:
                return self.received_messages[-1]
            return None
        except asyncio.TimeoutError:
            return None

    def clear_received(self):
        """Clear received messages."""
        self.received_messages.clear()
        self._message_event.clear()


# Public aliases for external use
MockUserNode = _MockUserNodeImpl
TestUserNode = _MockUserNodeImpl
