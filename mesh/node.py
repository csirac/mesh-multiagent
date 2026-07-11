"""
Base Node class for mesh participants.

All participants (user and agent) share this base functionality:
- Connect to router
- Send/receive messages
- Maintain conversation history
- Handle incoming messages
- Persist and restore history
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Awaitable

from .protocol import (
    Attachment,
    Message,
    MessageType,
    ControlAction,
    make_message,
    make_shutdown_ack,
)
from .transport import Connection, connect, connect_ws
from .config import NodeConfig
from .paths import HISTORY_DIR

logger = logging.getLogger(__name__)

# Heartbeat configuration
HEARTBEAT_INTERVAL = 60.0      # Send ping every 60 seconds of idle time
HEARTBEAT_TIMEOUT = 15.0       # Wait 15 seconds for pong response

# Default history directory (uses real home from /etc/passwd, not $HOME)
DEFAULT_HISTORY_DIR = HISTORY_DIR


@dataclass
class HistoryEntry:
    """A single entry in the node's conversation history."""
    message: Message
    direction: str  # "incoming" or "outgoing"

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "message": asdict(self.message),
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HistoryEntry":
        """Deserialize from dictionary."""
        msg_data = data["message"]
        # Reconstruct the Message object
        message = Message(
            id=msg_data["id"],
            type=MessageType(msg_data["type"]),
            from_node=msg_data["from_node"],
            to_node=msg_data["to_node"],
            content=msg_data["content"],
            timestamp=msg_data["timestamp"],
            in_reply_to=msg_data.get("in_reply_to"),
            metadata=msg_data.get("metadata", {}),
            attachments=[
                Attachment.from_dict(a)
                for a in msg_data.get("attachments", [])
                if isinstance(a, dict)
            ],
        )
        return cls(message=message, direction=data["direction"])


@dataclass
class SummaryState:
    """
    State of a conversation summary.

    When history grows too large, older messages are summarized.
    The summary replaces the summarized portion in context sent to the LLM.
    Full history is always preserved on disk.
    """
    summary_text: str              # The summarized content
    messages_summarized: int       # Number of history entries covered by this summary
    created_at: str                # ISO timestamp when summary was created
    token_estimate: int            # Approximate tokens in the summary

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "summary_text": self.summary_text,
            "messages_summarized": self.messages_summarized,
            "created_at": self.created_at,
            "token_estimate": self.token_estimate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SummaryState":
        """Deserialize from dictionary."""
        return cls(
            summary_text=data["summary_text"],
            messages_summarized=data["messages_summarized"],
            created_at=data["created_at"],
            token_estimate=data["token_estimate"],
        )


class Node(ABC):
    """
    Base class for all mesh nodes.

    Handles:
    - Connection to router
    - Registration
    - Message send/receive
    - Unified conversation history
    - History persistence
    """

    def __init__(
        self,
        config: NodeConfig,
        history_file: Path | str | None = None,
        persist: bool = False,
    ):
        """
        Initialize a node.

        Args:
            config: Node configuration.
            history_file: Path to history file. If None and persist=True,
                         uses default path based on node ID.
            persist: Whether to persist history to disk.
        """
        self.config = config
        self.node_id = config.id
        self._conn: Connection | None = None
        self._running = False
        self._history: list[HistoryEntry] = []
        self._message_handlers: list[Callable[[Message], Awaitable[None]]] = []
        # Pending response futures for request/response patterns
        self._pending_requests: dict[str, asyncio.Future] = {}

        # Persistence settings
        self._persist = persist
        self._history_file: Path | None = None
        if history_file:
            self._history_file = Path(history_file)
        elif persist:
            # Default path: ~/.mesh/history/{node_id}.json
            # Replace colons with dashes for filesystem compatibility
            safe_id = self.node_id.replace(":", "-")
            self._history_file = DEFAULT_HISTORY_DIR / f"{safe_id}.json"

        # Heartbeat state
        self._last_activity: float = 0.0  # monotonic time of last send/receive
        self._pending_pong: asyncio.Future | None = None
        self._heartbeat_task: asyncio.Task | None = None

        # Pending sends queue - messages that failed to send due to connection issues
        # Will be retried after reconnection
        self._pending_sends: list[Message] = []
        self._pending_sends_lock = asyncio.Lock()

        # Append-only persistence: track how many entries are already on disk
        self._persisted_count: int = 0
        self._save_scheduled: bool = False

        # Remote shutdown support
        self._stop_event: asyncio.Event | None = None  # Set by receive_loop for shutdown
        self._auth_token: str | None = None  # Auth token for validating shutdown requests

        # SQLite conversation tracking (for archiving back to loaded conversation)
        self._message_store: "MessageStore | None" = None
        self._loaded_conversation_id: str | None = None

    @property
    def history(self) -> list[HistoryEntry]:
        """The node's conversation history."""
        return self._history

    @property
    def history_file(self) -> Path | None:
        """Path to history file, if persistence is enabled."""
        return self._history_file

    @property
    def is_connected(self) -> bool:
        """Whether the node is connected to the router."""
        return self._conn is not None and not self._conn.is_closed

    def set_auth_token(self, token: str) -> None:
        """Set the auth token for validating shutdown requests."""
        self._auth_token = token

    def load_history(self) -> int:
        """
        Load history from disk.

        Supports both legacy JSON array format and new JSON Lines format.
        After loading, converts legacy files to JSONL on next save.

        Returns:
            Number of entries loaded.
        """
        if not self._history_file or not self._history_file.exists():
            return 0

        try:
            with open(self._history_file, "r") as f:
                first_char = f.read(1)
                if not first_char:
                    return 0
                f.seek(0)

                if first_char == "[":
                    # Legacy JSON array format — load all at once
                    data = json.load(f)
                    entries = [HistoryEntry.from_dict(entry) for entry in data]
                    self._history = entries
                    # Mark 0 persisted so next save rewrites as JSONL
                    self._persisted_count = 0
                    logger.info(
                        f"Loaded {len(entries)} history entries from {self._history_file} "
                        f"(legacy JSON array, will convert to JSONL on next save)"
                    )
                else:
                    # JSON Lines format — one entry per line
                    entries = []
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entries.append(HistoryEntry.from_dict(json.loads(line)))
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            logger.warning(f"Skipping corrupt line {line_num} in {self._history_file}: {e}")
                    self._history = entries
                    self._persisted_count = len(entries)
                    logger.info(f"Loaded {len(entries)} history entries from {self._history_file} (JSONL)")

            return len(entries)
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
            logger.error(f"Failed to load history from {self._history_file}: {e}")
            return 0

    def load_history_from_store(
        self,
        store: "MessageStore",
        conversation_id: str,
        limit: int = 10000,
    ) -> int:
        """
        Load history from SQLite MessageStore instead of JSON file.

        Args:
            store: MessageStore instance to load from
            conversation_id: The conversation ID to load (e.g., "chat:research-project")
            limit: Maximum number of messages to load

        Returns:
            Number of entries loaded.
        """
        try:
            messages = store.get_conversation_history(conversation_id, limit=limit)
            if not messages:
                logger.info(f"No messages found for conversation: {conversation_id}")
                return 0

            # Convert to HistoryEntry format
            # Determine direction based on from_node vs our node_id
            entries = []
            for msg in messages:
                # If from_node matches our node_id, it's outgoing; otherwise incoming
                if msg.from_node == self.config.id:
                    direction = "outgoing"
                else:
                    direction = "incoming"
                entries.append(HistoryEntry(message=msg, direction=direction))

            self._history = entries

            # Store the message store and conversation ID for archiving new messages
            self._message_store = store
            self._loaded_conversation_id = conversation_id

            logger.info(
                f"Loaded {len(entries)} history entries from store "
                f"for conversation: {conversation_id} (archiving enabled)"
            )

            # Also try to load summary if available
            summary_data = store.get_summary(conversation_id)
            if summary_data:
                self._summary = SummaryState(
                    summary_text=summary_data["summary_text"],
                    messages_summarized=summary_data["messages_summarized"],
                    created_at=summary_data["created_at"],
                    token_estimate=summary_data["token_estimate"],
                )
                logger.info(
                    f"Loaded summary: {self._summary.messages_summarized} messages summarized"
                )

            return len(entries)

        except Exception as e:
            logger.error(f"Failed to load history from store: {e}")
            return 0

    def save_history(self) -> bool:
        """
        Append new history entries to disk in JSON Lines format.

        Only writes entries that haven't been persisted yet (append-only).
        If the file was loaded from legacy JSON array format, rewrites
        the entire file as JSONL on the first save.

        Returns:
            True if saved successfully, False otherwise.
        """
        if not self._history_file:
            return False

        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)

            new_entries = self._history[self._persisted_count:]
            if not new_entries:
                return True  # Nothing new to write

            if self._persisted_count == 0 and self._history:
                # First save (or legacy conversion): write entire file as JSONL
                with open(self._history_file, "w") as f:
                    for entry in self._history:
                        f.write(json.dumps(entry.to_dict(), separators=(",", ":")) + "\n")
                logger.debug(f"Wrote {len(self._history)} history entries to {self._history_file} (full JSONL)")
            else:
                # Append only new entries
                with open(self._history_file, "a") as f:
                    for entry in new_entries:
                        f.write(json.dumps(entry.to_dict(), separators=(",", ":")) + "\n")
                logger.debug(f"Appended {len(new_entries)} history entries to {self._history_file}")

            self._persisted_count = len(self._history)
            return True
        except (OSError, IOError) as e:
            logger.error(f"Failed to save history to {self._history_file}: {e}")
            return False

    def schedule_save(self) -> None:
        """
        Schedule a debounced history save.

        Batches rapid-fire messages into a single disk write after 2 seconds
        of quiet. Safe to call from sync or async context.
        """
        if not self._persist or not self._history_file:
            return
        if self._save_scheduled:
            return  # Already scheduled
        self._save_scheduled = True
        try:
            loop = asyncio.get_running_loop()
            loop.call_later(2.0, self._do_scheduled_save)
        except RuntimeError:
            # No running loop — save synchronously (startup/shutdown)
            self.save_history()
            self._save_scheduled = False

    def _do_scheduled_save(self) -> None:
        """Execute the debounced save."""
        self._save_scheduled = False
        self.save_history()

    def clear_history(self) -> None:
        """Clear in-memory history (does not delete the file)."""
        self._history = []

    def delete_history_file(self) -> bool:
        """
        Delete the history file from disk.

        Returns:
            True if deleted (or didn't exist), False on error.
        """
        if not self._history_file:
            return True
        try:
            if self._history_file.exists():
                self._history_file.unlink()
                logger.info(f"Deleted history file: {self._history_file}")
            return True
        except OSError as e:
            logger.error(f"Failed to delete history file: {e}")
            return False

    @property
    def summary_file(self) -> Path | None:
        """Path to summary file (derived from history file)."""
        if not self._history_file:
            return None
        return self._history_file.with_suffix(".summary.json")

    def load_summary(self) -> SummaryState | None:
        """
        Load summary from disk.

        Returns:
            SummaryState if loaded, None otherwise.
        """
        summary_file = self.summary_file
        if not summary_file or not summary_file.exists():
            return None

        try:
            with open(summary_file, "r") as f:
                data = json.load(f)
            summary = SummaryState.from_dict(data)
            logger.info(
                f"Loaded summary from {summary_file}: "
                f"{summary.messages_summarized} messages, "
                f"~{summary.token_estimate} tokens"
            )
            return summary
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to load summary from {summary_file}: {e}")
            return None

    def save_summary(self, summary: SummaryState) -> bool:
        """
        Save summary to disk.

        Returns:
            True if saved successfully, False otherwise.
        """
        summary_file = self.summary_file
        if not summary_file:
            return False

        try:
            # Ensure directory exists
            summary_file.parent.mkdir(parents=True, exist_ok=True)

            with open(summary_file, "w") as f:
                json.dump(summary.to_dict(), f, indent=2)
            logger.debug(
                f"Saved summary to {summary_file}: "
                f"{summary.messages_summarized} messages, "
                f"~{summary.token_estimate} tokens"
            )
            return True
        except (OSError, IOError) as e:
            logger.error(f"Failed to save summary to {summary_file}: {e}")
            return False

    def delete_summary_file(self) -> bool:
        """
        Delete the summary file from disk.

        Returns:
            True if deleted (or didn't exist), False on error.
        """
        summary_file = self.summary_file
        if not summary_file:
            return True
        try:
            if summary_file.exists():
                summary_file.unlink()
                logger.info(f"Deleted summary file: {summary_file}")
            return True
        except OSError as e:
            logger.error(f"Failed to delete summary file: {e}")
            return False

    def add_message_handler(
        self,
        handler: Callable[[Message], Awaitable[None]],
    ) -> None:
        """Add a handler for incoming messages."""
        self._message_handlers.append(handler)

    def _get_registration_content(self) -> dict:
        """Get the content dict for the registration message.

        Subclasses can override to add additional metadata (e.g., description).
        """
        content = {"action": ControlAction.REGISTER.value}
        # Include auth token if configured
        if self.config.auth_token:
            content["auth_token"] = self.config.auth_token
        return content

    async def connect(self) -> None:
        """Connect to the router and register.

        This method can be called multiple times (e.g., for reconnection).
        Previous connection state is reset.
        """
        # Reset state for reconnection
        self._running = False
        if self._conn and not self._conn.is_closed:
            await self._conn.close()
            self._conn = None

        # Use WebSocket if ws_url is configured, otherwise TCP
        if self.config.ws_url:
            self._conn = await connect_ws(self.config.ws_url)
        else:
            self._conn = await connect(
                self.config.router_host,
                self.config.router_port,
                use_tls=self.config.use_tls,
                server_hostname=self.config.tls_server_hostname,
            )

        # Register with router
        register_msg = Message(
            from_node=self.node_id,
            to_node="router",
            type=MessageType.CONTROL,
            content=self._get_registration_content(),
        )
        await self._conn.send(register_msg)

        # Wait for ACK
        ack = await self._conn.receive()
        if ack is None:
            raise ConnectionError("Failed to receive registration ACK")

        content = ack.content if isinstance(ack.content, dict) else {}
        if content.get("action") != ControlAction.ACK.value:
            raise ConnectionError(f"Unexpected response: {ack.content}")

        # Check for auth error
        if content.get("status") == "error":
            error_msg = content.get("error", "unknown error")
            raise ConnectionError(f"Registration failed: {error_msg}")

        logger.info(f"Node {self.node_id} registered with router")
        self._running = True
        self._last_activity = asyncio.get_event_loop().time()

    async def disconnect(self) -> None:
        """Disconnect from the router."""
        self._running = False
        # Cancel heartbeat task
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        if self._conn:
            await self._conn.close()
            self._conn = None
        logger.info(f"Node {self.node_id} disconnected")

    def _get_status_summary(self) -> dict:
        """Build lightweight status summary for heartbeat pings.

        Base class returns minimal info. AgentNode overrides with
        router state, history stats, context tokens, and memory metrics.
        """
        import time as _time
        return {"uptime_s": round(_time.monotonic() - self._start_time, 1) if hasattr(self, '_start_time') else 0}

    def mark_activity(self) -> None:
        """Mark that activity occurred (send/receive). Resets heartbeat timer."""
        self._last_activity = asyncio.get_event_loop().time()

    async def _heartbeat_loop(self) -> None:
        """
        Background task that sends periodic pings to detect dead connections.

        Only sends pings when the connection has been idle for HEARTBEAT_INTERVAL.
        This means if the agent is actively processing (sending/receiving messages),
        no pings are sent — the activity itself proves the connection is alive.
        """
        loop = asyncio.get_event_loop()

        while self._running and self._conn and not self._conn.is_closed:
            now = loop.time()
            time_since_activity = now - self._last_activity

            if time_since_activity >= HEARTBEAT_INTERVAL:
                # Connection has been idle — send a ping
                try:
                    logger.debug(f"Sending heartbeat ping (idle for {time_since_activity:.0f}s)")
                    ping_content = {"action": ControlAction.PING.value}
                    try:
                        ping_content["status_summary"] = self._get_status_summary()
                    except Exception as e:
                        logger.debug(f"Failed to build status summary for ping: {e}")
                    ping_msg = Message(
                        from_node=self.node_id,
                        to_node="router",
                        type=MessageType.CONTROL,
                        content=ping_content,
                    )

                    # Create future for pong response
                    self._pending_pong = loop.create_future()
                    await self._conn.send(ping_msg)

                    # Wait for pong with timeout
                    try:
                        await asyncio.wait_for(self._pending_pong, timeout=HEARTBEAT_TIMEOUT)
                        logger.debug("Heartbeat pong received")
                        self._last_activity = loop.time()  # Reset activity on successful pong
                    except asyncio.TimeoutError:
                        logger.warning(f"Heartbeat timeout — no pong received in {HEARTBEAT_TIMEOUT}s")
                        # Connection is dead — break out to trigger reconnect
                        self._running = False
                        if self._conn and not self._conn.is_closed:
                            await self._conn.close()
                        break
                    finally:
                        self._pending_pong = None

                except Exception as e:
                    logger.warning(f"Heartbeat ping failed: {e}")
                    # Connection error — break out to trigger reconnect
                    self._running = False
                    break

                # Wait full interval before next check
                await asyncio.sleep(HEARTBEAT_INTERVAL)
            else:
                # Wait until next potential ping time
                wait_time = HEARTBEAT_INTERVAL - time_since_activity
                await asyncio.sleep(wait_time)

    async def send(
        self,
        to_node: str,
        content: str,
        in_reply_to: str | None = None,
        attachments: list[Attachment] | None = None,
    ) -> Message:
        """Send a message to another node.

        If the connection is closed, the message is queued for retry after reconnection.
        """
        msg = make_message(
            from_node=self.node_id,
            to_node=to_node,
            content=content,
            in_reply_to=in_reply_to,
            attachments=attachments,
        )
        await self._send_with_retry(msg)
        return msg

    async def send_message(self, msg: Message) -> None:
        """Send a pre-constructed message.

        If the connection is closed, the message is queued for retry after reconnection.
        """
        await self._send_with_retry(msg)

    async def _send_with_retry(self, msg: Message) -> None:
        """
        Internal method to send a message with retry-on-reconnect support.

        If the send fails due to a closed connection, the message is queued
        and will be retried when flush_pending_sends() is called after reconnection.
        """
        try:
            if not self._conn or self._conn.is_closed:
                raise ConnectionError("Not connected to router")

            await self._conn.send(msg)
            self.mark_activity()

            # Record in history and persist
            self._history.append(HistoryEntry(message=msg, direction="outgoing"))
            self.schedule_save()

        except (ConnectionError, OSError, asyncio.IncompleteReadError) as e:
            # Connection failed - queue for retry after reconnection
            async with self._pending_sends_lock:
                self._pending_sends.append(msg)
            logger.warning(
                f"Send failed ({e}), queued message for retry after reconnect "
                f"(to={msg.to_node}, {len(self._pending_sends)} pending)"
            )
            # Don't raise - the message is queued and will be retried
            # But do record in history so context is preserved
            self._history.append(HistoryEntry(message=msg, direction="outgoing"))
            self.schedule_save()

    async def flush_pending_sends(self) -> int:
        """
        Retry sending any messages that failed due to connection issues.

        Call this after successfully reconnecting to the router.

        Returns:
            Number of messages successfully sent.
        """
        if not self._pending_sends:
            return 0

        async with self._pending_sends_lock:
            pending = self._pending_sends.copy()
            self._pending_sends.clear()

        sent_count = 0
        failed = []

        for msg in pending:
            try:
                if not self._conn or self._conn.is_closed:
                    failed.extend(pending[pending.index(msg):])
                    break

                await self._conn.send(msg)
                self.mark_activity()
                sent_count += 1
                logger.info(f"Retried pending send: to={msg.to_node}, id={msg.id[:8]}...")

            except (ConnectionError, OSError, asyncio.IncompleteReadError) as e:
                logger.warning(f"Retry send failed ({e}), re-queueing message")
                failed.append(msg)

        # Re-queue any that still failed
        if failed:
            async with self._pending_sends_lock:
                self._pending_sends.extend(failed)

        if sent_count > 0:
            logger.info(f"Flushed {sent_count} pending sends ({len(failed)} still pending)")

        return sent_count

    @property
    def pending_send_count(self) -> int:
        """Number of messages waiting to be sent after reconnection."""
        return len(self._pending_sends)

    async def _request_node_list_raw(self, timeout: float = 5.0) -> dict:
        """Request node list from router, returning the full response content."""
        if not self._conn:
            raise ConnectionError("Not connected to router")

        msg = Message(
            from_node=self.node_id,
            to_node="router",
            type=MessageType.CONTROL,
            content={"action": ControlAction.LIST_NODES.value},
        )

        request_key = f"list_nodes_{msg.id}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict] = loop.create_future()
        self._pending_requests[request_key] = future

        await self._conn.send(msg)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return {}
        finally:
            self._pending_requests.pop(request_key, None)

    async def request_node_list(self, timeout: float = 5.0) -> list[str]:
        """Request list of connected node IDs from router."""
        content = await self._request_node_list_raw(timeout=timeout)
        return content.get("nodes", [])

    async def request_node_list_with_status(self, timeout: float = 5.0) -> tuple[list[str], dict]:
        """Request connected nodes and their heartbeat-lite status from router.

        Returns:
            Tuple of (node_ids, status_dict) where status_dict maps node_id
            to heartbeat-lite data (state, context_tokens, history_turns, etc.)
        """
        content = await self._request_node_list_raw(timeout=timeout)
        return content.get("nodes", []), content.get("status", {})

    async def request_node_list_with_status_raw(self, timeout: float = 5.0) -> dict:
        """Request connected nodes with full response (including cc_usage).

        Returns:
            Full response dict with keys: nodes, status, cc_usage
        """
        return await self._request_node_list_raw(timeout=timeout)

    async def request_channel_list(self, timeout: float = 5.0) -> list[dict]:
        """
        Request list of channels from router.

        Returns list of channel dicts with keys:
        - name: Channel name
        - description: Channel description
        - member_count: Number of members
        - is_member: Whether this node is a member
        """
        if not self._conn:
            raise ConnectionError("Not connected to router")

        msg = Message(
            from_node=self.node_id,
            to_node="router",
            type=MessageType.CONTROL,
            content={"action": ControlAction.CHANNEL_LIST.value},
        )

        request_key = f"channel_list_{msg.id}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[dict]] = loop.create_future()
        self._pending_requests[request_key] = future

        await self._conn.send(msg)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return []
        finally:
            self._pending_requests.pop(request_key, None)

    async def request_channel_members(
        self, channel_name: str, timeout: float = 5.0
    ) -> list[dict]:
        """
        Request members of a specific channel from router.

        Args:
            channel_name: Name of the channel (without 'channel:' prefix)

        Returns list of member dicts with keys:
        - node_id: The member's node ID
        - online: Whether the member is currently online
        """
        if not self._conn:
            raise ConnectionError("Not connected to router")

        msg = Message(
            from_node=self.node_id,
            to_node="router",
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_MEMBERS.value,
                "channel_name": channel_name,
            },
        )

        request_key = f"channel_members_{msg.id}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[dict]] = loop.create_future()
        self._pending_requests[request_key] = future

        await self._conn.send(msg)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return []
        finally:
            self._pending_requests.pop(request_key, None)

    async def send_control_and_wait(
        self,
        content: dict,
        timeout: float = 10.0,
    ) -> Message:
        """Send a CONTROL message to the router and await its in_reply_to response."""
        if not self._conn:
            raise ConnectionError("Not connected to router")

        msg = Message(
            from_node=self.node_id,
            to_node="router",
            type=MessageType.CONTROL,
            content=content,
        )

        request_key = f"_send_and_wait_{msg.id}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Message] = loop.create_future()
        self._pending_requests[request_key] = future

        await self._conn.send(msg)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending_requests.pop(request_key, None)

    async def receive_loop(self, stop_event: asyncio.Event | None = None) -> None:
        """
        Main loop for receiving and handling messages.

        Args:
            stop_event: Optional event to signal shutdown. If provided, the loop
                       will exit cleanly when the event is set, allowing graceful
                       shutdown in response to SIGTERM/SIGINT.
        """
        if not self._conn:
            raise ConnectionError("Not connected to router")

        # Store stop_event so shutdown handler can access it
        self._stop_event = stop_event

        # Track background tasks to prevent them from being garbage collected
        background_tasks: set[asyncio.Task] = set()

        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        try:
            while self._running and not self._conn.is_closed:
                # If stop_event is provided, race between receiving and stop signal
                if stop_event:
                    receive_task = asyncio.create_task(self._conn.receive())
                    stop_task = asyncio.create_task(stop_event.wait())

                    done, pending = await asyncio.wait(
                        {receive_task, stop_task},
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                    # Check if stop was signaled
                    if stop_task in done:
                        logger.info("Stop signal received, exiting receive loop")
                        break

                    # Otherwise get the message result
                    msg = receive_task.result()
                else:
                    # No stop_event provided, just receive normally
                    msg = await self._conn.receive()

                if msg is None:
                    break

                # Mark activity on every received message
                self.mark_activity()

                # Handle PONG responses for heartbeat
                if msg.type == MessageType.CONTROL:
                    content = msg.content if isinstance(msg.content, dict) else {}
                    if content.get("action") == ControlAction.PONG.value:
                        if self._pending_pong and not self._pending_pong.done():
                            self._pending_pong.set_result(True)
                        continue  # Don't pass pong to other handlers

                # Record in history (except control messages, confirm responses, presence,
                # and our own messages echoed back from channels)
                if msg.type not in (MessageType.CONTROL, MessageType.CONFIRM_RESPONSE, MessageType.PRESENCE):
                    # Skip adding our own messages (echoed from channels) to history
                    # to avoid duplicates and prevent agents from processing their own output
                    if msg.from_node != self.node_id:
                        self._history.append(HistoryEntry(message=msg, direction="incoming"))
                        self.schedule_save()
                    else:
                        logger.debug(f"Skipping history add for own message: {msg.id[:8]}...")

                # High-priority messages (confirmations) are handled inline
                # to avoid race conditions with pending confirmations
                if msg.type == MessageType.CONFIRM_RESPONSE:
                    await self._handle_message(msg)
                else:
                    # Other messages are handled in background tasks so the
                    # receive loop can continue (prevents deadlock when waiting
                    # for confirmations during tool execution)
                    task = asyncio.create_task(self._handle_message(msg))
                    background_tasks.add(task)
                    task.add_done_callback(background_tasks.discard)

            # Wait for any remaining tasks
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)

        finally:
            # Cancel heartbeat task on exit
            if self._heartbeat_task and not self._heartbeat_task.done():
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            # Final flush of any unsaved history
            self.save_history()

        logger.info(f"Node {self.node_id} receive loop ended")

    async def _handle_message(self, msg: Message) -> None:
        """Handle an incoming message."""
        # Check for generic _send_and_wait responses (keyed by in_reply_to)
        if msg.in_reply_to:
            wait_key = f"_send_and_wait_{msg.in_reply_to}"
            future = self._pending_requests.get(wait_key)
            if future and not future.done():
                future.set_result(msg)
                return

        # Check if this is a response to a pending request
        if msg.type == MessageType.CONTROL:
            content = msg.content if isinstance(msg.content, dict) else {}
            action = content.get("action")

            # Handle list_nodes response
            if action == ControlAction.LIST_NODES.value:
                # Resolve with full content dict (callers extract what they need)
                for key in list(self._pending_requests.keys()):
                    if key.startswith("list_nodes_"):
                        future = self._pending_requests.get(key)
                        if future and not future.done():
                            future.set_result(content)
                            return  # Don't pass to other handlers

            # Handle channel_list response
            if action == ControlAction.CHANNEL_LIST.value:
                channels = content.get("channels", [])
                for key in list(self._pending_requests.keys()):
                    if key.startswith("channel_list_"):
                        future = self._pending_requests.get(key)
                        if future and not future.done():
                            future.set_result(channels)
                            return

            # Handle channel_members response
            if action == ControlAction.CHANNEL_MEMBERS.value:
                members = content.get("members", [])
                for key in list(self._pending_requests.keys()):
                    if key.startswith("channel_members_"):
                        future = self._pending_requests.get(key)
                        if future and not future.done():
                            future.set_result(members)
                            return

            # Handle remote shutdown request
            if action == ControlAction.SHUTDOWN.value:
                await self._handle_shutdown_request(msg, content)
                return

            # Handle reset context request
            if action == ControlAction.RESET_CONTEXT.value:
                await self._handle_reset_context(msg, content)
                return

        # Call registered handlers
        for handler in self._message_handlers:
            try:
                await handler(msg)
            except Exception as e:
                logger.error(f"Handler error: {e}")

        # Call the subclass implementation
        await self.on_message(msg)

    async def _handle_shutdown_request(self, msg: Message, content: dict) -> None:
        """
        Handle a remote shutdown request.

        Validates the auth token, sends an ACK, and triggers graceful shutdown.
        """
        from_node = msg.from_node
        auth_token = content.get("auth_token", "")
        reason = content.get("reason", "")

        # Validate auth token
        if not self._auth_token:
            logger.warning(f"Shutdown request from {from_node} rejected: no auth token configured")
            return

        if auth_token != self._auth_token:
            logger.warning(f"Shutdown request from {from_node} rejected: invalid auth token")
            return

        logger.info(f"Received valid shutdown request from {from_node}" +
                   (f" (reason: {reason})" if reason else ""))

        # Send acknowledgment before shutting down
        ack = make_shutdown_ack(
            from_node=self.node_id,
            to_node=from_node,
            in_reply_to=msg.id,
        )
        try:
            await self.send(ack)
            logger.info(f"Sent shutdown ACK to {from_node}")
        except Exception as e:
            logger.error(f"Failed to send shutdown ACK: {e}")

        # Trigger graceful shutdown by setting the stop event
        if self._stop_event:
            logger.info("Triggering graceful shutdown via stop_event")
            self._stop_event.set()
        else:
            # Fallback: just mark as not running
            logger.warning("No stop_event available, forcing _running = False")
            self._running = False

    async def _handle_reset_context(self, msg: Message, content: dict) -> None:
        """
        Handle a reset context request.

        Clears the agent's conversation history. Used by test harnesses
        to ensure clean slate between problems.
        """
        from_node = msg.from_node
        reason = content.get("reason", "")

        logger.info(f"Received reset_context request from {from_node}" +
                   (f" (reason: {reason})" if reason else ""))

        # Clear the history
        history_size = len(self._history)
        self.clear_history()

        logger.info(f"Cleared {history_size} history entries for {self.node_id}")

        # Send acknowledgment
        ack = Message(
            from_node=self.node_id,
            to_node=from_node,
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.ACK.value,
                "reset_context": True,
                "cleared_entries": history_size,
            },
            in_reply_to=msg.id,
        )
        try:
            await self._conn.send(ack)
        except Exception as e:
            logger.error(f"Failed to send reset_context ACK: {e}")

    @abstractmethod
    async def on_message(self, msg: Message) -> None:
        """
        Handle an incoming message. Override in subclasses.

        This is called after the message is recorded in history.
        """
        pass

    def get_history_for_llm(self) -> list[dict]:
        """
        Format history for LLM context.

        Returns a list of {"role": ..., "content": ...} dicts.
        """
        messages = []
        for entry in self._history:
            msg = entry.message
            if msg.type == MessageType.MESSAGE:
                role = "assistant" if entry.direction == "outgoing" else "user"
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                # Prepend sender info for multi-party clarity
                if entry.direction == "incoming":
                    content = f"[From {msg.from_node}]: {content}"
                messages.append({"role": role, "content": content})
        return messages
