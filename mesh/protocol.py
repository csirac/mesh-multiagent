# SPDX-License-Identifier: Apache-2.0
"""
Message protocol for agent mesh communication.

All nodes (user or agent) exchange messages through this protocol.
Messages are JSON-serializable dataclasses with routing information.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MessageType(str, Enum):
    """Types of messages in the mesh."""
    MESSAGE = "message"           # Normal conversation message
    TOOL_REQUEST = "tool_request" # Request a node to execute a tool
    TOOL_RESULT = "tool_result"   # Result of tool execution
    CONTROL = "control"           # Spawn, kill, status, pause, resume
    CONFIRM_REQUEST = "confirm_request"   # Request user confirmation for tool
    CONFIRM_RESPONSE = "confirm_response" # User's response to confirmation request
    PRESENCE = "presence"         # Node joined/left announcement
    STATUS_REQUEST = "status_request"     # Request agent's recent context
    STATUS_RESPONSE = "status_response"   # Agent's recent context response
    TOOL_ACTIVITY = "tool_activity"       # Real-time tool call/result notifications
    STATUS = "status"                     # One-way status/phase notifications


class ControlAction(str, Enum):
    """Control actions for managing nodes."""
    SPAWN = "spawn"
    KILL = "kill"
    STATUS = "status"
    PAUSE = "pause"
    RESUME = "resume"
    LIST_NODES = "list_nodes"
    REGISTER = "register"         # Node registering with router
    ACK = "ack"                   # Acknowledgment
    REGISTER_PUSH_TOKEN = "register_push_token"  # Register FCM token for push notifications
    # Channel operations
    CHANNEL_CREATE = "channel_create"    # Create a new channel
    CHANNEL_DELETE = "channel_delete"    # Delete a channel (users only)
    CHANNEL_JOIN = "channel_join"        # Join a channel
    CHANNEL_LEAVE = "channel_leave"      # Leave a channel
    CHANNEL_LIST = "channel_list"        # List all channels
    CHANNEL_MEMBERS = "channel_members"  # List members of a channel
    CHANNEL_INVITE = "channel_invite"    # Invite a node to join a channel
    CHANNEL_REMOVE_MEMBER = "channel_remove_member"  # Remove a member from a channel
    # Message sync operations
    HISTORY_SYNC = "history_sync"        # Request message history
    HISTORY_RESPONSE = "history_response"  # Server sends batch of messages
    MARK_READ = "mark_read"              # Mark messages as read
    # Heartbeat
    PING = "ping"                        # Client keepalive ping
    PONG = "pong"                        # Server keepalive response
    # Remote shutdown
    SHUTDOWN = "shutdown"                # Request agent to shut down
    SHUTDOWN_ACK = "shutdown_ack"        # Acknowledgment before shutdown
    # Context management
    RESET_CONTEXT = "reset_context"      # Clear agent's conversation history
    # Agent management (router-side)
    LIST_AGENTS = "list_agents"          # List configured agent types
    START_AGENT = "start_agent"          # Start an agent process
    STOP_AGENT = "stop_agent"            # Stop an agent (via SHUTDOWN)
    # CC usage
    CC_USAGE = "cc_usage"                # Request Claude Code account usage
    # Scratchpad sync
    SCRATCHPAD_GET = "scratchpad_get"        # Request scratchpad content
    SCRATCHPAD_SET = "scratchpad_set"        # Push updated scratchpad content
    SCRATCHPAD_RESPONSE = "scratchpad_response"  # Router response with scratchpad data
    # Per-conversation todo sync
    TODO_GET = "todo_get"                    # Request todo list content
    TODO_MUTATE = "todo_mutate"              # Add/update/delete/reorder todos
    TODO_RESPONSE = "todo_response"          # Router response with todo data
    # Calendar read-only sync
    CALENDAR_GET = "calendar_get"            # Request calendar events
    CALENDAR_RESPONSE = "calendar_response"  # Router response with calendar events


def generate_message_id() -> str:
    """Generate a unique message ID."""
    return f"msg-{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    """Current UTC timestamp in ISO format (for consistent sorting)."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Attachment:
    """A per-message reference to an uploaded attachment blob.

    Stored messages keep ``url`` unset. The router fills signed URLs only on
    transient delivery/rendering copies.
    """
    id: str
    name: str
    size: int
    mime: str
    sha256: str
    url: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Attachment":
        """Hydrate an attachment reference, ignoring unknown/client URL fields."""
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            size=int(data.get("size", 0) or 0),
            mime=str(data.get("mime", "application/octet-stream") or "application/octet-stream"),
            sha256=str(data.get("sha256", "")),
            url=None,
        )

    def canonical(self) -> "Attachment":
        """Return a persisted-safe copy with no signed URL."""
        return Attachment(
            id=self.id,
            name=self.name,
            size=self.size,
            mime=self.mime,
            sha256=self.sha256,
            url=None,
        )


def to_local_display(ts: "datetime | str | None") -> str:
    """Convert a timestamp to local time for LLM display.

    Storage stays UTC — this is purely for rendering in history XML so the
    LLM sees timestamps that match the system clock and ``current_time`` tool.

    Returns e.g. ``"2026-03-09 14:51 CDT"``.
    """
    if ts is None:
        return ""
    try:
        if isinstance(ts, str):
            if not ts:
                return ""
            dt = datetime.fromisoformat(ts)
        else:
            dt = ts  # already a datetime
        # Convert to local timezone
        if dt.tzinfo is not None:
            dt = dt.astimezone()  # system local
        else:
            # Naive datetime — assume UTC (our storage convention)
            dt = dt.replace(tzinfo=timezone.utc).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M %Z")
    except (ValueError, TypeError):
        return str(ts)


@dataclass
class Message:
    """
    Core message structure for mesh communication.

    Every message has:
    - id: Unique identifier
    - from_node: Sender node ID (e.g., "user:yourname", "agent:researcher")
    - to_node: Recipient node ID or "router" for control messages
    - type: MessageType indicating the purpose
    - content: The actual payload (string for messages, dict for structured data)
    - timestamp: When the message was created
    - in_reply_to: Optional reference to a previous message ID
    - metadata: Optional additional data (tool name, control action, etc.)
    """
    from_node: str
    to_node: str
    type: MessageType
    content: str | dict[str, Any]
    id: str = field(default_factory=generate_message_id)
    timestamp: str = field(default_factory=now_iso)
    in_reply_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    attachments: list[Attachment] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        d = asdict(self)
        d["type"] = self.type.value
        return json.dumps(d)

    @classmethod
    def from_json(cls, data: str | bytes) -> Message:
        """Deserialize from JSON string."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        d = json.loads(data)
        d["type"] = MessageType(d["type"])
        d["attachments"] = [
            Attachment.from_dict(a)
            for a in d.get("attachments", [])
            if isinstance(a, dict)
        ]
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def reply(
        self,
        content: str | dict[str, Any],
        type: MessageType = MessageType.MESSAGE,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Create a reply to this message, swapping from/to."""
        return Message(
            from_node=self.to_node,
            to_node=self.from_node,
            type=type,
            content=content,
            in_reply_to=self.id,
            metadata=metadata or {},
        )


# Convenience constructors

def make_message(
    from_node: str,
    to_node: str,
    content: str,
    in_reply_to: str | None = None,
    attachments: list[Attachment] | None = None,
) -> Message:
    """Create a standard conversation message."""
    return Message(
        from_node=from_node,
        to_node=to_node,
        type=MessageType.MESSAGE,
        content=content,
        in_reply_to=in_reply_to,
        attachments=[a.canonical() for a in (attachments or [])],
    )


def make_control(
    from_node: str,
    action: ControlAction,
    target_node: str | None = None,
    config: dict[str, Any] | None = None,
) -> Message:
    """Create a control message for the router."""
    return Message(
        from_node=from_node,
        to_node="router",
        type=MessageType.CONTROL,
        content={"action": action.value, "target": target_node},
        metadata={"config": config} if config else {},
    )


def make_tool_request(
    from_node: str,
    to_node: str,
    tool_name: str,
    tool_args: dict[str, Any],
) -> Message:
    """Create a tool execution request."""
    return Message(
        from_node=from_node,
        to_node=to_node,
        type=MessageType.TOOL_REQUEST,
        content=tool_args,
        metadata={"tool": tool_name},
    )


def make_tool_result(
    from_node: str,
    to_node: str,
    result: Any,
    in_reply_to: str,
    success: bool = True,
    error: str | None = None,
) -> Message:
    """Create a tool result response."""
    return Message(
        from_node=from_node,
        to_node=to_node,
        type=MessageType.TOOL_RESULT,
        content={"result": result, "success": success, "error": error},
        in_reply_to=in_reply_to,
    )


def make_confirm_request(
    from_node: str,
    to_node: str,
    tool_name: str,
    tool_args: dict[str, Any],
    preview: str,
) -> Message:
    """
    Create a confirmation request for a sensitive tool.

    Args:
        from_node: The agent requesting confirmation
        to_node: The user to confirm with (original sender)
        tool_name: Name of the tool requiring confirmation
        tool_args: Arguments passed to the tool
        preview: Human-readable preview of the action
    """
    return Message(
        from_node=from_node,
        to_node=to_node,
        type=MessageType.CONFIRM_REQUEST,
        content={
            "tool_name": tool_name,
            "tool_args": tool_args,
            "preview": preview,
        },
    )


def make_confirm_response(
    from_node: str,
    to_node: str,
    in_reply_to: str,
    confirmed: bool,
) -> Message:
    """
    Create a response to a confirmation request.

    Args:
        from_node: The user responding
        to_node: The agent that requested confirmation
        in_reply_to: Message ID of the confirmation request
        confirmed: Whether the user approved the action
    """
    return Message(
        from_node=from_node,
        to_node=to_node,
        type=MessageType.CONFIRM_RESPONSE,
        content={"confirmed": confirmed},
        in_reply_to=in_reply_to,
    )


# Wire format: length-prefixed messages
# Format: 4-byte big-endian length + JSON payload

def encode_for_wire(msg: Message) -> bytes:
    """Encode a message for transmission over TCP."""
    payload = msg.to_json().encode("utf-8")
    length = len(payload)
    return length.to_bytes(4, "big") + payload


def decode_length_prefix(data: bytes) -> int:
    """Decode the 4-byte length prefix."""
    return int.from_bytes(data[:4], "big")


# =============================================================================
# Node ID parsing utilities
# =============================================================================

def parse_node_id(node_id: str) -> tuple[str, str, str | None]:
    """
    Parse a node ID into its components.

    Node ID formats:
    - Users: "user:{nickname}" -> ("user", nickname, None)
    - Agents: "agent:{type}:{nickname}" -> ("agent", type, nickname)
    - Legacy agents: "agent:{type}" -> ("agent", type, None)

    Returns:
        Tuple of (node_type, type_or_nickname, nickname_or_none)

    Examples:
        parse_node_id("user:yourname") -> ("user", "yourname", None)
        parse_node_id("agent:coder:alice") -> ("agent", "coder", "alice")
        parse_node_id("agent:coder") -> ("agent", "coder", None)
    """
    parts = node_id.split(":", 2)

    if len(parts) == 2:
        # "user:yourname" or "agent:coder" (legacy)
        return (parts[0], parts[1], None)
    elif len(parts) == 3:
        # "agent:coder:alice"
        return (parts[0], parts[1], parts[2])
    else:
        # Single component - shouldn't happen but handle gracefully
        return (parts[0], "", None)


def get_display_name(node_id: str) -> str:
    """
    Get a human-friendly display name for a node.

    For users: returns the nickname
    For agents: returns the nickname (or type if no nickname)

    Examples:
        get_display_name("user:yourname") -> "Alan"
        get_display_name("agent:coder:alice") -> "Alice"
        get_display_name("agent:coder") -> "Coder"
    """
    node_type, type_or_nick, nickname = parse_node_id(node_id)

    if node_type == "user":
        # User nickname is in type_or_nick
        return type_or_nick.capitalize()
    elif node_type == "agent":
        # Prefer nickname if available, otherwise use agent type
        name = nickname if nickname else type_or_nick
        return name.capitalize()
    else:
        return node_id


def build_agent_node_id(agent_type: str, nickname: str) -> str:
    """
    Build a full agent node ID from type and nickname.

    Example:
        build_agent_node_id("coder", "alice") -> "agent:coder:alice"
    """
    return f"agent:{agent_type}:{nickname}"


def build_user_node_id(nickname: str) -> str:
    """
    Build a user node ID from nickname.

    Example:
        build_user_node_id("yourname") -> "user:yourname"
    """
    return f"user:{nickname}"


# =============================================================================
# Presence messages
# =============================================================================

def make_presence(
    from_node: str,
    event: str,  # "join" or "leave"
    nickname: str,
    node_type: str,  # "user" or agent type like "coder"
    description: str = "",
    llm_backend: str = "",
    llm_model: str = "",
    hostname: str = "",
) -> Message:
    """
    Create a presence message announcing a node joining or leaving.

    Args:
        from_node: The node ID of the joining/leaving node
        event: "join" or "leave"
        nickname: The human-friendly nickname
        node_type: "user" for users, or agent type (e.g., "coder") for agents
        description: Optional description (e.g., project name)
        llm_backend: LLM backend name (e.g., "openai", "anthropic", "claude-code")
        llm_model: LLM model name (e.g., "gpt-5.1", "opus")
        hostname: Hostname where the node is running
    """
    content = {
        "event": event,
        "nickname": nickname,
        "node_type": node_type,
    }
    if description:
        content["description"] = description
    if llm_backend:
        content["llm_backend"] = llm_backend
    if llm_model:
        content["llm_model"] = llm_model
    if hostname:
        content["hostname"] = hostname
    return Message(
        from_node=from_node,
        to_node="broadcast",  # Router will broadcast to all nodes
        type=MessageType.PRESENCE,
        content=content,
    )


def make_status_request(
    from_node: str,
    to_node: str,
    num_messages: int = 5,
    diagnostics: bool = False,
) -> Message:
    """
    Request an agent's recent context (for supervision).

    Args:
        from_node: The user requesting status
        to_node: The agent to query
        num_messages: Number of recent messages to return
        diagnostics: If True, request full diagnostic report in addition to context.
                     When diagnostics=True with num_messages=0, only diagnostics are returned.
    """
    content: dict[str, Any] = {"num_messages": num_messages}
    if diagnostics:
        content["diagnostics"] = True
    return Message(
        from_node=from_node,
        to_node=to_node,
        type=MessageType.STATUS_REQUEST,
        content=content,
    )


def make_status_response(
    from_node: str,
    to_node: str,
    in_reply_to: str,
    context: list[dict],
    summary: str | None = None,
    current_activity: str | None = None,
    hostname: str | None = None,
    model: str | None = None,
    backend: str | None = None,
    working_directory: str | None = None,
    diagnostics: dict | None = None,
    status_summary: dict | None = None,
) -> Message:
    """
    Respond with agent's recent context.

    Args:
        from_node: The agent responding
        to_node: The user who requested
        in_reply_to: The status request message ID
        context: List of recent messages in format:
                 {"from": node_id, "content": str, "timestamp": str}
        summary: Optional summary of older context
        current_activity: Optional real-time activity (e.g., in-progress CC tool calls)
        hostname: Optional hostname where agent is running
        model: Optional LLM model name
        backend: Optional LLM backend type
        working_directory: Optional bash working directory
        diagnostics: Optional full diagnostic report (6 sections)
        status_summary: Optional heartbeat-lite status (state, context_tokens,
                        history_pct, memory_pool, memory_active, uptime_s, etc.)
    """
    content: dict[str, Any] = {"context": context}
    if summary:
        content["summary"] = summary
    if current_activity:
        content["current_activity"] = current_activity
    if hostname:
        content["hostname"] = hostname
    if model:
        content["model"] = model
    if backend:
        content["backend"] = backend
    if working_directory:
        content["working_directory"] = working_directory
    if diagnostics:
        content["diagnostics"] = diagnostics
    if status_summary:
        content["status_summary"] = status_summary
    return Message(
        from_node=from_node,
        to_node=to_node,
        type=MessageType.STATUS_RESPONSE,
        content=content,
        in_reply_to=in_reply_to,
    )


def make_register_push_token(
    from_node: str,
    fcm_token: str,
) -> Message:
    """
    Create a message to register an FCM push notification token.

    Args:
        from_node: The node registering the token
        fcm_token: The Firebase Cloud Messaging token from the device
    """
    return Message(
        from_node=from_node,
        to_node="router",
        type=MessageType.CONTROL,
        content={
            "action": ControlAction.REGISTER_PUSH_TOKEN.value,
            "fcm_token": fcm_token,
        },
    )


# =============================================================================
# Channel operations
# =============================================================================

def is_channel_address(address: str) -> bool:
    """Check if an address is a channel (e.g., 'channel:research')."""
    return address.startswith("channel:")


def parse_channel_name(address: str) -> str | None:
    """
    Extract channel name from a channel address.

    Returns None if not a valid channel address.

    Example:
        parse_channel_name("channel:research") -> "research"
        parse_channel_name("user:yourname") -> None
    """
    if address.startswith("channel:"):
        return address[8:]  # len("channel:") == 8
    return None


def build_channel_address(name: str) -> str:
    """
    Build a channel address from a channel name.

    Example:
        build_channel_address("research") -> "channel:research"
    """
    return f"channel:{name}"


def make_channel_create(
    from_node: str,
    channel_name: str,
    description: str = "",
) -> Message:
    """
    Create a request to create a new channel.

    Args:
        from_node: The node creating the channel (must be a user)
        channel_name: Name of the channel (without 'channel:' prefix)
        description: Optional description of the channel
    """
    return Message(
        from_node=from_node,
        to_node="router",
        type=MessageType.CONTROL,
        content={
            "action": ControlAction.CHANNEL_CREATE.value,
            "channel_name": channel_name,
            "description": description,
        },
    )


def make_channel_delete(
    from_node: str,
    channel_name: str,
) -> Message:
    """
    Create a request to delete a channel.

    Args:
        from_node: The node deleting the channel (must be a user)
        channel_name: Name of the channel to delete
    """
    return Message(
        from_node=from_node,
        to_node="router",
        type=MessageType.CONTROL,
        content={
            "action": ControlAction.CHANNEL_DELETE.value,
            "channel_name": channel_name,
        },
    )


def make_channel_join(
    from_node: str,
    channel_name: str,
) -> Message:
    """
    Create a request to join a channel.

    Args:
        from_node: The node joining the channel
        channel_name: Name of the channel to join
    """
    return Message(
        from_node=from_node,
        to_node="router",
        type=MessageType.CONTROL,
        content={
            "action": ControlAction.CHANNEL_JOIN.value,
            "channel_name": channel_name,
        },
    )


def make_channel_leave(
    from_node: str,
    channel_name: str,
) -> Message:
    """
    Create a request to leave a channel.

    Args:
        from_node: The node leaving the channel
        channel_name: Name of the channel to leave
    """
    return Message(
        from_node=from_node,
        to_node="router",
        type=MessageType.CONTROL,
        content={
            "action": ControlAction.CHANNEL_LEAVE.value,
            "channel_name": channel_name,
        },
    )


def make_channel_list(from_node: str) -> Message:
    """
    Create a request to list all channels.

    Args:
        from_node: The node requesting the list
    """
    return Message(
        from_node=from_node,
        to_node="router",
        type=MessageType.CONTROL,
        content={
            "action": ControlAction.CHANNEL_LIST.value,
        },
    )


def make_channel_members(
    from_node: str,
    channel_name: str,
) -> Message:
    """
    Create a request to list members of a channel.

    Args:
        from_node: The node requesting the member list
        channel_name: Name of the channel to query
    """
    return Message(
        from_node=from_node,
        to_node="router",
        type=MessageType.CONTROL,
        content={
            "action": ControlAction.CHANNEL_MEMBERS.value,
            "channel_name": channel_name,
        },
    )


def make_channel_invite(
    from_node: str,
    channel_name: str,
    node_id: str,
) -> Message:
    """
    Create an invite request to add a node to a channel.

    Args:
        from_node: The node sending the invite (must be a user)
        channel_name: Name of the channel
        node_id: The node ID to invite
    """
    return Message(
        from_node=from_node,
        to_node="router",
        type=MessageType.CONTROL,
        content={
            "action": ControlAction.CHANNEL_INVITE.value,
            "channel_name": channel_name,
            "node_id": node_id,
        },
    )


def make_channel_remove_member(
    from_node: str,
    channel_name: str,
    node_id: str,
    reason: str = "",
) -> Message:
    """
    Create a request to remove a member from a channel.

    Args:
        from_node: The node requesting the removal (must be a user)
        channel_name: Name of the channel
        node_id: The node ID to remove
        reason: Optional reason for the removal
    """
    content: dict[str, Any] = {
        "action": ControlAction.CHANNEL_REMOVE_MEMBER.value,
        "channel_name": channel_name,
        "node_id": node_id,
    }
    if reason:
        content["reason"] = reason
    return Message(
        from_node=from_node,
        to_node="router",
        type=MessageType.CONTROL,
        content=content,
    )


# =============================================================================
# Message history sync
# =============================================================================

def make_history_sync(
    from_node: str,
    conversation_id: str | None = None,
    since: str | None = None,
    limit: int = 500,
) -> Message:
    """
    Create a request to sync message history.

    Args:
        from_node: The node requesting sync
        conversation_id: Optional specific conversation to sync
        since: ISO timestamp - only return messages after this time
        limit: Maximum number of messages to return
    """
    content: dict[str, Any] = {
        "action": ControlAction.HISTORY_SYNC.value,
        "limit": limit,
    }
    if conversation_id:
        content["conversation_id"] = conversation_id
    if since:
        content["since"] = since

    return Message(
        from_node=from_node,
        to_node="router",
        type=MessageType.CONTROL,
        content=content,
    )


def make_history_response(
    to_node: str,
    messages: list[dict[str, Any]],
    read_receipts: dict[str, str] | None = None,
    conversation_id: str | None = None,
    has_more: bool = False,
) -> Message:
    """
    Create a history sync response.

    Args:
        to_node: The node that requested the sync
        messages: List of message dicts (serialized Message objects)
        read_receipts: Dict mapping conversation_id -> last_read_timestamp
        conversation_id: The specific conversation (None if all)
        has_more: Whether more messages are available
    """
    content: dict[str, Any] = {
        "action": ControlAction.HISTORY_RESPONSE.value,
        "messages": messages,
        "has_more": has_more,
    }
    if conversation_id:
        content["conversation_id"] = conversation_id
    if read_receipts:
        content["read_receipts"] = read_receipts

    return Message(
        from_node="router",
        to_node=to_node,
        type=MessageType.CONTROL,
        content=content,
    )


def make_mark_read(
    from_node: str,
    conversation_id: str,
    up_to_timestamp: str,
) -> Message:
    """
    Create a request to mark messages as read.

    Args:
        from_node: The node marking messages as read
        conversation_id: The conversation ID
        up_to_timestamp: ISO timestamp - mark all messages up to this point as read
    """
    return Message(
        from_node=from_node,
        to_node="router",
        type=MessageType.CONTROL,
        content={
            "action": ControlAction.MARK_READ.value,
            "conversation_id": conversation_id,
            "up_to_timestamp": up_to_timestamp,
        },
    )


# =============================================================================
# Remote agent shutdown
# =============================================================================

def make_shutdown_request(
    from_node: str,
    target_node: str,
    auth_token: str,
    reason: str = "",
) -> Message:
    """
    Create a shutdown request for a remote agent.

    Args:
        from_node: The node requesting the shutdown (user or agent)
        target_node: The agent to shut down (e.g., "agent:assistant:alice")
        auth_token: The MESH_AUTH_TOKEN for authentication
        reason: Optional reason for the shutdown
    """
    return Message(
        from_node=from_node,
        to_node=target_node,
        type=MessageType.CONTROL,
        content={
            "action": ControlAction.SHUTDOWN.value,
            "auth_token": auth_token,
            "reason": reason,
        },
    )


def make_shutdown_ack(
    from_node: str,
    to_node: str,
    in_reply_to: str,
) -> Message:
    """
    Create a shutdown acknowledgment before the agent shuts down.

    Args:
        from_node: The agent acknowledging shutdown
        to_node: The node that requested the shutdown
        in_reply_to: The shutdown request message ID
    """
    return Message(
        from_node=from_node,
        to_node=to_node,
        type=MessageType.CONTROL,
        content={
            "action": ControlAction.SHUTDOWN_ACK.value,
        },
        in_reply_to=in_reply_to,
    )


# =============================================================================
# Context management
# =============================================================================

def make_reset_context(
    from_node: str,
    target_node: str,
    reason: str = "",
) -> Message:
    """
    Create a reset context request for a remote agent.

    Clears the agent's conversation history. Used by test harnesses
    to ensure clean slate between problems.

    Args:
        from_node: The node requesting the reset
        target_node: The agent to reset (e.g., "agent:evalplus:agentic")
        reason: Optional reason for the reset
    """
    return Message(
        from_node=from_node,
        to_node=target_node,
        type=MessageType.CONTROL,
        content={
            "action": ControlAction.RESET_CONTEXT.value,
            "reason": reason,
        },
    )


# =============================================================================
# Tool activity notifications (real-time streaming)
# =============================================================================

def make_tool_activity(
    from_node: str,
    to_node: str,
    event_type: str,
    tool_name: str,
    tool_source: str,
    data: dict[str, Any] | None = None,
    in_reply_to: str | None = None,
) -> Message:
    """
    Create a tool activity notification for real-time streaming.

    Sent by agents to the user who triggered the current turn, allowing
    them to see tool calls and results as they happen.

    Args:
        from_node: The agent executing the tool
        to_node: The user to notify (original trigger sender)
        event_type: "tool_call" or "tool_result"
        tool_name: Name of the tool (e.g., "bash_exec", "gmail_send_message")
        tool_source: "cc" for Claude Code tools, "mesh" for mesh tools
        data: Event-specific data:
            For tool_call: {"args": {...}, "preview": "..."}
            For tool_result: {"result": "...", "success": bool, "error": "..."}
        in_reply_to: Optional reference to the triggering message
    """
    return Message(
        from_node=from_node,
        to_node=to_node,
        type=MessageType.TOOL_ACTIVITY,
        content={
            "event_type": event_type,
            "tool_name": tool_name,
            "tool_source": tool_source,
            "data": data or {},
        },
        in_reply_to=in_reply_to,
    )


# =============================================================================
# Scratchpad sync
# =============================================================================

def make_scratchpad_get(
    from_node: str,
    conversation_ids: list[str] | None = None,
) -> Message:
    """Request scratchpad content from the router."""
    content: dict[str, Any] = {
        "action": ControlAction.SCRATCHPAD_GET.value,
    }
    if conversation_ids is not None:
        content["conversation_ids"] = conversation_ids
    return Message(
        from_node=from_node,
        to_node="router",
        type=MessageType.CONTROL,
        content=content,
    )


def make_scratchpad_set(
    from_node: str,
    conversation_id: str,
    content_text: str,
    client_timestamp: str,
) -> Message:
    """Push updated scratchpad content.

    client_timestamp is the updated_at the client last received from
    the server — used for optimistic concurrency.
    """
    return Message(
        from_node=from_node,
        to_node="router",
        type=MessageType.CONTROL,
        content={
            "action": ControlAction.SCRATCHPAD_SET.value,
            "conversation_id": conversation_id,
            "text": content_text,
            "client_timestamp": client_timestamp,
        },
    )


def make_scratchpad_response(
    to_node: str,
    notes: dict[str, Any],
    in_reply_to: str | None = None,
) -> Message:
    """Router response with scratchpad data.

    notes is {conversation_id: {content, updated_at, updated_by}, ...}
    """
    return Message(
        from_node="router",
        to_node=to_node,
        type=MessageType.CONTROL,
        content={
            "action": ControlAction.SCRATCHPAD_RESPONSE.value,
            "notes": notes,
        },
        in_reply_to=in_reply_to,
    )


# =============================================================================
# Per-conversation todo sync
# =============================================================================

def make_todo_get(
    from_node: str,
    conversation_ids: list[str] | None = None,
    include_done: bool = True,
) -> Message:
    """Request todo lists from the router."""
    content: dict[str, Any] = {
        "action": ControlAction.TODO_GET.value,
        "include_done": include_done,
    }
    if conversation_ids is not None:
        content["conversation_ids"] = conversation_ids
    return Message(
        from_node=from_node,
        to_node="router",
        type=MessageType.CONTROL,
        content=content,
    )


def make_todo_mutate(
    from_node: str,
    conversation_id: str,
    op: str,
    payload: dict[str, Any] | None = None,
    expected_version: int | None = None,
) -> Message:
    """Request a todo mutation from the router."""
    content: dict[str, Any] = {
        "action": ControlAction.TODO_MUTATE.value,
        "conversation_id": conversation_id,
        "op": op,
        "payload": payload or {},
    }
    if expected_version is not None:
        content["expected_version"] = expected_version
    return Message(
        from_node=from_node,
        to_node="router",
        type=MessageType.CONTROL,
        content=content,
    )


def make_todo_response(
    to_node: str,
    todos: dict[str, Any],
    section_order: dict[str, Any] | None = None,
    accepted: bool | None = None,
    conversation_id: str | None = None,
    server_state: dict[str, Any] | None = None,
    error: str | None = None,
    in_reply_to: str | None = None,
) -> Message:
    """Router response with todo lists keyed by conversation id."""
    content: dict[str, Any] = {
        "action": ControlAction.TODO_RESPONSE.value,
        "todos": todos,
    }
    if section_order is not None:
        content["section_order"] = section_order
    if accepted is not None:
        content["accepted"] = accepted
    if conversation_id is not None:
        content["conversation_id"] = conversation_id
    if server_state is not None:
        content["server_state"] = server_state
    if error is not None:
        content["error"] = error
    return Message(
        from_node="router",
        to_node=to_node,
        type=MessageType.CONTROL,
        content=content,
        in_reply_to=in_reply_to,
    )


def make_calendar_response(
    to_node: str,
    date: str,
    events: list[dict[str, Any]],
    errors: list[str] | None = None,
    timezone: str | None = None,
    accounts: list[str] | None = None,
    in_reply_to: str | None = None,
) -> Message:
    """Router response with calendar events for a date."""
    content: dict[str, Any] = {
        "action": ControlAction.CALENDAR_RESPONSE.value,
        "date": date,
        "events": events,
        "errors": errors or [],
    }
    if timezone:
        content["timezone"] = timezone
    if accounts is not None:
        content["accounts"] = accounts
    return Message(
        from_node="router",
        to_node=to_node,
        type=MessageType.CONTROL,
        content=content,
        in_reply_to=in_reply_to,
    )
