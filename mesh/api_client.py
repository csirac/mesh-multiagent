"""
Simple API client for programmatic mesh access.

Provides an easy-to-use interface for sending and receiving messages
without needing the full TUI.

Usage:
    from mesh.api_client import MeshClient

    async with MeshClient(
        ws_url="wss://your-host.example.com/mesh/ws",
        auth_token="your-token",
        nickname="mybot",
    ) as client:
        # Send a message to an agent
        response = await client.send("claude", "What time is it?")
        print(response.content)

        # Send to a channel
        await client.send("channel:mesh-dev", "Hello from my script!")

        # Listen for messages
        async for msg in client.listen():
            print(f"{msg.from_node}: {msg.content}")
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import AsyncIterator
from contextlib import asynccontextmanager

from .protocol import (
    Message,
    MessageType,
    ControlAction,
    make_message,
    build_user_node_id,
)
from .transport import connect, connect_ws, Connection, WebSocketClientConnection

logger = logging.getLogger(__name__)


class MeshClient:
    """
    Simple client for programmatic mesh access.

    Supports both TCP (local) and WebSocket (remote) connections.
    """

    def __init__(
        self,
        nickname: str,
        auth_token: str | None = None,
        ws_url: str | None = None,
        host: str = "127.0.0.1",
        port: int = 7700,
        use_tls: bool = False,
    ):
        """
        Initialize the mesh client.

        Args:
            nickname: Your user nickname (will become user:{nickname})
            auth_token: Authentication token (from MESH_AUTH_TOKEN or per-user token)
            ws_url: WebSocket URL for remote access (e.g., wss://your-host.example.com/mesh/ws)
                    If provided, uses WebSocket instead of TCP.
            host: TCP host (default: 127.0.0.1)
            port: TCP port (default: 7700)
            use_tls: Enable TLS for TCP connection
        """
        self.nickname = nickname
        self.node_id = build_user_node_id(nickname)
        self.auth_token = auth_token or os.environ.get("MESH_AUTH_TOKEN")
        self.ws_url = ws_url or os.environ.get("MESH_WS_URL")
        self.host = host
        self.port = int(os.environ.get("MESH_PORT", port))
        self.use_tls = use_tls

        self._conn: Connection | WebSocketClientConnection | None = None
        self._connected = False
        self._roster: dict[str, str] = {}  # nickname -> node_id
        self._pending_responses: dict[str, asyncio.Future] = {}  # msg_id -> future

    async def connect(self) -> None:
        """Connect to the mesh router and register."""
        if self._connected:
            return

        # Connect via WebSocket or TCP
        if self.ws_url:
            logger.info(f"Connecting via WebSocket to {self.ws_url}")
            self._conn = await connect_ws(self.ws_url)
        else:
            logger.info(f"Connecting via TCP to {self.host}:{self.port}")
            self._conn = await connect(self.host, self.port, self.use_tls)

        # Register with router
        register_msg = Message(
            from_node=self.node_id,
            to_node="router",
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.REGISTER.value,
                "node_type": "user",
                "nickname": self.nickname,
                "auth_token": self.auth_token,
            },
        )
        await self._conn.send(register_msg)

        # Wait for ACK
        response = await self._conn.receive()
        if response is None:
            raise ConnectionError("Connection closed during registration")

        if response.type == MessageType.CONTROL:
            content = response.content if isinstance(response.content, dict) else {}
            if content.get("action") == ControlAction.ACK.value:
                # Router sends status: "registered" on success, or error: "..." on failure
                if content.get("status") == "registered" or content.get("success"):
                    self._connected = True
                    logger.info(f"Registered as {self.node_id}")
                else:
                    error = content.get("error", "Unknown error")
                    raise ConnectionError(f"Registration failed: {error}")
            else:
                raise ConnectionError(f"Unexpected response: {response.content}")
        else:
            raise ConnectionError(f"Expected CONTROL message, got {response.type}")

    async def disconnect(self) -> None:
        """Disconnect from the mesh."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            self._connected = False
            logger.info("Disconnected from mesh")

    async def __aenter__(self) -> "MeshClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    def _resolve_target(self, target: str) -> str:
        """
        Resolve a target to a full node ID.

        - "claude" -> "agent:assistant:claude" (if in roster)
        - "channel:dev" -> "channel:dev" (pass through)
        - "user:yourname" -> "user:yourname" (pass through)
        """
        if ":" in target:
            return target

        # Check roster
        if target.lower() in self._roster:
            return self._roster[target.lower()]

        # Assume it's a user nickname
        return build_user_node_id(target)

    async def send(
        self,
        to: str,
        content: str,
        wait_response: bool = True,
        timeout: float = 60.0,
    ) -> Message | None:
        """
        Send a message and optionally wait for a response.

        Args:
            to: Target - can be:
                - nickname (e.g., "claude", "bob")
                - full node ID (e.g., "agent:assistant:claude")
                - channel (e.g., "channel:mesh-dev")
            content: Message content
            wait_response: If True, wait for a reply
            timeout: How long to wait for a response (seconds)

        Returns:
            The response Message if wait_response=True, else None
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        target = self._resolve_target(to)
        msg = make_message(self.node_id, target, content)

        await self._conn.send(msg)
        logger.debug(f"Sent to {target}: {content[:50]}...")

        if not wait_response:
            return None

        # Wait for a response addressed to us from the target
        return await self._wait_for_response(target, timeout)

    async def _wait_for_response(
        self,
        from_node: str,
        timeout: float,
    ) -> Message | None:
        """Wait for a response from a specific node."""
        deadline = asyncio.get_event_loop().time() + timeout

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.warning(f"Timeout waiting for response from {from_node}")
                return None

            try:
                msg = await asyncio.wait_for(
                    self._conn.receive(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                return None

            if msg is None:
                raise ConnectionError("Connection closed while waiting for response")

            # Handle different message types
            if msg.type == MessageType.MESSAGE:
                # Check if this is from our target
                if msg.from_node == from_node or from_node.startswith("channel:"):
                    return msg
                # Otherwise it's from someone else - could queue it
                logger.debug(f"Ignoring message from {msg.from_node} while waiting for {from_node}")

            elif msg.type == MessageType.PRESENCE:
                # Update roster
                self._handle_presence(msg)

            elif msg.type == MessageType.TOOL_ACTIVITY:
                # Log tool activity
                content = msg.content if isinstance(msg.content, dict) else {}
                tool = content.get("tool_name", "?")
                event = content.get("event_type", "?")
                logger.debug(f"Tool activity: {tool} ({event})")

    def _handle_presence(self, msg: Message) -> None:
        """Update roster from presence message."""
        content = msg.content if isinstance(msg.content, dict) else {}
        event = content.get("event", "")
        nickname = content.get("nickname", "")

        if event == "join" and nickname:
            self._roster[nickname.lower()] = msg.from_node
        elif event == "leave" and nickname:
            self._roster.pop(nickname.lower(), None)

    async def listen(self, timeout: float | None = None) -> AsyncIterator[Message]:
        """
        Listen for incoming messages.

        Args:
            timeout: How long to listen (None = forever)

        Yields:
            Messages as they arrive
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        deadline = None
        if timeout:
            deadline = asyncio.get_event_loop().time() + timeout

        while True:
            remaining = None
            if deadline:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    return

            try:
                msg = await asyncio.wait_for(
                    self._conn.receive(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                return

            if msg is None:
                logger.info("Connection closed")
                return

            # Handle presence updates
            if msg.type == MessageType.PRESENCE:
                self._handle_presence(msg)
                continue

            # Yield messages to caller
            if msg.type == MessageType.MESSAGE:
                yield msg

    async def reset_context(
        self,
        target: str,
        reason: str = "",
        timeout: float = 10.0,
    ) -> bool:
        """
        Reset an agent's conversation context.

        Sends a reset_context control message and waits for ACK.

        Args:
            target: Agent node ID (e.g., "agent:evalplus:glm47-v02")
            reason: Optional reason for the reset
            timeout: How long to wait for ACK

        Returns:
            True if agent acknowledged the reset
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        from mesh.protocol import make_reset_context

        msg = make_reset_context(self.node_id, target, reason)
        await self._conn.send(msg)
        logger.debug(f"Sent reset_context to {target}: {reason}")

        # Wait for ACK
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.warning(f"Timeout waiting for reset_context ACK from {target}")
                return False

            try:
                response = await asyncio.wait_for(
                    self._conn.receive(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for reset_context ACK from {target}")
                return False

            if response is None:
                return False

            if response.type == MessageType.CONTROL:
                content = response.content if isinstance(response.content, dict) else {}
                if content.get("reset_context"):
                    logger.debug(f"Got reset_context ACK from {target}")
                    return True

            # Ignore other messages while waiting for ACK

    async def create_channel(self, channel_name: str, description: str = "") -> bool:
        """
        Create a channel.

        Args:
            channel_name: Name without 'channel:' prefix
            description: Optional channel description

        Returns:
            True if successful (or already exists)
        """
        msg = Message(
            from_node=self.node_id,
            to_node="router",
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_CREATE.value,
                "channel_name": channel_name,
                "description": description,
            },
        )
        await self._conn.send(msg)

        # Wait for ACK
        response = await self._conn.receive()
        if response and response.type == MessageType.CONTROL:
            content = response.content if isinstance(response.content, dict) else {}
            # Success if created or already exists
            return content.get("success", False) or "already exists" in content.get("error", "")
        return False

    async def join_channel(self, channel_name: str) -> bool:
        """
        Join a channel.

        Args:
            channel_name: Name without 'channel:' prefix

        Returns:
            True if successful
        """
        msg = Message(
            from_node=self.node_id,
            to_node="router",
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_JOIN.value,
                "channel_name": channel_name,
            },
        )
        await self._conn.send(msg)

        # Wait for ACK
        response = await self._conn.receive()
        if response and response.type == MessageType.CONTROL:
            content = response.content if isinstance(response.content, dict) else {}
            return content.get("success", False)
        return False

    async def list_nodes(self) -> list[dict]:
        """
        List all connected nodes.

        Returns:
            List of node info dicts (node IDs as strings)
        """
        msg = Message(
            from_node=self.node_id,
            to_node="router",
            type=MessageType.CONTROL,
            content={"action": ControlAction.LIST_NODES.value},
        )
        await self._conn.send(msg)

        response = await self._conn.receive()
        if response and response.type == MessageType.CONTROL:
            content = response.content if isinstance(response.content, dict) else {}
            return content.get("nodes", [])
        return []

    async def list_agents(self) -> dict:
        """
        List available agent configurations and connected agents.

        Returns:
            Dict with:
                - configured: List of agent configs from mesh.yaml (agent_type, llm_model, etc.)
                - connected: List of currently connected agent node IDs
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        msg = Message(
            from_node=self.node_id,
            to_node="router",
            type=MessageType.CONTROL,
            content={"action": ControlAction.LIST_AGENTS.value},
        )
        await self._conn.send(msg)

        # Wait for the list_agents response, skipping presence messages
        for _ in range(20):  # Max 20 messages to skip
            response = await self._conn.receive()
            if response is None:
                break
            if response.type == MessageType.PRESENCE:
                self._handle_presence(response)
                continue
            if response.type == MessageType.CONTROL:
                content = response.content if isinstance(response.content, dict) else {}
                if content.get("action") == ControlAction.LIST_AGENTS.value:
                    return {
                        "configured": content.get("configured", []),
                        "connected": content.get("connected", []),
                    }
        return {"configured": [], "connected": []}

    async def start_agent(
        self,
        agent_type: str,
        nickname: str,
        backend: str | None = None,
        model: str | None = None,
        fresh: bool = False,
        controller: str | None = None,
        effort: str | None = None,
        timeout: float = 10.0,
    ) -> dict:
        """
        Start a new agent on the server.

        Args:
            agent_type: Agent type (e.g., "assistant", "coder", "researcher")
            nickname: Unique nickname for the agent
            backend: LLM backend name (optional)
            model: Model override (optional)
            fresh: Start without history (default: False)
            controller: Controller mode (optional):
                - "passthrough": No controller (direct LLM)
                - "task-fsm-v0": v0.1 task-based controller
                - "phase-flow-v02": v0.2 adaptive phase-flow controller
            effort: Effort level for v0.2 controller (optional):
                - "low": Fast, less thorough
                - "medium": Balanced (default)
                - "high": Thorough, more deliberate
            timeout: How long to wait for response

        Returns:
            Dict with:
                - success: bool
                - node_id: Full node ID if successful
                - pid: Process ID if successful
                - error: Error message if failed
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        content = {
            "action": ControlAction.START_AGENT.value,
            "agent_type": agent_type,
            "nickname": nickname,
        }
        if backend:
            content["backend"] = backend
        if model:
            content["model"] = model
        if fresh:
            content["fresh"] = True
        if controller:
            content["controller"] = controller
        if effort:
            content["effort"] = effort

        msg = Message(
            from_node=self.node_id,
            to_node="router",
            type=MessageType.CONTROL,
            content=content,
        )
        await self._conn.send(msg)

        try:
            response = await asyncio.wait_for(
                self._conn.receive(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return {"success": False, "error": "Timeout waiting for response"}

        if response and response.type == MessageType.CONTROL:
            content = response.content if isinstance(response.content, dict) else {}
            return {
                "success": content.get("success", False),
                "node_id": content.get("node_id"),
                "pid": content.get("pid"),
                "error": content.get("error"),
            }
        return {"success": False, "error": "Unexpected response"}

    async def stop_agent(
        self,
        target: str,
        reason: str = "",
        timeout: float = 10.0,
    ) -> bool:
        """
        Stop a running agent by sending a SHUTDOWN command.

        Args:
            target: Agent node ID (e.g., "agent:assistant:v02") or nickname
            reason: Optional reason for shutdown

        Returns:
            True if shutdown acknowledged, False otherwise
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        # Resolve target to full node ID
        target_id = self._resolve_target(target)

        # Build shutdown request
        msg = Message(
            from_node=self.node_id,
            to_node=target_id,
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.SHUTDOWN.value,
                "auth_token": self.auth_token,
                "reason": reason,
            },
        )
        await self._conn.send(msg)
        logger.info(f"Sent shutdown request to {target_id}")

        # Wait for acknowledgment
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            remaining = deadline - asyncio.get_event_loop().time()
            try:
                response = await asyncio.wait_for(
                    self._conn.receive(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for shutdown ack from {target_id}")
                return False

            if response is None:
                return False

            if response.type == MessageType.CONTROL:
                content = response.content if isinstance(response.content, dict) else {}
                if content.get("action") == ControlAction.SHUTDOWN_ACK.value:
                    logger.info(f"Agent {target_id} acknowledged shutdown")
                    return True

            # Handle presence updates (agent leaving)
            if response.type == MessageType.PRESENCE:
                self._handle_presence(response)
                content = response.content if isinstance(response.content, dict) else {}
                if content.get("event") == "leave" and response.from_node == target_id:
                    logger.info(f"Agent {target_id} has disconnected")
                    return True

        return False


# Convenience function for one-shot messages
async def send_message(
    to: str,
    content: str,
    nickname: str = "api-client",
    auth_token: str | None = None,
    ws_url: str | None = None,
    wait_response: bool = True,
    timeout: float = 60.0,
) -> Message | None:
    """
    Send a single message and optionally wait for a response.

    This is a convenience wrapper that handles connection/disconnection.

    Args:
        to: Target (nickname, node ID, or channel)
        content: Message content
        nickname: Your nickname
        auth_token: Auth token (uses MESH_AUTH_TOKEN env var if not provided)
        ws_url: WebSocket URL (uses MESH_WS_URL env var if not provided)
        wait_response: Whether to wait for a response
        timeout: Response timeout in seconds

    Returns:
        Response message if wait_response=True, else None

    Example:
        response = await send_message("claude", "What time is it?")
        print(response.content)
    """
    async with MeshClient(
        nickname=nickname,
        auth_token=auth_token,
        ws_url=ws_url,
    ) as client:
        return await client.send(to, content, wait_response, timeout)
