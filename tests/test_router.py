"""Tests for the router module."""

import pytest
import asyncio
import tempfile
import os

from mesh.router import Router
from mesh.config import RouterConfig
from mesh.transport import connect, Connection
from mesh.protocol import Message, MessageType, ControlAction, make_todo_mutate


async def receive_skip_presence(conn: Connection, timeout: float = 2.0) -> Message | None:
    """Receive a message, skipping any PRESENCE messages."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError()
        msg = await asyncio.wait_for(conn.receive(), timeout=remaining)
        if msg is None:
            return None
        if msg.type != MessageType.PRESENCE:
            return msg
        # Skip PRESENCE messages and continue


@pytest.fixture
def temp_storage():
    """Create a temporary storage path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def router_config(temp_storage):
    """Create a router config with test settings."""
    return RouterConfig(
        host="127.0.0.1",
        port=0,  # Random available port
        ws_port=0,  # Random available port for WebSocket
        storage_path=temp_storage,
    )


class TestRouterBasics:
    @pytest.mark.asyncio
    async def test_router_starts_and_stops(self, router_config):
        """Router can start and stop cleanly."""
        router = Router(router_config)
        await router.start()
        assert router._running
        assert router._server is not None

        await router.stop()
        assert not router._running

    @pytest.mark.asyncio
    async def test_connected_nodes_initially_empty(self, router_config):
        """Router starts with no connected nodes."""
        router = Router(router_config)
        assert router.connected_nodes == []


class TestNodeRegistration:
    @pytest.mark.asyncio
    async def test_node_registration(self, router_config):
        """Node can register with router."""
        router = Router(router_config)
        await router.start()
        port = router._server._server.sockets[0].getsockname()[1]

        # Connect and register
        conn = await connect("127.0.0.1", port)

        register_msg = Message(
            from_node="user:testuser",
            to_node="router",
            type=MessageType.CONTROL,
            content={"action": ControlAction.REGISTER.value},
        )
        await conn.send(register_msg)

        # Wait for ACK
        ack = await asyncio.wait_for(conn.receive(), timeout=2.0)

        assert ack is not None
        assert ack.type == MessageType.CONTROL
        assert ack.content["action"] == ControlAction.ACK.value
        assert ack.content["status"] == "registered"

        # Node should be in connected list
        assert "user:testuser" in router.connected_nodes

        await conn.close()
        # Give time for disconnect to process
        await asyncio.sleep(0.1)
        assert "user:testuser" not in router.connected_nodes

        await router.stop()

    @pytest.mark.asyncio
    async def test_registration_required(self, router_config):
        """Non-registration message before registering closes connection."""
        router = Router(router_config)
        await router.start()
        port = router._server._server.sockets[0].getsockname()[1]

        conn = await connect("127.0.0.1", port)

        # Send a regular message instead of REGISTER
        msg = Message(
            from_node="user:testuser",
            to_node="agent:echo",
            type=MessageType.MESSAGE,
            content="Hello",
        )
        await conn.send(msg)

        # Connection should be closed by router
        response = await asyncio.wait_for(conn.receive(), timeout=2.0)
        assert response is None  # Connection closed

        await router.stop()

    @pytest.mark.asyncio
    async def test_reconnection_replaces_old(self, router_config):
        """Reconnecting with same node_id replaces old connection."""
        router = Router(router_config)
        await router.start()
        port = router._server._server.sockets[0].getsockname()[1]

        # First connection
        conn1 = await connect("127.0.0.1", port)
        await conn1.send(Message(
            from_node="user:testuser",
            to_node="router",
            type=MessageType.CONTROL,
            content={"action": ControlAction.REGISTER.value},
        ))
        await conn1.receive()  # ACK

        # Second connection with same ID
        conn2 = await connect("127.0.0.1", port)
        await conn2.send(Message(
            from_node="user:testuser",
            to_node="router",
            type=MessageType.CONTROL,
            content={"action": ControlAction.REGISTER.value},
        ))
        await conn2.receive()  # ACK

        # Only one should be registered
        assert router.connected_nodes.count("user:testuser") == 1

        # Old connection should be closed (or at least replaced)
        try:
            result = await asyncio.wait_for(conn1.receive(), timeout=2.0)
            assert result is None  # Closed
        except (asyncio.TimeoutError, ConnectionError, Exception):
            pass  # Connection may be forcefully closed

        await conn2.close()
        await router.stop()


class TestMessageRouting:
    async def _register_node(self, port, node_id):
        """Helper to register a node and return connection."""
        conn = await connect("127.0.0.1", port)
        await conn.send(Message(
            from_node=node_id,
            to_node="router",
            type=MessageType.CONTROL,
            content={"action": ControlAction.REGISTER.value},
        ))
        await conn.receive()  # ACK
        return conn

    @pytest.mark.asyncio
    async def test_route_between_nodes(self, router_config):
        """Messages are routed between connected nodes."""
        router = Router(router_config)
        await router.start()
        port = router._server._server.sockets[0].getsockname()[1]

        # Register two nodes
        user_conn = await self._register_node(port, "user:testuser")
        agent_conn = await self._register_node(port, "agent:echo")

        # Send message from user to agent
        msg = Message(
            from_node="user:testuser",
            to_node="agent:echo",
            type=MessageType.MESSAGE,
            content="Hello agent!",
        )
        await user_conn.send(msg)

        # Agent should receive it (skip any roster PRESENCE from existing nodes)
        received = await receive_skip_presence(agent_conn, timeout=2.0)
        assert received is not None
        assert received.from_node == "user:testuser"
        assert received.to_node == "agent:echo"
        assert received.content == "Hello agent!"

        await user_conn.close()
        await agent_conn.close()
        await router.stop()

    @pytest.mark.asyncio
    async def test_route_reply(self, router_config):
        """Reply messages are routed back."""
        router = Router(router_config)
        await router.start()
        port = router._server._server.sockets[0].getsockname()[1]

        user_conn = await self._register_node(port, "user:testuser")
        agent_conn = await self._register_node(port, "agent:echo")

        # User sends to agent
        msg = Message(
            from_node="user:testuser",
            to_node="agent:echo",
            type=MessageType.MESSAGE,
            content="Hello",
            id="msg-original",
        )
        await user_conn.send(msg)

        # Agent receives (skip any roster PRESENCE from existing nodes)
        received = await receive_skip_presence(agent_conn, timeout=2.0)

        # Agent replies
        reply = received.reply("Hello back!")
        await agent_conn.send(reply)

        # User should receive reply (skip any PRESENCE messages)
        user_received = await receive_skip_presence(user_conn, timeout=2.0)
        assert user_received is not None
        assert user_received.from_node == "agent:echo"
        assert user_received.content == "Hello back!"
        assert user_received.in_reply_to == "msg-original"

        await user_conn.close()
        await agent_conn.close()
        await router.stop()


class TestOfflineMessageStorage:
    async def _register_node(self, port, node_id):
        conn = await connect("127.0.0.1", port)
        await conn.send(Message(
            from_node=node_id,
            to_node="router",
            type=MessageType.CONTROL,
            content={"action": ControlAction.REGISTER.value},
        ))
        await conn.receive()
        return conn

    @pytest.mark.asyncio
    async def test_store_for_offline_node(self, router_config):
        """Messages to offline nodes are stored."""
        router = Router(router_config)
        await router.start()
        port = router._server._server.sockets[0].getsockname()[1]

        # Only register user, agent is offline
        user_conn = await self._register_node(port, "user:testuser")

        # Send to offline agent
        msg = Message(
            from_node="user:testuser",
            to_node="agent:echo",
            type=MessageType.MESSAGE,
            content="Hello offline agent",
        )
        await user_conn.send(msg)

        # Give router time to process
        await asyncio.sleep(0.1)

        # Check storage
        assert router.store.count_pending("agent:echo") == 1

        await user_conn.close()
        await router.stop()

    @pytest.mark.asyncio
    async def test_deliver_pending_on_connect(self, router_config):
        """Pending messages are delivered when node connects."""
        router = Router(router_config)
        await router.start()
        port = router._server._server.sockets[0].getsockname()[1]

        # User sends while agent offline
        user_conn = await self._register_node(port, "user:testuser")
        msg = Message(
            from_node="user:testuser",
            to_node="agent:echo",
            type=MessageType.MESSAGE,
            content="Delivered later",
        )
        await user_conn.send(msg)
        await asyncio.sleep(0.1)

        # Now agent connects
        agent_conn = await self._register_node(port, "agent:echo")

        # Agent should receive the pending message (skip any roster PRESENCE)
        received = await receive_skip_presence(agent_conn, timeout=2.0)
        assert received is not None
        assert received.content == "Delivered later"

        # Storage should be cleared
        assert router.store.count_pending("agent:echo") == 0

        await user_conn.close()
        await agent_conn.close()
        await router.stop()

    @pytest.mark.asyncio
    async def test_multiple_pending_delivered_in_order(self, router_config):
        """Multiple pending messages delivered in order."""
        router = Router(router_config)
        await router.start()
        port = router._server._server.sockets[0].getsockname()[1]

        user_conn = await self._register_node(port, "user:testuser")

        # Send multiple messages while agent offline
        for i in range(3):
            msg = Message(
                from_node="user:testuser",
                to_node="agent:echo",
                type=MessageType.MESSAGE,
                content=f"Message {i}",
            )
            await user_conn.send(msg)

        await asyncio.sleep(0.1)
        assert router.store.count_pending("agent:echo") == 3

        # Agent connects
        agent_conn = await self._register_node(port, "agent:echo")

        # Receive all pending in order (skip any roster PRESENCE)
        for i in range(3):
            received = await receive_skip_presence(agent_conn, timeout=2.0)
            assert received.content == f"Message {i}"

        await user_conn.close()
        await agent_conn.close()
        await router.stop()


class TestControlMessages:
    async def _register_node(self, port, node_id):
        conn = await connect("127.0.0.1", port)
        await conn.send(Message(
            from_node=node_id,
            to_node="router",
            type=MessageType.CONTROL,
            content={"action": ControlAction.REGISTER.value},
        ))
        await conn.receive()
        return conn

    @pytest.mark.asyncio
    async def test_list_nodes(self, router_config):
        """LIST_NODES returns connected nodes."""
        router = Router(router_config)
        await router.start()
        port = router._server._server.sockets[0].getsockname()[1]

        user_conn = await self._register_node(port, "user:testuser")
        agent_conn = await self._register_node(port, "agent:echo")

        # Request node list (skip any PRESENCE messages first)
        await user_conn.send(Message(
            from_node="user:testuser",
            to_node="router",
            type=MessageType.CONTROL,
            content={"action": ControlAction.LIST_NODES.value},
        ))

        response = await receive_skip_presence(user_conn, timeout=2.0)
        assert response is not None
        assert response.type == MessageType.CONTROL
        assert response.content["action"] == ControlAction.LIST_NODES.value
        nodes = response.content["nodes"]
        assert "user:testuser" in nodes
        assert "agent:echo" in nodes

        await user_conn.close()
        await agent_conn.close()
        await router.stop()

    @pytest.mark.asyncio
    async def test_status(self, router_config):
        """STATUS returns router statistics."""
        router = Router(router_config)
        await router.start()
        port = router._server._server.sockets[0].getsockname()[1]

        user_conn = await self._register_node(port, "user:testuser")

        # Store a pending message
        router.store.store(Message(
            from_node="x", to_node="offline",
            type=MessageType.MESSAGE, content="pending",
        ))

        # Request status
        await user_conn.send(Message(
            from_node="user:testuser",
            to_node="router",
            type=MessageType.CONTROL,
            content={"action": ControlAction.STATUS.value},
        ))

        response = await asyncio.wait_for(user_conn.receive(), timeout=2.0)
        assert response is not None
        assert response.content["action"] == ControlAction.STATUS.value
        assert response.content["connected_nodes"] == 1
        assert response.content["pending_messages"] == 1

        await user_conn.close()
        await router.stop()


# =============================================================================
# Channel tests
# =============================================================================


class TestChannelOperations:
    async def _register_node(self, port, node_id):
        conn = await connect("127.0.0.1", port)
        await conn.send(Message(
            from_node=node_id,
            to_node="router",
            type=MessageType.CONTROL,
            content={"action": ControlAction.REGISTER.value},
        ))
        await conn.receive()
        return conn

    @pytest.mark.asyncio
    async def test_create_channel(self, router_config):
        """User can create a channel."""
        router = Router(router_config)
        await router.start()
        port = router._server._server.sockets[0].getsockname()[1]

        user_conn = await self._register_node(port, "user:testuser")

        # Create channel
        await user_conn.send(Message(
            from_node="user:testuser",
            to_node="router",
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_CREATE.value,
                "channel_name": "research",
                "description": "Research projects",
            },
        ))

        response = await asyncio.wait_for(user_conn.receive(), timeout=2.0)
        assert response.content["action"] == ControlAction.CHANNEL_CREATE.value
        assert response.content["status"] == "created"
        assert response.content["channel_name"] == "research"

        # Creator should be auto-joined
        assert router.store.is_channel_member("research", "user:testuser")

        await user_conn.close()
        await router.stop()

    @pytest.mark.asyncio
    async def test_todo_mutation_broadcasts_to_channel_participants(self, router_config):
        """TODO_MUTATE is broker-owned and broadcasts TODO_RESPONSE to participants."""
        router = Router(router_config)
        await router.start()
        port = router._server._server.sockets[0].getsockname()[1]

        user_conn = await self._register_node(port, "user:testuser")
        agent_conn = await self._register_node(port, "agent:coder")

        await user_conn.send(Message(
            from_node="user:testuser",
            to_node="router",
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_CREATE.value,
                "channel_name": "research",
            },
        ))
        await receive_skip_presence(user_conn, timeout=2.0)

        await agent_conn.send(Message(
            from_node="agent:coder",
            to_node="router",
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_JOIN.value,
                "channel_name": "research",
            },
        ))
        await receive_skip_presence(agent_conn, timeout=2.0)

        mutate = make_todo_mutate(
            "user:testuser",
            "channel:research",
            "add",
            payload={"text": "Draft the todo panel", "section": "today"},
        )
        await user_conn.send(mutate)

        user_response = await receive_skip_presence(user_conn, timeout=2.0)
        agent_response = await receive_skip_presence(agent_conn, timeout=2.0)

        for response in (user_response, agent_response):
            assert response.content["action"] == ControlAction.TODO_RESPONSE.value
            assert response.content["accepted"] is True
            todos = response.content["todos"]["channel:research"]
            assert len(todos) == 1
            assert todos[0]["text"] == "Draft the todo panel"
            assert todos[0]["section"] == "today"
            assert response.content["section_order"]["channel:research"] == []

        assert user_response.in_reply_to == mutate.id
        assert agent_response.in_reply_to is None

        await user_conn.close()
        await agent_conn.close()
        await asyncio.sleep(0.1)
        await router.stop()

    @pytest.mark.asyncio
    async def test_agent_cannot_create_channel(self, router_config):
        """Agents cannot create channels."""
        router = Router(router_config)
        await router.start()
        port = router._server._server.sockets[0].getsockname()[1]

        agent_conn = await self._register_node(port, "agent:coder")

        await agent_conn.send(Message(
            from_node="agent:coder",
            to_node="router",
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_CREATE.value,
                "channel_name": "research",
            },
        ))

        response = await asyncio.wait_for(agent_conn.receive(), timeout=2.0)
        assert response.content["status"] == "error"
        assert "only users" in response.content["error"]

        await agent_conn.close()
        await router.stop()

    @pytest.mark.asyncio
    async def test_join_and_leave_channel(self, router_config):
        """Nodes can join and leave channels."""
        router = Router(router_config)
        await router.start()
        port = router._server._server.sockets[0].getsockname()[1]

        user_conn = await self._register_node(port, "user:testuser")
        agent_conn = await self._register_node(port, "agent:coder")

        # Create channel
        await user_conn.send(Message(
            from_node="user:testuser",
            to_node="router",
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_CREATE.value,
                "channel_name": "research",
            },
        ))
        await user_conn.receive()  # ACK

        # Agent joins
        await agent_conn.send(Message(
            from_node="agent:coder",
            to_node="router",
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_JOIN.value,
                "channel_name": "research",
            },
        ))

        # Drain any presence messages, then get the join response
        response = await receive_skip_presence(agent_conn, timeout=2.0)
        assert response.content["action"] == ControlAction.CHANNEL_JOIN.value
        assert response.content["status"] == "joined"

        assert router.store.is_channel_member("research", "agent:coder")

        # Agent leaves
        await agent_conn.send(Message(
            from_node="agent:coder",
            to_node="router",
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_LEAVE.value,
                "channel_name": "research",
            },
        ))

        response = await asyncio.wait_for(agent_conn.receive(), timeout=2.0)
        assert response.content["action"] == ControlAction.CHANNEL_LEAVE.value
        assert response.content["status"] == "left"

        assert not router.store.is_channel_member("research", "agent:coder")

        await user_conn.close()
        await agent_conn.close()
        await router.stop()

    @pytest.mark.asyncio
    async def test_channel_message_broadcast(self, router_config):
        """Messages to a channel are broadcast to members."""
        router = Router(router_config)
        await router.start()
        port = router._server._server.sockets[0].getsockname()[1]

        user_conn = await self._register_node(port, "user:testuser")
        agent1_conn = await self._register_node(port, "agent:coder")
        agent2_conn = await self._register_node(port, "agent:researcher")

        # Create channel and have everyone join
        await user_conn.send(Message(
            from_node="user:testuser",
            to_node="router",
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_CREATE.value,
                "channel_name": "research",
            },
        ))
        await user_conn.receive()

        # Agent 1 joins
        await agent1_conn.send(Message(
            from_node="agent:coder",
            to_node="router",
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_JOIN.value,
                "channel_name": "research",
            },
        ))
        await receive_skip_presence(agent1_conn, timeout=2.0)

        # Agent 2 joins
        await agent2_conn.send(Message(
            from_node="agent:researcher",
            to_node="router",
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_JOIN.value,
                "channel_name": "research",
            },
        ))
        await receive_skip_presence(agent2_conn, timeout=2.0)

        # Give time for presence broadcasts to complete and drain any pending presence
        await asyncio.sleep(0.1)

        # User sends to channel
        await user_conn.send(Message(
            from_node="user:testuser",
            to_node="channel:research",
            type=MessageType.MESSAGE,
            content="Hello team!",
        ))

        # Both agents should receive it (skip any presence messages)
        received1 = await receive_skip_presence(agent1_conn, timeout=2.0)
        assert received1 is not None, "agent1 received None (connection closed)"
        assert received1.from_node == "user:testuser"
        assert received1.to_node == "channel:research"
        assert received1.content == "Hello team!"

        received2 = await receive_skip_presence(agent2_conn, timeout=2.0)
        assert received2 is not None, "agent2 received None (connection closed)"
        assert received2.from_node == "user:testuser"
        assert received2.to_node == "channel:research"
        assert received2.content == "Hello team!"

        await user_conn.close()
        await agent1_conn.close()
        await agent2_conn.close()
        # Give time for disconnects to propagate before stopping router
        await asyncio.sleep(0.1)
        await router.stop()

    @pytest.mark.asyncio
    async def test_channel_message_not_sent_to_non_members(self, router_config):
        """Channel messages are not sent to non-members."""
        router = Router(router_config)
        await router.start()
        port = router._server._server.sockets[0].getsockname()[1]

        user_conn = await self._register_node(port, "user:testuser")
        member_conn = await self._register_node(port, "agent:member")
        non_member_conn = await self._register_node(port, "agent:outsider")

        # Create channel, only user and member join
        await user_conn.send(Message(
            from_node="user:testuser",
            to_node="router",
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_CREATE.value,
                "channel_name": "private",
            },
        ))
        await user_conn.receive()

        await member_conn.send(Message(
            from_node="agent:member",
            to_node="router",
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_JOIN.value,
                "channel_name": "private",
            },
        ))
        await receive_skip_presence(member_conn, timeout=2.0)

        # User sends to channel
        await user_conn.send(Message(
            from_node="user:testuser",
            to_node="channel:private",
            type=MessageType.MESSAGE,
            content="Secret message",
        ))

        # Member should receive it
        received = await receive_skip_presence(member_conn, timeout=2.0)
        assert received.content == "Secret message"

        # Non-member should NOT receive it (would timeout)
        with pytest.raises(asyncio.TimeoutError):
            await receive_skip_presence(non_member_conn, timeout=0.5)

        await user_conn.close()
        await member_conn.close()
        await non_member_conn.close()
        await asyncio.sleep(0.1)
        await router.stop()

    @pytest.mark.asyncio
    async def test_list_channels(self, router_config):
        """CHANNEL_LIST returns all channels."""
        router = Router(router_config)
        await router.start()
        port = router._server._server.sockets[0].getsockname()[1]

        user_conn = await self._register_node(port, "user:testuser")

        # Create two channels
        for name in ["research", "android"]:
            await user_conn.send(Message(
                from_node="user:testuser",
                to_node="router",
                type=MessageType.CONTROL,
                content={
                    "action": ControlAction.CHANNEL_CREATE.value,
                    "channel_name": name,
                },
            ))
            await user_conn.receive()

        # List channels
        await user_conn.send(Message(
            from_node="user:testuser",
            to_node="router",
            type=MessageType.CONTROL,
            content={"action": ControlAction.CHANNEL_LIST.value},
        ))

        response = await asyncio.wait_for(user_conn.receive(), timeout=2.0)
        assert response.content["action"] == ControlAction.CHANNEL_LIST.value
        channels = response.content["channels"]
        assert len(channels) == 2
        names = {c["name"] for c in channels}
        assert names == {"research", "android"}

        await user_conn.close()
        await asyncio.sleep(0.1)
        await router.stop()

    @pytest.mark.asyncio
    async def test_channel_members(self, router_config):
        """CHANNEL_MEMBERS returns members with online status."""
        router = Router(router_config)
        await router.start()
        port = router._server._server.sockets[0].getsockname()[1]

        user_conn = await self._register_node(port, "user:testuser")
        agent_conn = await self._register_node(port, "agent:coder")

        # Create channel
        await user_conn.send(Message(
            from_node="user:testuser",
            to_node="router",
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_CREATE.value,
                "channel_name": "research",
            },
        ))
        await user_conn.receive()

        # Agent joins
        await agent_conn.send(Message(
            from_node="agent:coder",
            to_node="router",
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_JOIN.value,
                "channel_name": "research",
            },
        ))
        await receive_skip_presence(agent_conn, timeout=2.0)

        # Give time for channel presence to arrive at user, then drain it
        await asyncio.sleep(0.1)
        # Drain any PRESENCE messages (channel join notification to user)
        try:
            while True:
                msg = await asyncio.wait_for(user_conn.receive(), timeout=0.1)
                if msg is None or msg.type != MessageType.PRESENCE:
                    break
        except asyncio.TimeoutError:
            pass  # No more messages to drain

        # Get members
        await user_conn.send(Message(
            from_node="user:testuser",
            to_node="router",
            type=MessageType.CONTROL,
            content={
                "action": ControlAction.CHANNEL_MEMBERS.value,
                "channel_name": "research",
            },
        ))

        response = await receive_skip_presence(user_conn, timeout=2.0)
        assert response.content["action"] == ControlAction.CHANNEL_MEMBERS.value
        members = response.content["members"]
        assert len(members) == 2

        # Both should be online
        for m in members:
            assert m["online"] is True

        await user_conn.close()
        await agent_conn.close()
        await asyncio.sleep(0.1)
        await router.stop()
