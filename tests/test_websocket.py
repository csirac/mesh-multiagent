"""Tests for WebSocket support in the router."""

import pytest
import asyncio
import tempfile
import os

import aiohttp

from mesh.router import Router
from mesh.config import RouterConfig
from mesh.protocol import Message, MessageType, ControlAction
from mesh.transport import connect


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
    """Create a router config with WebSocket enabled."""
    return RouterConfig(
        host="127.0.0.1",
        port=0,  # Random TCP port
        storage_path=temp_storage,
        ws_enabled=True,
        ws_port=0,  # Random WebSocket port - but we'll use a fixed one
    )


class TestWebSocketBasics:
    @pytest.mark.asyncio
    async def test_websocket_server_starts(self, temp_storage):
        """WebSocket server starts alongside TCP server."""
        config = RouterConfig(
            host="127.0.0.1",
            port=0,
            storage_path=temp_storage,
            ws_enabled=True,
            ws_port=18080,  # Fixed port for testing
        )
        router = Router(config)
        await router.start()

        assert router._running
        assert router._ws_runner is not None

        # Check health endpoint
        async with aiohttp.ClientSession() as session:
            async with session.get("http://127.0.0.1:18080/health") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"
                assert data["connected_nodes"] == 0

        await router.stop()

    @pytest.mark.asyncio
    async def test_websocket_disabled(self, temp_storage):
        """WebSocket server doesn't start when disabled."""
        config = RouterConfig(
            host="127.0.0.1",
            port=0,
            storage_path=temp_storage,
            ws_enabled=False,
        )
        router = Router(config)
        await router.start()

        assert router._running
        assert router._ws_runner is None

        await router.stop()


class TestWebSocketRegistration:
    @pytest.mark.asyncio
    async def test_websocket_node_registration(self, temp_storage):
        """Node can register via WebSocket."""
        config = RouterConfig(
            host="127.0.0.1",
            port=0,
            storage_path=temp_storage,
            ws_enabled=True,
            ws_port=18081,
        )
        router = Router(config)
        await router.start()

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect("http://127.0.0.1:18081/ws") as ws:
                # Send registration
                register_msg = Message(
                    from_node="user:websocket",
                    to_node="router",
                    type=MessageType.CONTROL,
                    content={"action": ControlAction.REGISTER.value},
                )
                await ws.send_str(register_msg.to_json())

                # Receive ACK
                msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                assert msg.type == aiohttp.WSMsgType.TEXT
                ack = Message.from_json(msg.data)
                assert ack.type == MessageType.CONTROL
                assert ack.content["action"] == ControlAction.ACK.value
                assert ack.content["status"] == "registered"

                # Check node is registered
                assert "user:websocket" in router.connected_nodes

        # After disconnect
        await asyncio.sleep(0.1)
        assert "user:websocket" not in router.connected_nodes

        await router.stop()


class TestWebSocketMessaging:
    @pytest.mark.asyncio
    async def test_websocket_to_tcp_messaging(self, temp_storage):
        """WebSocket node can send messages to TCP node."""
        config = RouterConfig(
            host="127.0.0.1",
            port=0,
            storage_path=temp_storage,
            ws_enabled=True,
            ws_port=18082,
        )
        router = Router(config)
        await router.start()
        tcp_port = router._server._server.sockets[0].getsockname()[1]

        # Register TCP node
        tcp_conn = await connect("127.0.0.1", tcp_port)
        await tcp_conn.send(Message(
            from_node="agent:tcp",
            to_node="router",
            type=MessageType.CONTROL,
            content={"action": ControlAction.REGISTER.value},
        ))
        await tcp_conn.receive()  # ACK

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect("http://127.0.0.1:18082/ws") as ws:
                # Register WebSocket node
                register_msg = Message(
                    from_node="user:websocket",
                    to_node="router",
                    type=MessageType.CONTROL,
                    content={"action": ControlAction.REGISTER.value},
                )
                await ws.send_str(register_msg.to_json())
                await ws.receive()  # ACK

                # Send message from WebSocket to TCP
                chat_msg = Message(
                    from_node="user:websocket",
                    to_node="agent:tcp",
                    type=MessageType.MESSAGE,
                    content="Hello from WebSocket!",
                )
                await ws.send_str(chat_msg.to_json())

        # TCP node should receive the message (may need to skip PRESENCE messages)
        received = None
        for _ in range(5):  # Try a few times to skip PRESENCE
            received = await asyncio.wait_for(tcp_conn.receive(), timeout=2.0)
            if received and received.type == MessageType.MESSAGE:
                break

        assert received is not None
        assert received.type == MessageType.MESSAGE
        assert received.from_node == "user:websocket"
        assert received.content == "Hello from WebSocket!"

        await tcp_conn.close()
        await router.stop()

    @pytest.mark.asyncio
    async def test_tcp_to_websocket_messaging(self, temp_storage):
        """TCP node can send messages to WebSocket node."""
        config = RouterConfig(
            host="127.0.0.1",
            port=0,
            storage_path=temp_storage,
            ws_enabled=True,
            ws_port=18083,
        )
        router = Router(config)
        await router.start()
        tcp_port = router._server._server.sockets[0].getsockname()[1]

        # Register TCP node
        tcp_conn = await connect("127.0.0.1", tcp_port)
        await tcp_conn.send(Message(
            from_node="agent:tcp",
            to_node="router",
            type=MessageType.CONTROL,
            content={"action": ControlAction.REGISTER.value},
        ))
        await tcp_conn.receive()  # ACK

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect("http://127.0.0.1:18083/ws") as ws:
                # Register WebSocket node
                register_msg = Message(
                    from_node="user:websocket",
                    to_node="router",
                    type=MessageType.CONTROL,
                    content={"action": ControlAction.REGISTER.value},
                )
                await ws.send_str(register_msg.to_json())
                await ws.receive()  # ACK

                # Send message from TCP to WebSocket
                chat_msg = Message(
                    from_node="agent:tcp",
                    to_node="user:websocket",
                    type=MessageType.MESSAGE,
                    content="Hello from TCP!",
                )
                await tcp_conn.send(chat_msg)

                # WebSocket should receive the message (may need to skip PRESENCE)
                received = None
                for _ in range(5):
                    msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        parsed = Message.from_json(msg.data)
                        if parsed.type == MessageType.MESSAGE:
                            received = parsed
                            break

                assert received is not None
                assert received.from_node == "agent:tcp"
                assert received.content == "Hello from TCP!"

        await tcp_conn.close()
        await router.stop()

    @pytest.mark.asyncio
    async def test_websocket_bidirectional(self, temp_storage):
        """Two WebSocket nodes can communicate."""
        config = RouterConfig(
            host="127.0.0.1",
            port=0,
            storage_path=temp_storage,
            ws_enabled=True,
            ws_port=18084,
        )
        router = Router(config)
        await router.start()

        async with aiohttp.ClientSession() as session:
            # Connect two WebSocket clients
            async with session.ws_connect("http://127.0.0.1:18084/ws") as ws1:
                async with session.ws_connect("http://127.0.0.1:18084/ws") as ws2:
                    # Register both
                    await ws1.send_str(Message(
                        from_node="user:alice",
                        to_node="router",
                        type=MessageType.CONTROL,
                        content={"action": ControlAction.REGISTER.value},
                    ).to_json())
                    await ws1.receive()  # ACK

                    await ws2.send_str(Message(
                        from_node="user:bob",
                        to_node="router",
                        type=MessageType.CONTROL,
                        content={"action": ControlAction.REGISTER.value},
                    ).to_json())
                    await ws2.receive()  # ACK

                    # Alice sends to Bob
                    await ws1.send_str(Message(
                        from_node="user:alice",
                        to_node="user:bob",
                        type=MessageType.MESSAGE,
                        content="Hi Bob!",
                    ).to_json())

                    # Bob receives (skip PRESENCE messages)
                    received = None
                    for _ in range(5):
                        msg = await asyncio.wait_for(ws2.receive(), timeout=2.0)
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            parsed = Message.from_json(msg.data)
                            if parsed.type == MessageType.MESSAGE:
                                received = parsed
                                break

                    assert received is not None
                    assert received.from_node == "user:alice"
                    assert received.content == "Hi Bob!"

        await router.stop()


class TestWebSocketOfflineStorage:
    @pytest.mark.asyncio
    async def test_websocket_messages_stored_for_offline(self, temp_storage):
        """Messages from WebSocket to offline node are stored."""
        config = RouterConfig(
            host="127.0.0.1",
            port=0,
            storage_path=temp_storage,
            ws_enabled=True,
            ws_port=18085,
        )
        router = Router(config)
        await router.start()

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect("http://127.0.0.1:18085/ws") as ws:
                # Register
                await ws.send_str(Message(
                    from_node="user:websocket",
                    to_node="router",
                    type=MessageType.CONTROL,
                    content={"action": ControlAction.REGISTER.value},
                ).to_json())
                await ws.receive()  # ACK

                # Send to offline node
                await ws.send_str(Message(
                    from_node="user:websocket",
                    to_node="agent:offline",
                    type=MessageType.MESSAGE,
                    content="Hello offline!",
                ).to_json())

        # Give time for processing
        await asyncio.sleep(0.1)

        # Check storage
        assert router.store.count_pending("agent:offline") == 1

        await router.stop()

    @pytest.mark.asyncio
    async def test_pending_delivered_to_websocket(self, temp_storage):
        """Pending messages are delivered when WebSocket node connects."""
        config = RouterConfig(
            host="127.0.0.1",
            port=0,
            storage_path=temp_storage,
            ws_enabled=True,
            ws_port=18086,
        )
        router = Router(config)
        await router.start()
        tcp_port = router._server._server.sockets[0].getsockname()[1]

        # TCP node sends while WebSocket is offline
        tcp_conn = await connect("127.0.0.1", tcp_port)
        await tcp_conn.send(Message(
            from_node="agent:tcp",
            to_node="router",
            type=MessageType.CONTROL,
            content={"action": ControlAction.REGISTER.value},
        ))
        await tcp_conn.receive()  # ACK

        await tcp_conn.send(Message(
            from_node="agent:tcp",
            to_node="user:websocket",
            type=MessageType.MESSAGE,
            content="Message while offline",
        ))

        await asyncio.sleep(0.1)
        assert router.store.count_pending("user:websocket") == 1

        # Now WebSocket connects
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect("http://127.0.0.1:18086/ws") as ws:
                # Register
                await ws.send_str(Message(
                    from_node="user:websocket",
                    to_node="router",
                    type=MessageType.CONTROL,
                    content={"action": ControlAction.REGISTER.value},
                ).to_json())
                await ws.receive()  # ACK

                # Should receive pending message
                received = None
                for _ in range(5):
                    msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        parsed = Message.from_json(msg.data)
                        if parsed.type == MessageType.MESSAGE:
                            received = parsed
                            break

                assert received is not None
                assert received.content == "Message while offline"

        # Storage should be cleared
        assert router.store.count_pending("user:websocket") == 0

        await tcp_conn.close()
        await router.stop()
