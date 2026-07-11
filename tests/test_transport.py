"""Tests for the transport module."""

import pytest
import asyncio

from mesh.transport import Connection, Server, connect
from mesh.protocol import Message, MessageType


class TestConnection:
    """Tests for the Connection class using mock streams."""

    @pytest.fixture
    def mock_streams(self):
        """Create mock StreamReader and StreamWriter."""
        reader = asyncio.StreamReader()

        # Create a simple mock writer
        class MockWriter:
            def __init__(self):
                self.data = b""
                self._closed = False
                self._extra = {"peername": ("127.0.0.1", 12345)}

            def write(self, data):
                self.data += data

            async def drain(self):
                pass

            def get_extra_info(self, key):
                return self._extra.get(key)

            def close(self):
                self._closed = True

            async def wait_closed(self):
                pass

        return reader, MockWriter()

    @pytest.mark.asyncio
    async def test_connection_remote_address(self, mock_streams):
        """Connection reports remote address."""
        reader, writer = mock_streams
        conn = Connection(reader, writer)
        assert conn.remote_address == "127.0.0.1:12345"

    @pytest.mark.asyncio
    async def test_connection_send(self, mock_streams):
        """Connection sends length-prefixed message."""
        reader, writer = mock_streams
        conn = Connection(reader, writer)

        msg = Message(
            from_node="user:test",
            to_node="agent:test",
            type=MessageType.MESSAGE,
            content="Hello",
        )
        await conn.send(msg)

        # Check that data was written
        assert len(writer.data) > 4  # At least length prefix + some content
        # First 4 bytes are length
        length = int.from_bytes(writer.data[:4], "big")
        assert len(writer.data) == 4 + length

    @pytest.mark.asyncio
    async def test_connection_receive(self, mock_streams):
        """Connection receives and parses messages."""
        reader, writer = mock_streams
        conn = Connection(reader, writer)

        # Prepare test message
        msg = Message(
            from_node="sender",
            to_node="receiver",
            type=MessageType.MESSAGE,
            content="Test content",
        )
        # Encode and feed to reader
        payload = msg.to_json().encode("utf-8")
        length_prefix = len(payload).to_bytes(4, "big")
        reader.feed_data(length_prefix + payload)
        reader.feed_eof()

        # Receive should return the message
        received = await conn.receive()
        assert received is not None
        assert received.from_node == "sender"
        assert received.content == "Test content"

    @pytest.mark.asyncio
    async def test_connection_receive_eof(self, mock_streams):
        """Connection returns None on EOF."""
        reader, writer = mock_streams
        conn = Connection(reader, writer)

        # Feed EOF immediately
        reader.feed_eof()

        received = await conn.receive()
        assert received is None
        assert conn.is_closed

    @pytest.mark.asyncio
    async def test_connection_close(self, mock_streams):
        """Connection closes properly."""
        reader, writer = mock_streams
        conn = Connection(reader, writer)

        assert not conn.is_closed
        await conn.close()
        assert conn.is_closed
        assert writer._closed

    @pytest.mark.asyncio
    async def test_connection_send_when_closed(self, mock_streams):
        """Sending on closed connection raises error."""
        reader, writer = mock_streams
        conn = Connection(reader, writer)
        await conn.close()

        msg = Message(
            from_node="a", to_node="b",
            type=MessageType.MESSAGE, content="test",
        )
        with pytest.raises(ConnectionError):
            await conn.send(msg)

    @pytest.mark.asyncio
    async def test_connection_node_id(self, mock_streams):
        """Connection can have a node_id assigned."""
        reader, writer = mock_streams
        conn = Connection(reader, writer)

        assert conn.node_id is None
        conn.node_id = "user:testuser"
        assert conn.node_id == "user:testuser"


class TestServerClientIntegration:
    """Integration tests using real TCP connections."""

    @pytest.mark.asyncio
    async def test_server_accepts_connection(self):
        """Server accepts client connections."""
        connected = asyncio.Event()
        received_conn = None

        async def on_connection(conn):
            nonlocal received_conn
            received_conn = conn
            connected.set()

        server = Server("127.0.0.1", 0, on_connection)  # Port 0 = random available port
        await server.start()

        # Get the actual port
        port = server._server.sockets[0].getsockname()[1]

        # Connect as client
        client_conn = await connect("127.0.0.1", port)

        # Wait for server to accept
        await asyncio.wait_for(connected.wait(), timeout=2.0)

        assert received_conn is not None
        assert client_conn is not None

        await client_conn.close()
        await server.stop()

    @pytest.mark.asyncio
    async def test_message_exchange(self):
        """Client and server can exchange messages."""
        server_received = []
        client_ready = asyncio.Event()

        async def on_connection(conn):
            # Wait for message from client
            msg = await conn.receive()
            server_received.append(msg)
            # Send reply
            reply = msg.reply("Received!")
            await conn.send(reply)

        server = Server("127.0.0.1", 0, on_connection)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        # Connect as client
        client_conn = await connect("127.0.0.1", port)

        # Send message
        msg = Message(
            from_node="client",
            to_node="server",
            type=MessageType.MESSAGE,
            content="Hello from client",
        )
        await client_conn.send(msg)

        # Receive reply
        reply = await asyncio.wait_for(client_conn.receive(), timeout=2.0)

        assert len(server_received) == 1
        assert server_received[0].content == "Hello from client"
        assert reply is not None
        assert reply.content == "Received!"

        await client_conn.close()
        await server.stop()

    @pytest.mark.asyncio
    async def test_multiple_messages(self):
        """Multiple messages can be sent on same connection."""
        messages_received = []

        async def on_connection(conn):
            while True:
                msg = await conn.receive()
                if msg is None:
                    break
                messages_received.append(msg.content)

        server = Server("127.0.0.1", 0, on_connection)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        client_conn = await connect("127.0.0.1", port)

        # Send multiple messages
        for i in range(5):
            msg = Message(
                from_node="client",
                to_node="server",
                type=MessageType.MESSAGE,
                content=f"Message {i}",
            )
            await client_conn.send(msg)

        # Close to signal end
        await client_conn.close()

        # Give server time to process
        await asyncio.sleep(0.1)

        assert messages_received == [f"Message {i}" for i in range(5)]

        await server.stop()

    @pytest.mark.asyncio
    async def test_multiple_clients(self):
        """Server handles multiple concurrent clients."""
        connections = []
        all_connected = asyncio.Event()
        expected_clients = 3

        async def on_connection(conn):
            connections.append(conn)
            if len(connections) >= expected_clients:
                all_connected.set()
            # Keep connection open
            while True:
                msg = await conn.receive()
                if msg is None:
                    break

        server = Server("127.0.0.1", 0, on_connection)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        # Connect multiple clients
        clients = []
        for i in range(expected_clients):
            conn = await connect("127.0.0.1", port)
            clients.append(conn)

        # Wait for all to be accepted
        await asyncio.wait_for(all_connected.wait(), timeout=2.0)

        assert len(connections) == expected_clients

        # Cleanup
        for client in clients:
            await client.close()
        await server.stop()


class TestConnectFunction:
    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """connect raises on connection failure."""
        with pytest.raises(OSError):
            # Try to connect to a port that's not listening
            await connect("127.0.0.1", 59999)
