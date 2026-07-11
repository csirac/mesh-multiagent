# SPDX-License-Identifier: Apache-2.0
"""
Transport layer for mesh communication.

Provides async TCP connection handling with length-prefixed message framing.
Also supports WebSocket connections for browser/mobile clients.
Designed to be network-ready (can work over localhost or remote addresses).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable, TYPE_CHECKING

from .protocol import Message, encode_for_wire, decode_length_prefix

if TYPE_CHECKING:
    import aiohttp.web

logger = logging.getLogger(__name__)


class Connection:
    """
    A single TCP connection that can send/receive Messages.

    Handles the length-prefixed wire format transparently.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        node_id: str | None = None,
    ):
        self.reader = reader
        self.writer = writer
        self.node_id = node_id  # Set after registration
        self._closed = False

    @property
    def remote_address(self) -> str:
        """Get the remote address as a string."""
        peername = self.writer.get_extra_info("peername")
        if peername:
            return f"{peername[0]}:{peername[1]}"
        return "unknown"

    async def send(self, msg: Message) -> None:
        """Send a message over this connection."""
        if self._closed:
            raise ConnectionError("Connection is closed")
        data = encode_for_wire(msg)
        self.writer.write(data)
        await self.writer.drain()
        logger.debug(
            f"SEND [{self.node_id or self.remote_address}] "
            f"type={msg.type.value} from={msg.from_node} to={msg.to_node} "
            f"id={msg.id[:8]}... content_preview={str(msg.content)[:100]!r}"
        )

    async def receive(self) -> Message | None:
        """
        Receive a message from this connection.

        Returns None if connection is closed.
        """
        if self._closed:
            return None

        try:
            # Read 4-byte length prefix
            length_bytes = await self.reader.readexactly(4)
            length = decode_length_prefix(length_bytes)

            # Read the payload
            payload = await self.reader.readexactly(length)
            msg = Message.from_json(payload)
            logger.debug(
                f"RECV [{self.node_id or self.remote_address}] "
                f"type={msg.type.value} from={msg.from_node} to={msg.to_node} "
                f"id={msg.id[:8]}... content_preview={str(msg.content)[:100]!r}"
            )
            return msg

        except asyncio.IncompleteReadError:
            logger.info(f"Connection closed by {self.node_id or self.remote_address}")
            self._closed = True
            return None
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            self._closed = True
            return None

    async def close(self) -> None:
        """Close this connection."""
        if not self._closed:
            self._closed = True
            self.writer.close()
            await self.writer.wait_closed()
            logger.debug(f"Closed connection to {self.node_id or self.remote_address}")

    @property
    def is_closed(self) -> bool:
        return self._closed


class WebSocketConnection:
    """
    A WebSocket connection that can send/receive Messages.

    Uses JSON text frames (no length prefix needed for WebSocket).
    Compatible with browser/mobile WebSocket clients.
    """

    def __init__(
        self,
        ws: "aiohttp.web.WebSocketResponse",
        node_id: str | None = None,
        remote_address: str = "unknown",
    ):
        self.ws = ws
        self.node_id = node_id
        self._remote_address = remote_address
        self._closed = False

    @property
    def remote_address(self) -> str:
        """Get the remote address as a string."""
        return self._remote_address

    async def send(self, msg: Message) -> None:
        """Send a message over this connection."""
        if self._closed:
            raise ConnectionError("Connection is closed")
        await self.ws.send_str(msg.to_json())
        logger.debug(
            f"SEND [WS {self.node_id or self.remote_address}] "
            f"type={msg.type.value} from={msg.from_node} to={msg.to_node} "
            f"id={msg.id[:8]}... content_preview={str(msg.content)[:100]!r}"
        )

    async def receive(self) -> Message | None:
        """
        Receive a message from this connection.

        Returns None if connection is closed.
        """
        if self._closed:
            return None

        try:
            import aiohttp
            ws_msg = await self.ws.receive()

            if ws_msg.type == aiohttp.WSMsgType.TEXT:
                msg = Message.from_json(ws_msg.data)
                logger.debug(
                    f"RECV [WS {self.node_id or self.remote_address}] "
                    f"type={msg.type.value} from={msg.from_node} to={msg.to_node} "
                    f"id={msg.id[:8]}... content_preview={str(msg.content)[:100]!r}"
                )
                return msg
            elif ws_msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                logger.info(f"WebSocket closed by {self.node_id or self.remote_address}")
                self._closed = True
                return None
            else:
                # Ignore other message types (PING, PONG, BINARY, etc.)
                return await self.receive()

        except Exception as e:
            logger.error(f"Error receiving WebSocket message: {e}")
            self._closed = True
            return None

    async def close(self) -> None:
        """Close this connection."""
        if not self._closed:
            self._closed = True
            await self.ws.close()
            logger.debug(f"Closed WebSocket connection to {self.node_id or self.remote_address}")

    @property
    def is_closed(self) -> bool:
        return self._closed


class Server:
    """
    TCP server that accepts connections and dispatches to a handler.

    Used by the Router to accept node connections.
    """

    def __init__(
        self,
        host: str,
        port: int,
        on_connection: Callable[[Connection], Awaitable[None]],
    ):
        self.host = host
        self.port = port
        self.on_connection = on_connection
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        """Start the server."""
        self._server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port,
        )
        addr = self._server.sockets[0].getsockname()
        logger.info(f"Server listening on {addr[0]}:{addr[1]}")

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a new client connection."""
        conn = Connection(reader, writer)
        logger.info(f"New connection from {conn.remote_address}")
        try:
            await self.on_connection(conn)
        except Exception as e:
            logger.error(f"Error handling connection: {e}")
        finally:
            await conn.close()

    async def stop(self) -> None:
        """Stop the server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("Server stopped")

    async def serve_forever(self) -> None:
        """Run the server until cancelled."""
        if self._server:
            await self._server.serve_forever()


async def connect(
    host: str,
    port: int,
    use_tls: bool = False,
    server_hostname: str | None = None,
) -> Connection:
    """
    Connect to a server as a client.

    Used by nodes to connect to the router.

    Args:
        host: Router hostname or IP
        port: Router port
        use_tls: If True, wrap connection in TLS
        server_hostname: Hostname for TLS verification (defaults to host)
    """
    ssl_ctx = None
    if use_tls:
        import ssl
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = True
        ssl_ctx.verify_mode = ssl.CERT_REQUIRED

    reader, writer = await asyncio.open_connection(
        host,
        port,
        ssl=ssl_ctx,
        server_hostname=server_hostname or host if use_tls else None,
    )
    conn = Connection(reader, writer)
    tls_info = " (TLS)" if use_tls else ""
    logger.info(f"Connected to {host}:{port}{tls_info}")
    return conn


class WebSocketClientConnection:
    """
    A WebSocket client connection that can send/receive Messages.

    Used for connecting to a mesh router over WebSocket (e.g., from remote clients).
    """

    def __init__(
        self,
        ws: "aiohttp.ClientWebSocketResponse",
        session: "aiohttp.ClientSession",
        url: str,
    ):
        self.ws = ws
        self._session = session
        self.url = url
        self._closed = False
        self.node_id: str | None = None

    @property
    def remote_address(self) -> str:
        """Get the remote URL as a string."""
        return self.url

    async def send(self, msg: Message) -> None:
        """Send a message over this connection."""
        if self._closed:
            raise ConnectionError("Connection is closed")
        await self.ws.send_str(msg.to_json())
        logger.debug(
            f"SEND [WS-CLIENT {self.url}] "
            f"type={msg.type.value} from={msg.from_node} to={msg.to_node} "
            f"id={msg.id[:8]}... content_preview={str(msg.content)[:100]!r}"
        )

    async def receive(self) -> Message | None:
        """
        Receive a message from this connection.

        Returns None if connection is closed.
        """
        if self._closed:
            return None

        try:
            import aiohttp
            ws_msg = await self.ws.receive()

            if ws_msg.type == aiohttp.WSMsgType.TEXT:
                msg = Message.from_json(ws_msg.data)
                logger.debug(
                    f"RECV [WS-CLIENT {self.url}] "
                    f"type={msg.type.value} from={msg.from_node} to={msg.to_node} "
                    f"id={msg.id[:8]}... content_preview={str(msg.content)[:100]!r}"
                )
                return msg
            elif ws_msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                logger.info(f"WebSocket client closed: {self.url}")
                self._closed = True
                return None
            else:
                # Ignore other message types (PING, PONG, BINARY, etc.)
                return await self.receive()

        except Exception as e:
            logger.error(f"Error receiving WebSocket message: {e}")
            self._closed = True
            return None

    async def close(self) -> None:
        """Close this connection."""
        if not self._closed:
            self._closed = True
            await self.ws.close()
            await self._session.close()
            logger.debug(f"Closed WebSocket client connection to {self.url}")

    @property
    def is_closed(self) -> bool:
        return self._closed


async def connect_ws(
    url: str,
) -> WebSocketClientConnection:
    """
    Connect to a mesh router over WebSocket.

    Args:
        url: WebSocket URL (ws:// or wss://)

    Returns:
        WebSocketClientConnection ready for send/receive

    Example:
        conn = await connect_ws("wss://your-host.example.com/mesh/ws")
        await conn.send(msg)
        response = await conn.receive()
    """
    import aiohttp

    # Create session and connect
    session = aiohttp.ClientSession()
    ws = await session.ws_connect(url)

    logger.info(f"Connected via WebSocket to {url}")
    return WebSocketClientConnection(ws, session, url)
