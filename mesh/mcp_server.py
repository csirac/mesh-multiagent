"""
MCP server: JSON-RPC stdio bridge between Claude Code and mesh tools.

CC discovers tools via `tools/list` at startup. When CC calls a tool,
this server executes it — either locally (tools with real handlers) or
by routing through the mesh WebSocket connection (send_message, etc.).

Runnable standalone:
    python -m mesh.mcp_server --router ws://localhost:8765 --token $TOKEN --node-id agent:sysadmin:bob
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

from .mcp_tools import get_mcp_tools, get_mcp_tool_map
from .protocol import Message, MessageType, make_message
from .tools import ToolCall, ToolDefinition, execute_tool, get_registry

logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2024-11-05"

# Tools that are routed through the mesh WebSocket rather than executed locally
MESH_ROUTED_TOOLS = {"send_message"}

# Tools that require agent-local state (WS connection, scheduler, etc.)
# When agent_socket_path is set, these are routed to agent_node.py via Unix socket.
AGENT_LOCAL_TOOLS = {
    "send_message", "channel_list", "channel_members",
    "schedule_wake", "schedule_list", "schedule_cancel",
    "agent_shutdown", "mesh_status", "agent_status",
}

# MCP schema for send_message (always injected, even if not in the local registry)
SEND_MESSAGE_SCHEMA = {
    "name": "send_message",
    "description": (
        "Send a message to a user or agent on the mesh network. "
        "Use this to communicate status updates, ask questions, or deliver results."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "Target node ID (e.g., 'user:yourname', 'agent:coder:sobek'). Use 'channel:general' for broadcast.",
            },
            "content": {
                "type": "string",
                "description": "The message text to send.",
            },
        },
        "required": ["to", "content"],
    },
}


class MCPServer:
    """MCP bridge: CC tool calls -> mesh tool execution."""

    def __init__(
        self,
        router_url: str,
        auth_token: str,
        node_id: str,
        tool_whitelist: list[str] | None = None,
        agent_socket_path: str | None = None,
    ):
        self.router_url = router_url
        self.auth_token = auth_token
        self.node_id = node_id
        self.tool_whitelist = tool_whitelist
        self.agent_socket_path = agent_socket_path
        self._registry = get_registry()
        self._tool_map = get_mcp_tool_map(tool_whitelist, self._registry)
        self._tool_schemas = get_mcp_tools(tool_whitelist, self._registry)
        self._ws = None
        self._ws_connected = asyncio.Event()

    async def _connect_ws(self) -> None:
        """Establish WebSocket connection to the mesh router and register."""
        try:
            import websockets
            self._ws = await websockets.connect(self.router_url)

            # Register with the router
            register_msg = json.dumps({
                "type": "register",
                "node_id": self.node_id,
                "auth_token": self.auth_token,
            })
            await self._ws.send(register_msg)

            # Wait for registration ack
            response = await asyncio.wait_for(self._ws.recv(), timeout=10)
            data = json.loads(response)
            if data.get("type") == "registered":
                logger.info(f"MCP server registered as {self.node_id}")
                self._ws_connected.set()
            else:
                logger.warning(f"Unexpected registration response: {data}")
                self._ws_connected.set()  # Proceed anyway
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            # Set event anyway so callers don't block forever
            self._ws_connected.set()

    async def run(self) -> None:
        """Main loop: read JSON-RPC from stdin, dispatch, write results to stdout."""
        # Connect to mesh router in background
        asyncio.create_task(self._connect_ws())

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break  # EOF
                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                try:
                    request = json.loads(line_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON-RPC request: {e}")
                    continue

                response = await self._dispatch(request)
                if response is not None:
                    self._write_response(response)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"MCP server error: {e}", exc_info=True)

    def _write_response(self, response: dict) -> None:
        """Write a JSON-RPC response to stdout."""
        data = json.dumps(response) + "\n"
        sys.stdout.write(data)
        sys.stdout.flush()

    async def _dispatch(self, request: dict) -> dict | None:
        """Route a JSON-RPC request to the appropriate handler."""
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            result = self._handle_initialize(params)
            return {"jsonrpc": "2.0", "id": req_id, "result": result}

        elif method == "notifications/initialized":
            # Client acknowledgment — no response needed
            return None

        elif method == "tools/list":
            result = self._handle_tools_list()
            return {"jsonrpc": "2.0", "id": req_id, "result": result}

        elif method == "tools/call":
            name = params.get("name", "")
            arguments = params.get("arguments", {})
            try:
                result = await self._handle_tools_call(name, arguments)
                return {"jsonrpc": "2.0", "id": req_id, "result": result}
            except Exception as e:
                logger.error(f"Tool call error ({name}): {e}", exc_info=True)
                return self._error_response(req_id, -32000, str(e))

        elif method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        else:
            logger.debug(f"Unknown MCP method: {method}")
            return self._error_response(req_id, -32601, f"Method not found: {method}")

    def _handle_initialize(self, params: dict) -> dict:
        """Handle the initialize handshake."""
        logger.info(f"MCP initialize: client={params.get('clientInfo', {}).get('name', 'unknown')}")
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": "mesh-mcp",
                "version": "0.1.0",
            },
        }

    def _handle_tools_list(self) -> dict:
        """Return available tool schemas.

        Always includes send_message for mesh communication, even if
        it's not in the local tool registry.
        """
        tools = list(self._tool_schemas)

        # Inject send_message if not already present from the registry
        tool_names = {t["name"] for t in tools}
        if "send_message" not in tool_names:
            tools.append(SEND_MESSAGE_SCHEMA)

        return {"tools": tools}

    async def _handle_tools_call(self, name: str, arguments: dict) -> dict:
        """Execute a tool call and return the result."""
        if name in AGENT_LOCAL_TOOLS and self.agent_socket_path:
            # Preferred: route through agent's Unix socket (has full agent state)
            result_text = await self._execute_via_agent_socket(name, arguments)
        elif name in MESH_ROUTED_TOOLS and self._ws:
            # Fallback: route via direct WS (standalone mode, no agent socket)
            result_text = await self._execute_mesh_tool(name, arguments)
        elif name in self._tool_map:
            call = ToolCall(name=name, arguments=arguments, raw_xml="")
            result_text = await execute_tool(self._registry, call)
        else:
            result_text = f"Unknown tool: {name}"

        return {
            "content": [
                {"type": "text", "text": result_text},
            ],
        }

    async def _execute_mesh_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool that requires mesh routing."""
        if name == "send_message":
            return await self._execute_send_message(arguments)
        return f"Unsupported mesh tool: {name}"

    async def _execute_send_message(self, arguments: dict) -> str:
        """Send a message through the mesh WebSocket connection."""
        to = arguments.get("to", "")
        content = arguments.get("content", "")

        if not to or not content:
            return "Error: 'to' and 'content' are required"

        # Wait for WS connection (with timeout)
        try:
            await asyncio.wait_for(self._ws_connected.wait(), timeout=10)
        except asyncio.TimeoutError:
            return "Error: WebSocket connection not established"

        if not self._ws:
            return "Error: WebSocket not connected"

        try:
            msg = make_message(
                from_node=self.node_id,
                to_node=to,
                content=content,
            )
            await self._ws.send(msg.to_json())
            logger.info(f"MCP send_message: {self.node_id} → {to} ({len(content)} chars)")
            return f"Message sent to {to}"
        except Exception as e:
            logger.error(f"send_message failed: {e}")
            return f"Error sending message: {e}"

    async def _execute_via_agent_socket(self, name: str, arguments: dict) -> str:
        """Route a tool call to agent_node.py via Unix domain socket."""
        import aiohttp
        try:
            connector = aiohttp.UnixConnector(path=self.agent_socket_path)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    "http://localhost/tool",
                    json={"name": name, "arguments": arguments},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    data = await resp.json()
                    return data.get("result", "No result")
        except Exception as e:
            logger.error(f"Agent socket call failed ({name}): {e}")
            return f"Error executing {name}: {e}"

    @staticmethod
    def _error_response(req_id: Any, code: int, message: str) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": code,
                "message": message,
            },
        }


async def main():
    """Entry point for standalone execution."""
    parser = argparse.ArgumentParser(description="Mesh MCP server")
    parser.add_argument("--router", required=True, help="WebSocket URL of the mesh router")
    parser.add_argument("--token", required=True, help="Authentication token")
    parser.add_argument("--node-id", required=True, help="Node ID for this agent")
    parser.add_argument("--tools", nargs="*", default=None, help="Tool whitelist")
    parser.add_argument("--agent-socket", default=None, help="Unix socket path for agent-local tools")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    server = MCPServer(
        router_url=args.router,
        auth_token=args.token,
        node_id=args.node_id,
        tool_whitelist=args.tools,
        agent_socket_path=args.agent_socket,
    )
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
