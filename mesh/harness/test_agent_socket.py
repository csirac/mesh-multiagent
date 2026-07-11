"""
Tests for agent socket routing — harness subprocess → parent agent_node.

Starts a local Unix domain socket server (mimicking agent_node's /tool endpoint),
then verifies that _call_agent_socket and _execute_tool_calls route correctly.
"""

import asyncio
import os
import tempfile

import pytest
import aiohttp
from aiohttp import web

from mesh.harness.loop import (
    _call_agent_socket,
    _execute_tool_calls,
    AGENT_LOCAL_TOOLS,
)
from mesh.tools import ToolCall


@pytest.fixture
def socket_server():
    """Start a Unix domain socket server that echoes tool calls."""
    calls_received = []

    async def handle_tool(request: web.Request) -> web.Response:
        data = await request.json()
        calls_received.append(data)
        name = data.get("name", "")
        if name == "channel_list":
            return web.json_response({"result": "general, random, dev"})
        if name == "send_message":
            to = data.get("arguments", {}).get("to", "unknown")
            return web.json_response({"result": f"Message sent to {to}"})
        if name == "mesh_status":
            return web.json_response({"result": "3 nodes online"})
        return web.json_response({"result": f"OK: {name}"})

    sock_path = os.path.join(tempfile.gettempdir(), f"test_mesh_agent_{os.getpid()}.sock")
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    app = web.Application()
    app.router.add_post("/tool", handle_tool)

    loop = asyncio.new_event_loop()

    async def start():
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.UnixSite(runner, sock_path)
        await site.start()
        return runner

    runner = loop.run_until_complete(start())

    import threading
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()

    yield sock_path, calls_received

    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=3)
    loop.run_until_complete(runner.cleanup())
    loop.close()
    if os.path.exists(sock_path):
        os.unlink(sock_path)


class TestCallAgentSocket:

    def test_channel_list(self, socket_server):
        sock_path, calls = socket_server
        result = asyncio.get_event_loop().run_until_complete(
            _call_agent_socket(sock_path, "channel_list", {})
        )
        assert "general" in result
        assert len(calls) == 1
        assert calls[0]["name"] == "channel_list"

    def test_send_message(self, socket_server):
        sock_path, calls = socket_server
        result = asyncio.get_event_loop().run_until_complete(
            _call_agent_socket(sock_path, "send_message", {"to": "dev", "content": "hello"})
        )
        assert "Message sent to dev" in result
        assert calls[-1]["arguments"]["to"] == "dev"

    def test_socket_not_found(self):
        with pytest.raises(Exception):
            asyncio.get_event_loop().run_until_complete(
                _call_agent_socket("/tmp/nonexistent_socket.sock", "test", {})
            )


class TestExecuteToolCallsRouting:

    def test_agent_local_tool_routed_via_socket(self, socket_server):
        sock_path, calls = socket_server
        from mesh.tools import get_registry
        registry = get_registry()

        tool_calls = [ToolCall(name="channel_list", arguments={}, raw_xml="")]
        results = asyncio.get_event_loop().run_until_complete(
            _execute_tool_calls(tool_calls, registry, agent_socket_path=sock_path)
        )
        assert len(results) == 1
        name, result_str, success = results[0]
        assert name == "channel_list"
        assert success
        assert "general" in result_str

    def test_local_tool_not_routed_via_socket(self, socket_server):
        """Non-agent-local tools should use the local registry, not the socket."""
        sock_path, calls = socket_server
        from mesh.tools import get_registry
        import mesh.tool_implementations  # noqa: register tools
        import mesh.harness.tools  # noqa: register harness tools (shell)
        registry = get_registry()

        tool_calls = [ToolCall(name="shell", arguments={"command": "echo local-exec"}, raw_xml="")]
        results = asyncio.get_event_loop().run_until_complete(
            _execute_tool_calls(tool_calls, registry, agent_socket_path=sock_path)
        )
        assert len(results) == 1
        name, result_str, success = results[0]
        assert name == "shell"
        assert success
        assert "local-exec" in result_str
        assert len(calls) == 0  # socket should NOT have been called

    def test_no_socket_falls_back_to_registry(self):
        """Without socket, agent-local tools fall through to registry."""
        from mesh.tools import get_registry
        registry = get_registry()

        tool_calls = [ToolCall(name="channel_list", arguments={}, raw_xml="")]
        results = asyncio.get_event_loop().run_until_complete(
            _execute_tool_calls(tool_calls, registry, agent_socket_path=None)
        )
        assert len(results) == 1
        name, result_str, success = results[0]
        assert name == "channel_list"
        # Without socket, it should either execute locally or report unknown tool
        # (depends on whether it's registered)


class TestAgentLocalToolsSet:

    def test_expected_tools_present(self):
        assert "send_message" in AGENT_LOCAL_TOOLS
        assert "channel_list" in AGENT_LOCAL_TOOLS
        assert "mesh_status" in AGENT_LOCAL_TOOLS

    def test_coding_tools_not_in_agent_local(self):
        assert "shell" not in AGENT_LOCAL_TOOLS
        assert "apply_patch" not in AGENT_LOCAL_TOOLS
        assert "file_read" not in AGENT_LOCAL_TOOLS
