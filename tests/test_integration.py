"""End-to-end integration tests for the mesh system."""

import pytest
import asyncio
import tempfile
import os

from mesh.router import Router
from mesh.agent_node import SimpleAgentNode, AgentNode
from mesh.user_node import TestUserNode
from mesh.config import RouterConfig, NodeConfig
from mesh.protocol import Message, MessageType


@pytest.fixture
def temp_storage():
    """Create a temporary storage path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
async def mesh_setup(temp_storage):
    """Set up a complete mesh with router and nodes."""
    # Start router with random ports to avoid conflicts with production
    router_config = RouterConfig(
        host="127.0.0.1",
        port=0,  # Random available port
        ws_port=0,  # Random available port for WebSocket
        storage_path=temp_storage,
    )
    router = Router(router_config)
    await router.start()
    port = router._server._server.sockets[0].getsockname()[1]

    yield router, port

    await router.stop()


class TestFullMessageFlow:
    """Test complete message flows through the system."""

    @pytest.mark.asyncio
    async def test_user_to_agent_roundtrip(self, mesh_setup):
        """Complete message roundtrip: user -> agent -> user."""
        router, port = mesh_setup

        # Create user (TestUserNode doesn't auto-reply) and agent (SimpleAgentNode echoes)
        user = TestUserNode(NodeConfig(id="user:testuser", router_port=port))
        agent = SimpleAgentNode(NodeConfig(id="agent:echo", router_port=port))

        await user.connect()
        await agent.connect()

        # Start receive loops
        user_task = asyncio.create_task(user.receive_loop())
        agent_task = asyncio.create_task(agent.receive_loop())

        try:
            # User sends message
            await user.send("agent:echo", "Hello, agent!")

            # Wait for roundtrip
            await asyncio.sleep(0.3)

            # Verify user received response (outgoing + incoming)
            assert len(user.history) == 2
            assert user.history[0].direction == "outgoing"
            assert user.history[0].message.content == "Hello, agent!"
            assert user.history[1].direction == "incoming"
            assert "Hello, agent!" in user.history[1].message.content

            # Verify agent processed the message (incoming + outgoing)
            assert len(agent.history) == 2
            assert agent.history[0].direction == "incoming"
            assert agent.history[1].direction == "outgoing"

        finally:
            await user.disconnect()
            await agent.disconnect()
            user_task.cancel()
            agent_task.cancel()
            try:
                await user_task
            except asyncio.CancelledError:
                pass
            try:
                await agent_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_multi_agent_chain(self, mesh_setup):
        """Message passes through multiple agents."""
        router, port = mesh_setup

        # Create a chain: user -> agent1 -> agent2 -> user
        class ForwardingAgent(AgentNode):
            def __init__(self, config, forward_to, reply_to=None):
                super().__init__(config)
                self.forward_to = forward_to
                self.reply_to = reply_to  # Who to send final reply to

            async def _process_with_llm(self, trigger_msg):
                if self.forward_to:
                    # Forward to next agent using send_message
                    content = f"Forwarded by {self.node_id}: {trigger_msg.content}"
                    await self._execute_send_message(
                        {"to": self.forward_to, "content": content},
                        trigger_msg
                    )
                else:
                    # End of chain, reply back to configured destination
                    target = self.reply_to or trigger_msg.from_node
                    content = f"Chain complete: {trigger_msg.content}"
                    await self.send(target, content, in_reply_to=trigger_msg.id)

        user = TestUserNode(NodeConfig(id="user:testuser", router_port=port))
        agent1 = ForwardingAgent(NodeConfig(id="agent:first", router_port=port), forward_to="agent:second")
        agent2 = ForwardingAgent(NodeConfig(id="agent:second", router_port=port), forward_to=None, reply_to="user:testuser")

        await user.connect()
        await agent1.connect()
        await agent2.connect()

        user_task = asyncio.create_task(user.receive_loop())
        agent1_task = asyncio.create_task(agent1.receive_loop())
        agent2_task = asyncio.create_task(agent2.receive_loop())

        try:
            await user.send("agent:first", "Start chain")
            await asyncio.sleep(0.5)

            # User should have received final response
            responses = [h for h in user.history if h.direction == "incoming"]
            assert len(responses) >= 1
            # Final response should contain chain info
            final = responses[-1].message.content
            assert "Chain complete" in final

        finally:
            await user.disconnect()
            await agent1.disconnect()
            await agent2.disconnect()
            for task in [user_task, agent1_task, agent2_task]:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


class TestOfflineDelivery:
    """Test message delivery when nodes are offline."""

    @pytest.mark.asyncio
    async def test_message_delivered_after_reconnect(self, mesh_setup):
        """Messages sent while offline are delivered on reconnect."""
        router, port = mesh_setup

        user = SimpleAgentNode(NodeConfig(id="user:testuser", router_port=port))
        await user.connect()

        # Send to offline agent
        await user.send("agent:offline", "You were offline")
        await asyncio.sleep(0.1)

        # Verify stored
        assert router.store.count_pending("agent:offline") == 1

        # Now agent comes online
        agent = SimpleAgentNode(NodeConfig(id="agent:offline", router_port=port))
        await agent.connect()

        # Start receive loop
        agent_task = asyncio.create_task(agent.receive_loop())

        try:
            await asyncio.sleep(0.2)

            # Agent should have received the pending message
            assert len(agent.history) >= 1
            assert agent.history[0].message.content == "You were offline"

            # Storage should be cleared
            assert router.store.count_pending("agent:offline") == 0

        finally:
            await user.disconnect()
            await agent.disconnect()
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_multiple_pending_messages(self, mesh_setup):
        """Multiple pending messages delivered in order."""
        router, port = mesh_setup

        user = SimpleAgentNode(NodeConfig(id="user:testuser", router_port=port))
        await user.connect()

        # Send multiple messages while agent offline
        for i in range(5):
            await user.send("agent:sleepy", f"Message {i}")

        await asyncio.sleep(0.1)
        assert router.store.count_pending("agent:sleepy") == 5

        # Agent comes online
        agent = SimpleAgentNode(NodeConfig(id="agent:sleepy", router_port=port))
        await agent.connect()
        agent_task = asyncio.create_task(agent.receive_loop())

        try:
            await asyncio.sleep(0.3)

            # Should have all messages in order
            contents = [h.message.content for h in agent.history if h.direction == "incoming"]
            assert contents == [f"Message {i}" for i in range(5)]

        finally:
            await user.disconnect()
            await agent.disconnect()
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass


class TestConcurrency:
    """Test concurrent message handling."""

    @pytest.mark.asyncio
    async def test_multiple_simultaneous_senders(self, mesh_setup):
        """Agent handles messages from multiple senders."""
        router, port = mesh_setup

        agent = SimpleAgentNode(NodeConfig(id="agent:busy", router_port=port))
        await agent.connect()
        agent_task = asyncio.create_task(agent.receive_loop())

        # Create multiple users (TestUserNode doesn't auto-reply)
        users = []
        user_tasks = []
        for i in range(3):
            user = TestUserNode(NodeConfig(id=f"user:sender{i}", router_port=port))
            await user.connect()
            users.append(user)
            user_tasks.append(asyncio.create_task(user.receive_loop()))

        try:
            # All users send simultaneously
            await asyncio.gather(*[
                user.send("agent:busy", f"From {user.node_id}")
                for user in users
            ])

            await asyncio.sleep(0.5)

            # Each user should have received a response
            for user in users:
                responses = [h for h in user.history if h.direction == "incoming"]
                assert len(responses) >= 1

        finally:
            for user in users:
                await user.disconnect()
            await agent.disconnect()
            for task in user_tasks + [agent_task]:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_rapid_message_exchange(self, mesh_setup):
        """System handles rapid message exchange."""
        router, port = mesh_setup

        user = TestUserNode(NodeConfig(id="user:rapid", router_port=port))
        agent = SimpleAgentNode(NodeConfig(id="agent:rapid", router_port=port))

        await user.connect()
        await agent.connect()

        user_task = asyncio.create_task(user.receive_loop())
        agent_task = asyncio.create_task(agent.receive_loop())

        try:
            # Send many messages rapidly
            for i in range(20):
                await user.send("agent:rapid", f"Rapid {i}")

            # Wait for all to process
            await asyncio.sleep(1.0)

            # Check all responses received
            outgoing = [h for h in user.history if h.direction == "outgoing"]
            incoming = [h for h in user.history if h.direction == "incoming"]

            assert len(outgoing) == 20
            assert len(incoming) == 20

        finally:
            await user.disconnect()
            await agent.disconnect()
            user_task.cancel()
            agent_task.cancel()
            try:
                await user_task
            except asyncio.CancelledError:
                pass
            try:
                await agent_task
            except asyncio.CancelledError:
                pass


class TestNodeLifecycle:
    """Test node connect/disconnect scenarios."""

    @pytest.mark.asyncio
    async def test_reconnection(self, mesh_setup):
        """Node can disconnect and reconnect."""
        router, port = mesh_setup

        node = SimpleAgentNode(NodeConfig(id="user:reconnect", router_port=port))

        await node.connect()
        assert node.is_connected
        assert "user:reconnect" in router.connected_nodes

        await node.disconnect()
        assert not node.is_connected
        await asyncio.sleep(0.1)

        # Reconnect
        await node.connect()
        assert node.is_connected
        assert "user:reconnect" in router.connected_nodes

        await node.disconnect()

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, mesh_setup):
        """System handles graceful shutdown."""
        router, port = mesh_setup

        nodes = []
        tasks = []
        for i in range(3):
            node = SimpleAgentNode(NodeConfig(id=f"node:{i}", router_port=port))
            await node.connect()
            nodes.append(node)
            tasks.append(asyncio.create_task(node.receive_loop()))

        # Verify all connected
        assert len(router.connected_nodes) == 3

        # Disconnect all gracefully
        for node in nodes:
            await node.disconnect()

        # Cancel tasks
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Give router time to process
        await asyncio.sleep(0.1)

        # All should be disconnected
        assert len(router.connected_nodes) == 0


class TestRouterResilience:
    """Test router behavior under various conditions."""

    @pytest.mark.asyncio
    async def test_router_survives_client_crash(self, mesh_setup):
        """Router continues after abrupt client disconnect."""
        router, port = mesh_setup

        # Create and connect a node
        node1 = SimpleAgentNode(NodeConfig(id="user:crasher", router_port=port))
        await node1.connect()

        # Simulate crash by closing connection without proper disconnect
        await node1._conn.close()

        # Router should still be running
        await asyncio.sleep(0.1)
        assert router._running

        # New node can still connect
        node2 = SimpleAgentNode(NodeConfig(id="user:survivor", router_port=port))
        await node2.connect()
        assert "user:survivor" in router.connected_nodes

        await node2.disconnect()

    @pytest.mark.asyncio
    async def test_message_to_self(self, mesh_setup):
        """Node can send message to itself."""
        router, port = mesh_setup

        node = SimpleAgentNode(NodeConfig(id="user:self", router_port=port))
        await node.connect()
        task = asyncio.create_task(node.receive_loop())

        try:
            await node.send("user:self", "Hello myself")
            await asyncio.sleep(1.0)

            # Should have at least the outgoing message
            assert len(node.history) >= 1
            outgoing = [h for h in node.history if h.direction == "outgoing"]
            assert len(outgoing) >= 1

        finally:
            await node.disconnect()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
