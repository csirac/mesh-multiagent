"""Tests for the node module."""

import pytest
import asyncio
import tempfile
import os

from mesh.node import Node, HistoryEntry
from mesh.agent_node import SimpleAgentNode, AgentNode
from mesh.user_node import TestUserNode
from mesh.router import Router
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
async def router(temp_storage):
    """Create and start a router for testing."""
    config = RouterConfig(
        host="127.0.0.1",
        port=0,  # Random available port
        ws_port=0,  # Random available port for WebSocket
        storage_path=temp_storage,
    )
    r = Router(config)
    await r.start()
    port = r._server._server.sockets[0].getsockname()[1]
    yield r, port
    await r.stop()


class TestNodeConfig:
    def test_node_config_defaults(self):
        """NodeConfig has sensible defaults."""
        config = NodeConfig(id="user:testuser")
        assert config.router_host == "127.0.0.1"
        assert config.router_port == 7700
        assert config.llm_model is None
        assert config.system_prompt == ""


class TestNodeConnection:
    @pytest.mark.asyncio
    async def test_node_connects_and_registers(self, router):
        """Node can connect and register with router."""
        r, port = router
        config = NodeConfig(id="user:test", router_port=port)
        node = SimpleAgentNode(config)

        await node.connect()
        assert node.is_connected
        assert "user:test" in r.connected_nodes

        await node.disconnect()
        assert not node.is_connected

    @pytest.mark.asyncio
    async def test_node_disconnect(self, router):
        """Node disconnects cleanly."""
        r, port = router
        config = NodeConfig(id="user:test", router_port=port)
        node = SimpleAgentNode(config)

        await node.connect()
        await node.disconnect()

        # Give time for router to process disconnect
        await asyncio.sleep(0.1)
        assert "user:test" not in r.connected_nodes


class TestNodeSendReceive:
    @pytest.mark.asyncio
    async def test_node_send_message(self, router):
        """Node can send a message."""
        r, port = router

        # Create user (TestUserNode doesn't echo) and agent (SimpleAgentNode echoes)
        user_config = NodeConfig(id="user:testuser", router_port=port)
        agent_config = NodeConfig(id="agent:echo", router_port=port)

        user = TestUserNode(user_config)
        agent = SimpleAgentNode(agent_config)

        await user.connect()
        await agent.connect()

        # Start receive loops
        user_task = asyncio.create_task(user.receive_loop())
        agent_task = asyncio.create_task(agent.receive_loop())

        # User sends message
        sent_msg = await user.send("agent:echo", "Hello!")

        # Give time for delivery
        await asyncio.sleep(0.2)

        # Check agent history - received the message and sent echo
        assert len(agent.history) == 2
        assert agent.history[0].message.content == "Hello!"
        assert agent.history[0].direction == "incoming"
        assert agent.history[1].direction == "outgoing"  # Echo reply

        # Check user history - sent message and received echo
        assert len(user.history) == 2
        assert user.history[0].message.content == "Hello!"
        assert user.history[0].direction == "outgoing"
        assert user.history[1].direction == "incoming"  # Echo reply

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
    async def test_node_request_node_list(self, router):
        """Node can request list of connected nodes."""
        r, port = router

        node1_config = NodeConfig(id="user:testuser", router_port=port)
        node2_config = NodeConfig(id="agent:echo", router_port=port)

        node1 = SimpleAgentNode(node1_config)
        node2 = SimpleAgentNode(node2_config)

        await node1.connect()
        await node2.connect()

        # Start receive loop so we can get the response
        recv_task = asyncio.create_task(node1.receive_loop())

        try:
            nodes = await node1.request_node_list()
            assert "user:testuser" in nodes
            assert "agent:echo" in nodes
        finally:
            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

        await node1.disconnect()
        await node2.disconnect()


class TestSimpleAgentNode:
    @pytest.mark.asyncio
    async def test_echo_response(self, router):
        """SimpleAgentNode echoes messages back."""
        r, port = router

        user_config = NodeConfig(id="user:testuser", router_port=port)
        agent_config = NodeConfig(id="agent:echo", router_port=port)

        user = TestUserNode(user_config)  # User doesn't echo
        agent = SimpleAgentNode(agent_config)  # Agent echoes

        await user.connect()
        await agent.connect()

        # Start loops
        user_task = asyncio.create_task(user.receive_loop())
        agent_task = asyncio.create_task(agent.receive_loop())

        # User sends message
        await user.send("agent:echo", "Test message")

        # Wait for round trip
        await asyncio.sleep(0.3)

        # User should have received echo
        # History: outgoing (to agent), incoming (from agent)
        assert len(user.history) == 2
        assert user.history[1].direction == "incoming"
        assert "Test message" in user.history[1].message.content
        assert user.history[1].message.from_node == "agent:echo"

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


class TestNodeHistory:
    @pytest.mark.asyncio
    async def test_history_records_both_directions(self, router):
        """History records both sent and received messages."""
        r, port = router

        user_config = NodeConfig(id="user:testuser", router_port=port)
        agent_config = NodeConfig(id="agent:echo", router_port=port)

        user = TestUserNode(user_config)  # User doesn't echo
        agent = SimpleAgentNode(agent_config)  # Agent echoes

        await user.connect()
        await agent.connect()

        user_task = asyncio.create_task(user.receive_loop())
        agent_task = asyncio.create_task(agent.receive_loop())

        # User sends
        await user.send("agent:echo", "First")
        await asyncio.sleep(0.2)
        await user.send("agent:echo", "Second")
        await asyncio.sleep(0.2)

        # User history should have: out, in, out, in
        assert len(user.history) == 4
        assert user.history[0].direction == "outgoing"
        assert user.history[0].message.content == "First"
        assert user.history[1].direction == "incoming"  # Echo back
        assert user.history[2].direction == "outgoing"
        assert user.history[2].message.content == "Second"
        assert user.history[3].direction == "incoming"

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
    async def test_get_history_for_llm(self, router):
        """get_history_for_llm formats messages for LLM."""
        r, port = router

        user_config = NodeConfig(id="user:testuser", router_port=port)
        agent_config = NodeConfig(id="agent:echo", router_port=port)

        user = TestUserNode(user_config)  # User doesn't echo
        agent = SimpleAgentNode(agent_config)  # Agent echoes

        await user.connect()
        await agent.connect()

        user_task = asyncio.create_task(user.receive_loop())
        agent_task = asyncio.create_task(agent.receive_loop())

        await user.send("agent:echo", "Hello")
        await asyncio.sleep(0.2)

        # Agent's perspective
        llm_messages = agent.get_history_for_llm()
        assert len(llm_messages) == 2
        # First message is from user (incoming) -> role: user
        assert llm_messages[0]["role"] == "user"
        assert "[From user:testuser]" in llm_messages[0]["content"]
        # Second message is echo back (outgoing) -> role: assistant
        assert llm_messages[1]["role"] == "assistant"

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


class TestAgentNodeRouting:
    @pytest.mark.asyncio
    async def test_explicit_send_message_routing(self, router):
        """AgentNode uses _execute_send_message to route messages."""
        r, port = router

        # Create a custom agent that simulates using send_message tool
        class RoutingTestAgent(AgentNode):
            async def _process_with_llm(self, trigger_msg):
                # Simulate what happens when LLM calls send_message tool
                content = f"Processed: {trigger_msg.content}"
                result = await self._execute_send_message(
                    {"to": "user:testuser", "content": content},
                    trigger_msg
                )
                # Verify send_message succeeded
                assert "successfully" in result.lower()

        user_config = NodeConfig(id="user:testuser", router_port=port)
        agent_config = NodeConfig(id="agent:router_test", router_port=port)

        user = TestUserNode(user_config)  # User doesn't echo
        agent = RoutingTestAgent(agent_config)

        await user.connect()
        await agent.connect()

        user_task = asyncio.create_task(user.receive_loop())
        agent_task = asyncio.create_task(agent.receive_loop())

        await user.send("agent:router_test", "Test routing")
        await asyncio.sleep(0.2)

        # User should receive response via send_message
        assert len(user.history) >= 2
        response = user.history[1].message
        assert response.from_node == "agent:router_test"
        assert "Processed: Test routing" in response.content

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


class TestMessageHandlers:
    @pytest.mark.asyncio
    async def test_add_message_handler(self, router):
        """Custom message handlers are called."""
        r, port = router

        config = NodeConfig(id="user:testuser", router_port=port)
        node = SimpleAgentNode(config)

        received_messages = []

        async def custom_handler(msg):
            # Only record MESSAGE types (not PRESENCE)
            if msg.type == MessageType.MESSAGE:
                received_messages.append(msg)

        node.add_message_handler(custom_handler)

        await node.connect()
        receive_task = asyncio.create_task(node.receive_loop())

        # Create another node to send a message
        other_config = NodeConfig(id="agent:sender", router_port=port)
        other = SimpleAgentNode(other_config)
        await other.connect()

        await other.send("user:testuser", "Handler test")
        await asyncio.sleep(0.2)

        assert len(received_messages) == 1
        assert received_messages[0].content == "Handler test"

        await node.disconnect()
        await other.disconnect()
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass


class TestHistoryPersistence:
    """Tests for history save/load functionality."""

    @pytest.fixture
    def temp_history_file(self):
        """Create a temporary history file path."""
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        # Remove it so the test can create it
        if os.path.exists(path):
            os.unlink(path)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_save_and_load_history(self, router, temp_history_file):
        """History can be saved and loaded from disk."""
        r, port = router

        config = NodeConfig(id="user:testuser", router_port=port)
        node = SimpleAgentNode(config, history_file=temp_history_file, persist=True)

        await node.connect()

        # Create another node to send messages
        other_config = NodeConfig(id="agent:echo", router_port=port)
        other = SimpleAgentNode(other_config)
        await other.connect()

        # Start receive loops
        recv1 = asyncio.create_task(node.receive_loop())
        recv2 = asyncio.create_task(other.receive_loop())

        # Send messages
        await node.send("agent:echo", "Hello")
        await asyncio.sleep(0.5)

        # Should have history
        assert len(node.history) > 0

        # Force save (debounced save has 2s delay)
        node.save_history()

        # Check file was created
        assert os.path.exists(temp_history_file)

        # Disconnect
        await node.disconnect()
        await other.disconnect()
        recv1.cancel()
        recv2.cancel()
        try:
            await recv1
        except asyncio.CancelledError:
            pass
        try:
            await recv2
        except asyncio.CancelledError:
            pass

        # Create new node with same history file
        config2 = NodeConfig(id="user:testuser", router_port=port)
        node2 = SimpleAgentNode(config2, history_file=temp_history_file, persist=True)

        # Load history
        loaded = node2.load_history()
        assert loaded > 0
        assert len(node2.history) == len(node.history)
        assert node2.history[0].message.content == "Hello"

    def test_history_entry_serialization(self):
        """HistoryEntry can be serialized and deserialized."""
        msg = Message(
            id="msg-123",
            type=MessageType.MESSAGE,
            from_node="user:testuser",
            to_node="agent:coder:alice",
            content="Test message",
            timestamp="2026-01-22T10:00:00",
            in_reply_to=None,
        )
        entry = HistoryEntry(message=msg, direction="outgoing")

        # Serialize
        data = entry.to_dict()
        assert data["direction"] == "outgoing"
        assert data["message"]["content"] == "Test message"
        assert data["message"]["type"] == "message"

        # Deserialize
        restored = HistoryEntry.from_dict(data)
        assert restored.direction == "outgoing"
        assert restored.message.content == "Test message"
        assert restored.message.type == MessageType.MESSAGE
        assert restored.message.from_node == "user:testuser"

    def test_clear_history(self):
        """clear_history() removes in-memory history."""
        config = NodeConfig(id="user:testuser")
        node = SimpleAgentNode(config, persist=False)

        # Manually add to history for testing
        msg = Message(
            id="msg-1",
            type=MessageType.MESSAGE,
            from_node="agent:test",
            to_node="user:testuser",
            content="Test",
            timestamp="2026-01-22T10:00:00",
        )
        node._history.append(HistoryEntry(message=msg, direction="incoming"))
        assert len(node.history) == 1

        node.clear_history()
        assert len(node.history) == 0

    def test_delete_history_file(self, temp_history_file):
        """delete_history_file() removes the file from disk."""
        config = NodeConfig(id="user:testuser")
        node = SimpleAgentNode(config, history_file=temp_history_file, persist=True)

        # Create the file
        node._history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_history_file, "w") as f:
            f.write("[]")
        assert os.path.exists(temp_history_file)

        # Delete
        result = node.delete_history_file()
        assert result is True
        assert not os.path.exists(temp_history_file)

    def test_default_history_path(self):
        """When persist=True without explicit path, uses default."""
        config = NodeConfig(id="user:testuser")
        node = SimpleAgentNode(config, persist=True)

        assert node.history_file is not None
        assert "user-testuser.json" in str(node.history_file)
        assert ".mesh/history" in str(node.history_file)


class TestSummaryState:
    """Tests for SummaryState and summary persistence."""

    from mesh.node import SummaryState

    def test_summary_state_serialization(self):
        """SummaryState can be serialized and deserialized."""
        from mesh.node import SummaryState

        summary = SummaryState(
            summary_text="This is a test summary of the conversation.",
            messages_summarized=25,
            created_at="2026-01-22T11:00:00Z",
            token_estimate=150,
        )

        # Serialize
        data = summary.to_dict()
        assert data["summary_text"] == "This is a test summary of the conversation."
        assert data["messages_summarized"] == 25
        assert data["created_at"] == "2026-01-22T11:00:00Z"
        assert data["token_estimate"] == 150

        # Deserialize
        restored = SummaryState.from_dict(data)
        assert restored.summary_text == "This is a test summary of the conversation."
        assert restored.messages_summarized == 25
        assert restored.created_at == "2026-01-22T11:00:00Z"
        assert restored.token_estimate == 150

    @pytest.fixture
    def temp_summary_file(self):
        """Create a temporary summary file path."""
        fd, path = tempfile.mkstemp(suffix=".summary.json")
        os.close(fd)
        if os.path.exists(path):
            os.unlink(path)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_summary_file_path(self):
        """Summary file path is derived from history file."""
        config = NodeConfig(id="user:testuser")
        node = SimpleAgentNode(config, persist=True)

        assert node.summary_file is not None
        assert str(node.summary_file).endswith(".summary.json")
        # Should be based on history file path
        history_stem = str(node.history_file).replace(".json", "")
        summary_path = str(node.summary_file).replace(".summary.json", "")
        assert history_stem == summary_path

    def test_save_and_load_summary(self, temp_summary_file):
        """Summary can be saved and loaded from disk."""
        from mesh.node import SummaryState
        from pathlib import Path

        # Create node with temp history file (summary file will be derived)
        history_file = temp_summary_file.replace(".summary.json", ".json")
        config = NodeConfig(id="user:testuser")
        node = SimpleAgentNode(config, history_file=history_file, persist=True)

        # Create a summary
        summary = SummaryState(
            summary_text="Test summary content here.",
            messages_summarized=10,
            created_at="2026-01-22T12:00:00Z",
            token_estimate=50,
        )

        # Save
        result = node.save_summary(summary)
        assert result is True
        assert node.summary_file.exists()

        # Load on a new node
        node2 = SimpleAgentNode(config, history_file=history_file, persist=True)
        loaded = node2.load_summary()

        assert loaded is not None
        assert loaded.summary_text == "Test summary content here."
        assert loaded.messages_summarized == 10
        assert loaded.token_estimate == 50

        # Cleanup
        if Path(history_file).exists():
            Path(history_file).unlink()

    def test_delete_summary_file(self, temp_summary_file):
        """delete_summary_file() removes the summary file from disk."""
        from mesh.node import SummaryState
        from pathlib import Path

        history_file = temp_summary_file.replace(".summary.json", ".json")
        config = NodeConfig(id="user:testuser")
        node = SimpleAgentNode(config, history_file=history_file, persist=True)

        # Save a summary first
        summary = SummaryState(
            summary_text="To be deleted",
            messages_summarized=5,
            created_at="2026-01-22T12:00:00Z",
            token_estimate=20,
        )
        node.save_summary(summary)
        assert node.summary_file.exists()

        # Delete
        result = node.delete_summary_file()
        assert result is True
        assert not node.summary_file.exists()


class TestTokenEstimation:
    """Tests for token estimation functions."""

    def test_estimate_tokens_basic(self):
        """estimate_tokens returns reasonable count."""
        from mesh.llm import estimate_tokens

        text = "Hello world, this is a test."
        tokens = estimate_tokens(text)
        # Should be around 7-10 tokens for this text
        assert 5 <= tokens <= 15

    def test_estimate_tokens_empty(self):
        """estimate_tokens handles empty string."""
        from mesh.llm import estimate_tokens

        tokens = estimate_tokens("")
        assert tokens == 0

    def test_estimate_history_tokens(self):
        """estimate_history_tokens sums tokens across messages."""
        from mesh.llm import estimate_history_tokens, HistoryMessage

        history = [
            HistoryMessage(
                from_node="user:testuser",
                content="Hello, how are you?",
                timestamp="2026-01-22T10:00:00Z",
            ),
            HistoryMessage(
                from_node="agent:assistant",
                content="I'm doing well, thank you!",
                timestamp="2026-01-22T10:00:01Z",
            ),
        ]

        total = estimate_history_tokens(history, base_overhead=0)
        # Each message has content tokens + overhead (~35)
        # Total should be reasonable for these short messages
        assert total > 70  # At least 2 messages * 35 overhead
        assert total < 150  # Not too large for short messages


class TestAgentSummarization:
    """Tests for AgentNode summarization functionality."""

    def test_agent_summarization_params(self):
        """AgentNode accepts and stores summarization parameters."""
        # Config values take precedence over CLI args when config has non-None values.
        # To test CLI override, set config to None-like values first.
        config = NodeConfig(id="agent:test", history_soft_limit_tokens=100000)
        agent = AgentNode(
            config,
            soft_limit=50000,  # This is overridden by config
            target_ratio=0.3,
        )

        assert agent._soft_limit == 100000  # Config wins
        assert agent._target_ratio == 0.3
        assert agent._target == 30000  # 100000 * 0.3

    def test_agent_default_summarization_params(self):
        """AgentNode uses config defaults for summarization parameters."""
        config = NodeConfig(id="agent:test")
        agent = AgentNode(config)

        # Config default (70K) takes precedence over class default (50K)
        assert agent._soft_limit == config.history_soft_limit_tokens
        assert agent._target_ratio == AgentNode.DEFAULT_TARGET_RATIO
        assert agent._target == int(config.history_soft_limit_tokens * AgentNode.DEFAULT_TARGET_RATIO)

    def test_agent_summary_state_initialized_none(self):
        """AgentNode starts with no summary."""
        config = NodeConfig(id="agent:test")
        agent = AgentNode(config)

        assert agent._summary is None
        assert agent._summarizing is False

    def test_build_history_without_summary(self):
        """_build_history_for_llm returns all messages when no summary."""
        from mesh.node import HistoryEntry
        from mesh.protocol import Message, MessageType

        config = NodeConfig(id="agent:test")
        agent = AgentNode(config)

        # Add some history
        for i in range(3):
            msg = Message(
                id=f"msg-{i}",
                type=MessageType.MESSAGE,
                from_node="user:testuser",
                to_node="agent:test",
                content=f"Message {i}",
                timestamp=f"2026-01-22T10:0{i}:00Z",
            )
            agent._history.append(HistoryEntry(message=msg, direction="incoming"))

        history = agent._build_history_for_llm()
        assert len(history) == 3
        assert history[0].content == "Message 0"
        assert history[2].content == "Message 2"

    def test_build_history_with_summary(self):
        """_build_history_for_llm returns summary + recent when summary exists."""
        from mesh.node import HistoryEntry, SummaryState
        from mesh.protocol import Message, MessageType

        config = NodeConfig(id="agent:test")
        agent = AgentNode(config)

        # Add 5 messages to history
        for i in range(5):
            msg = Message(
                id=f"msg-{i}",
                type=MessageType.MESSAGE,
                from_node="user:testuser",
                to_node="agent:test",
                content=f"Message {i}",
                timestamp=f"2026-01-22T10:0{i}:00Z",
            )
            agent._history.append(HistoryEntry(message=msg, direction="incoming"))

        # Set summary that covers first 3 messages
        agent._summary = SummaryState(
            summary_text="Summary of messages 0, 1, and 2",
            messages_summarized=3,
            created_at="2026-01-22T10:05:00Z",
            token_estimate=20,
        )

        history = agent._build_history_for_llm()

        # Should have: summary (1) + remaining messages (2) = 3
        assert len(history) == 3
        # First should be the summary
        assert "[Earlier summary]" in history[0].content
        assert "Summary of messages 0, 1, and 2" in history[0].content
        assert history[0].from_node == "system"
        # Remaining should be messages 3 and 4
        assert history[1].content == "Message 3"
        assert history[2].content == "Message 4"
