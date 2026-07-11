"""Tests for the storage module."""

import pytest
import tempfile
import os
from pathlib import Path

from mesh.storage import MessageStore
from mesh.protocol import Message, MessageType


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def store(temp_db):
    """Create a MessageStore with a temporary database."""
    return MessageStore(temp_db)


class TestMessageStoreInit:
    def test_creates_db_file(self, temp_db):
        """Store creates the database file."""
        os.unlink(temp_db)  # Remove it first
        assert not os.path.exists(temp_db)

        store = MessageStore(temp_db)
        assert os.path.exists(temp_db)

    def test_creates_parent_dirs(self):
        """Store creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "deep" / "messages.db"
            assert not path.parent.exists()

            store = MessageStore(path)
            assert path.exists()


class TestMessageStoreBasics:
    def test_store_and_retrieve(self, store):
        """Store a message and retrieve it."""
        msg = Message(
            from_node="user:testuser",
            to_node="agent:echo",
            type=MessageType.MESSAGE,
            content="Hello",
            id="msg-test123",
        )
        store.store(msg)

        pending = store.get_pending("agent:echo")
        assert len(pending) == 1
        assert pending[0].id == "msg-test123"
        assert pending[0].content == "Hello"
        assert pending[0].from_node == "user:testuser"
        assert pending[0].to_node == "agent:echo"
        assert pending[0].type == MessageType.MESSAGE

    def test_store_dict_content(self, store):
        """Store a message with dict content."""
        msg = Message(
            from_node="router",
            to_node="user:testuser",
            type=MessageType.CONTROL,
            content={"action": "status", "nodes": 5},
            id="msg-control",
        )
        store.store(msg)

        pending = store.get_pending("user:testuser")
        assert len(pending) == 1
        assert pending[0].content == {"action": "status", "nodes": 5}

    def test_get_pending_filters_by_node(self, store):
        """get_pending only returns messages for the specified node."""
        msg1 = Message(
            from_node="user:testuser",
            to_node="agent:echo",
            type=MessageType.MESSAGE,
            content="To echo",
        )
        msg2 = Message(
            from_node="user:testuser",
            to_node="agent:researcher",
            type=MessageType.MESSAGE,
            content="To researcher",
        )
        store.store(msg1)
        store.store(msg2)

        echo_pending = store.get_pending("agent:echo")
        assert len(echo_pending) == 1
        assert echo_pending[0].content == "To echo"

        researcher_pending = store.get_pending("agent:researcher")
        assert len(researcher_pending) == 1
        assert researcher_pending[0].content == "To researcher"

    def test_get_pending_ordered_by_time(self, store):
        """Messages are returned in creation order."""
        for i in range(5):
            msg = Message(
                from_node="user:testuser",
                to_node="agent:echo",
                type=MessageType.MESSAGE,
                content=f"Message {i}",
                id=f"msg-{i}",
            )
            store.store(msg)

        pending = store.get_pending("agent:echo")
        assert len(pending) == 5
        for i, msg in enumerate(pending):
            assert msg.content == f"Message {i}"

    def test_get_pending_empty(self, store):
        """get_pending returns empty list if no messages."""
        pending = store.get_pending("nonexistent:node")
        assert pending == []


class TestMessageStoreRemove:
    def test_remove_message(self, store):
        """Remove a specific message by ID."""
        msg = Message(
            from_node="user:testuser",
            to_node="agent:echo",
            type=MessageType.MESSAGE,
            content="Hello",
            id="msg-to-remove",
        )
        store.store(msg)
        assert len(store.get_pending("agent:echo")) == 1

        store.remove("msg-to-remove")
        assert len(store.get_pending("agent:echo")) == 0

    def test_remove_nonexistent(self, store):
        """Removing nonexistent message doesn't error."""
        store.remove("msg-nonexistent")  # Should not raise

    def test_remove_all_for_node(self, store):
        """Remove all messages for a specific node."""
        for i in range(5):
            store.store(Message(
                from_node="user:testuser",
                to_node="agent:echo",
                type=MessageType.MESSAGE,
                content=f"Message {i}",
            ))
        # Also add message for different node
        store.store(Message(
            from_node="user:testuser",
            to_node="agent:other",
            type=MessageType.MESSAGE,
            content="Other",
        ))

        count = store.remove_all("agent:echo")
        assert count == 5
        assert len(store.get_pending("agent:echo")) == 0
        assert len(store.get_pending("agent:other")) == 1

    def test_remove_all_empty(self, store):
        """remove_all returns 0 if no messages to remove."""
        count = store.remove_all("nonexistent:node")
        assert count == 0


class TestMessageStoreCount:
    def test_count_pending_all(self, store):
        """Count all pending messages."""
        store.store(Message(
            from_node="a", to_node="b",
            type=MessageType.MESSAGE, content="1",
        ))
        store.store(Message(
            from_node="a", to_node="c",
            type=MessageType.MESSAGE, content="2",
        ))
        store.store(Message(
            from_node="a", to_node="b",
            type=MessageType.MESSAGE, content="3",
        ))

        assert store.count_pending() == 3

    def test_count_pending_by_node(self, store):
        """Count pending messages for a specific node."""
        for i in range(3):
            store.store(Message(
                from_node="a", to_node="node1",
                type=MessageType.MESSAGE, content=f"{i}",
            ))
        for i in range(2):
            store.store(Message(
                from_node="a", to_node="node2",
                type=MessageType.MESSAGE, content=f"{i}",
            ))

        assert store.count_pending("node1") == 3
        assert store.count_pending("node2") == 2
        assert store.count_pending("node3") == 0

    def test_count_empty(self, store):
        """Count is 0 for empty store."""
        assert store.count_pending() == 0
        assert store.count_pending("any:node") == 0


class TestMessageStoreEdgeCases:
    def test_replace_on_duplicate_id(self, store):
        """Storing with same ID replaces existing message."""
        msg1 = Message(
            from_node="user:testuser",
            to_node="agent:echo",
            type=MessageType.MESSAGE,
            content="Original",
            id="msg-duplicate",
        )
        store.store(msg1)

        msg2 = Message(
            from_node="user:testuser",
            to_node="agent:echo",
            type=MessageType.MESSAGE,
            content="Updated",
            id="msg-duplicate",
        )
        store.store(msg2)

        pending = store.get_pending("agent:echo")
        assert len(pending) == 1
        assert pending[0].content == "Updated"

    def test_special_characters_in_content(self, store):
        """Handle special characters in content."""
        msg = Message(
            from_node="user:testuser",
            to_node="agent:echo",
            type=MessageType.MESSAGE,
            content='Quote: "test"\nNewline\tTab\r\nCRLF',
            id="msg-special",
        )
        store.store(msg)

        pending = store.get_pending("agent:echo")
        assert pending[0].content == 'Quote: "test"\nNewline\tTab\r\nCRLF'

    def test_unicode_content(self, store):
        """Handle unicode content."""
        msg = Message(
            from_node="user:testuser",
            to_node="agent:echo",
            type=MessageType.MESSAGE,
            content="Hello 世界 🌍 émojis",
            id="msg-unicode",
        )
        store.store(msg)

        pending = store.get_pending("agent:echo")
        assert pending[0].content == "Hello 世界 🌍 émojis"

    def test_metadata_preserved(self, store):
        """Metadata is stored and retrieved correctly."""
        msg = Message(
            from_node="agent:coder",
            to_node="agent:executor",
            type=MessageType.TOOL_REQUEST,
            content={"cmd": "ls"},
            metadata={"tool": "bash", "timeout": 30},
            id="msg-meta",
        )
        store.store(msg)

        pending = store.get_pending("agent:executor")
        assert pending[0].metadata == {"tool": "bash", "timeout": 30}

    def test_in_reply_to_preserved(self, store):
        """in_reply_to is stored and retrieved correctly."""
        msg = Message(
            from_node="agent:echo",
            to_node="user:testuser",
            type=MessageType.MESSAGE,
            content="Reply",
            in_reply_to="msg-original-123",
            id="msg-reply",
        )
        store.store(msg)

        pending = store.get_pending("user:testuser")
        assert pending[0].in_reply_to == "msg-original-123"

    def test_null_in_reply_to(self, store):
        """null in_reply_to is handled correctly."""
        msg = Message(
            from_node="user:testuser",
            to_node="agent:echo",
            type=MessageType.MESSAGE,
            content="First message",
            in_reply_to=None,
        )
        store.store(msg)

        pending = store.get_pending("agent:echo")
        assert pending[0].in_reply_to is None


# =============================================================================
# Channel tests
# =============================================================================


class TestChannelCreate:
    def test_create_channel(self, store):
        """Create a new channel."""
        created = store.create_channel("research", "user:testuser", "Research projects")
        assert created is True

        channel = store.get_channel("research")
        assert channel is not None
        assert channel["name"] == "research"
        assert channel["description"] == "Research projects"
        assert channel["created_by"] == "user:testuser"
        assert channel["created_at"] is not None

    def test_create_duplicate_channel(self, store):
        """Creating duplicate channel returns False."""
        store.create_channel("research", "user:testuser")
        created = store.create_channel("research", "user:bob")
        assert created is False

    def test_channel_exists(self, store):
        """Check if channel exists."""
        assert store.channel_exists("research") is False
        store.create_channel("research", "user:testuser")
        assert store.channel_exists("research") is True


class TestChannelDelete:
    def test_delete_channel(self, store):
        """Delete a channel."""
        store.create_channel("research", "user:testuser")
        assert store.channel_exists("research") is True

        deleted = store.delete_channel("research")
        assert deleted is True
        assert store.channel_exists("research") is False

    def test_delete_nonexistent_channel(self, store):
        """Deleting nonexistent channel returns False."""
        deleted = store.delete_channel("nonexistent")
        assert deleted is False

    def test_delete_removes_members(self, store):
        """Deleting channel removes all members."""
        store.create_channel("research", "user:testuser")
        store.join_channel("research", "user:testuser")
        store.join_channel("research", "agent:coder")

        store.delete_channel("research")

        # Members should be gone (foreign key cascade)
        members = store.get_channel_members("research")
        assert members == []


class TestChannelMembership:
    def test_join_channel(self, store):
        """Join a channel."""
        store.create_channel("research", "user:testuser")
        joined = store.join_channel("research", "agent:coder")
        assert joined is True

        members = store.get_channel_members("research")
        assert "agent:coder" in members

    def test_join_nonexistent_channel(self, store):
        """Joining nonexistent channel returns False."""
        joined = store.join_channel("nonexistent", "user:testuser")
        assert joined is False

    def test_join_twice_returns_false(self, store):
        """Joining same channel twice returns False."""
        store.create_channel("research", "user:testuser")
        store.join_channel("research", "agent:coder")
        joined = store.join_channel("research", "agent:coder")
        assert joined is False

    def test_leave_channel(self, store):
        """Leave a channel."""
        store.create_channel("research", "user:testuser")
        store.join_channel("research", "agent:coder")

        left = store.leave_channel("research", "agent:coder")
        assert left is True

        members = store.get_channel_members("research")
        assert "agent:coder" not in members

    def test_leave_nonmember_returns_false(self, store):
        """Leaving channel when not a member returns False."""
        store.create_channel("research", "user:testuser")
        left = store.leave_channel("research", "agent:coder")
        assert left is False

    def test_is_channel_member(self, store):
        """Check channel membership."""
        store.create_channel("research", "user:testuser")
        assert store.is_channel_member("research", "agent:coder") is False

        store.join_channel("research", "agent:coder")
        assert store.is_channel_member("research", "agent:coder") is True

    def test_get_node_channels(self, store):
        """Get all channels a node belongs to."""
        store.create_channel("research", "user:testuser")
        store.create_channel("android", "user:testuser")
        store.create_channel("other", "user:testuser")

        store.join_channel("research", "agent:coder")
        store.join_channel("android", "agent:coder")

        channels = store.get_node_channels("agent:coder")
        assert set(channels) == {"research", "android"}


class TestChannelList:
    def test_list_channels_empty(self, store):
        """List channels when none exist."""
        channels = store.list_channels()
        assert channels == []

    def test_list_channels(self, store):
        """List all channels with member counts."""
        store.create_channel("research", "user:testuser", "Research projects")
        store.create_channel("android", "user:testuser", "Android dev")

        store.join_channel("research", "user:testuser")
        store.join_channel("research", "agent:coder")
        store.join_channel("android", "user:testuser")

        channels = store.list_channels()
        assert len(channels) == 2

        research = next(c for c in channels if c["name"] == "research")
        assert research["member_count"] == 2
        assert research["description"] == "Research projects"

        android = next(c for c in channels if c["name"] == "android")
        assert android["member_count"] == 1


class TestConversationTodos:
    def test_add_list_and_update_todos(self, store):
        """Todos are stored per conversation with status/version tracking."""
        conv_id = MessageStore.compute_conversation_id("user:testuser", "agent:coder:sobek")
        todo = store.add_todo(
            conv_id,
            "Draft plan",
            created_by="user:testuser",
            section="today",
        )

        assert todo["conversation_id"] == conv_id
        assert todo["section"] == "today"
        assert todo["status"] == "open"
        assert todo["position"] == 0
        assert todo["version"] == 1

        listed = store.list_todos(conv_id)
        assert [t["id"] for t in listed] == [todo["id"]]

        accepted, updated = store.update_todo(
            todo["id"],
            updated_by="agent:coder:sobek",
            status="in_progress",
            section="medium-term",
            update_section=True,
            expected_version=todo["version"],
        )
        assert accepted is True
        assert updated["status"] == "in_progress"
        assert updated["section"] == "medium-term"
        assert updated["version"] == 2

    def test_todo_section_order_can_be_set_and_cleared(self, store):
        """Todo section order is stored per conversation."""
        conv_id = "channel:mesh-infra"

        order = store.set_todo_section_order(
            conv_id,
            ["today", "medium-term", "today", "", "backlog"],
            updated_by="user:testuser",
        )
        assert order == ["today", "medium-term", "backlog"]
        assert store.get_todo_section_order(conv_id) == order

        cleared = store.set_todo_section_order(conv_id, None, updated_by="user:testuser")
        assert cleared == []
        assert store.get_todo_section_order(conv_id) == []

    def test_todo_expected_version_conflict_returns_current(self, store):
        """Version mismatches reject the mutation and return current state."""
        conv_id = "channel:mesh-infra"
        todo = store.add_todo(conv_id, "Check router context", created_by="user:testuser")

        accepted, current = store.update_todo(
            todo["id"],
            updated_by="agent:coder:sobek",
            status="done",
            expected_version=todo["version"] + 1,
        )

        assert accepted is False
        assert current["id"] == todo["id"]
        assert current["status"] == "open"

    def test_done_hidden_unless_included_and_remove_is_soft_delete(self, store):
        """Done items stay persisted, and remove hides via deleted_at."""
        conv_id = "channel:mesh-infra"
        todo = store.add_todo(conv_id, "Ship todos", created_by="user:testuser")

        accepted, done = store.update_todo(
            todo["id"], updated_by="user:testuser", status="done"
        )
        assert accepted is True
        assert done["completed_at"] is not None

        assert store.list_todos(conv_id) == []
        assert len(store.list_todos(conv_id, include_done=True)) == 1

        removed, deleted = store.delete_todo(todo["id"], updated_by="user:testuser")
        assert removed is True
        assert deleted["deleted_at"] is not None
        assert store.list_todos(conv_id, include_done=True) == []
