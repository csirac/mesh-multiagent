"""Tests for the protocol module."""

import pytest
import json
from datetime import datetime, timezone

from mesh.protocol import (
    Message,
    MessageType,
    ControlAction,
    generate_message_id,
    now_iso,
    make_message,
    make_control,
    make_tool_request,
    make_tool_result,
    make_todo_get,
    make_todo_mutate,
    make_todo_response,
    encode_for_wire,
    decode_length_prefix,
)


class TestMessageId:
    def test_generate_unique_ids(self):
        """Each call should produce a unique ID."""
        ids = [generate_message_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_id_format(self):
        """ID should start with msg- prefix."""
        msg_id = generate_message_id()
        assert msg_id.startswith("msg-")
        assert len(msg_id) == 16  # "msg-" + 12 hex chars


class TestTimestamp:
    def test_now_iso_format(self):
        """Timestamp should be valid ISO format."""
        ts = now_iso()
        # Should parse without error
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        assert dt.tzinfo is not None  # Should have timezone


class TestMessage:
    def test_basic_creation(self):
        """Create a message with required fields."""
        msg = Message(
            from_node="user:testuser",
            to_node="agent:echo",
            type=MessageType.MESSAGE,
            content="Hello",
        )
        assert msg.from_node == "user:testuser"
        assert msg.to_node == "agent:echo"
        assert msg.type == MessageType.MESSAGE
        assert msg.content == "Hello"
        assert msg.id.startswith("msg-")
        assert msg.in_reply_to is None
        assert msg.metadata == {}

    def test_with_all_fields(self):
        """Create a message with all fields specified."""
        msg = Message(
            from_node="user:testuser",
            to_node="agent:echo",
            type=MessageType.MESSAGE,
            content="Hello",
            id="msg-custom123",
            timestamp="2026-01-21T12:00:00Z",
            in_reply_to="msg-previous",
            metadata={"key": "value"},
        )
        assert msg.id == "msg-custom123"
        assert msg.timestamp == "2026-01-21T12:00:00Z"
        assert msg.in_reply_to == "msg-previous"
        assert msg.metadata == {"key": "value"}

    def test_dict_content(self):
        """Content can be a dictionary."""
        msg = Message(
            from_node="router",
            to_node="user:testuser",
            type=MessageType.CONTROL,
            content={"action": "list_nodes", "nodes": ["a", "b"]},
        )
        assert msg.content == {"action": "list_nodes", "nodes": ["a", "b"]}


class TestMessageSerialization:
    def test_to_json(self):
        """Serialize message to JSON string."""
        msg = Message(
            from_node="user:testuser",
            to_node="agent:echo",
            type=MessageType.MESSAGE,
            content="Hello",
            id="msg-test123456",
            timestamp="2026-01-21T12:00:00Z",
        )
        json_str = msg.to_json()
        data = json.loads(json_str)

        assert data["from_node"] == "user:testuser"
        assert data["to_node"] == "agent:echo"
        assert data["type"] == "message"  # Enum serialized as string
        assert data["content"] == "Hello"
        assert data["id"] == "msg-test123456"

    def test_from_json_string(self):
        """Deserialize message from JSON string."""
        json_str = json.dumps({
            "from_node": "agent:echo",
            "to_node": "user:testuser",
            "type": "message",
            "content": "Reply",
            "id": "msg-reply12345",
            "timestamp": "2026-01-21T12:00:01Z",
            "in_reply_to": "msg-original",
            "metadata": {},
        })
        msg = Message.from_json(json_str)

        assert msg.from_node == "agent:echo"
        assert msg.to_node == "user:testuser"
        assert msg.type == MessageType.MESSAGE
        assert msg.content == "Reply"
        assert msg.in_reply_to == "msg-original"

    def test_from_json_bytes(self):
        """Deserialize from bytes."""
        json_bytes = b'{"from_node": "a", "to_node": "b", "type": "message", "content": "test", "id": "msg-123", "timestamp": "2026-01-21T12:00:00Z", "in_reply_to": null, "metadata": {}}'
        msg = Message.from_json(json_bytes)
        assert msg.content == "test"

    def test_roundtrip(self):
        """Serialize then deserialize produces equivalent message."""
        original = Message(
            from_node="user:testuser",
            to_node="agent:echo",
            type=MessageType.TOOL_REQUEST,
            content={"arg": "value"},
            metadata={"tool": "search"},
        )
        json_str = original.to_json()
        restored = Message.from_json(json_str)

        assert restored.from_node == original.from_node
        assert restored.to_node == original.to_node
        assert restored.type == original.type
        assert restored.content == original.content
        assert restored.id == original.id
        assert restored.metadata == original.metadata


class TestMessageReply:
    def test_reply_swaps_from_to(self):
        """Reply swaps from_node and to_node."""
        original = Message(
            from_node="user:testuser",
            to_node="agent:echo",
            type=MessageType.MESSAGE,
            content="Hello",
            id="msg-original",
        )
        reply = original.reply("Hello back!")

        assert reply.from_node == "agent:echo"  # Was to_node
        assert reply.to_node == "user:testuser"    # Was from_node
        assert reply.content == "Hello back!"
        assert reply.in_reply_to == "msg-original"
        assert reply.type == MessageType.MESSAGE

    def test_reply_with_different_type(self):
        """Reply can have a different type."""
        original = Message(
            from_node="user:testuser",
            to_node="agent:echo",
            type=MessageType.TOOL_REQUEST,
            content={"name": "search"},
            id="msg-request",
        )
        reply = original.reply(
            {"result": "found"},
            type=MessageType.TOOL_RESULT,
            metadata={"success": True},
        )

        assert reply.type == MessageType.TOOL_RESULT
        assert reply.content == {"result": "found"}
        assert reply.metadata == {"success": True}


class TestConvenienceConstructors:
    def test_make_message(self):
        """make_message creates a standard MESSAGE type."""
        msg = make_message("user:testuser", "agent:echo", "Hello")
        assert msg.type == MessageType.MESSAGE
        assert msg.from_node == "user:testuser"
        assert msg.to_node == "agent:echo"
        assert msg.content == "Hello"

    def test_make_message_with_reply_to(self):
        """make_message can set in_reply_to."""
        msg = make_message("agent:echo", "user:testuser", "Reply", in_reply_to="msg-123")
        assert msg.in_reply_to == "msg-123"

    def test_make_control(self):
        """make_control creates a CONTROL message to router."""
        msg = make_control("user:testuser", ControlAction.LIST_NODES)
        assert msg.type == MessageType.CONTROL
        assert msg.from_node == "user:testuser"
        assert msg.to_node == "router"
        assert msg.content["action"] == "list_nodes"

    def test_make_control_with_target(self):
        """make_control can specify a target node."""
        msg = make_control("user:testuser", ControlAction.KILL, target_node="agent:echo")
        assert msg.content["target"] == "agent:echo"

    def test_make_control_with_config(self):
        """make_control can include config in metadata."""
        msg = make_control(
            "user:testuser",
            ControlAction.SPAWN,
            target_node="agent:new",
            config={"model": "gpt-4"},
        )
        assert msg.metadata["config"] == {"model": "gpt-4"}

    def test_make_tool_request(self):
        """make_tool_request creates a TOOL_REQUEST message."""
        msg = make_tool_request(
            "agent:coder",
            "agent:executor",
            "bash",
            {"command": "ls -la"},
        )
        assert msg.type == MessageType.TOOL_REQUEST
        assert msg.from_node == "agent:coder"
        assert msg.to_node == "agent:executor"
        assert msg.content == {"command": "ls -la"}
        assert msg.metadata["tool"] == "bash"

    def test_make_tool_result(self):
        """make_tool_result creates a TOOL_RESULT message."""
        msg = make_tool_result(
            "agent:executor",
            "agent:coder",
            "file1.txt\nfile2.txt",
            in_reply_to="msg-request",
            success=True,
        )
        assert msg.type == MessageType.TOOL_RESULT
        assert msg.content["result"] == "file1.txt\nfile2.txt"
        assert msg.content["success"] is True
        assert msg.content["error"] is None
        assert msg.in_reply_to == "msg-request"

    def test_make_tool_result_with_error(self):
        """make_tool_result can indicate failure."""
        msg = make_tool_result(
            "agent:executor",
            "agent:coder",
            None,
            in_reply_to="msg-request",
            success=False,
            error="Command not found",
        )
        assert msg.content["success"] is False
        assert msg.content["error"] == "Command not found"


class TestWireFormat:
    def test_encode_for_wire(self):
        """encode_for_wire produces length-prefixed bytes."""
        msg = Message(
            from_node="a",
            to_node="b",
            type=MessageType.MESSAGE,
            content="test",
            id="msg-123456789",
            timestamp="2026-01-21T12:00:00Z",
        )
        wire_data = encode_for_wire(msg)

        # First 4 bytes are length
        length = int.from_bytes(wire_data[:4], "big")
        payload = wire_data[4:]
        assert len(payload) == length

        # Payload should be valid JSON
        data = json.loads(payload.decode("utf-8"))
        assert data["content"] == "test"

    def test_decode_length_prefix(self):
        """decode_length_prefix extracts the length."""
        data = (100).to_bytes(4, "big") + b"x" * 100
        length = decode_length_prefix(data)
        assert length == 100

    def test_wire_roundtrip(self):
        """Encode then manually decode produces same message."""
        original = Message(
            from_node="user:testuser",
            to_node="agent:echo",
            type=MessageType.MESSAGE,
            content="Hello, mesh!",
        )
        wire_data = encode_for_wire(original)

        # Manual decode
        length = decode_length_prefix(wire_data)
        payload = wire_data[4:4+length]
        restored = Message.from_json(payload)

        assert restored.from_node == original.from_node
        assert restored.to_node == original.to_node
        assert restored.content == original.content
        assert restored.id == original.id


class TestMessageTypes:
    def test_message_type_values(self):
        """MessageType enum has expected values."""
        assert MessageType.MESSAGE.value == "message"
        assert MessageType.TOOL_REQUEST.value == "tool_request"
        assert MessageType.TOOL_RESULT.value == "tool_result"
        assert MessageType.CONTROL.value == "control"
        assert MessageType.CONFIRM_REQUEST.value == "confirm_request"
        assert MessageType.CONFIRM_RESPONSE.value == "confirm_response"

    def test_control_action_values(self):
        """ControlAction enum has expected values."""
        assert ControlAction.SPAWN.value == "spawn"
        assert ControlAction.KILL.value == "kill"
        assert ControlAction.STATUS.value == "status"
        assert ControlAction.PAUSE.value == "pause"
        assert ControlAction.RESUME.value == "resume"
        assert ControlAction.LIST_NODES.value == "list_nodes"
        assert ControlAction.REGISTER.value == "register"
        assert ControlAction.ACK.value == "ack"
        assert ControlAction.TODO_GET.value == "todo_get"
        assert ControlAction.TODO_MUTATE.value == "todo_mutate"
        assert ControlAction.TODO_RESPONSE.value == "todo_response"


class TestTodoControlMessages:
    def test_make_todo_messages(self):
        """Todo control factories create broker-routed control messages."""
        get_msg = make_todo_get("user:testuser", ["channel:mesh-infra"], include_done=True)
        assert get_msg.type == MessageType.CONTROL
        assert get_msg.to_node == "router"
        assert get_msg.content["action"] == ControlAction.TODO_GET.value
        assert get_msg.content["conversation_ids"] == ["channel:mesh-infra"]

        mutate_msg = make_todo_mutate(
            "agent:coder:sobek",
            "channel:mesh-infra",
            "add",
            payload={"text": "Draft plan"},
            expected_version=2,
        )
        assert mutate_msg.content["action"] == ControlAction.TODO_MUTATE.value
        assert mutate_msg.content["op"] == "add"
        assert mutate_msg.content["payload"]["text"] == "Draft plan"
        assert mutate_msg.content["expected_version"] == 2

        response = make_todo_response(
            "user:testuser",
            {"channel:mesh-infra": [{"id": "todo-1"}]},
            section_order={"channel:mesh-infra": ["today", "medium-term"]},
            accepted=True,
            conversation_id="channel:mesh-infra",
            in_reply_to=mutate_msg.id,
        )
        assert response.from_node == "router"
        assert response.content["action"] == ControlAction.TODO_RESPONSE.value
        assert response.content["section_order"]["channel:mesh-infra"] == ["today", "medium-term"]
        assert response.content["accepted"] is True
        assert response.in_reply_to == mutate_msg.id


class TestConfirmMessages:
    def test_make_confirm_request(self):
        """make_confirm_request creates a CONFIRM_REQUEST message."""
        from mesh.protocol import make_confirm_request
        msg = make_confirm_request(
            from_node="agent:assistant",
            to_node="user:testuser",
            tool_name="gmail_send_message",
            tool_args={"to": "bob@example.com", "subject": "Hello"},
            preview="Send email to bob@example.com\nSubject: Hello",
        )
        assert msg.type == MessageType.CONFIRM_REQUEST
        assert msg.from_node == "agent:assistant"
        assert msg.to_node == "user:testuser"
        assert msg.content["tool_name"] == "gmail_send_message"
        assert msg.content["tool_args"]["to"] == "bob@example.com"
        assert "Send email" in msg.content["preview"]

    def test_make_confirm_response_confirmed(self):
        """make_confirm_response creates a CONFIRM_RESPONSE with confirmed=True."""
        from mesh.protocol import make_confirm_response
        msg = make_confirm_response(
            from_node="user:testuser",
            to_node="agent:assistant",
            in_reply_to="msg-request-123",
            confirmed=True,
        )
        assert msg.type == MessageType.CONFIRM_RESPONSE
        assert msg.from_node == "user:testuser"
        assert msg.to_node == "agent:assistant"
        assert msg.in_reply_to == "msg-request-123"
        assert msg.content["confirmed"] is True

    def test_make_confirm_response_rejected(self):
        """make_confirm_response creates a CONFIRM_RESPONSE with confirmed=False."""
        from mesh.protocol import make_confirm_response
        msg = make_confirm_response(
            from_node="user:testuser",
            to_node="agent:assistant",
            in_reply_to="msg-request-123",
            confirmed=False,
        )
        assert msg.content["confirmed"] is False

    def test_confirm_messages_serialize(self):
        """Confirm messages serialize and deserialize correctly."""
        from mesh.protocol import make_confirm_request, make_confirm_response

        # Test request
        request = make_confirm_request(
            from_node="agent:assistant",
            to_node="user:testuser",
            tool_name="test_tool",
            tool_args={"arg1": "value1"},
            preview="Test preview",
        )
        json_str = request.to_json()
        restored = Message.from_json(json_str)
        assert restored.type == MessageType.CONFIRM_REQUEST
        assert restored.content["tool_name"] == "test_tool"

        # Test response
        response = make_confirm_response(
            from_node="user:testuser",
            to_node="agent:assistant",
            in_reply_to=request.id,
            confirmed=True,
        )
        json_str = response.to_json()
        restored = Message.from_json(json_str)
        assert restored.type == MessageType.CONFIRM_RESPONSE
        assert restored.content["confirmed"] is True
