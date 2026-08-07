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
    make_history_sync,
    make_todo_get,
    make_todo_mutate,
    make_todo_response,
    make_conversation_notes_get,
    make_conversation_notes_set,
    make_autonomous_control,
    make_autonomous_control_response,
    parse_autonomous_control,
    build_autonomous_wake_prompt,
    AUTONOMOUS_CONTROL_OPS,
    AUTONOMOUS_WAKE_HEADER,
    make_conversation_notes_response,
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

    def test_make_history_sync_with_before(self):
        """make_history_sync can request older pages in one conversation."""
        msg = make_history_sync(
            "user:testuser",
            conversation_id="channel:rec-fishing",
            before="2026-01-01T00:03:00.000Z",
            limit=200,
        )

        assert msg.type == MessageType.CONTROL
        assert msg.to_node == "router"
        assert msg.content == {
            "action": ControlAction.HISTORY_SYNC.value,
            "limit": 200,
            "conversation_id": "channel:rec-fishing",
            "before": "2026-01-01T00:03:00.000Z",
        }

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
        assert ControlAction.CONVERSATION_NOTES_GET.value == "conversation_notes_get"
        assert ControlAction.CONVERSATION_NOTES_SET.value == "conversation_notes_set"
        assert ControlAction.CONVERSATION_NOTES_RESPONSE.value == "conversation_notes_response"
        assert ControlAction.AUTONOMOUS_CONTROL.value == "autonomous_control"
        assert ControlAction.AUTONOMOUS_CONTROL_RESPONSE.value == "autonomous_control_response"


class TestAutonomousControlMessages:
    def test_make_autonomous_control_addresses_the_router(self):
        msg = make_autonomous_control("user:testuser", "status", agent="agent:coder:autopilot")
        assert msg.type == MessageType.CONTROL
        assert msg.to_node == "router"
        assert msg.content["action"] == ControlAction.AUTONOMOUS_CONTROL.value
        assert msg.content["payload"]["op"] == "status"
        assert msg.content["payload"]["agent"] == "agent:coder:autopilot"

    def test_round_trip_each_op(self):
        """Every op survives make -> parse with its op-specific fields intact."""
        cases = {
            "status": {},
            "wake": {"project": "project:bluesky-rl", "wake_time": "in 30 minutes"},
            "cancel": {"wake_id": "wake-dbf4b6f4"},
            "budget": {"project": "project:bluesky-rl", "count": 10},
            "budget-reset": {"project": "project:bluesky-rl"},
            "active": {"project": "project:bluesky-rl", "value": True},
            "report": {"project": "project:bluesky-rl", "since": "2026-08-01"},
        }
        assert set(cases) == set(AUTONOMOUS_CONTROL_OPS)
        for op, extra in cases.items():
            msg = make_autonomous_control(
                "user:testuser", op, agent="agent:coder:autopilot-rl", **extra
            )
            parsed = parse_autonomous_control(msg.content)
            assert parsed["op"] == op
            assert parsed["agent"] == "agent:coder:autopilot-rl"
            for field, value in extra.items():
                assert parsed[field] == value
            if op == "budget-reset":
                assert parsed["count"] is None

    def test_parse_accepts_bare_payload_and_full_content(self):
        msg = make_autonomous_control("user:testuser", "status", agent="agent:researcher:reme")
        assert (
            parse_autonomous_control(msg.content)
            == parse_autonomous_control(msg.content["payload"])
        )

    def test_parse_rejects_malformed_payloads(self):
        with pytest.raises(ValueError):
            parse_autonomous_control(None)
        with pytest.raises(ValueError, match="unknown op"):
            parse_autonomous_control({"op": "detonate"})
        with pytest.raises(ValueError, match="unknown op"):
            parse_autonomous_control({})
        with pytest.raises(ValueError, match="wake_time required"):
            parse_autonomous_control({"op": "wake"})
        with pytest.raises(ValueError, match="wake_id required"):
            parse_autonomous_control({"op": "cancel"})
        with pytest.raises(ValueError, match="project required for op=report"):
            parse_autonomous_control({"op": "report"})
        with pytest.raises(ValueError, match="since must be YYYY-MM-DD"):
            parse_autonomous_control(
                {"op": "report", "project": "project:x", "since": "2026/08/01"}
            )
        with pytest.raises(ValueError, match="since must be YYYY-MM-DD"):
            parse_autonomous_control(
                {"op": "report", "project": "project:x", "since": "2026-8-1"}
            )
        with pytest.raises(ValueError, match="count required"):
            parse_autonomous_control({"op": "budget", "project": "project:x"})
        with pytest.raises(ValueError, match="project required for op=budget-reset"):
            parse_autonomous_control({"op": "budget-reset"})
        with pytest.raises(ValueError, match="count must be an integer"):
            parse_autonomous_control(
                {"op": "budget", "project": "project:x", "count": "many"}
            )
        with pytest.raises(ValueError, match="project required for op=active"):
            parse_autonomous_control({"op": "active", "value": True})
        with pytest.raises(ValueError, match="value required for op=active"):
            parse_autonomous_control({"op": "active", "project": "project:x"})
        with pytest.raises(ValueError, match="value must be one of"):
            parse_autonomous_control(
                {"op": "active", "project": "project:x", "value": "maybe"}
            )

    def test_active_value_is_normalized_to_a_bool_at_the_boundary(self):
        """Operators type words; every consumer downstream sees a real bool."""
        for spelling in ("on", "ON", "true", "yes", "1", "enable", True):
            parsed = parse_autonomous_control(
                {"op": "active", "project": "project:x", "value": spelling}
            )
            assert parsed["value"] is True, spelling
        for spelling in ("off", "OFF", "false", "no", "0", "disable", False):
            parsed = parse_autonomous_control(
                {"op": "active", "project": "project:x", "value": spelling}
            )
            assert parsed["value"] is False, spelling

    def test_active_false_survives_make_and_parse(self):
        """`value=False` must not be dropped the way a falsy optional would."""
        msg = make_autonomous_control(
            "user:testuser", "active", agent="agent:coder:autopilot-rl",
            project="project:bluesky-rl", value=False,
        )
        assert parse_autonomous_control(msg.content)["value"] is False

    def test_response_carries_result_and_error(self):
        ok = make_autonomous_control_response(
            "user:testuser", "wake", agent="agent:coder:autopilot-rl",
            result={"wake_id": "wake-1"}, in_reply_to="msg-1",
        )
        assert ok.content["action"] == ControlAction.AUTONOMOUS_CONTROL_RESPONSE.value
        assert ok.content["accepted"] is True
        assert ok.content["result"]["wake_id"] == "wake-1"
        assert ok.in_reply_to == "msg-1"

        bad = make_autonomous_control_response(
            "user:testuser", "wake", accepted=False, error="nope"
        )
        assert bad.content["accepted"] is False
        assert bad.content["error"] == "nope"

    def test_wake_prompt_carries_the_session_contract(self):
        prompt = build_autonomous_wake_prompt(
            "project:bluesky-rl",
            "/home/testuser/.mesh/digests/project-bluesky-rl.md",
            max_workers_this_session=10,
        )
        assert prompt.startswith(AUTONOMOUS_WAKE_HEADER)
        assert "project_entity_key: project:bluesky-rl" in prompt
        assert "project_dossier: /home/testuser/.mesh/digests/project-bluesky-rl.md" in prompt
        assert "report_to: user:operator" in prompt
        assert "max_workers_this_session: 10" in prompt
        assert "never call dossier_spend_budget yourself" in prompt

        with_extra = build_autonomous_wake_prompt(
            "project:bluesky-rl", "/tmp/d.md", 3, extra_instructions="Focus on T-004."
        )
        assert with_extra.endswith("\n\nFocus on T-004.")


class TestTodoControlMessages:
    def test_make_todo_messages(self):
        """Todo control factories create broker-routed control messages."""
        get_msg = make_todo_get("user:testuser", ["channel:mesh-infra"], include_done=True)
        assert get_msg.type == MessageType.CONTROL
        assert get_msg.to_node == "router"
        assert get_msg.content["action"] == ControlAction.TODO_GET.value
        assert get_msg.content["conversation_ids"] == ["channel:mesh-infra"]

        mutate_msg = make_todo_mutate(
            "agent:coder:coder1",
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


class TestConversationNotesControlMessages:
    def test_make_conversation_notes_messages(self):
        """Conversation note control factories create broker-routed messages."""
        get_msg = make_conversation_notes_get("agent:coder:coder1", "channel:mesh-infra")
        assert get_msg.type == MessageType.CONTROL
        assert get_msg.to_node == "router"
        assert get_msg.content["action"] == ControlAction.CONVERSATION_NOTES_GET.value
        assert get_msg.content["conversation_id"] == "channel:mesh-infra"

        set_msg = make_conversation_notes_set(
            "agent:coder:coder1",
            "channel:mesh-infra",
            "Operations: docs/operations/computehost.md",
        )
        assert set_msg.content["action"] == ControlAction.CONVERSATION_NOTES_SET.value
        assert set_msg.content["content"] == "Operations: docs/operations/computehost.md"

        response = make_conversation_notes_response(
            "user:testuser",
            {"channel:mesh-infra": {"content": "Worklog: docs/worklog.md"}},
            accepted=True,
            conversation_id="channel:mesh-infra",
            in_reply_to=set_msg.id,
        )
        assert response.from_node == "router"
        assert response.content["action"] == ControlAction.CONVERSATION_NOTES_RESPONSE.value
        assert response.content["accepted"] is True
        assert response.content["notes"]["channel:mesh-infra"]["content"] == "Worklog: docs/worklog.md"
        assert response.in_reply_to == set_msg.id


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
