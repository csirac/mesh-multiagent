"""Tests for the LLM client."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from mesh.llm import LLMClient, LLMConfig, HistoryMessage


class EventCapture:
    """Simple callback for LLM streaming-event tests."""

    def __init__(self):
        self.events = []

    def on_cc_tool_event(self, event):
        self.events.append(event)

    def on_todos(self, todos):
        pass


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig()
        assert config.api_key == ""
        assert config.base_url == "https://api.openai.com/v1"
        assert config.model == "gpt-4"
        assert config.max_tokens == 4096
        assert config.temperature == 0.7

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LLMConfig(
            api_key="test-key",
            base_url="http://localhost:11434/v1",
            model="llama3",
            max_tokens=2048,
            temperature=0.5,
        )
        assert config.api_key == "test-key"
        assert config.base_url == "http://localhost:11434/v1"
        assert config.model == "llama3"
        assert config.max_tokens == 2048
        assert config.temperature == 0.5

    def test_from_env(self, monkeypatch):
        """Test loading config from environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://test.com/v1")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-3.5-turbo")

        config = LLMConfig.from_env()
        assert config.api_key == "env-key"
        assert config.base_url == "http://test.com/v1"
        assert config.model == "gpt-3.5-turbo"

    def test_from_env_with_prefix(self, monkeypatch):
        """Test loading config with custom prefix."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "claude-key")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")

        config = LLMConfig.from_env(prefix="ANTHROPIC")
        assert config.api_key == "claude-key"
        assert config.base_url == "https://api.anthropic.com/v1"


class TestHistoryMessage:
    """Tests for HistoryMessage."""

    def test_creation(self):
        """Test creating a history message."""
        msg = HistoryMessage(
            from_node="user:testuser",
            content="Hello, world!",
            timestamp="2026-01-21T20:00:00Z",
        )
        assert msg.from_node == "user:testuser"
        assert msg.content == "Hello, world!"
        assert msg.timestamp == "2026-01-21T20:00:00Z"


class TestLLMClient:
    """Tests for LLMClient."""

    def test_format_history_xml_empty(self):
        """Test formatting empty history."""
        config = LLMConfig(api_key="test")
        client = LLMClient(config)

        result = client.format_history_xml([], "agent:test")

        assert "<history>" in result
        assert "</history>" in result
        assert "<identity>" in result
        assert "You are agent:test" in result
        assert "<instructions>" in result

    def test_format_history_xml_with_messages(self):
        """Test formatting history with messages."""
        config = LLMConfig(api_key="test")
        client = LLMClient(config)

        history = [
            HistoryMessage("user:testuser", "Hello!", "2026-01-21T20:00:00Z"),
            HistoryMessage("agent:test", "Hi there!", "2026-01-21T20:00:01Z"),
        ]

        result = client.format_history_xml(history, "agent:test")

        assert '<message from="user:testuser"' in result
        assert "Hello!" in result
        assert '<message from="agent:test"' in result
        assert "Hi there!" in result

    def test_format_history_xml_with_system_prompt(self):
        """Test formatting with custom system prompt."""
        config = LLMConfig(api_key="test")
        client = LLMClient(config)

        result = client.format_history_xml(
            [], "agent:test", system_prompt="You are helpful."
        )

        assert "<system>" in result
        assert "You are helpful." in result

    def test_format_history_xml_message_received(self):
        """Trigger message extracted from history into <message_received>."""
        config = LLMConfig(api_key="test")
        client = LLMClient(config)

        trigger = HistoryMessage("user:testuser", "Fix the bug", "2026-01-21T20:00:02Z")
        history = [
            HistoryMessage("user:testuser", "Earlier context", "2026-01-21T20:00:00Z"),
            HistoryMessage("agent:test", "Got it", "2026-01-21T20:00:01Z"),
            trigger,
        ]

        result = client.format_history_xml(history, "agent:test", trigger_msg=trigger)

        assert "<message_received" in result
        assert "Fix the bug" in result
        # Trigger should NOT be inside <history> block
        history_block = result.split("</history>")[0]
        assert "Fix the bug" not in history_block
        assert "Earlier context" in history_block
        assert "Got it" in history_block

    def test_format_history_xml_no_trigger(self):
        """Without trigger_msg, no <message_received> XML block rendered."""
        config = LLMConfig(api_key="test")
        client = LLMClient(config)

        history = [
            HistoryMessage("user:testuser", "Hello", "2026-01-21T20:00:00Z"),
        ]

        result = client.format_history_xml(history, "agent:test", trigger_msg=None)

        # No actual <message_received> block — the word may appear in default instructions text
        assert "<message_received from=" not in result
        assert "Hello" in result

    def test_format_history_xml_trigger_with_to_node(self):
        """Trigger with to_node renders it in <message_received> attrs."""
        config = LLMConfig(api_key="test")
        client = LLMClient(config)

        trigger = HistoryMessage("user:testuser", "Channel msg", "2026-01-21T20:00:00Z",
                                 to_node="channel:general")
        result = client.format_history_xml([trigger], "agent:test", trigger_msg=trigger)

        assert 'to="channel:general"' in result
        assert "<message_received" in result

    def test_format_history_xml_default_instructions_with_trigger(self):
        """Default instructions reference <message_received> when trigger is present."""
        config = LLMConfig(api_key="test")
        client = LLMClient(config)

        trigger = HistoryMessage("user:testuser", "Do stuff", "2026-01-21T20:00:00Z")
        result = client.format_history_xml([trigger], "agent:test", trigger_msg=trigger)

        assert "message_received" in result.split("<instructions>")[1]

    @pytest.mark.asyncio
    async def test_complete_success(self):
        """Test successful completion."""
        config = LLMConfig(api_key="test-key", base_url="http://test.com/v1")
        client = LLMClient(config)

        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from LLM!"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_response
            mock_ensure.return_value = mock_http_client

            result = await client.complete("Test prompt")

        assert result == "Hello from LLM!"
        mock_http_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_with_history(self):
        """Test complete_with_history convenience method."""
        config = LLMConfig(api_key="test-key")
        client = LLMClient(config)

        history = [
            HistoryMessage("user:testuser", "What is 2+2?", "2026-01-21T20:00:00Z"),
        ]

        # Mock complete method
        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "4"

            result = await client.complete_with_history(
                history=history,
                node_id="agent:math",
                system_prompt="You do math.",
            )

        assert result == "4"
        # Check that complete was called with formatted XML
        call_args = mock_complete.call_args
        prompt = call_args[0][0]
        assert "What is 2+2?" in prompt
        assert "agent:math" in prompt
        assert "You do math." in prompt

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        config = LLMConfig(api_key="test")

        async with LLMClient(config) as client:
            assert client._client is not None

        # Client should be closed after exiting context
        assert client._client is None

    def test_codex_item_command_execution_events_emit_tool_activity(self):
        """Current Codex JSONL command_execution items produce tool callbacks."""
        config = LLMConfig(backend="codex")
        client = LLMClient(config)
        capture = EventCapture()
        usage = {}
        text_blocks = []

        client._process_codex_event(
            {
                "type": "item.started",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                    "command": "/bin/bash -lc 'printf hello > /tmp/test'",
                    "status": "in_progress",
                },
            },
            usage,
            text_blocks,
            capture,
        )
        client._process_codex_event(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                    "command": "/bin/bash -lc 'printf hello > /tmp/test'",
                    "aggregated_output": "hello",
                    "exit_code": 0,
                    "status": "completed",
                },
            },
            usage,
            text_blocks,
            capture,
        )

        assert text_blocks == []
        assert [(e.event_type, e.tool_name, e.call_id) for e in capture.events] == [
            ("tool_call", "codex:shell", "item_1"),
            ("tool_result", "codex:shell", "item_1"),
        ]
        assert capture.events[0].data["command"].startswith("/bin/bash")
        assert capture.events[1].data["exit_code"] == 0
        assert capture.events[1].data["output"] == "hello"

    def test_codex_item_agent_message_still_appends_text(self):
        """Codex item.completed agent_message remains the assistant response."""
        config = LLMConfig(backend="codex")
        client = LLMClient(config)
        capture = EventCapture()
        usage = {}
        text_blocks = []

        client._process_codex_event(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_2",
                    "type": "agent_message",
                    "text": "DONE",
                },
            },
            usage,
            text_blocks,
            capture,
        )

        assert text_blocks == ["DONE"]
        assert capture.events == []

    @pytest.mark.asyncio
    async def test_deepseek_multi_turn_threads_reasoning_content(self):
        """Native multi-turn sends prior reasoning and keeps high effort enabled."""
        config = LLMConfig(
            backend="openai",
            model="deepseek-v4-pro",
            base_url="https://api.deepseek.com/v1",
            api_key="test-key",
            reasoning_effort="high",
        )
        client = LLMClient(config)
        captured = {}

        class Response:
            status_code = 200
            text = "ok"

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [{"message": {
                        "role": "assistant",
                        "content": "done",
                        "reasoning_content": "continued reasoning",
                    }}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                }

        class HTTPClient:
            async def post(self, url, headers, json):
                captured.update({"url": url, "headers": headers, "json": json})
                return Response()

        messages = [
            {"role": "user", "content": "edit the digest"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "reasoning from turn one",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "file_read", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]

        with patch.object(client, "_ensure_client", return_value=HTTPClient()):
            content, tool_calls, usage = await client.complete_multi_turn(
                messages, tools=[], parallel_tool_calls=False,
            )

        payload = captured["json"]
        assert payload["messages"][1]["reasoning_content"] == "reasoning from turn one"
        assert payload["reasoning_effort"] == "high"
        assert payload["thinking"] == {"type": "enabled"}
        assert "parallel_tool_calls" not in payload
        assert content == "done"
        assert tool_calls == []
        assert usage["input_tokens"] == 10
        assert client._last_reasoning_content == "continued reasoning"
        assert client.supports_native_reasoning_multiturn is True


class TestLLMConfigBackend:
    """Tests for LLM backend config in MeshConfig."""

    def test_llm_backend_config_env_expansion(self, monkeypatch):
        """Test environment variable expansion in api_key."""
        from mesh.config import LLMBackendConfig

        monkeypatch.setenv("MY_API_KEY", "secret-key-123")

        config = LLMBackendConfig(api_key="${MY_API_KEY}")
        assert config.api_key == "secret-key-123"

    def test_llm_backend_config_no_expansion(self):
        """Test that regular keys are not modified."""
        from mesh.config import LLMBackendConfig

        config = LLMBackendConfig(api_key="regular-key")
        assert config.api_key == "regular-key"

    def test_mesh_config_with_llm_backends(self):
        """Test MeshConfig with LLM backends."""
        from mesh.config import MeshConfig

        data = {
            "router": {"host": "localhost", "port": 7700},
            "llm_backends": {
                "openai": {
                    "api_key": "test-key",
                    "base_url": "https://api.openai.com/v1",
                    "default_model": "gpt-4",
                },
                "local": {
                    "api_key": "",
                    "base_url": "http://localhost:11434/v1",
                    "default_model": "llama3",
                },
            },
            "nodes": {
                "agent:test": {
                    "llm_backend": "openai",
                    "llm_model": "gpt-4-turbo",
                },
            },
        }

        config = MeshConfig.from_dict(data)

        assert "openai" in config.llm_backends
        assert "local" in config.llm_backends
        assert config.llm_backends["openai"].api_key == "test-key"
        assert config.llm_backends["local"].default_model == "llama3"

    def test_get_llm_config_for_node(self):
        """Test getting LLM config for a specific node."""
        from mesh.config import MeshConfig

        data = {
            "llm_backends": {
                "default": {
                    "api_key": "default-key",
                    "default_model": "gpt-4",
                },
            },
            "nodes": {
                "agent:test": {
                    "llm_backend": "default",
                    "llm_model": "gpt-4-turbo",
                },
            },
        }

        config = MeshConfig.from_dict(data)
        llm_config = config.get_llm_config_for_node("agent:test")

        assert llm_config is not None
        assert llm_config.api_key == "default-key"

    def test_get_llm_config_for_nonexistent_node(self):
        """Test getting LLM config for nonexistent node."""
        from mesh.config import MeshConfig

        config = MeshConfig()
        llm_config = config.get_llm_config_for_node("agent:nonexistent")

        assert llm_config is None


class TestOpenAIToolConversion:
    """Tests for OpenAI native tool support."""

    def test_convert_openai_tool_calls_single(self):
        """Test converting a single OpenAI tool call."""
        config = LLMConfig(api_key="test")
        client = LLMClient(config)

        openai_tool_calls = [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Austin", "units": "celsius"}',
                }
            }
        ]

        result = client._convert_openai_tool_calls(openai_tool_calls)

        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert result[0].arguments == {"location": "Austin", "units": "celsius"}
        assert "get_weather" in result[0].raw_xml

    def test_convert_openai_tool_calls_multiple(self):
        """Test converting multiple OpenAI tool calls."""
        config = LLMConfig(api_key="test")
        client = LLMClient(config)

        openai_tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "tool_a",
                    "arguments": '{"x": 1}',
                }
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "tool_b",
                    "arguments": '{"y": "test"}',
                }
            }
        ]

        result = client._convert_openai_tool_calls(openai_tool_calls)

        assert len(result) == 2
        assert result[0].name == "tool_a"
        assert result[0].arguments == {"x": 1}
        assert result[1].name == "tool_b"
        assert result[1].arguments == {"y": "test"}

    def test_convert_openai_tool_calls_empty_args(self):
        """Test converting tool call with no arguments."""
        config = LLMConfig(api_key="test")
        client = LLMClient(config)

        openai_tool_calls = [
            {
                "id": "call_xyz",
                "type": "function",
                "function": {
                    "name": "current_time",
                    "arguments": "{}",
                }
            }
        ]

        result = client._convert_openai_tool_calls(openai_tool_calls)

        assert len(result) == 1
        assert result[0].name == "current_time"
        assert result[0].arguments == {}

    def test_convert_openai_tool_calls_invalid_json(self):
        """Test converting tool call with invalid JSON arguments."""
        config = LLMConfig(api_key="test")
        client = LLMClient(config)

        openai_tool_calls = [
            {
                "id": "call_bad",
                "type": "function",
                "function": {
                    "name": "test",
                    "arguments": "not valid json",
                }
            }
        ]

        result = client._convert_openai_tool_calls(openai_tool_calls)

        assert len(result) == 1
        assert result[0].name == "test"
        assert result[0].arguments == {}  # Falls back to empty dict

    def test_convert_openai_tool_calls_ignores_non_function(self):
        """Test that non-function tool calls are ignored."""
        config = LLMConfig(api_key="test")
        client = LLMClient(config)

        openai_tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "valid", "arguments": "{}"}
            },
            {
                "id": "call_2",
                "type": "other",  # Not a function
                "data": "something"
            }
        ]

        result = client._convert_openai_tool_calls(openai_tool_calls)

        assert len(result) == 1
        assert result[0].name == "valid"


class TestCompleteWithTools:
    """Tests for complete_with_tools method - verifies native function calling flow."""

    @pytest.fixture
    def tool_registry(self):
        """Create a tool registry with test tools."""
        from mesh.tools import ToolRegistry, ToolParameter
        registry = ToolRegistry()
        registry.register(
            "get_weather",
            "Get weather for a location",
            parameters=[
                ToolParameter("location", "string", "City name"),
                ToolParameter("units", "string", "Units", required=False),
            ],
        )
        registry.register(
            "current_time",
            "Get current time",
            parameters=[],
        )
        return registry

    @pytest.mark.asyncio
    async def test_openai_backend_with_tools_uses_native_function_calling(self, tool_registry):
        """Verify OpenAI backend passes tools to API in native format."""
        config = LLMConfig(api_key="test-key", backend="openai", model="gpt-4")
        client = LLMClient(config)

        history = [
            HistoryMessage("user:testuser", "What's the weather in Austin?", "2026-01-21T20:00:00Z"),
        ]

        # Mock _complete_openai to capture what's passed
        captured_tools = None
        async def mock_complete_openai(prompt, model, max_tokens, temperature, tools=None, images=None):
            nonlocal captured_tools
            captured_tools = tools
            # Return a tool call response
            return ("", [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Austin"}',
                    }
                }
            ])

        with patch.object(client, "_complete_openai", side_effect=mock_complete_openai):
            content, tool_calls = await client.complete_with_tools(
                history=history,
                node_id="agent:weather",
                system_prompt="You are a weather assistant.",
                tool_registry=tool_registry,
                tool_names=["get_weather", "current_time"],
            )

        # Verify tools were passed in OpenAI format
        assert captured_tools is not None
        assert len(captured_tools) == 2

        # Check format matches OpenAI function calling spec
        tool_names = {t["function"]["name"] for t in captured_tools}
        assert tool_names == {"get_weather", "current_time"}

        # Verify each tool has the right structure
        for tool in captured_tools:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]
            # strict mode was removed to avoid constrained decoding overhead

        # Verify tool calls were returned correctly
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].arguments == {"location": "Austin"}

    @pytest.mark.asyncio
    async def test_openai_backend_no_tools_does_not_pass_tools(self, tool_registry):
        """Verify OpenAI backend with empty tool_names doesn't use native calling."""
        config = LLMConfig(api_key="test-key", backend="openai", model="gpt-4")
        client = LLMClient(config)

        history = [
            HistoryMessage("user:testuser", "Hello!", "2026-01-21T20:00:00Z"),
        ]

        # Mock complete to capture the call
        captured_prompt = None
        async def mock_complete(prompt, **kwargs):
            nonlocal captured_prompt
            captured_prompt = prompt
            return "Hello back!"

        with patch.object(client, "complete", side_effect=mock_complete):
            content, tool_calls = await client.complete_with_tools(
                history=history,
                node_id="agent:test",
                system_prompt="You are helpful.",
                tool_registry=tool_registry,
                tool_names=None,  # No tools enabled
            )

        # Should use complete() instead of _complete_openai with tools
        assert captured_prompt is not None
        assert content == "Hello back!"
        assert tool_calls == []
        # No tool instructions should be in prompt
        assert "<tools>" not in captured_prompt

    @pytest.mark.asyncio
    async def test_openai_backend_empty_tool_list(self, tool_registry):
        """Verify empty tool_names list doesn't use native calling."""
        config = LLMConfig(api_key="test-key", backend="openai", model="gpt-4")
        client = LLMClient(config)

        history = [
            HistoryMessage("user:testuser", "Hello!", "2026-01-21T20:00:00Z"),
        ]

        captured_prompt = None
        async def mock_complete(prompt, **kwargs):
            nonlocal captured_prompt
            captured_prompt = prompt
            return "Hi there!"

        with patch.object(client, "complete", side_effect=mock_complete):
            content, tool_calls = await client.complete_with_tools(
                history=history,
                node_id="agent:test",
                system_prompt="You are helpful.",
                tool_registry=tool_registry,
                tool_names=[],  # Empty list
            )

        assert content == "Hi there!"
        assert tool_calls == []

    @pytest.mark.asyncio
    async def test_openai_backend_tool_call_returns_content_and_calls(self, tool_registry):
        """Verify both content and tool_calls are returned when LLM returns both."""
        config = LLMConfig(api_key="test-key", backend="openai", model="gpt-4")
        client = LLMClient(config)

        history = [
            HistoryMessage("user:testuser", "What's the weather?", "2026-01-21T20:00:00Z"),
        ]

        async def mock_complete_openai(prompt, model, max_tokens, temperature, tools=None, images=None):
            # LLM returns both content and tool calls
            return ("Let me check that for you.", [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Austin", "units": "celsius"}',
                    }
                }
            ])

        with patch.object(client, "_complete_openai", side_effect=mock_complete_openai):
            content, tool_calls = await client.complete_with_tools(
                history=history,
                node_id="agent:weather",
                system_prompt="You are a weather assistant.",
                tool_registry=tool_registry,
                tool_names=["get_weather"],
            )

        assert content == "Let me check that for you."
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].arguments == {"location": "Austin", "units": "celsius"}

    @pytest.mark.asyncio
    async def test_openai_backend_no_tool_call_in_response(self, tool_registry):
        """Verify handling when LLM doesn't call any tools."""
        config = LLMConfig(api_key="test-key", backend="openai", model="gpt-4")
        client = LLMClient(config)

        history = [
            HistoryMessage("user:testuser", "Hello!", "2026-01-21T20:00:00Z"),
        ]

        async def mock_complete_openai(prompt, model, max_tokens, temperature, tools=None, images=None):
            # LLM returns just content, no tool calls
            return "Hello! How can I help you today?"

        with patch.object(client, "_complete_openai", side_effect=mock_complete_openai):
            content, tool_calls = await client.complete_with_tools(
                history=history,
                node_id="agent:test",
                system_prompt="You are helpful.",
                tool_registry=tool_registry,
                tool_names=["get_weather"],
            )

        assert content == "Hello! How can I help you today?"
        assert tool_calls == []

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_openai_prompt_excludes_xml_tool_syntax(self, tool_registry):
        """Verify OpenAI backend includes common guidance but not XML tool syntax."""
        config = LLMConfig(api_key="test-key", backend="openai", model="gpt-4")
        client = LLMClient(config)

        history = [
            HistoryMessage("user:testuser", "What's the weather?", "2026-01-21T20:00:00Z"),
        ]

        captured_prompt = None
        async def mock_complete_openai(prompt, model, max_tokens, temperature, tools=None, images=None):
            nonlocal captured_prompt
            captured_prompt = prompt
            return "The weather is nice."

        with patch.object(client, "_complete_openai", side_effect=mock_complete_openai):
            await client.complete_with_tools(
                history=history,
                node_id="agent:weather",
                system_prompt="You are a weather assistant.",
                tool_registry=tool_registry,
                tool_names=["get_weather"],
            )

        # OpenAI gets common guidance (in <tools> block) but not XML syntax
        assert captured_prompt is not None
        assert "<tools>" in captured_prompt  # Has the tools section
        # But does NOT contain XML mesh_call syntax examples
        assert '<mesh_call name="' not in captured_prompt

    @pytest.mark.asyncio
    async def test_openai_multiple_tool_calls(self, tool_registry):
        """Verify multiple tool calls are handled correctly."""
        config = LLMConfig(api_key="test-key", backend="openai", model="gpt-4")
        client = LLMClient(config)

        history = [
            HistoryMessage("user:testuser", "Weather in Austin and current time?", "2026-01-21T20:00:00Z"),
        ]

        async def mock_complete_openai(prompt, model, max_tokens, temperature, tools=None, images=None):
            return ("", [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Austin"}',
                    }
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "current_time",
                        "arguments": "{}",
                    }
                },
            ])

        with patch.object(client, "_complete_openai", side_effect=mock_complete_openai):
            content, tool_calls = await client.complete_with_tools(
                history=history,
                node_id="agent:assistant",
                system_prompt="You are helpful.",
                tool_registry=tool_registry,
                tool_names=["get_weather", "current_time"],
            )

        assert len(tool_calls) == 2
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].arguments == {"location": "Austin"}
        assert tool_calls[1].name == "current_time"
        assert tool_calls[1].arguments == {}
