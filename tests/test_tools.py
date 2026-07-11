"""Tests for the tool registry and execution system."""

import pytest
from mesh.tools import (
    ToolRegistry,
    ToolDefinition,
    ToolParameter,
    ToolCall,
    get_registry,
    register_tool,
    tool,
    parse_tool_calls,
    has_tool_call,
    execute_tool,
    execute_tool_calls,
)


class TestToolParameter:
    def test_basic_creation(self):
        param = ToolParameter(
            name="location",
            type="string",
            description="The city name",
        )
        assert param.name == "location"
        assert param.type == "string"
        assert param.description == "The city name"
        assert param.required is True
        assert param.default is None

    def test_optional_with_default(self):
        param = ToolParameter(
            name="units",
            type="string",
            description="Temperature units",
            required=False,
            default="celsius",
        )
        assert param.required is False
        assert param.default == "celsius"


class TestToolDefinition:
    def test_basic_creation(self):
        tool_def = ToolDefinition(
            name="weather",
            description="Get weather for a location",
        )
        assert tool_def.name == "weather"
        assert tool_def.description == "Get weather for a location"
        assert tool_def.parameters == []
        assert tool_def.handler is None
        assert tool_def.requires_confirmation is False

    def test_with_requires_confirmation(self):
        tool_def = ToolDefinition(
            name="send_email",
            description="Send an email",
            requires_confirmation=True,
        )
        assert tool_def.requires_confirmation is True

    def test_with_parameters(self):
        tool_def = ToolDefinition(
            name="weather",
            description="Get weather",
            parameters=[
                ToolParameter("location", "string", "City name"),
                ToolParameter("units", "string", "Units", required=False),
            ],
        )
        assert len(tool_def.parameters) == 2

    def test_format_for_prompt_no_params(self):
        tool_def = ToolDefinition(
            name="current_time",
            description="Get the current time",
        )
        formatted = tool_def.format_for_prompt()
        assert "**current_time**" in formatted
        assert "Get the current time" in formatted
        assert "Params:" not in formatted  # No params

    def test_format_for_prompt_with_params(self):
        tool_def = ToolDefinition(
            name="weather",
            description="Get weather",
            parameters=[
                ToolParameter("location", "string", "City name"),
                ToolParameter("units", "string", "Units", required=False, default="celsius"),
            ],
        )
        formatted = tool_def.format_for_prompt()
        assert "**weather**" in formatted
        assert "Get weather" in formatted
        assert "Params:" in formatted
        assert "location: string" in formatted
        assert "units: string?" in formatted  # ? indicates optional

    def test_format_detailed_help(self):
        """Test detailed help format."""
        tool_def = ToolDefinition(
            name="weather",
            description="Get weather info",
            parameters=[
                ToolParameter("location", "string", "City name"),
                ToolParameter("units", "string", "Units", required=False, default="celsius"),
            ],
        )
        help_text = tool_def.format_detailed_help()
        assert "## weather" in help_text
        assert "Get weather info" in help_text
        assert "### Parameters" in help_text
        assert "**location**" in help_text
        assert "(required)" in help_text
        assert "(optional)" in help_text
        assert 'default="celsius"' in help_text
        assert "### Usage" in help_text
        assert "mesh-tool weather" in help_text

    def test_to_openai_function_no_params(self):
        """Test conversion to OpenAI function format without parameters."""
        tool_def = ToolDefinition(
            name="current_time",
            description="Get the current time",
        )
        result = tool_def.to_openai_function()
        assert result["type"] == "function"
        assert result["function"]["name"] == "current_time"
        assert result["function"]["description"] == "Get the current time"
        assert result["function"]["parameters"]["type"] == "object"
        assert result["function"]["parameters"]["properties"] == {}
        assert result["function"]["parameters"]["required"] == []
        # strict mode was removed to avoid constrained decoding slowdowns
        assert "strict" not in result["function"]

    def test_to_openai_function_with_params(self):
        """Test conversion to OpenAI function format with parameters."""
        tool_def = ToolDefinition(
            name="weather",
            description="Get weather info",
            parameters=[
                ToolParameter("location", "string", "City name"),
                ToolParameter("units", "string", "Units", required=False),
            ],
        )
        result = tool_def.to_openai_function()
        assert result["type"] == "function"
        assert result["function"]["name"] == "weather"

        props = result["function"]["parameters"]["properties"]
        assert "location" in props
        assert props["location"]["type"] == "string"
        assert props["location"]["description"] == "City name"
        assert "units" in props
        # Without strict mode, optional params keep their plain type
        assert props["units"]["type"] == "string"

        # Without strict mode, only required params are in required list
        assert result["function"]["parameters"]["required"] == ["location"]

    def test_to_openai_function_all_types(self):
        """Test conversion with different parameter types."""
        tool_def = ToolDefinition(
            name="test",
            description="Test all types",
            parameters=[
                ToolParameter("s", "string", "A string"),
                ToolParameter("i", "integer", "An integer"),
                ToolParameter("n", "number", "A number"),
                ToolParameter("b", "boolean", "A boolean"),
                ToolParameter("a", "array", "An array"),
                ToolParameter("o", "object", "An object"),
            ],
        )
        result = tool_def.to_openai_function()
        props = result["function"]["parameters"]["properties"]

        assert props["s"]["type"] == "string"
        assert props["i"]["type"] == "integer"
        assert props["n"]["type"] == "number"
        assert props["b"]["type"] == "boolean"
        assert props["a"]["type"] == "array"
        assert "items" in props["a"]  # Arrays have items
        assert props["o"]["type"] == "object"

    def test_to_openai_responses_function_no_params(self):
        """Test conversion to OpenAI Responses API format (no params)."""
        tool_def = ToolDefinition(
            name="current_time",
            description="Get the current time",
            parameters=[],
        )
        result = tool_def.to_openai_responses_function()
        # Responses API has flattened structure (no nested "function")
        assert result["type"] == "function"
        assert result["name"] == "current_time"
        assert result["description"] == "Get the current time"
        assert result["parameters"]["type"] == "object"
        assert result["parameters"]["properties"] == {}
        assert result["parameters"]["required"] == []
        # strict mode was removed
        assert "strict" not in result

    def test_to_openai_responses_function_with_params(self):
        """Test conversion to OpenAI Responses API format with parameters."""
        tool_def = ToolDefinition(
            name="weather",
            description="Get weather info",
            parameters=[
                ToolParameter("location", "string", "City name"),
                ToolParameter("units", "string", "Units", required=False),
            ],
        )
        result = tool_def.to_openai_responses_function()
        # Responses API has flattened structure
        assert result["type"] == "function"
        assert result["name"] == "weather"

        props = result["parameters"]["properties"]
        assert "location" in props
        assert props["location"]["type"] == "string"
        assert props["location"]["description"] == "City name"
        assert "units" in props
        # Without strict mode, optional params keep plain type
        assert props["units"]["type"] == "string"

        # Without strict mode, only required params are in required list
        assert result["parameters"]["required"] == ["location"]


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        registry.register("test_tool", "A test tool")
        tool_def = registry.get("test_tool")
        assert tool_def is not None
        assert tool_def.name == "test_tool"

    def test_get_nonexistent(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_list_names(self):
        registry = ToolRegistry()
        registry.register("tool1", "First tool")
        registry.register("tool2", "Second tool")
        names = registry.list_names()
        assert "tool1" in names
        assert "tool2" in names

    def test_get_subset(self):
        registry = ToolRegistry()
        registry.register("tool1", "First")
        registry.register("tool2", "Second")
        registry.register("tool3", "Third")

        subset = registry.get_subset(["tool1", "tool3"])
        assert len(subset) == 2
        assert "tool1" in subset
        assert "tool3" in subset
        assert "tool2" not in subset

    def test_get_subset_ignores_missing(self):
        registry = ToolRegistry()
        registry.register("tool1", "First")

        subset = registry.get_subset(["tool1", "nonexistent"])
        assert len(subset) == 1
        assert "tool1" in subset

    def test_format_tools_prompt_all(self):
        registry = ToolRegistry()
        registry.register("tool1", "First tool")
        registry.register("tool2", "Second tool")

        prompt = registry.format_tools_prompt()
        assert "<tools>" in prompt
        assert "tool1" in prompt
        assert "tool2" in prompt
        assert "</tools>" in prompt

    def test_format_tools_prompt_subset(self):
        registry = ToolRegistry()
        registry.register("tool1", "First tool")
        registry.register("tool2", "Second tool")
        registry.register("tool3", "Third tool")

        prompt = registry.format_tools_prompt(["tool1", "tool3"])
        assert "tool1" in prompt
        assert "tool3" in prompt
        assert "tool2" not in prompt

    def test_format_tools_prompt_empty(self):
        registry = ToolRegistry()
        prompt = registry.format_tools_prompt()
        assert prompt == ""

    def test_get_openai_tools_all(self):
        """Test getting all tools in OpenAI format."""
        registry = ToolRegistry()
        registry.register(
            "tool1", "First tool",
            parameters=[ToolParameter("x", "string", "A param")],
        )
        registry.register("tool2", "Second tool")

        tools = registry.get_openai_tools()
        assert len(tools) == 2
        assert all(t["type"] == "function" for t in tools)

        names = {t["function"]["name"] for t in tools}
        assert names == {"tool1", "tool2"}

    def test_get_openai_tools_subset(self):
        """Test getting a subset of tools in OpenAI format."""
        registry = ToolRegistry()
        registry.register("tool1", "First tool")
        registry.register("tool2", "Second tool")
        registry.register("tool3", "Third tool")

        tools = registry.get_openai_tools(["tool1", "tool3"])
        assert len(tools) == 2

        names = {t["function"]["name"] for t in tools}
        assert names == {"tool1", "tool3"}

    def test_get_openai_tools_ignores_missing(self):
        """Test that missing tools are ignored."""
        registry = ToolRegistry()
        registry.register("tool1", "First tool")

        tools = registry.get_openai_tools(["tool1", "nonexistent"])
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "tool1"

    def test_get_openai_responses_tools_all(self):
        """Test getting all tools in OpenAI Responses API format."""
        registry = ToolRegistry()
        registry.register(
            "tool1", "First tool",
            parameters=[ToolParameter("x", "string", "A param")],
        )
        registry.register("tool2", "Second tool")

        tools = registry.get_openai_responses_tools()
        assert len(tools) == 2
        assert all(t["type"] == "function" for t in tools)

        # Responses API has flattened format - name at top level
        names = {t["name"] for t in tools}
        assert names == {"tool1", "tool2"}

    def test_get_openai_responses_tools_subset(self):
        """Test getting a subset of tools in OpenAI Responses API format."""
        registry = ToolRegistry()
        registry.register("tool1", "First tool")
        registry.register("tool2", "Second tool")
        registry.register("tool3", "Third tool")

        tools = registry.get_openai_responses_tools(["tool1", "tool3"])
        assert len(tools) == 2

        names = {t["name"] for t in tools}
        assert names == {"tool1", "tool3"}

    def test_register_with_handler(self):
        registry = ToolRegistry()

        def my_handler(x: int) -> int:
            return x * 2

        registry.register("double", "Double a number", handler=my_handler)
        tool_def = registry.get("double")
        assert tool_def.handler is my_handler

    def test_decorator_registration(self):
        registry = ToolRegistry()

        @registry.register_decorator("decorated", "A decorated tool")
        def my_tool():
            return "hello"

        tool_def = registry.get("decorated")
        assert tool_def is not None
        assert tool_def.handler is my_tool


class TestGlobalRegistry:
    def test_get_registry_returns_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_builtin_tools_registered(self):
        registry = get_registry()
        # These are registered in tools.py
        assert registry.get("echo") is not None
        assert registry.get("current_time") is not None


class TestParseToolCalls:
    def test_single_tool_call(self):
        text = """Let me check that for you.

<mesh_call name="weather">
<location>Austin, TX</location>
</mesh_call>

I'll have the result shortly."""

        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "weather"
        assert calls[0].arguments == {"location": "Austin, TX"}

    def test_multiple_params(self):
        text = """<mesh_call name="search">
<query>python asyncio</query>
<limit>10</limit>
</mesh_call>"""

        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].arguments["query"] == "python asyncio"
        assert calls[0].arguments["limit"] == 10  # Should be parsed as int

    def test_multiple_tool_calls(self):
        text = """<mesh_call name="tool1">
<arg>value1</arg>
</mesh_call>

Some text between.

<mesh_call name="tool2">
<arg>value2</arg>
</mesh_call>"""

        calls = parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0].name == "tool1"
        assert calls[1].name == "tool2"

    def test_no_tool_calls(self):
        text = "Just a regular message without any tool calls."
        calls = parse_tool_calls(text)
        assert len(calls) == 0

    def test_empty_tool(self):
        text = """<mesh_call name="current_time">
</mesh_call>"""

        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "current_time"
        assert calls[0].arguments == {}

    def test_json_value_in_param(self):
        text = """<mesh_call name="config">
<settings>{"debug": true, "level": 5}</settings>
</mesh_call>"""

        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].arguments["settings"] == {"debug": True, "level": 5}

    def test_raw_xml_preserved(self):
        text = """<mesh_call name="test">
<arg>value</arg>
</mesh_call>"""

        calls = parse_tool_calls(text)
        assert calls[0].raw_xml == text.strip()

    def test_ignores_tool_calls_in_code_blocks(self):
        """Tool calls inside markdown code blocks should not be parsed."""
        text = """Here's an example of the syntax:

```xml
<mesh_call name="example">
<param>value</param>
</mesh_call>
```

And here's the actual call:

<mesh_call name="real_tool">
<arg>real_value</arg>
</mesh_call>
"""
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "real_tool"
        assert calls[0].arguments == {"arg": "real_value"}

    def test_all_code_block_tool_calls_ignored(self):
        """When all tool calls are in code blocks, result should be empty."""
        text = """Here's how to use it:

```
<mesh_call name="example">
<param>value</param>
</mesh_call>
```
"""
        calls = parse_tool_calls(text)
        assert len(calls) == 0


class TestHasToolCall:
    def test_has_tool_call_true(self):
        text = '<mesh_call name="test"><arg>val</arg></mesh_call>'
        assert has_tool_call(text) is True

    def test_has_tool_call_false(self):
        text = "No tools here"
        assert has_tool_call(text) is False

    def test_has_tool_call_partial(self):
        # Incomplete/malformed shouldn't match
        text = '<mesh_call name="test"'  # Missing closing >
        assert has_tool_call(text) is False

    def test_has_tool_call_ignores_code_blocks(self):
        """has_tool_call should ignore tool calls in code blocks."""
        text = """Here's an example:

```xml
<mesh_call name="example">
<param>value</param>
</mesh_call>
```
"""
        assert has_tool_call(text) is False


class TestExecuteTool:
    @pytest.fixture
    def registry_with_tools(self):
        registry = ToolRegistry()

        def sync_tool(x: int, y: int = 0) -> int:
            return x + y

        async def async_tool(msg: str) -> str:
            return f"async: {msg}"

        def error_tool():
            raise ValueError("Tool error!")

        registry.register("add", "Add numbers", handler=sync_tool)
        registry.register("async_echo", "Async echo", handler=async_tool)
        registry.register("error", "Always errors", handler=error_tool)
        registry.register("no_handler", "No handler")

        return registry

    @pytest.mark.asyncio
    async def test_execute_sync_tool(self, registry_with_tools):
        call = ToolCall(name="add", arguments={"x": 5, "y": 3}, raw_xml="")
        result = await execute_tool(registry_with_tools, call)
        assert result == "8"

    @pytest.mark.asyncio
    async def test_execute_async_tool(self, registry_with_tools):
        call = ToolCall(name="async_echo", arguments={"msg": "hello"}, raw_xml="")
        result = await execute_tool(registry_with_tools, call)
        assert result == "async: hello"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, registry_with_tools):
        call = ToolCall(name="nonexistent", arguments={}, raw_xml="")
        result = await execute_tool(registry_with_tools, call)
        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_execute_tool_no_handler(self, registry_with_tools):
        call = ToolCall(name="no_handler", arguments={}, raw_xml="")
        result = await execute_tool(registry_with_tools, call)
        assert "no handler" in result

    @pytest.mark.asyncio
    async def test_execute_tool_error(self, registry_with_tools):
        call = ToolCall(name="error", arguments={}, raw_xml="")
        result = await execute_tool(registry_with_tools, call)
        assert "Error" in result
        assert "Tool error!" in result

    @pytest.mark.asyncio
    async def test_execute_tool_wrong_args(self, registry_with_tools):
        call = ToolCall(name="add", arguments={"wrong": 5}, raw_xml="")
        result = await execute_tool(registry_with_tools, call)
        assert "Invalid arguments" in result or "Error" in result


class TestExecuteToolCalls:
    @pytest.mark.asyncio
    async def test_execute_multiple(self):
        registry = ToolRegistry()

        def tool1() -> str:
            return "result1"

        def tool2() -> str:
            return "result2"

        registry.register("t1", "Tool 1", handler=tool1)
        registry.register("t2", "Tool 2", handler=tool2)

        calls = [
            ToolCall(name="t1", arguments={}, raw_xml=""),
            ToolCall(name="t2", arguments={}, raw_xml=""),
        ]

        result = await execute_tool_calls(registry, calls)
        assert '<mesh_result name="t1">' in result
        assert "result1" in result
        assert '<mesh_result name="t2">' in result
        assert "result2" in result


class TestBuiltinTools:
    @pytest.mark.asyncio
    async def test_echo_tool(self):
        registry = get_registry()
        call = ToolCall(name="echo", arguments={"text": "hello world"}, raw_xml="")
        result = await execute_tool(registry, call)
        assert "hello world" in result

    @pytest.mark.asyncio
    async def test_current_time_tool(self):
        registry = get_registry()
        call = ToolCall(name="current_time", arguments={}, raw_xml="")
        result = await execute_tool(registry, call)
        # Should be an ISO timestamp
        assert "202" in result or "T" in result  # Basic check for timestamp format
