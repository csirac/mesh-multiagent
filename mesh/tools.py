"""
Tool registry for mesh agents.

Defines tools that agents can use, with XML-based invocation format.
Each tool has a name, description, parameters schema, and implementation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable, TYPE_CHECKING

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "integer", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolDefinition:
    """Definition of a tool."""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable[..., Awaitable[Any]] | Callable[..., Any] | None = None
    requires_confirmation: bool = False  # If True, requires user confirmation before execution

    def format_for_prompt(self) -> str:
        """Format this tool for inclusion in the LLM prompt."""
        lines = [f"- **{self.name}**"]
        lines.append(f"  {self.description}")

        if self.parameters:
            params_str = ", ".join(
                f"{p.name}: {p.type}" + ("?" if not p.required else "")
                for p in self.parameters
            )
            lines.append(f"  Params: {params_str}")

        return "\n".join(lines)

    def format_detailed_help(self) -> str:
        """Format detailed help for this tool."""
        lines = [f"## {self.name}"]
        lines.append("")
        lines.append(self.description)
        lines.append("")

        if self.parameters:
            lines.append("### Parameters")
            for param in self.parameters:
                required_str = "(required)" if param.required else "(optional)"
                default_str = f", default={json.dumps(param.default)}" if param.default is not None else ""
                lines.append(f"- **{param.name}** ({param.type}, {required_str}{default_str})")
                lines.append(f"  {param.description}")
        else:
            lines.append("No parameters.")

        lines.append("")
        lines.append("### Usage")
        if self.parameters:
            lines.append("```bash")
            parts = [f"mesh-tool {self.name}"]
            for param in self.parameters:
                example = _get_example_value(param.type)
                parts.append(f"--{param.name} {example}")
            lines.append(" ".join(parts))
            lines.append("```")
        else:
            lines.append("```bash")
            lines.append(f"mesh-tool {self.name}")
            lines.append("```")

        return "\n".join(lines)

    def _build_parameters_schema(self, strict: bool = False) -> tuple[dict, list]:
        """Build parameter properties and required list for JSON schema.

        Args:
            strict: If True, all parameters are listed as required (for OpenAI strict mode).
                    Optional parameters become nullable to allow omission.
        """
        properties = {}
        required = []

        for param in self.parameters:
            # Map our type names to JSON schema types
            json_type = _map_to_json_schema_type(param.type)

            if strict and not param.required:
                # In strict mode, optional params use nullable type
                # ["type", "null"] allows the model to pass null for optional params
                prop: dict[str, Any] = {
                    "type": [json_type, "null"],
                    "description": param.description + " (optional, can be null)",
                }
            else:
                prop = {
                    "type": json_type,
                    "description": param.description,
                }

            # Handle array and object types
            if param.type == "array":
                prop["items"] = {"type": "string"}  # Default to string items
            elif param.type == "object":
                prop["additionalProperties"] = True

            properties[param.name] = prop

            # In strict mode, all params go in required; otherwise only required params
            if strict or param.required:
                required.append(param.name)

        return properties, required

    def to_openai_function(self) -> dict:
        """
        Convert this tool to OpenAI Chat Completions API function calling format.

        Returns a dict suitable for the `tools` parameter in OpenAI Chat Completions API.

        Format:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {...}
            }
        }
        """
        properties, required = self._build_parameters_schema(strict=False)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
        }

    def to_openai_responses_function(self) -> dict:
        """
        Convert this tool to OpenAI Responses API function calling format.

        Returns a dict suitable for the `tools` parameter in OpenAI Responses API.

        Format (flattened compared to Chat Completions):
        {
            "type": "function",
            "name": "...",
            "description": "...",
            "parameters": {...}
        }
        """
        properties, required = self._build_parameters_schema(strict=False)

        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def to_anthropic_tool(self) -> dict:
        """
        Convert this tool to Anthropic Messages API tool format.

        Returns a dict suitable for the `tools` parameter in Anthropic Messages API.

        Format:
        {
            "name": "...",
            "description": "...",
            "input_schema": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }
        """
        # Anthropic doesn't use strict mode, but we keep the same schema for consistency
        properties, required = self._build_parameters_schema(strict=False)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }


def _get_example_value(param_type: str) -> str:
    """Get an example value for a parameter type."""
    examples = {
        "string": "example text",
        "integer": "42",
        "number": "3.14",
        "boolean": "true",
        "object": '{"key": "value"}',
        "array": '["item1", "item2"]',
    }
    return examples.get(param_type, "value")


def _map_to_json_schema_type(param_type: str) -> str:
    """Map our parameter types to JSON schema types."""
    type_map = {
        "string": "string",
        "integer": "integer",
        "number": "number",
        "boolean": "boolean",
        "object": "object",
        "array": "array",
    }
    return type_map.get(param_type, "string")


class ToolRegistry:
    """
    Registry of available tools.

    Tools are registered globally but agents only get access to the ones
    configured in their node config.
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: list[ToolParameter] | None = None,
        handler: Callable[..., Awaitable[Any]] | Callable[..., Any] | None = None,
        requires_confirmation: bool = False,
    ) -> ToolDefinition:
        """Register a tool."""
        tool = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or [],
            handler=handler,
            requires_confirmation=requires_confirmation,
        )
        self._tools[name] = tool
        logger.debug(f"Registered tool: {name} (requires_confirmation={requires_confirmation})")
        return tool

    def register_decorator(
        self,
        name: str,
        description: str,
        parameters: list[ToolParameter] | None = None,
        requires_confirmation: bool = False,
    ):
        """Decorator to register a function as a tool handler."""
        def decorator(fn: Callable) -> Callable:
            self.register(name, description, parameters, fn, requires_confirmation)
            return fn
        return decorator

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_subset(self, names: list[str]) -> dict[str, ToolDefinition]:
        """Get a subset of tools by name."""
        return {name: self._tools[name] for name in names if name in self._tools}

    def list_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_tool_help(self, tool_name: str) -> str:
        """Get detailed help for a specific tool."""
        tool_def = self._tools.get(tool_name)
        if tool_def is None:
            available = ", ".join(sorted(self._tools.keys()))
            return f"Unknown tool '{tool_name}'. Available tools: {available}"
        return tool_def.format_detailed_help()

    def get_all_tools_help(self) -> str:
        """Get detailed help for all tools."""
        lines = ["# Tool Reference"]
        lines.append("")
        for name in sorted(self._tools.keys()):
            lines.append(self._tools[name].format_detailed_help())
            lines.append("")
            lines.append("---")
            lines.append("")
        return "\n".join(lines)

    def get_openai_tools(self, tool_names: list[str] | None = None) -> list[dict]:
        """
        Get tools formatted for OpenAI Chat Completions API function calling.

        Args:
            tool_names: List of tool names to include. If None, includes all tools.

        Returns:
            List of tool definitions in OpenAI Chat Completions function calling format.
        """
        if tool_names is None:
            tools = list(self._tools.values())
        else:
            tools = [self._tools[n] for n in tool_names if n in self._tools]

        return [tool.to_openai_function() for tool in tools]

    def get_openai_responses_tools(self, tool_names: list[str] | None = None) -> list[dict]:
        """
        Get tools formatted for OpenAI Responses API function calling.

        The Responses API uses a flattened format compared to Chat Completions:
        - name, description, parameters are at top level (not nested under "function")

        Args:
            tool_names: List of tool names to include. If None, includes all tools.

        Returns:
            List of tool definitions in OpenAI Responses API function calling format.
        """
        if tool_names is None:
            tools = list(self._tools.values())
        else:
            tools = [self._tools[n] for n in tool_names if n in self._tools]

        return [tool.to_openai_responses_function() for tool in tools]

    def get_anthropic_tools(self, tool_names: list[str] | None = None) -> list[dict]:
        """
        Get tools formatted for Anthropic Messages API tool use.

        Args:
            tool_names: List of tool names to include. If None, includes all tools.

        Returns:
            List of tool definitions in Anthropic Messages API tool format.
        """
        if tool_names is None:
            tools = list(self._tools.values())
        else:
            tools = [self._tools[n] for n in tool_names if n in self._tools]

        return [tool.to_anthropic_tool() for tool in tools]

    def format_tools_prompt(
        self,
        tool_names: list[str] | None = None,
        backend: str = "xml",
    ) -> str:
        """
        Generate the tool prompt section for the given tools.

        Args:
            tool_names: List of tool names to include. If None, includes all tools.
            backend: "xml" for XML-based backends (Claude Code, Z.AI, Google),
                     "openai" for OpenAI native function calling.

        Returns:
            Empty string if no tools are available.

        Tool instructions are loaded from mesh/prompts/ directory:
        - tool_guidance_common.md: Shared guidance for all backends
        - tool_syntax_xml.md: XML syntax for non-OpenAI backends
        - tool_syntax_openai.md: Brief guidance for OpenAI function calling

        For OpenAI backend, only guidance is included (no tool list), since
        tools are passed via the native function calling API.
        """
        if tool_names is None:
            tools = list(self._tools.values())
        else:
            tools = [self._tools[n] for n in tool_names if n in self._tools]

        if not tools:
            return ""

        # Load instructions based on backend
        instructions = _load_tool_instructions(backend)

        lines = [
            "<tools>",
            "",
            instructions,
        ]

        # For OpenAI, skip the tool list (tools are passed via API)
        # For other backends, include the full tool list
        if backend != "openai":
            lines.extend([
                "",
                "---",
                "",
                "## Available Tools",
                "",
            ])

            for tool_def in tools:
                lines.append(tool_def.format_for_prompt())

        lines.append("")
        lines.append("</tools>")
        return "\n".join(lines)


def _load_tool_instructions(backend: str = "xml") -> str:
    """
    Load tool instructions from the prompt files.

    Args:
        backend: "xml" for XML-based backends, "openai" for native function calling.

    Returns:
        Combined instructions string (common guidance + syntax instructions).
    """
    prompts_dir = Path(__file__).parent / "prompts"

    # Load common guidance (shared by all backends)
    common_file = prompts_dir / "tool_guidance_common.md"
    if common_file.exists():
        common_text = common_file.read_text().strip()
    else:
        logger.warning(f"Common tool guidance file not found: {common_file}")
        common_text = ""

    # Load syntax instructions based on backend
    if backend == "openai":
        syntax_file = prompts_dir / "tool_syntax_openai.md"
    else:
        syntax_file = prompts_dir / "tool_syntax_xml.md"

    if syntax_file.exists():
        syntax_text = syntax_file.read_text().strip()
    else:
        logger.warning(f"Tool syntax file not found: {syntax_file}")
        if backend == "openai":
            syntax_text = "Call tools using native function calling."
        else:
            syntax_text = "Use tools by calling them with XML format: <mesh_call name=\"tool_name\"><param>value</param></mesh_call>"

    # Combine common guidance and syntax instructions
    if common_text and syntax_text:
        return f"{common_text}\n\n---\n\n{syntax_text}"
    elif common_text:
        return common_text
    else:
        return syntax_text


# Global registry instance
_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _registry


def register_tool(
    name: str,
    description: str,
    parameters: list[ToolParameter] | None = None,
    handler: Callable[..., Awaitable[Any]] | Callable[..., Any] | None = None,
    requires_confirmation: bool = False,
) -> ToolDefinition:
    """Convenience function to register a tool in the global registry."""
    return _registry.register(name, description, parameters, handler, requires_confirmation)


def tool(
    name: str,
    description: str,
    parameters: list[ToolParameter] | None = None,
    requires_confirmation: bool = False,
):
    """Decorator to register a function as a tool."""
    return _registry.register_decorator(name, description, parameters, requires_confirmation)


# ============================================================================
# Tool call parsing
# ============================================================================

@dataclass
class ToolCall:
    """A parsed tool call from LLM output."""
    name: str
    arguments: dict[str, Any]
    raw_xml: str
    call_id: str = ""


def parse_tool_calls(text: str) -> list[ToolCall]:
    """
    Parse tool calls from LLM output.

    Supports two formats:
    1. Simple: <mesh_call name="weather_get"><location>Austin, TX</location></mesh_call>
    2. Named param: <mesh_call name="weather_get"><param name="location">Austin, TX</param></mesh_call>

    Tool calls inside markdown code blocks (```...```) are ignored to prevent
    accidental execution of examples in documentation/help output.

    Returns list of parsed tool calls.
    """
    calls = []

    # First, mask out code blocks so we don't parse tool calls inside them
    # This prevents examples in help text from being executed
    code_block_pattern = r'```[^`]*```'
    masked_text = re.sub(code_block_pattern, lambda m: ' ' * len(m.group(0)), text, flags=re.DOTALL)

    # Find all <mesh_call name="...">...</mesh_call> blocks in the masked text
    pattern = r'<mesh_call\s+name="([^"]+)">(.*?)</mesh_call>'
    matches = list(re.finditer(pattern, masked_text, re.DOTALL))

    # Extract tool calls from original text using positions from masked matches
    for match in matches:
        start = match.start()
        # Re-match at the same position in original text to get actual content
        original_match = re.match(pattern, text[start:], re.DOTALL)
        if not original_match:
            continue

        tool_name = original_match.group(1)
        tool_body = original_match.group(2).strip()
        raw_xml = original_match.group(0)

        # Parse parameters from the body
        args = {}

        # Format 1: <param name="...">value</param>
        named_param_pattern = r'<param\s+name="([^"]+)">(.*?)</param>'
        named_matches = list(re.finditer(named_param_pattern, tool_body, re.DOTALL))

        if named_matches:
            # Use named param format
            for pm in named_matches:
                param_name = pm.group(1).strip()
                param_value = pm.group(2).strip()
                # Try to parse as JSON for complex types
                try:
                    args[param_name] = json.loads(param_value)
                except (json.JSONDecodeError, ValueError):
                    args[param_name] = param_value
        else:
            # Format 2: <param_name>value</param_name>
            simple_param_pattern = r'<([^/>\s]+)>(.*?)</\1>'
            for pm in re.finditer(simple_param_pattern, tool_body, re.DOTALL):
                param_name = pm.group(1).strip()
                param_value = pm.group(2).strip()
                try:
                    args[param_name] = json.loads(param_value)
                except (json.JSONDecodeError, ValueError):
                    args[param_name] = param_value

        calls.append(ToolCall(name=tool_name, arguments=args, raw_xml=raw_xml))

    return calls


def has_tool_call(text: str) -> bool:
    """Check if text contains a tool call (outside of code blocks)."""
    # Mask out code blocks first
    code_block_pattern = r'```[^`]*```'
    masked_text = re.sub(code_block_pattern, '', text, flags=re.DOTALL)
    return bool(re.search(r'<mesh_call\s+name="[^"]+">', masked_text))


# ============================================================================
# Tool execution
# ============================================================================

async def execute_tool(
    registry: ToolRegistry,
    call: ToolCall,
) -> str:
    """
    Execute a tool call and return the result as a string.

    Returns error message if tool not found or execution fails.
    """
    tool_def = registry.get(call.name)

    if tool_def is None:
        return f"Error: Unknown tool '{call.name}'"

    if tool_def.handler is None:
        return f"Error: Tool '{call.name}' has no handler"

    try:
        logger.debug(f"Executing tool {call.name} with args: {call.arguments}")

        if asyncio.iscoroutinefunction(tool_def.handler):
            result = await tool_def.handler(**call.arguments)
        else:
            result = tool_def.handler(**call.arguments)

        logger.debug(f"Tool {call.name} result: {str(result)[:200]}...")
        return str(result)

    except TypeError as e:
        logger.error(f"Tool {call.name} argument error: {e}")
        return f"Error: Invalid arguments for '{call.name}': {e}"
    except Exception as e:
        logger.exception(f"Tool {call.name} execution failed: {e}")
        return f"Error executing '{call.name}': {e}"


async def execute_tool_calls(
    registry: ToolRegistry,
    calls: list[ToolCall],
) -> str:
    """
    Execute multiple tool calls and format results as XML.

    Returns formatted XML with all results.
    """
    results = []

    for call in calls:
        result = await execute_tool(registry, call)
        results.append(f'<mesh_result name="{call.name}">\n{result}\n</mesh_result>')

    return "\n\n".join(results)


# ============================================================================
# Built-in tools (examples)
# ============================================================================

# Example: Simple echo tool for testing
@tool(
    name="echo",
    description="Echoes back the input text. Useful for testing.",
    parameters=[
        ToolParameter(
            name="text",
            type="string",
            description="The text to echo back",
        ),
    ],
)
def echo_tool(text: str) -> str:
    return f"Echo: {text}"


# Example: Current time tool
@tool(
    name="current_time",
    description="Returns the current date and time.",
    parameters=[],
)
def current_time_tool() -> str:
    from datetime import datetime
    return datetime.now().astimezone().isoformat()
