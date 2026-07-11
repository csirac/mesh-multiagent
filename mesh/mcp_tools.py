"""
MCP tool schema definitions for the CC-session architecture.

Converts mesh ToolDefinition objects to MCP-compatible JSON Schema format
for use with Claude Code's native tool calling via --mcp-config.
"""
from __future__ import annotations

from typing import Any

from .tools import ToolDefinition, ToolRegistry, get_registry


def tool_to_mcp_schema(tool: ToolDefinition) -> dict[str, Any]:
    """
    Convert a mesh ToolDefinition to MCP tool schema format.

    MCP format:
    {
        "name": "tool_name",
        "description": "What it does",
        "inputSchema": {
            "type": "object",
            "properties": { ... },
            "required": [ ... ]
        }
    }
    """
    properties, required = tool._build_parameters_schema(strict=False)
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def get_mcp_tools(
    tool_names: list[str] | None = None,
    registry: ToolRegistry | None = None,
) -> list[dict[str, Any]]:
    """
    Get MCP tool schemas for the specified tools.

    Args:
        tool_names: List of tool names to include. None = all registered tools.
        registry: Tool registry to use. None = global registry.

    Returns:
        List of MCP tool schema dicts.
    """
    reg = registry or get_registry()
    if tool_names is not None:
        tools = reg.get_subset(tool_names)
    else:
        tools = reg.get_subset(reg.list_names())
    return [tool_to_mcp_schema(tool) for tool in tools.values()]


def get_mcp_tool_map(
    tool_names: list[str] | None = None,
    registry: ToolRegistry | None = None,
) -> dict[str, ToolDefinition]:
    """
    Get a name→ToolDefinition map for fast lookup during tool dispatch.

    Args:
        tool_names: List of tool names to include. None = all registered tools.
        registry: Tool registry to use. None = global registry.

    Returns:
        Dict mapping tool name to ToolDefinition.
    """
    reg = registry or get_registry()
    if tool_names is not None:
        return reg.get_subset(tool_names)
    else:
        return reg.get_subset(reg.list_names())
