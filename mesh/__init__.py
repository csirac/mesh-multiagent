# SPDX-License-Identifier: Apache-2.0
# hello-world mesh - multi-agent communication framework

from .protocol import Message, MessageType, ControlAction
from .config import MeshConfig, RouterConfig, NodeConfig, LLMBackendConfig, load_config
from .node import Node
from .router import Router

# Lazy imports for optional dependencies (httpx, etc.)
def __getattr__(name):
    if name in ('LLMClient', 'LLMConfig', 'HistoryMessage', 'CCToolEvent', 'LLMStreamCallback'):
        from .llm import LLMClient, LLMConfig, HistoryMessage, CCToolEvent, LLMStreamCallback
        return globals()[name]
    if name in ('AgentNode', 'SimpleAgentNode'):
        from .agent_node import AgentNode, SimpleAgentNode
        return globals()[name]
    if name in ('ToolRegistry', 'ToolDefinition', 'ToolParameter', 'ToolCall',
                'get_registry', 'register_tool', 'tool', 'parse_tool_calls',
                'has_tool_call', 'execute_tool', 'execute_tool_calls'):
        from .tools import (
            ToolRegistry, ToolDefinition, ToolParameter, ToolCall,
            get_registry, register_tool, tool, parse_tool_calls,
            has_tool_call, execute_tool, execute_tool_calls,
        )
        return globals()[name]
    if name in ('MarkdownRenderer',):
        from .markdown_renderer import MarkdownRenderer
        return MarkdownRenderer
    if name in ('WezMathRenderer',):
        from .wez_math_renderer import WezMathRenderer
        return WezMathRenderer
    raise AttributeError(f"module 'mesh' has no attribute {name!r}")

__all__ = [
    "Message",
    "MessageType",
    "ControlAction",
    "MeshConfig",
    "RouterConfig",
    "NodeConfig",
    "LLMBackendConfig",
    "load_config",
    "LLMClient",
    "LLMConfig",
    "HistoryMessage",
    "CCToolEvent",
    "LLMStreamCallback",
    "Node",
    "AgentNode",
    "SimpleAgentNode",
    "Router",
    # Tools
    "ToolRegistry",
    "ToolDefinition",
    "ToolParameter",
    "ToolCall",
    "get_registry",
    "register_tool",
    "tool",
    "parse_tool_calls",
    "has_tool_call",
    "execute_tool",
    "execute_tool_calls",
    # Rendering
    "MarkdownRenderer",
    "WezMathRenderer",
]
