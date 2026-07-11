"""
Configuration for model evaluation framework.

Defines:
- Model configurations (backend, model, settings)
- Eval-specific router config (isolated from production)
- Agent type definitions for evaluators
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EvalModelConfig:
    """Configuration for a model under evaluation."""
    name: str                     # Display name (e.g., "gpt51", "deepseek")
    backend_type: str             # "openai", "anthropic", "claude-code", "zai"
    model: str                    # Model identifier
    api_key_env: str              # Environment variable for API key
    base_url: Optional[str] = None
    max_tokens: int = 8192
    temperature: float = 0.7
    # OpenAI-specific
    reasoning_effort: Optional[str] = None  # "low", "medium", "high"
    # Anthropic-specific
    thinking_budget: Optional[int] = None
    # Claude Code / ZAI specific
    cc_allowed_tools: list[str] = field(default_factory=list)
    cc_model: str = "sonnet"     # For claude-code backend


# Model configurations for evaluation
MODEL_CONFIGS: dict[str, EvalModelConfig] = {
    # OpenAI GPT-5.1 with reasoning
    "gpt51": EvalModelConfig(
        name="gpt51",
        backend_type="openai",
        model="gpt-5.1",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        max_tokens=16384,
        reasoning_effort="medium",
    ),

    # DeepSeek V3 via Synthetic Anthropic API
    "deepseek": EvalModelConfig(
        name="deepseek",
        backend_type="anthropic",
        model="hf:deepseek-ai/DeepSeek-V3",
        api_key_env="SYNTHETIC_API_KEY",
        base_url="https://api.synthetic.new/anthropic/v1",
        max_tokens=8192,
    ),

    # GLM 4.7 via Synthetic Anthropic API
    "glm": EvalModelConfig(
        name="glm",
        backend_type="anthropic",
        model="hf:zai-org/GLM-4.7",
        api_key_env="SYNTHETIC_API_KEY",
        base_url="https://api.synthetic.new/anthropic/v1",
        max_tokens=8192,
    ),

    # ZAI backend (Claude Code subprocess with ZAI proxy)
    "zai": EvalModelConfig(
        name="zai",
        backend_type="zai",
        model="glm-4.7",
        api_key_env="ZAI_API_KEY",
        max_tokens=4096,
        cc_allowed_tools=[
            "Read", "Edit", "Write", "Bash", "Glob", "Grep", "Task",
        ],
    ),

    # Claude Code with Sonnet
    "cc-sonnet": EvalModelConfig(
        name="cc-sonnet",
        backend_type="claude-code",
        model="sonnet",
        api_key_env="",  # Uses Claude CLI auth
        max_tokens=16000,
        cc_model="sonnet",
        cc_allowed_tools=[
            "Read", "Edit", "Write", "Bash", "Glob", "Grep", "Task",
        ],
    ),

    # Claude Code with Opus
    "cc-opus": EvalModelConfig(
        name="cc-opus",
        backend_type="claude-code",
        model="opus",
        api_key_env="",  # Uses Claude CLI auth
        max_tokens=16000,
        cc_model="opus",
        cc_allowed_tools=[
            "Read", "Edit", "Write", "Bash", "Glob", "Grep", "Task",
        ],
    ),
}


def get_model_config(model_name: str) -> EvalModelConfig:
    """Get configuration for a model by name."""
    if model_name not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    return MODEL_CONFIGS[model_name]


def list_models() -> list[str]:
    """List all available model names."""
    return list(MODEL_CONFIGS.keys())


# =============================================================================
# Eval Router Configuration
# =============================================================================

EVAL_ROUTER_HOST = "127.0.0.1"
EVAL_ROUTER_PORT = 9999
EVAL_ROUTER_WS_PORT = 9998  # Not used but defined for completeness


@dataclass
class EvalRouterConfig:
    """Configuration for the isolated eval router."""
    host: str = EVAL_ROUTER_HOST
    port: int = EVAL_ROUTER_PORT
    ws_port: int = EVAL_ROUTER_WS_PORT
    storage_path: str = "/tmp/mesh_eval/router_messages.db"
    fcm_enabled: bool = False
    auth_enabled: bool = False


# =============================================================================
# Coder Tools for Evaluation
# =============================================================================

# Tools available to evaluator agents (subset focused on coding tasks)
EVAL_AGENT_TOOLS = [
    "send_message",
    "sleep",
    "set_working_directory",
    "get_working_directory",
    "bash_exec",
    "file_read",
    "file_edit",
    "file_create",
    "file_write",
    "file_diff",
]


# =============================================================================
# System Prompt for Evaluator Agents
# =============================================================================

EVAL_CODER_PROMPT = """You are an AI assistant being evaluated on software engineering tasks.

## Your Environment

You have access to a sandbox directory where your task files are located. You can:
- Read files with `file_read`
- Edit files with `file_edit` (exact string replacement)
- Write/overwrite files with `file_write`
- Apply diffs with `file_diff` (unified diff format)
- Run commands with `bash_exec`

## File Editing Guide

1. **Read first**: Always use `file_read` before editing to see current contents
2. **Choose the right tool**:
   - `file_write`: For creating new files or complete rewrites
   - `file_edit`: For targeted changes when you know the exact string
   - `file_diff`: For adding/modifying functions in existing classes
3. **Test your changes**: Run tests with `bash_exec` after making changes

## Task Completion

When you complete the task:
1. Send a message summarizing what you did
2. Include the test results if applicable

Focus on completing the task correctly and efficiently.
"""
