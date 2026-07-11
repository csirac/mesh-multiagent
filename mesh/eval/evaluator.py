"""
Evaluator - runs a single model against tasks.

Each evaluator:
1. Creates a sandbox for the task
2. Runs the model using direct API calls (OpenAI/Anthropic) or CC subprocess
3. Validates the result
4. Reports back via mesh messaging
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

from .tasks import EvalTask, TaskResult
from ..llm import _build_subprocess_env

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for an evaluator."""
    model_name: str
    backend_type: str  # "openai", "anthropic", "claude-code", "zai"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_id: Optional[str] = None
    max_tokens: int = 8192
    temperature: float = 0.7
    reasoning_effort: Optional[str] = None  # For OpenAI
    anthropic_thinking_budget: Optional[int] = None  # For Anthropic
    zai_api_key: Optional[str] = None  # For ZAI backend
    cc_allowed_tools: list[str] = field(default_factory=lambda: [
        "Read", "Edit", "Write", "Bash", "Glob", "Grep"
    ])


# Pre-configured model configs
MODEL_CONFIGS = {
    "gpt51": EvalConfig(
        model_name="gpt51",
        backend_type="openai",
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1",
        model_id="gpt-5.1",
        max_tokens=16384,
        reasoning_effort="medium",
    ),
    "deepseek": EvalConfig(
        model_name="deepseek",
        backend_type="anthropic",
        api_key=os.environ.get("SYNTHETIC_API_KEY"),
        base_url="https://api.synthetic.new/anthropic/v1",
        model_id="hf:deepseek-ai/DeepSeek-V3",
        max_tokens=8192,
        # Note: DeepSeek V3 doesn't support anthropic thinking API
    ),
    "glm": EvalConfig(
        model_name="glm",
        backend_type="anthropic",
        api_key=os.environ.get("SYNTHETIC_API_KEY"),
        base_url="https://api.synthetic.new/anthropic/v1",
        model_id="hf:zai-org/GLM-4.7",
        max_tokens=8192,
        anthropic_thinking_budget=10000,
    ),
    "zai": EvalConfig(
        model_name="zai",
        backend_type="zai",
        zai_api_key=os.environ.get("ZAI_API_KEY"),
        model_id="glm-4.7",
        max_tokens=8192,
    ),
    "cc-sonnet": EvalConfig(
        model_name="cc-sonnet",
        backend_type="claude-code",
        model_id="sonnet",
        max_tokens=16000,
    ),
    "cc-opus": EvalConfig(
        model_name="cc-opus",
        backend_type="claude-code",
        model_id="opus",
        max_tokens=16000,
    ),
}


# Tool definitions for OpenAI format
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read a file with line numbers. Use instead of cat/head/tail.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"},
                    "start_line": {"type": "integer", "description": "Start line (1-indexed, optional)"},
                    "end_line": {"type": "integer", "description": "End line (1-indexed, optional)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "Create or overwrite a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_edit",
            "description": "Perform exact string replacement in a file. Use file_read first to see content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"},
                    "old_string": {"type": "string", "description": "Exact string to replace"},
                    "new_string": {"type": "string", "description": "Replacement string"},
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_diff",
            "description": "Apply a unified diff patch to a file. Best for adding methods or multiple changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"},
                    "diff": {"type": "string", "description": "Unified diff content (like git diff output)"},
                    "fuzz": {"type": "integer", "description": "Fuzz level: 0=exact, 1=trim whitespace, 2=normalize (default 1)"},
                },
                "required": ["path", "diff"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_create",
            "description": "Create a new file (fails if exists). Use file_write to overwrite.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash_exec",
            "description": "Execute a bash command and return stdout/stderr/exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute"},
                    "timeout": {"type": "number", "description": "Timeout in seconds (default 120)"},
                },
                "required": ["command"],
            },
        },
    },
]


# Tool definitions for Anthropic format
ANTHROPIC_TOOLS = [
    {
        "name": "file_read",
        "description": "Read a file with line numbers. Use instead of cat/head/tail.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
                "start_line": {"type": "integer", "description": "Start line (1-indexed, optional)"},
                "end_line": {"type": "integer", "description": "End line (1-indexed, optional)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "file_write",
        "description": "Create or overwrite a file with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "file_edit",
        "description": "Perform exact string replacement in a file. Use file_read first to see content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
                "old_string": {"type": "string", "description": "Exact string to replace"},
                "new_string": {"type": "string", "description": "Replacement string"},
                "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "file_diff",
        "description": "Apply a unified diff patch to a file. Best for adding methods or multiple changes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
                "diff": {"type": "string", "description": "Unified diff content (like git diff output)"},
                "fuzz": {"type": "integer", "description": "Fuzz level: 0=exact, 1=trim whitespace, 2=normalize (default 1)"},
            },
            "required": ["path", "diff"],
        },
    },
    {
        "name": "file_create",
        "description": "Create a new file (fails if exists). Use file_write to overwrite.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "bash_exec",
        "description": "Execute a bash command and return stdout/stderr/exit code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to execute"},
                "timeout": {"type": "number", "description": "Timeout in seconds (default 120)"},
            },
            "required": ["command"],
        },
    },
]


def build_system_prompt(task: EvalTask, sandbox_dir: Path) -> str:
    """Build the system prompt for the evaluation."""
    return f"""You are an expert software engineer completing a coding task.

## Working Directory
Your working directory is: {sandbox_dir}
All file operations should use absolute paths within this directory.

## File Operations Guide

| Task | Best Tool | When to Use |
|------|-----------|-------------|
| Read file | `file_read` | Always read before editing |
| New file | `file_write` | Creating from scratch |
| Overwrite file | `file_write` | Complete replacement |
| Small edit | `file_edit` | Change a specific string |
| Add method | `file_diff` | Adding to existing class/function |
| Multiple changes | `file_diff` | Several related edits |

### Workflow
1. **Always read first**: Use `file_read` to see current contents
2. **Edit carefully**: Match exact strings for `file_edit`
3. **Use diffs for additions**: `file_diff` is better for adding new code
4. **Verify changes**: Read the file again after editing
5. **Run tests**: Use bash_exec to run pytest

## Task Instructions

{task.description}

## Important
- Work within the sandbox directory: {sandbox_dir}
- Complete the task as instructed
- Run tests to verify your solution
- Make sure all tests pass before finishing"""


def execute_tool(name: str, args: dict, sandbox_dir: Path) -> str:
    """Execute a tool synchronously."""
    # Import here to avoid circular imports
    from ..tool_implementations import (
        file_read, file_write, file_edit, file_diff, file_create, bash_exec
    )

    # Ensure paths are within sandbox
    if "path" in args:
        path = Path(args["path"])
        if not path.is_absolute():
            path = sandbox_dir / path
        # Security check
        try:
            path.resolve().relative_to(sandbox_dir.resolve())
        except ValueError:
            return f"Error: Path {path} is outside sandbox"
        args["path"] = str(path)

    # Handle bash command - set cwd to sandbox
    if name == "bash_exec":
        command = args.get("command", "")
        timeout = args.get("timeout", 120)
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=sandbox_dir,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr] {result.stderr}"
            if result.returncode != 0:
                output += f"\nExit code {result.returncode}"
            return output[:20000] if len(output) > 20000 else output
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"

    # Execute file tools
    tool_map = {
        "file_read": file_read,
        "file_write": file_write,
        "file_edit": file_edit,
        "file_diff": file_diff,
        "file_create": file_create,
    }

    if name not in tool_map:
        return f"Unknown tool: {name}"

    try:
        result = tool_map[name](**args)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


class Evaluator:
    """Runs models against evaluation tasks."""

    def __init__(self, config: EvalConfig, sandbox_base: Optional[Path] = None):
        self.config = config
        self.sandbox_base = sandbox_base or Path(tempfile.gettempdir()) / "mesh_eval"

    async def run_task(self, task: EvalTask) -> TaskResult:
        """Run a single task and return the result."""
        # Create sandbox
        sandbox_dir = self.sandbox_base / f"{task.id}_{self.config.model_name}_{int(time.time())}"
        task.setup_sandbox(sandbox_dir)

        logger.info(f"Running task {task.id} with model {self.config.model_name}")
        logger.info(f"Sandbox: {sandbox_dir}")

        # Track tool usage
        tool_log: list[dict] = []
        tools_used: dict[str, int] = {}

        start_time = time.time()
        error = None
        final_response = ""

        try:
            if self.config.backend_type in ("claude-code", "zai"):
                # Use Claude Code subprocess
                tool_log, tools_used, final_response, error = await self._run_with_cc(
                    task, sandbox_dir, start_time
                )
            elif self.config.backend_type == "openai":
                # Use OpenAI API
                tool_log, tools_used, final_response, error = await self._run_with_openai(
                    task, sandbox_dir, start_time
                )
            elif self.config.backend_type == "anthropic":
                # Use Anthropic API
                tool_log, tools_used, final_response, error = await self._run_with_anthropic(
                    task, sandbox_dir, start_time
                )
            else:
                error = f"Unknown backend: {self.config.backend_type}"

        except Exception as e:
            error = str(e)
            logger.exception(f"Error running task {task.id}")

        elapsed = time.time() - start_time

        # Validate result
        try:
            passed, total, details = task.validate(sandbox_dir)
        except Exception as e:
            passed, total, details = 0, 1, {"validation_error": str(e)}

        # Calculate score
        score = passed / total if total > 0 else 0.0

        return TaskResult(
            task_id=task.id,
            model=self.config.model_name,
            success=error is None and passed == total,
            score=score,
            elapsed_seconds=elapsed,
            tool_calls=len(tool_log),
            tools_used=tools_used,
            tests_passed=passed,
            tests_total=total,
            error=error,
            details={
                "sandbox_dir": str(sandbox_dir),
                "tool_log": tool_log[-20:],  # Last 20 calls
                "validation": details,
                "final_response": final_response[:1000] if final_response else None,
            },
        )

    async def _run_with_cc(
        self, task: EvalTask, sandbox_dir: Path, start_time: float
    ) -> tuple[list[dict], dict[str, int], str, Optional[str]]:
        """Run task using Claude Code subprocess."""
        import shutil

        # Check if claude is available
        if not shutil.which("claude"):
            return [], {}, "", "Claude CLI not found"

        # Build prompt
        system_prompt = build_system_prompt(task, sandbox_dir)
        full_prompt = f"{system_prompt}\n\n---\n\nPlease complete this task:\n{task.description}"

        # Build command
        cmd = ["claude", "-p", full_prompt, "--model", self.config.model_id or "sonnet"]

        # Add allowed tools
        for tool in self.config.cc_allowed_tools:
            cmd.extend(["--allowedTools", tool])

        # Build curated environment — allowlist blocks LD_PRELOAD etc.
        env = _build_subprocess_env()
        env.pop("CLAUDE_CODE_ENTRYPOINT", None)
        if self.config.backend_type == "zai" and self.config.zai_api_key:
            env["ZAI_API_KEY"] = self.config.zai_api_key

        # Run subprocess
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=task.timeout_seconds,
                cwd=sandbox_dir,
                env=env,
            )
            # CC subprocess is a single "tool call"
            tool_log = [{"name": "claude_code_subprocess", "args": {}, "iteration": 1}]
            tools_used = {"claude_code_subprocess": 1}
            return tool_log, tools_used, result.stdout, None
        except subprocess.TimeoutExpired:
            return [], {}, "", "Task timed out"
        except Exception as e:
            return [], {}, "", str(e)

    async def _run_with_openai(
        self, task: EvalTask, sandbox_dir: Path, start_time: float
    ) -> tuple[list[dict], dict[str, int], str, Optional[str]]:
        """Run task using OpenAI API."""
        tool_log: list[dict] = []
        tools_used: dict[str, int] = {}

        system_prompt = build_system_prompt(task, sandbox_dir)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please complete this task:\n\n{task.description}"},
        ]

        async with httpx.AsyncClient(timeout=None) as client:
            iteration = 0
            while iteration < task.max_iterations:
                iteration += 1

                # Check timeout
                if time.time() - start_time > task.timeout_seconds:
                    return tool_log, tools_used, "", "Task timed out"

                # Build request
                request_body = {
                    "model": self.config.model_id,
                    "messages": messages,
                    "tools": OPENAI_TOOLS,
                }

                # Use max_completion_tokens for newer models, max_tokens for older ones
                if self.config.model_id.startswith(("o1", "o3", "gpt-4", "gpt-5")):
                    request_body["max_completion_tokens"] = self.config.max_tokens
                else:
                    request_body["max_tokens"] = self.config.max_tokens

                # Add reasoning effort for reasoning models
                # Note: reasoning_effort models don't support custom temperature
                if self.config.reasoning_effort:
                    request_body["reasoning_effort"] = self.config.reasoning_effort
                else:
                    request_body["temperature"] = self.config.temperature

                # Make request
                response = await client.post(
                    f"{self.config.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_body,
                )

                if response.status_code != 200:
                    return tool_log, tools_used, "", f"API error: {response.status_code} {response.text[:500]}"

                data = response.json()
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})

                # Check for tool calls
                tool_calls = message.get("tool_calls", [])

                if not tool_calls:
                    # Done
                    return tool_log, tools_used, message.get("content", ""), None

                # Execute tools
                messages.append(message)  # Add assistant message

                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    args = json.loads(func.get("arguments", "{}"))

                    # Log
                    tool_log.append({"iteration": iteration, "name": name, "args": args})
                    tools_used[name] = tools_used.get(name, 0) + 1

                    # Execute
                    result = execute_tool(name, args, sandbox_dir)

                    # Add result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "content": result,
                    })

        return tool_log, tools_used, "", "Max iterations reached"

    async def _run_with_anthropic(
        self, task: EvalTask, sandbox_dir: Path, start_time: float
    ) -> tuple[list[dict], dict[str, int], str, Optional[str]]:
        """Run task using Anthropic API."""
        tool_log: list[dict] = []
        tools_used: dict[str, int] = {}

        system_prompt = build_system_prompt(task, sandbox_dir)
        messages = [
            {"role": "user", "content": f"Please complete this task:\n\n{task.description}"},
        ]

        async with httpx.AsyncClient(timeout=None) as client:
            iteration = 0
            while iteration < task.max_iterations:
                iteration += 1

                # Check timeout
                if time.time() - start_time > task.timeout_seconds:
                    return tool_log, tools_used, "", "Task timed out"

                # Build request
                request_body = {
                    "model": self.config.model_id,
                    "messages": messages,
                    "system": system_prompt,
                    "tools": ANTHROPIC_TOOLS,
                    "max_tokens": self.config.max_tokens,
                }

                # Add thinking if configured
                if self.config.anthropic_thinking_budget:
                    request_body["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": self.config.anthropic_thinking_budget,
                    }

                # Make request
                response = await client.post(
                    f"{self.config.base_url}/messages",
                    headers={
                        "x-api-key": self.config.api_key,
                        "Content-Type": "application/json",
                        "anthropic-version": "2023-06-01",
                    },
                    json=request_body,
                )

                if response.status_code != 200:
                    return tool_log, tools_used, "", f"API error: {response.status_code} {response.text[:500]}"

                data = response.json()
                content = data.get("content", [])

                # Check for tool use
                tool_uses = [b for b in content if b.get("type") == "tool_use"]

                if not tool_uses:
                    # Done - extract text
                    texts = [b.get("text", "") for b in content if b.get("type") == "text"]
                    return tool_log, tools_used, "\n".join(texts), None

                # Execute tools
                messages.append({"role": "assistant", "content": content})

                tool_results = []
                for tc in tool_uses:
                    name = tc.get("name", "")
                    args = tc.get("input", {})

                    # Log
                    tool_log.append({"iteration": iteration, "name": name, "args": args})
                    tools_used[name] = tools_used.get(name, 0) + 1

                    # Execute
                    result = execute_tool(name, args, sandbox_dir)

                    # Add result
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.get("id"),
                        "content": result,
                    })

                messages.append({"role": "user", "content": tool_results})

        return tool_log, tools_used, "", "Max iterations reached"


async def run_single_evaluation(
    model_name: str,
    task: EvalTask,
    sandbox_base: Optional[Path] = None,
) -> TaskResult:
    """Convenience function to run a single evaluation."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]
    evaluator = Evaluator(config, sandbox_base)
    return await evaluator.run_task(task)
