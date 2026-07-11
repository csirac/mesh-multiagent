"""
Evaluator Agent - runs evaluation tasks as a mesh agent.

This module provides an agent that:
1. Connects to the eval router as agent:eval:{model_name}
2. Receives task assignments from the orchestrator
3. Sets up sandbox environments
4. Uses the full agent infrastructure (llm.py, tools.py) to complete tasks
5. Reports results back via mesh messages

The evaluator uses the actual hello-world LLM and tool infrastructure,
making the evaluation realistic and debugging infrastructure issues.
"""

import asyncio
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..config import NodeConfig, backend_config_to_llm_config, LLMBackendConfig
from ..llm import LLMClient, LLMConfig, HistoryMessage
from ..node import Node
from ..protocol import Message, MessageType, make_message, build_agent_node_id
from ..tools import get_registry, ToolCall

# Import tool_implementations to register all tools
import mesh.tool_implementations  # noqa: F401

from .config import EvalModelConfig, EVAL_AGENT_TOOLS, EVAL_CODER_PROMPT, EVAL_ROUTER_HOST, EVAL_ROUTER_PORT
from .tasks import EvalTask, TaskResult

logger = logging.getLogger(__name__)


@dataclass
class EvalAgentConfig:
    """Configuration for an evaluator agent."""
    model_config: EvalModelConfig
    sandbox_base: Path = field(default_factory=lambda: Path("/tmp/mesh_eval"))
    max_iterations: int = 50
    timeout_seconds: int = 600


def model_config_to_llm_config(mc: EvalModelConfig) -> LLMConfig:
    """Convert EvalModelConfig to LLMConfig for the LLM client."""
    import os

    # Get API key from environment
    api_key = os.environ.get(mc.api_key_env, "") if mc.api_key_env else ""

    config = LLMConfig(
        backend=mc.backend_type,
        api_key=api_key,
        model=mc.model,
        max_tokens=mc.max_tokens,
        temperature=mc.temperature,
    )

    # Set base URLs based on backend type
    if mc.base_url:
        if mc.backend_type == "anthropic":
            config.anthropic_base_url = mc.base_url
            config.anthropic_api_key = api_key  # Use same key
        else:
            config.base_url = mc.base_url

    # Backend-specific settings
    if mc.reasoning_effort:
        config.reasoning_effort = mc.reasoning_effort
    if mc.thinking_budget:
        config.anthropic_thinking_budget = mc.thinking_budget
    if mc.cc_allowed_tools:
        config.cc_allowed_tools = mc.cc_allowed_tools
    if mc.backend_type == "claude-code":
        config.model = mc.cc_model

    return config


class EvaluatorAgent(Node):
    """
    Agent that executes evaluation tasks.

    Connects to the eval router and waits for task assignments from
    the orchestrator. Each task is executed in an isolated sandbox.
    """

    def __init__(
        self,
        config: EvalAgentConfig,
        router_host: str = EVAL_ROUTER_HOST,
        router_port: int = EVAL_ROUTER_PORT,
    ):
        self.eval_config = config
        model_name = config.model_config.name

        # Build node config for the mesh
        node_config = NodeConfig(
            id=build_agent_node_id("eval", model_name),
            router_host=router_host,
            router_port=router_port,
            agent_type="eval",
            nickname=model_name,
        )
        super().__init__(node_config)

        # Initialize LLM client
        llm_config = model_config_to_llm_config(config.model_config)
        self.llm_client = LLMClient(llm_config)  # node_id not needed — eval uses API backends only, no subprocesses
        self.llm_config = llm_config

        # Tool registry with eval tools
        self.tool_registry = get_registry()
        self.available_tools = list(self.tool_registry.get_subset(EVAL_AGENT_TOOLS).values())

        # State
        self.current_task: Optional[EvalTask] = None
        self.current_sandbox: Optional[Path] = None
        self.task_start_time: Optional[float] = None
        self.tool_log: list[dict] = []

        logger.info(f"EvaluatorAgent initialized: {self.node_id}, backend={llm_config.backend}")

    async def on_message(self, msg: Message) -> None:
        """Handle incoming messages from the mesh (required by Node)."""
        if msg.type == MessageType.MESSAGE:
            # Check if it's a task assignment from orchestrator
            content = msg.content
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    pass

            if isinstance(content, dict) and content.get("type") == "task_assignment":
                await self._handle_task_assignment(msg, content)
            elif isinstance(content, dict) and content.get("type") == "status_request":
                await self._handle_status_request(msg)
            else:
                logger.debug(f"Ignoring non-task message: {msg.content[:100] if isinstance(msg.content, str) else msg.content}")

    async def _handle_status_request(self, msg: Message) -> None:
        """Respond to status requests from orchestrator."""
        status = {
            "type": "status_response",
            "model": self.eval_config.model_config.name,
            "ready": self.current_task is None,
            "current_task": self.current_task.id if self.current_task else None,
        }
        reply = make_message(
            from_node=self.node_id,
            to_node=msg.from_node,
            content=json.dumps(status),
            in_reply_to=msg.id,
        )
        await self.send(reply)

    async def _handle_task_assignment(self, msg: Message, content: dict) -> None:
        """Handle a task assignment from the orchestrator."""
        task_data = content.get("task", {})
        task_id = task_data.get("id")

        logger.info(f"Received task assignment: {task_id}")

        try:
            # Reconstruct EvalTask from serialized data
            from .tasks import EvalTask, TaskCategory
            task = EvalTask(
                id=task_data["id"],
                name=task_data["name"],
                category=TaskCategory(task_data["category"]),
                description=task_data["description"],
                setup_files=task_data["setup_files"],
                validation_script=task_data["validation_script"],
                max_iterations=task_data.get("max_iterations", 50),
                timeout_seconds=task_data.get("timeout_seconds", 600),
            )

            # Run the task
            result = await self.run_task(task)

            # Send result back to orchestrator
            result_msg = make_message(
                from_node=self.node_id,
                to_node=msg.from_node,
                content=json.dumps({
                    "type": "task_result",
                    "result": result.to_dict(),
                }),
                in_reply_to=msg.id,
            )
            await self.send(result_msg)

        except Exception as e:
            logger.exception(f"Error running task {task_id}")
            error_msg = make_message(
                from_node=self.node_id,
                to_node=msg.from_node,
                content=json.dumps({
                    "type": "task_error",
                    "task_id": task_id,
                    "error": str(e),
                }),
                in_reply_to=msg.id,
            )
            await self.send(error_msg)

    async def run_task(self, task: EvalTask) -> TaskResult:
        """
        Execute a single evaluation task.

        1. Set up sandbox with task files
        2. Run agent loop until completion or limit
        3. Validate results
        4. Return scored result
        """
        self.current_task = task
        self.task_start_time = time.time()
        self.tool_log = []

        model_name = self.eval_config.model_config.name
        sandbox_dir = self.eval_config.sandbox_base / model_name / task.id

        logger.info(f"[{model_name}] Starting task: {task.id}")

        try:
            # Clean and setup sandbox
            if sandbox_dir.exists():
                shutil.rmtree(sandbox_dir)
            task.setup_sandbox(sandbox_dir)
            self.current_sandbox = sandbox_dir

            logger.info(f"[{model_name}] Sandbox ready: {sandbox_dir}")

            # Build system prompt with task description
            system_prompt = f"{EVAL_CODER_PROMPT}\n\n## Your Task\n\n{task.description}\n\n## Sandbox Directory\n\nYour working directory is: {sandbox_dir}\n"

            # Get current timestamp for history messages
            from datetime import datetime, timezone
            now_iso = datetime.now(timezone.utc).isoformat()

            # Build initial message history with task as user message
            history: list[HistoryMessage] = [
                HistoryMessage(
                    from_node="user:evaluator",
                    content=f"Please complete this task:\n\n{task.description}",
                    timestamp=now_iso,
                ),
            ]

            # Tool names for the LLM client
            tool_names = EVAL_AGENT_TOOLS

            # Run agent loop
            iteration = 0
            max_iterations = min(task.max_iterations, self.eval_config.max_iterations)

            while iteration < max_iterations:
                iteration += 1
                elapsed = time.time() - self.task_start_time

                if elapsed > task.timeout_seconds:
                    logger.warning(f"[{model_name}] Task {task.id} timed out after {elapsed:.0f}s")
                    break

                logger.debug(f"[{model_name}] Iteration {iteration}/{max_iterations}")

                # Call LLM using complete_with_tools
                try:
                    text_content, tool_calls = await self.llm_client.complete_with_tools(
                        history=history,
                        node_id=self.node_id,
                        system_prompt=system_prompt,
                        tool_registry=self.tool_registry,
                        tool_names=tool_names,
                    )
                except Exception as e:
                    logger.error(f"[{model_name}] LLM error: {e}")
                    break

                # If no tool calls, we're done
                if not tool_calls:
                    logger.info(f"[{model_name}] Task {task.id} completed (no more tool calls)")
                    break

                # Add assistant response (with tool calls) to history
                # Synthesize XML representation for history if using native tools
                response_for_history = text_content or ""
                if not response_for_history and tool_calls:
                    response_for_history = "\n".join(tc.raw_xml for tc in tool_calls if hasattr(tc, 'raw_xml') and tc.raw_xml)
                if not response_for_history:
                    response_for_history = f"[Making {len(tool_calls)} tool calls]"

                history.append(HistoryMessage(
                    from_node=self.node_id,
                    content=response_for_history,
                    timestamp=now_iso,
                    source="in_flight",
                ))

                # Execute tool calls
                tool_result_strs = []
                for tc in tool_calls:
                    self.tool_log.append({
                        "name": tc.name,
                        "args": tc.arguments,
                        "iteration": iteration,
                    })

                    result = await self._execute_tool(tc, sandbox_dir)
                    tool_result_strs.append(f"[{tc.name}]\n{result}")

                # Add tool results as system message (same pattern as agent_node)
                tool_results_text = "\n\n".join(tool_result_strs)
                history.append(HistoryMessage(
                    from_node="system",
                    content=f"Tool execution results:\n{tool_results_text}",
                    timestamp=now_iso,
                    source="in_flight",
                ))

            # Validate results
            elapsed_seconds = time.time() - self.task_start_time
            passed, total, details = task.validate(sandbox_dir)

            # Calculate score
            score = passed / total if total > 0 else 0.0

            # Count tools used
            tools_used: dict[str, int] = {}
            for tl in self.tool_log:
                name = tl["name"]
                tools_used[name] = tools_used.get(name, 0) + 1

            result = TaskResult(
                task_id=task.id,
                model=model_name,
                success=passed == total and total > 0,
                score=score,
                elapsed_seconds=elapsed_seconds,
                tool_calls=len(self.tool_log),
                tools_used=tools_used,
                tests_passed=passed,
                tests_total=total,
                details=details,
            )

            logger.info(
                f"[{model_name}] Task {task.id} result: "
                f"score={score:.2f}, tests={passed}/{total}, tools={len(self.tool_log)}"
            )

            return result

        finally:
            self.current_task = None
            self.current_sandbox = None

    def _get_tool_definitions(self) -> list[dict]:
        """Get tool definitions for LLM."""
        tools = []
        for tool in self.available_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            p.name: {
                                "type": p.type,
                                "description": p.description,
                            }
                            for p in tool.parameters
                        },
                        "required": [p.name for p in tool.parameters if p.required],
                    },
                },
            })
        return tools

    async def _execute_tool(self, tc: ToolCall, sandbox_dir: Path) -> str:
        """Execute a tool call in the sandbox context."""
        tool = self.tool_registry.get(tc.name)
        if not tool:
            return f"Error: Unknown tool '{tc.name}'"

        try:
            # Inject sandbox directory into path-based arguments
            args = dict(tc.arguments)

            # For file operations, resolve paths relative to sandbox
            if tc.name in ("file_read", "file_edit", "file_create", "file_write", "file_diff"):
                if "path" in args:
                    path = args["path"]
                    if not path.startswith("/"):
                        args["path"] = str(sandbox_dir / path)
                    elif not path.startswith(str(sandbox_dir)):
                        # Ensure we stay in sandbox
                        args["path"] = str(sandbox_dir / Path(path).name)

            # For bash_exec, ensure we're in sandbox
            if tc.name == "bash_exec":
                cmd = args.get("command", "")
                # Prepend cd to sandbox if not already there
                if not cmd.strip().startswith("cd "):
                    args["command"] = f"cd {sandbox_dir} && {cmd}"

            # Execute tool
            result = tool.handler(**args)

            # Handle async results
            if asyncio.iscoroutine(result):
                result = await result

            return str(result) if result is not None else "Success"

        except Exception as e:
            logger.warning(f"Tool {tc.name} error: {e}")
            return f"Error: {e}"


async def run_evaluator(
    model_name: str,
    sandbox_base: Optional[Path] = None,
    router_host: str = EVAL_ROUTER_HOST,
    router_port: int = EVAL_ROUTER_PORT,
    max_iterations: int = 50,
) -> None:
    """
    Run an evaluator agent for the given model.

    This connects to the eval router and waits for task assignments.
    """
    from .config import get_model_config

    model_config = get_model_config(model_name)

    config = EvalAgentConfig(
        model_config=model_config,
        sandbox_base=sandbox_base or Path("/tmp/mesh_eval"),
        max_iterations=max_iterations,
    )

    agent = EvaluatorAgent(config, router_host, router_port)

    logger.info(f"Starting evaluator agent: {agent.node_id}")

    # Connect and run
    await agent.connect()
    try:
        await agent.receive_loop()
    finally:
        await agent.disconnect()
