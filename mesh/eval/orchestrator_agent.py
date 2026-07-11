"""
Orchestrator Agent - coordinates model evaluation via mesh.

The orchestrator:
1. Connects to the eval router as agent:eval:orchestrator
2. Manages evaluator agents (one per model)
3. Distributes tasks and collects results
4. Stores results in SQLite and generates reports

This is a generalizable pattern for coordinating multiple agents.
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import NodeConfig
from ..node import Node
from ..protocol import Message, MessageType, make_message, build_agent_node_id

from .config import (
    EvalModelConfig, MODEL_CONFIGS, EVAL_ROUTER_HOST, EVAL_ROUTER_PORT,
    list_models, get_model_config,
)
from .tasks import EvalTask, TaskResult, TaskCategory, get_all_tasks, get_tasks_by_category, get_task
from .orchestrator import ResultStore, EvalRun, generate_report

logger = logging.getLogger(__name__)


@dataclass
class AgentStatus:
    """Status of an evaluator agent."""
    model: str
    node_id: str
    ready: bool = False
    current_task: Optional[str] = None
    process: Optional[subprocess.Popen] = None


class OrchestratorAgent(Node):
    """
    Coordinates model evaluation across multiple evaluator agents.

    The orchestrator:
    1. Starts evaluator agents as subprocesses
    2. Waits for agents to connect and report ready
    3. Distributes tasks round-robin
    4. Collects results and stores in SQLite
    """

    def __init__(
        self,
        models: list[str],
        tasks: Optional[list[EvalTask]] = None,
        router_host: str = EVAL_ROUTER_HOST,
        router_port: int = EVAL_ROUTER_PORT,
        db_path: Optional[Path] = None,
        sandbox_base: Optional[Path] = None,
    ):
        node_config = NodeConfig(
            id=build_agent_node_id("eval", "orchestrator"),
            router_host=router_host,
            router_port=router_port,
            agent_type="eval",
            nickname="orchestrator",
        )
        super().__init__(node_config)

        self.models = models
        self.tasks = tasks or []
        self.sandbox_base = sandbox_base or Path("/tmp/mesh_eval")

        # Validate models
        for model in models:
            if model not in MODEL_CONFIGS:
                raise ValueError(f"Unknown model: {model}")

        # Initialize result store
        if db_path is None:
            from ..paths import real_home
            db_path = real_home() / ".hello-world" / "eval_results.db"
        self.store = ResultStore(db_path)

        # Agent tracking
        self.agents: dict[str, AgentStatus] = {}
        self.pending_results: dict[str, asyncio.Future] = {}

        # Current run
        self.current_run: Optional[EvalRun] = None

        logger.info(f"OrchestratorAgent initialized with models: {models}")

    async def start_evaluator_agents(self) -> None:
        """Start evaluator agent subprocesses for each model."""
        for model in self.models:
            agent_node_id = build_agent_node_id("eval", model)
            self.agents[model] = AgentStatus(
                model=model,
                node_id=agent_node_id,
            )

            # Start subprocess
            cmd = [
                sys.executable, "-m", "mesh.eval.run_evaluator",
                "--model", model,
                "--router-host", str(self.config.router_host),
                "--router-port", str(self.config.router_port),
                "--sandbox-base", str(self.sandbox_base),
            ]

            logger.info(f"Starting evaluator for {model}: {' '.join(cmd)}")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.agents[model].process = proc

        # Wait for agents to connect (poll with status requests)
        await self._wait_for_agents_ready(timeout=60)

    async def _wait_for_agents_ready(self, timeout: float = 60) -> None:
        """Wait for all evaluator agents to report ready."""
        start = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start < timeout:
            # Send status requests to all agents
            for model, status in self.agents.items():
                if not status.ready:
                    msg = make_message(
                        from_node=self.node_id,
                        to_node=status.node_id,
                        content=json.dumps({"type": "status_request"}),
                    )
                    try:
                        await self.send(msg)
                    except Exception:
                        pass  # Agent not connected yet

            await asyncio.sleep(2)

            # Check if all ready
            if all(s.ready for s in self.agents.values()):
                logger.info("All evaluator agents ready")
                return

        # Timeout - report which agents aren't ready
        not_ready = [m for m, s in self.agents.items() if not s.ready]
        logger.warning(f"Timeout waiting for agents: {not_ready}")

    async def on_message(self, msg: Message) -> None:
        """Handle incoming messages (results, status)."""
        if msg.type != MessageType.MESSAGE:
            return

        content = msg.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                return

        if not isinstance(content, dict):
            return

        msg_type = content.get("type")

        if msg_type == "status_response":
            await self._handle_status_response(msg, content)
        elif msg_type == "task_result":
            await self._handle_task_result(msg, content)
        elif msg_type == "task_error":
            await self._handle_task_error(msg, content)

    async def _handle_status_response(self, msg: Message, content: dict) -> None:
        """Handle status response from evaluator."""
        model = content.get("model")
        ready = content.get("ready", False)

        if model in self.agents:
            self.agents[model].ready = ready
            self.agents[model].current_task = content.get("current_task")
            logger.debug(f"Agent {model} status: ready={ready}")

    async def _handle_task_result(self, msg: Message, content: dict) -> None:
        """Handle task result from evaluator."""
        result_data = content.get("result", {})
        task_id = result_data.get("task_id")
        model = result_data.get("model")

        logger.info(f"Received result for {task_id} from {model}")

        # Reconstruct TaskResult
        result = TaskResult(
            task_id=result_data["task_id"],
            model=result_data["model"],
            success=result_data["success"],
            score=result_data["score"],
            elapsed_seconds=result_data["elapsed_seconds"],
            tool_calls=result_data["tool_calls"],
            tools_used=result_data["tools_used"],
            tests_passed=result_data["tests_passed"],
            tests_total=result_data["tests_total"],
            error=result_data.get("error"),
            details=result_data.get("details", {}),
        )

        # Resolve pending future if exists
        key = f"{model}:{task_id}"
        if key in self.pending_results:
            self.pending_results[key].set_result(result)

    async def _handle_task_error(self, msg: Message, content: dict) -> None:
        """Handle task error from evaluator."""
        task_id = content.get("task_id")
        error = content.get("error")
        model = msg.from_node.split(":")[-1]  # Extract model from node_id

        logger.error(f"Task {task_id} failed on {model}: {error}")

        # Create error result
        result = TaskResult(
            task_id=task_id,
            model=model,
            success=False,
            score=0.0,
            elapsed_seconds=0.0,
            tool_calls=0,
            tools_used={},
            tests_passed=0,
            tests_total=1,
            error=error,
        )

        key = f"{model}:{task_id}"
        if key in self.pending_results:
            self.pending_results[key].set_result(result)

    async def assign_task(self, model: str, task: EvalTask) -> TaskResult:
        """Assign a task to an evaluator and wait for result."""
        agent = self.agents.get(model)
        if not agent:
            raise ValueError(f"No agent for model: {model}")

        # Create future for result
        key = f"{model}:{task.id}"
        future: asyncio.Future[TaskResult] = asyncio.get_event_loop().create_future()
        self.pending_results[key] = future

        # Serialize task
        task_data = {
            "id": task.id,
            "name": task.name,
            "category": task.category.value,
            "description": task.description,
            "setup_files": task.setup_files,
            "validation_script": task.validation_script,
            "max_iterations": task.max_iterations,
            "timeout_seconds": task.timeout_seconds,
        }

        # Send task assignment
        msg = make_message(
            from_node=self.node_id,
            to_node=agent.node_id,
            content=json.dumps({
                "type": "task_assignment",
                "task": task_data,
            }),
        )
        await self.send(msg)

        logger.info(f"Assigned task {task.id} to {model}")

        # Wait for result with timeout
        try:
            result = await asyncio.wait_for(future, timeout=task.timeout_seconds + 60)
            return result
        except asyncio.TimeoutError:
            logger.error(f"Task {task.id} timed out waiting for {model}")
            return TaskResult(
                task_id=task.id,
                model=model,
                success=False,
                score=0.0,
                elapsed_seconds=task.timeout_seconds,
                tool_calls=0,
                tools_used={},
                tests_passed=0,
                tests_total=1,
                error="Orchestrator timeout waiting for result",
            )
        finally:
            self.pending_results.pop(key, None)

    async def run_evaluation(self) -> EvalRun:
        """Run the full evaluation."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run = EvalRun(
            run_id=run_id,
            models=self.models,
            tasks=[t.id for t in self.tasks],
            started_at=datetime.now(),
        )
        self.store.save_run(self.current_run)

        logger.info(f"Starting evaluation run: {run_id}")
        logger.info(f"Models: {self.models}")
        logger.info(f"Tasks: {[t.id for t in self.tasks]}")

        # Run tasks - each model runs its tasks sequentially, but models in parallel
        async def run_model_tasks(model: str) -> list[TaskResult]:
            results = []
            for task in self.tasks:
                logger.info(f"[{model}] Running: {task.id}")
                result = await self.assign_task(model, task)
                results.append(result)
                # Save result immediately
                self.store.save_result(run_id, result)
                logger.info(f"[{model}] {task.id}: score={result.score:.2f}")
            return results

        # Run all models in parallel
        all_results = await asyncio.gather(*[
            run_model_tasks(model) for model in self.models
        ])

        # Flatten results
        for model_results in all_results:
            self.current_run.results.extend(model_results)

        # Mark complete
        self.current_run.completed_at = datetime.now()
        self.store.save_run(self.current_run)

        logger.info(f"Evaluation run {run_id} complete")
        return self.current_run

    def stop_evaluator_agents(self) -> None:
        """Stop all evaluator agent subprocesses."""
        for model, status in self.agents.items():
            if status.process:
                logger.info(f"Stopping evaluator for {model}")
                status.process.terminate()
                try:
                    status.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    status.process.kill()


async def run_orchestrator(
    models: list[str],
    tasks: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
    router_host: str = EVAL_ROUTER_HOST,
    router_port: int = EVAL_ROUTER_PORT,
    db_path: Optional[Path] = None,
    sandbox_base: Optional[Path] = None,
) -> EvalRun:
    """
    Run an evaluation with the orchestrator.

    This is the main entry point for running evaluations via mesh.
    """
    # Determine tasks
    task_list: list[EvalTask]
    if tasks:
        task_list = [get_task(t) for t in tasks]
    elif categories:
        task_list = []
        for cat_name in categories:
            cat = TaskCategory(cat_name)
            task_list.extend(get_tasks_by_category(cat))
    else:
        task_list = get_all_tasks()

    orchestrator = OrchestratorAgent(
        models=models,
        tasks=task_list,
        router_host=router_host,
        router_port=router_port,
        db_path=db_path,
        sandbox_base=sandbox_base,
    )

    # Connect to router
    await orchestrator.connect()

    try:
        # Start evaluator agents
        await orchestrator.start_evaluator_agents()

        # Run evaluation
        run = await orchestrator.run_evaluation()

        return run

    finally:
        # Cleanup
        orchestrator.stop_evaluator_agents()
        await orchestrator.disconnect()
