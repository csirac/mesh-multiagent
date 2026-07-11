"""
Orchestrator - coordinates distributed model evaluation.

The orchestrator:
1. Loads tasks from the task library
2. Distributes tasks to evaluator agents
3. Collects and aggregates results
4. Generates reports
"""

import asyncio
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .tasks import (
    EvalTask, TaskResult, TaskCategory,
    get_all_tasks, get_tasks_by_category, get_task, TASK_LIBRARY,
)
from .evaluator import MODEL_CONFIGS, Evaluator, EvalConfig

logger = logging.getLogger(__name__)


@dataclass
class EvalRun:
    """A single evaluation run."""
    run_id: str
    models: list[str]
    tasks: list[str]
    started_at: datetime
    completed_at: Optional[datetime] = None
    results: list[TaskResult] = field(default_factory=list)


class ResultStore:
    """SQLite-backed storage for evaluation results."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    models TEXT,
                    tasks TEXT,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    task_id TEXT,
                    model TEXT,
                    success INTEGER,
                    score REAL,
                    elapsed_seconds REAL,
                    tool_calls INTEGER,
                    tools_used TEXT,
                    tests_passed INTEGER,
                    tests_total INTEGER,
                    error TEXT,
                    details TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
                CREATE INDEX IF NOT EXISTS idx_results_model ON results(model);
                CREATE INDEX IF NOT EXISTS idx_results_task ON results(task_id);
            """)

    def save_run(self, run: EvalRun):
        """Save a run to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, models, tasks, started_at, completed_at) VALUES (?, ?, ?, ?, ?)",
                (
                    run.run_id,
                    json.dumps(run.models),
                    json.dumps(run.tasks),
                    run.started_at.isoformat(),
                    run.completed_at.isoformat() if run.completed_at else None,
                ),
            )

    def save_result(self, run_id: str, result: TaskResult):
        """Save a task result to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO results
                   (run_id, task_id, model, success, score, elapsed_seconds, tool_calls,
                    tools_used, tests_passed, tests_total, error, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    result.task_id,
                    result.model,
                    1 if result.success else 0,
                    result.score,
                    result.elapsed_seconds,
                    result.tool_calls,
                    json.dumps(result.tools_used),
                    result.tests_passed,
                    result.tests_total,
                    result.error,
                    json.dumps(result.details),
                ),
            )

    def get_run(self, run_id: str) -> Optional[EvalRun]:
        """Get a run by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if not row:
                return None

            results = self.get_results(run_id)
            return EvalRun(
                run_id=row["run_id"],
                models=json.loads(row["models"]),
                tasks=json.loads(row["tasks"]),
                started_at=datetime.fromisoformat(row["started_at"]),
                completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                results=results,
            )

    def get_results(self, run_id: str) -> list[TaskResult]:
        """Get all results for a run."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM results WHERE run_id = ? ORDER BY created_at", (run_id,)
            ).fetchall()

            return [
                TaskResult(
                    task_id=row["task_id"],
                    model=row["model"],
                    success=bool(row["success"]),
                    score=row["score"],
                    elapsed_seconds=row["elapsed_seconds"],
                    tool_calls=row["tool_calls"],
                    tools_used=json.loads(row["tools_used"]),
                    tests_passed=row["tests_passed"],
                    tests_total=row["tests_total"],
                    error=row["error"],
                    details=json.loads(row["details"]) if row["details"] else {},
                )
                for row in rows
            ]

    def get_latest_run_id(self) -> Optional[str]:
        """Get the most recent run ID."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            return row[0] if row else None

    def list_runs(self, limit: int = 10) -> list[dict]:
        """List recent runs."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT run_id, models, tasks, started_at, completed_at FROM runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

            return [
                {
                    "run_id": row["run_id"],
                    "models": json.loads(row["models"]),
                    "tasks": json.loads(row["tasks"]),
                    "started_at": row["started_at"],
                    "completed_at": row["completed_at"],
                }
                for row in rows
            ]


class Orchestrator:
    """Coordinates model evaluation across multiple models and tasks."""

    def __init__(
        self,
        models: list[str],
        tasks: Optional[list[str]] = None,
        categories: Optional[list[TaskCategory]] = None,
        db_path: Optional[Path] = None,
        sandbox_base: Optional[Path] = None,
        parallel: bool = True,
    ):
        self.models = models
        self.parallel = parallel
        self.sandbox_base = sandbox_base or Path("/tmp/mesh_eval")

        # Validate models
        for model in models:
            if model not in MODEL_CONFIGS:
                raise ValueError(f"Unknown model: {model}. Available: {list(MODEL_CONFIGS.keys())}")

        # Determine tasks to run
        if tasks:
            self.tasks = [get_task(t) for t in tasks]
        elif categories:
            self.tasks = []
            for cat in categories:
                self.tasks.extend(get_tasks_by_category(cat))
        else:
            self.tasks = get_all_tasks()

        # Initialize storage
        if db_path is None:
            from ..paths import real_home
            db_path = real_home() / ".hello-world" / "eval_results.db"
        self.store = ResultStore(db_path)

    async def run(self) -> EvalRun:
        """Run the evaluation and return results."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run = EvalRun(
            run_id=run_id,
            models=self.models,
            tasks=[t.id for t in self.tasks],
            started_at=datetime.now(),
        )
        self.store.save_run(run)

        logger.info(f"Starting evaluation run {run_id}")
        logger.info(f"Models: {self.models}")
        logger.info(f"Tasks: {[t.id for t in self.tasks]}")

        # Create evaluators
        evaluators = {
            model: Evaluator(MODEL_CONFIGS[model], self.sandbox_base / model)
            for model in self.models
        }

        # Build task queue
        task_queue: list[tuple[str, EvalTask]] = []
        for model in self.models:
            for task in self.tasks:
                task_queue.append((model, task))

        # Run tasks
        if self.parallel:
            results = await self._run_parallel(evaluators, task_queue)
        else:
            results = await self._run_sequential(evaluators, task_queue)

        # Save results
        for result in results:
            self.store.save_result(run_id, result)
            run.results.append(result)

        # Mark run complete
        run.completed_at = datetime.now()
        self.store.save_run(run)

        logger.info(f"Evaluation run {run_id} complete")
        return run

    async def _run_parallel(
        self,
        evaluators: dict[str, Evaluator],
        task_queue: list[tuple[str, EvalTask]],
    ) -> list[TaskResult]:
        """Run tasks in parallel (one per model at a time)."""
        results: list[TaskResult] = []

        # Group by model for parallel execution
        model_tasks: dict[str, list[EvalTask]] = {m: [] for m in self.models}
        for model, task in task_queue:
            model_tasks[model].append(task)

        # Run each model's tasks sequentially, but models in parallel
        async def run_model_tasks(model: str, tasks: list[EvalTask]) -> list[TaskResult]:
            model_results = []
            evaluator = evaluators[model]
            for task in tasks:
                logger.info(f"[{model}] Running task: {task.id}")
                try:
                    result = await evaluator.run_task(task)
                    model_results.append(result)
                    logger.info(f"[{model}] {task.id}: score={result.score:.2f}, tools={result.tool_calls}")
                except Exception as e:
                    logger.exception(f"[{model}] Failed task {task.id}")
                    model_results.append(TaskResult(
                        task_id=task.id,
                        model=model,
                        success=False,
                        score=0.0,
                        elapsed_seconds=0.0,
                        tool_calls=0,
                        tools_used={},
                        tests_passed=0,
                        tests_total=1,
                        error=str(e),
                    ))
            return model_results

        # Run all models in parallel
        all_results = await asyncio.gather(*[
            run_model_tasks(model, tasks)
            for model, tasks in model_tasks.items()
        ])

        # Flatten results
        for model_results in all_results:
            results.extend(model_results)

        return results

    async def _run_sequential(
        self,
        evaluators: dict[str, Evaluator],
        task_queue: list[tuple[str, EvalTask]],
    ) -> list[TaskResult]:
        """Run tasks sequentially."""
        results: list[TaskResult] = []

        for model, task in task_queue:
            logger.info(f"[{model}] Running task: {task.id}")
            try:
                result = await evaluators[model].run_task(task)
                results.append(result)
                logger.info(f"[{model}] {task.id}: score={result.score:.2f}, tools={result.tool_calls}")
            except Exception as e:
                logger.exception(f"[{model}] Failed task {task.id}")
                results.append(TaskResult(
                    task_id=task.id,
                    model=model,
                    success=False,
                    score=0.0,
                    elapsed_seconds=0.0,
                    tool_calls=0,
                    tools_used={},
                    tests_passed=0,
                    tests_total=1,
                    error=str(e),
                ))

        return results


def generate_report(run: EvalRun, format: str = "markdown") -> str:
    """Generate a report from evaluation results."""
    if format == "markdown":
        return _generate_markdown_report(run)
    elif format == "json":
        return json.dumps({
            "run_id": run.run_id,
            "models": run.models,
            "tasks": run.tasks,
            "started_at": run.started_at.isoformat(),
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "results": [r.to_dict() for r in run.results],
        }, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")


def _generate_markdown_report(run: EvalRun) -> str:
    """Generate a Markdown report."""
    lines = [
        f"# Evaluation Report: {run.run_id}",
        "",
        f"**Started**: {run.started_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Completed**: {run.completed_at.strftime('%Y-%m-%d %H:%M:%S') if run.completed_at else 'In Progress'}",
        f"**Models**: {', '.join(run.models)}",
        f"**Tasks**: {len(run.tasks)}",
        "",
        "## Summary",
        "",
        "| Model | Tasks | Passed | Score | Avg Time | Avg Tools |",
        "|-------|-------|--------|-------|----------|-----------|",
    ]

    # Aggregate by model
    model_stats: dict[str, dict] = {}
    for result in run.results:
        if result.model not in model_stats:
            model_stats[result.model] = {
                "tasks": 0,
                "passed": 0,
                "total_score": 0.0,
                "total_time": 0.0,
                "total_tools": 0,
            }
        stats = model_stats[result.model]
        stats["tasks"] += 1
        stats["passed"] += 1 if result.success else 0
        stats["total_score"] += result.score
        stats["total_time"] += result.elapsed_seconds
        stats["total_tools"] += result.tool_calls

    for model, stats in sorted(model_stats.items()):
        avg_score = stats["total_score"] / stats["tasks"] if stats["tasks"] > 0 else 0
        avg_time = stats["total_time"] / stats["tasks"] if stats["tasks"] > 0 else 0
        avg_tools = stats["total_tools"] / stats["tasks"] if stats["tasks"] > 0 else 0
        lines.append(
            f"| {model} | {stats['tasks']} | {stats['passed']} | {avg_score:.1%} | {avg_time:.1f}s | {avg_tools:.1f} |"
        )

    lines.extend([
        "",
        "## Detailed Results",
        "",
        "| Task | Model | Success | Score | Time | Tools | Tests |",
        "|------|-------|---------|-------|------|-------|-------|",
    ])

    for result in sorted(run.results, key=lambda r: (r.task_id, r.model)):
        success_icon = "✅" if result.success else "❌"
        lines.append(
            f"| {result.task_id} | {result.model} | {success_icon} | {result.score:.1%} | "
            f"{result.elapsed_seconds:.1f}s | {result.tool_calls} | {result.tests_passed}/{result.tests_total} |"
        )

    # Tool usage breakdown
    lines.extend([
        "",
        "## Tool Usage",
        "",
    ])

    for model in run.models:
        model_results = [r for r in run.results if r.model == model]
        all_tools: dict[str, int] = {}
        for r in model_results:
            for tool, count in r.tools_used.items():
                all_tools[tool] = all_tools.get(tool, 0) + count

        if all_tools:
            lines.append(f"**{model}**: " + ", ".join(f"{t}={c}" for t, c in sorted(all_tools.items())))
        else:
            lines.append(f"**{model}**: (no tool usage recorded)")

    return "\n".join(lines)
