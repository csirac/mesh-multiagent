#!/usr/bin/env python3
"""
Run model evaluation using the mesh infrastructure.

This script:
1. Starts a local eval router (isolated from production)
2. Starts evaluator agents for each model
3. Runs the orchestrator to coordinate evaluation
4. Generates reports

Usage:
    # Run evaluation with specific models and tasks
    python -m mesh.eval.run_eval_mesh -m gpt51,deepseek -t bugfix_pagination

    # Run all models on all bugfix tasks
    python -m mesh.eval.run_eval_mesh -m all -c bugfix

    # List available models and tasks
    python -m mesh.eval.run_eval_mesh --list
"""

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def start_eval_router(host: str = "127.0.0.1", port: int = 9999) -> subprocess.Popen:
    """Start an isolated eval router."""
    # Create minimal config for eval router
    config_content = f"""
router:
  host: "{host}"
  port: {port}
  ws_port: {port - 1}
  storage_path: "/tmp/mesh_eval/router_messages.db"
  fcm_enabled: false
  auth_enabled: false

nodes: {{}}
llm_backends: {{}}
"""
    config_path = Path("/tmp/mesh_eval/eval_router_config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_content)

    # Start router
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent.parent / "run_router.py"),
        "--config", str(config_path),
        "--log", "/tmp/mesh_eval/router.log",
    ]

    logger.info(f"Starting eval router on {host}:{port}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for router to start
    time.sleep(2)

    if proc.poll() is not None:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        raise RuntimeError(f"Router failed to start: {stderr}")

    return proc


def list_available():
    """List available models and tasks."""
    from mesh.eval.config import list_models
    from mesh.eval.tasks import get_all_tasks, TaskCategory

    print("Available Models:")
    for model in list_models():
        print(f"  - {model}")

    print("\nAvailable Tasks:")
    tasks = get_all_tasks()
    by_category: dict[str, list] = {}
    for task in tasks:
        cat = task.category.value
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(task)

    for cat, cat_tasks in sorted(by_category.items()):
        print(f"\n  [{cat}]")
        for task in cat_tasks:
            print(f"    - {task.id}: {task.name}")


async def run_evaluation(
    models: list[str],
    tasks: list[str] | None,
    categories: list[str] | None,
    router_host: str,
    router_port: int,
) -> None:
    """Run the evaluation using mesh infrastructure."""
    from mesh.eval.orchestrator_agent import run_orchestrator
    from mesh.eval.orchestrator import generate_report

    run = await run_orchestrator(
        models=models,
        tasks=tasks,
        categories=categories,
        router_host=router_host,
        router_port=router_port,
    )

    # Generate and print report
    report = generate_report(run, format="markdown")
    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run model evaluation via mesh")
    parser.add_argument("--list", action="store_true", help="List available models and tasks")
    parser.add_argument("-m", "--models", help="Comma-separated model names (or 'all')")
    parser.add_argument("-t", "--tasks", help="Comma-separated task IDs")
    parser.add_argument("-c", "--categories", help="Comma-separated task categories")
    parser.add_argument("--router-host", default="127.0.0.1", help="Router host")
    parser.add_argument("--router-port", type=int, default=9999, help="Router port")
    parser.add_argument("--external-router", action="store_true", help="Use external router (don't start one)")
    args = parser.parse_args()

    if args.list:
        list_available()
        return

    if not args.models:
        parser.error("--models is required (use --list to see options)")

    # Parse models
    from mesh.eval.config import list_models
    if args.models.lower() == "all":
        models = list_models()
    else:
        models = [m.strip() for m in args.models.split(",")]

    # Parse tasks/categories
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    categories = [c.strip() for c in args.categories.split(",")] if args.categories else None

    router_proc = None

    try:
        # Start router if needed
        if not args.external_router:
            router_proc = start_eval_router(args.router_host, args.router_port)

        # Run evaluation
        asyncio.run(run_evaluation(
            models=models,
            tasks=tasks,
            categories=categories,
            router_host=args.router_host,
            router_port=args.router_port,
        ))

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        if router_proc:
            logger.info("Stopping eval router")
            router_proc.terminate()
            try:
                router_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                router_proc.kill()


if __name__ == "__main__":
    main()
