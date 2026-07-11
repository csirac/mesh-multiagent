"""
CLI runner for model evaluation.

Usage:
    python -m mesh.eval run --models gpt51,kimi,glm --category bugfix
    python -m mesh.eval run --models all --task bugfix_pagination
    python -m mesh.eval report --run-id latest
    python -m mesh.eval list
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .tasks import TaskCategory, TASK_LIBRARY, get_all_tasks
from .evaluator import MODEL_CONFIGS
from .orchestrator import Orchestrator, ResultStore, generate_report


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_models(models_str: str) -> list[str]:
    """Parse model specification."""
    if models_str.lower() == "all":
        return list(MODEL_CONFIGS.keys())
    return [m.strip() for m in models_str.split(",")]


def parse_categories(cats_str: str) -> list[TaskCategory]:
    """Parse category specification."""
    return [TaskCategory(c.strip().lower()) for c in cats_str.split(",")]


async def cmd_run(args):
    """Run evaluation."""
    models = parse_models(args.models)
    print(f"Models: {models}")

    tasks = None
    categories = None

    if args.task:
        tasks = [t.strip() for t in args.task.split(",")]
        print(f"Tasks: {tasks}")
    elif args.category:
        categories = parse_categories(args.category)
        print(f"Categories: {[c.value for c in categories]}")
    else:
        print("Running all tasks")

    # Create orchestrator
    orchestrator = Orchestrator(
        models=models,
        tasks=tasks,
        categories=categories,
        parallel=not args.sequential,
        sandbox_base=Path(args.sandbox) if args.sandbox else None,
    )

    print(f"\n{'='*60}")
    print(f"Starting evaluation: {len(orchestrator.tasks)} tasks × {len(models)} models")
    print(f"{'='*60}\n")

    # Run evaluation
    run = await orchestrator.run()

    # Generate report
    report = generate_report(run, format="markdown")
    print("\n" + report)

    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report)
        print(f"\nReport saved to: {output_path}")

    print(f"\nRun ID: {run.run_id}")
    return 0


def cmd_report(args):
    """Generate report for a run."""
    from ..paths import real_home
    db_path = real_home() / ".hello-world" / "eval_results.db"
    store = ResultStore(db_path)

    run_id = args.run_id
    if run_id == "latest":
        run_id = store.get_latest_run_id()
        if not run_id:
            print("No evaluation runs found")
            return 1

    run = store.get_run(run_id)
    if not run:
        print(f"Run not found: {run_id}")
        return 1

    report = generate_report(run, format=args.format)
    print(report)

    if args.output:
        Path(args.output).write_text(report)
        print(f"\nSaved to: {args.output}")

    return 0


def cmd_list(args):
    """List available tasks and models."""
    print("Available Models:")
    print("-" * 40)
    for name, config in MODEL_CONFIGS.items():
        print(f"  {name:12} - {config.backend_type:10} ({config.model_id})")

    print("\nAvailable Tasks:")
    print("-" * 40)
    for task in get_all_tasks():
        print(f"  {task.id:30} - {task.name} [{task.category.value}]")

    print("\nCategories:")
    print("-" * 40)
    for cat in TaskCategory:
        count = len([t for t in TASK_LIBRARY.values() if t.category == cat])
        print(f"  {cat.value:15} - {count} tasks")

    return 0


def cmd_runs(args):
    """List recent runs."""
    from ..paths import real_home
    db_path = real_home() / ".hello-world" / "eval_results.db"
    store = ResultStore(db_path)

    runs = store.list_runs(limit=args.limit)
    if not runs:
        print("No evaluation runs found")
        return 0

    print("Recent Evaluation Runs:")
    print("-" * 80)
    for run in runs:
        status = "✓" if run["completed_at"] else "…"
        print(f"  {status} {run['run_id']}  models={','.join(run['models'])}  tasks={len(run['tasks'])}")

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Model evaluation framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # run command
    run_parser = subparsers.add_parser("run", help="Run evaluation")
    run_parser.add_argument(
        "-m", "--models",
        required=True,
        help="Models to evaluate (comma-separated, or 'all')",
    )
    run_parser.add_argument(
        "-t", "--task",
        help="Specific task IDs to run (comma-separated)",
    )
    run_parser.add_argument(
        "-c", "--category",
        help="Task categories to run (comma-separated)",
    )
    run_parser.add_argument(
        "-o", "--output",
        help="Output file for report",
    )
    run_parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run sequentially instead of parallel",
    )
    run_parser.add_argument(
        "--sandbox",
        help="Base directory for sandboxes",
    )

    # report command
    report_parser = subparsers.add_parser("report", help="Generate report")
    report_parser.add_argument(
        "-r", "--run-id",
        default="latest",
        help="Run ID (default: latest)",
    )
    report_parser.add_argument(
        "-f", "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format",
    )
    report_parser.add_argument(
        "-o", "--output",
        help="Output file",
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List tasks and models")

    # runs command
    runs_parser = subparsers.add_parser("runs", help="List recent runs")
    runs_parser.add_argument(
        "-n", "--limit",
        type=int,
        default=10,
        help="Number of runs to show",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "run":
        return asyncio.run(cmd_run(args))
    elif args.command == "report":
        return cmd_report(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "runs":
        return cmd_runs(args)
    else:
        parser.print_help()
        return 1


def run_evaluation(
    models: list[str],
    tasks: list[str] = None,
    categories: list[str] = None,
    parallel: bool = True,
) -> dict:
    """
    Programmatic interface to run evaluation.

    Returns dict with run_id and results.
    """
    cats = [TaskCategory(c) for c in categories] if categories else None

    orchestrator = Orchestrator(
        models=models,
        tasks=tasks,
        categories=cats,
        parallel=parallel,
    )

    run = asyncio.run(orchestrator.run())

    return {
        "run_id": run.run_id,
        "models": run.models,
        "tasks": run.tasks,
        "results": [r.to_dict() for r in run.results],
    }


if __name__ == "__main__":
    sys.exit(main())
