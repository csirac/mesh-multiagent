#!/usr/bin/env python3
"""
EvalPlus benchmark runner for mesh agents.

Runs HumanEval+ problems through our mesh agents and generates
EvalPlus-compatible output for evaluation.

Usage:
    # Run 5 problems as a dry run
    python -m tests.live.run_evalplus --problems 5

    # Run specific problems
    python -m tests.live.run_evalplus --problem-ids HumanEval/0,HumanEval/4

    # Full run with specific agent
    python -m tests.live.run_evalplus --agent agent:assistant:v02 --controller phase-flow-v02

    # Evaluate results
    evalplus.evaluate --dataset humaneval --samples tests/live/evalplus_results/run_*.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evalplus.data import get_human_eval_plus

from tests.live.evalplus_provider import MeshDecoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HumanEval+ benchmark against mesh agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Problem selection
    parser.add_argument(
        "--problems", "-n",
        type=int,
        default=5,
        help="Number of problems to run (default: 5 for dry run)",
    )
    parser.add_argument(
        "--problem-ids",
        type=str,
        help="Comma-separated specific problem IDs (e.g., HumanEval/0,HumanEval/4)",
    )

    # Agent configuration
    parser.add_argument(
        "--agent",
        type=str,
        default="agent:assistant:v02",
        help="Agent node ID to use (default: agent:assistant:v02)",
    )
    parser.add_argument(
        "--controller",
        type=str,
        choices=["passthrough", "task-fsm-v0", "phase-flow-v02"],
        help="Controller mode for agent",
    )
    parser.add_argument(
        "--ws-url",
        type=str,
        default=os.environ.get("MESH_WS_URL", ""),
        help="WebSocket URL for mesh router",
    )
    parser.add_argument(
        "--auth-token",
        type=str,
        default=os.environ.get("MESH_AUTH_TOKEN"),
        help="Mesh authentication token",
    )

    # Execution options
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Timeout per problem in seconds (default: 90)",
    )
    parser.add_argument(
        "--reset-context",
        action="store_true",
        help="Spawn fresh agent per problem (not implemented yet)",
    )
    parser.add_argument(
        "--max-tokens-total",
        type=int,
        help="Stop if total tokens exceed this (cost cap for dry runs)",
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="tests/live/evalplus_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw agent responses for debugging",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def select_problems(args) -> list[tuple[str, dict]]:
    """Select which problems to run based on args."""
    dataset = get_human_eval_plus()

    if args.problem_ids:
        # Specific problem IDs requested
        ids = [id.strip() for id in args.problem_ids.split(",")]
        problems = [(id, dataset[id]) for id in ids if id in dataset]
        if len(problems) != len(ids):
            missing = set(ids) - set(p[0] for p in problems)
            logger.warning(f"Some problem IDs not found: {missing}")
        return problems

    # Select first N problems
    all_ids = sorted(dataset.keys())
    selected_ids = all_ids[:args.problems]
    return [(id, dataset[id]) for id in selected_ids]


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_log_dir = None
    if args.save_raw:
        raw_log_dir = output_dir / "raw"
        raw_log_dir.mkdir(exist_ok=True)

    # Select problems
    problems = select_problems(args)
    logger.info(f"Running {len(problems)} problems against {args.agent}")

    # Create decoder
    decoder = MeshDecoder(
        name=args.agent,
        ws_url=args.ws_url,
        auth_token=args.auth_token,
        timeout=args.timeout,
        controller=args.controller,
        reset_context=args.reset_context,
        raw_log_dir=str(raw_log_dir) if raw_log_dir else None,
    )

    # Run timestamp for output files
    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    results_file = output_dir / f"run_{run_id}.jsonl"
    summary_file = output_dir / f"run_{run_id}_summary.json"

    # Run problems
    results = []
    for i, (task_id, problem) in enumerate(problems, 1):
        logger.info(f"[{i}/{len(problems)}] Running {task_id}...")

        prompt = problem["prompt"]
        entry_point = problem["entry_point"]

        # Generate completion
        completions = decoder.codegen(prompt, do_sample=False, num_samples=1)
        completion = completions[0] if completions else ""

        # Check if extraction succeeded
        success = bool(completion and f"def {entry_point}" in completion)

        result = {
            "task_id": task_id,
            "completion": completion,
        }
        results.append(result)

        status = "✓" if success else "✗"
        logger.info(f"  {status} {task_id}: {len(completion)} chars")

        # Check cost cap
        if args.max_tokens_total and decoder.total_tokens > args.max_tokens_total:
            logger.warning(f"Token cap reached ({decoder.total_tokens} > {args.max_tokens_total}), stopping")
            break

    # Write results JSONL (EvalPlus-compatible)
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    logger.info(f"Results written to {results_file}")

    # Write summary
    stats = decoder.get_stats()
    summary = {
        "run_id": run_id,
        "agent": args.agent,
        "controller": args.controller,
        "ws_url": args.ws_url,
        "problems_attempted": len(results),
        "problems_requested": len(problems),
        **stats,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written to {summary_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("EvalPlus Run Summary")
    print("=" * 60)
    print(f"Agent:                 {args.agent}")
    print(f"Controller:            {args.controller or 'default'}")
    print(f"Problems attempted:    {len(results)}")
    print(f"Extraction success:    {stats['total_problems'] - stats['extraction_failures']}/{stats['total_problems']}")
    print(f"Total time:            {stats['total_time_seconds']:.1f}s")
    print(f"Avg time/problem:      {stats['avg_time_per_problem']:.1f}s")
    print("=" * 60)
    print(f"\nTo evaluate: evalplus.evaluate --dataset humaneval --samples {results_file}")


if __name__ == "__main__":
    main()
