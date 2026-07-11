#!/usr/bin/env python3
"""
EvalPlus baseline runner for direct API comparison.

Runs HumanEval+ problems directly against OpenAI models (bypassing mesh)
to establish baseline performance for comparison with agentic approaches.

Usage:
    # Run 5 problems with GPT-4o
    python -m tests.live.run_evalplus_baseline --model gpt-4o --problems 5

    # Run with o3-mini (reasoning model)
    python -m tests.live.run_evalplus_baseline --model o3-mini --problems 30

    # Full run
    python -m tests.live.run_evalplus_baseline --model gpt-4o --problems 164

    # Evaluate results
    evalplus.evaluate --dataset humaneval --samples tests/live/evalplus_results/baseline_*.jsonl
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

from tests.live.evalplus_provider import DirectOpenAIDecoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HumanEval+ baseline benchmark with direct OpenAI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model selection
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key (default: from OPENAI_API_KEY env)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        help="Custom base URL for OpenAI-compatible APIs",
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

    # Generation options
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for greedy)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens in response (default: 2048)",
    )

    # Cost control
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
        help="Save raw model responses for debugging",
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

    if not args.api_key:
        logger.error("OpenAI API key not provided. Set OPENAI_API_KEY or use --api-key")
        sys.exit(1)

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_log_dir = None
    if args.save_raw:
        raw_log_dir = output_dir / "raw_baseline"
        raw_log_dir.mkdir(exist_ok=True)

    # Select problems
    problems = select_problems(args)
    logger.info(f"Running {len(problems)} problems against {args.model} (direct API)")

    # Create decoder
    decoder = DirectOpenAIDecoder(
        name=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        raw_log_dir=str(raw_log_dir) if raw_log_dir else None,
    )

    # Run timestamp for output files
    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    model_safe = args.model.replace("/", "-").replace(":", "-")
    results_file = output_dir / f"baseline_{model_safe}_{run_id}.jsonl"
    summary_file = output_dir / f"baseline_{model_safe}_{run_id}_summary.json"

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
        "type": "baseline",
        "model": args.model,
        "temperature": args.temperature,
        "problems_attempted": len(results),
        "problems_requested": len(problems),
        **stats,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written to {summary_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("EvalPlus Baseline Run Summary")
    print("=" * 60)
    print(f"Model:                 {args.model}")
    print(f"Type:                  Direct OpenAI API (baseline)")
    print(f"Problems attempted:    {len(results)}")
    print(f"Extraction success:    {stats['total_problems'] - stats['extraction_failures']}/{stats['total_problems']}")
    print(f"Total time:            {stats['total_time_seconds']:.1f}s")
    print(f"Avg time/problem:      {stats['avg_time_per_problem']:.1f}s")
    print(f"Total tokens:          {stats['total_tokens']}")
    print("=" * 60)
    print(f"\nTo evaluate: evalplus.evaluate --dataset humaneval --samples {results_file}")


if __name__ == "__main__":
    main()
