#!/usr/bin/env python3
"""
LiveCodeBench baseline runner for direct API comparison.

Runs LiveCodeBench problems directly against OpenAI/Synthetic models
(bypassing mesh) for baseline comparison with agentic approaches.

Usage:
    # Run 10 easy problems with GPT-4o
    python -m tests.live.run_livecodebench_baseline --model gpt-4o --problems 10 --difficulty easy

    # Run with Synthetic backend (GLM-4.7)
    python -m tests.live.run_livecodebench_baseline \
        --model "hf:zai-org/GLM-4.7" --backend synthetic --problems 50

    # Full run with date filtering
    python -m tests.live.run_livecodebench_baseline \
        --model gpt-4o --problems 1055 --start-date 2024-07-01
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.live.livecodebench_provider import (
    load_problems,
    build_prompt,
    extract_code,
    evaluate_solution,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LiveCodeBench baseline with direct API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model selection
    parser.add_argument(
        "--model", "-m", type=str, default="gpt-4o",
        help="Model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--backend", type=str, default="openai",
        choices=["openai", "synthetic"],
        help="API backend (default: openai)",
    )
    parser.add_argument(
        "--api-key", type=str,
        help="API key (default: from env OPENAI_API_KEY or SYNTHETIC_API_KEY)",
    )
    parser.add_argument(
        "--base-url", type=str,
        help="Custom base URL for OpenAI-compatible APIs",
    )

    # Problem selection
    parser.add_argument(
        "--problems", "-n", type=int, default=10,
        help="Number of problems to run (default: 10)",
    )
    parser.add_argument(
        "--difficulty", type=str, choices=["easy", "medium", "hard"],
        help="Filter by difficulty",
    )
    parser.add_argument(
        "--platform", type=str, choices=["leetcode", "codeforces", "atcoder"],
        help="Filter by platform",
    )
    parser.add_argument(
        "--start-date", type=str,
        help="Only problems after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date", type=str,
        help="Only problems before this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--version", type=str, default="release_v6",
        help="Dataset version tag (default: release_v6)",
    )

    # Generation options
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=16384,
        help="Max tokens in response (default: 16384)",
    )
    parser.add_argument(
        "--reasoning-effort", type=str, default="medium",
        choices=["none", "low", "medium", "high"],
        help="Reasoning effort level (default: medium)",
    )

    # Evaluation options
    parser.add_argument(
        "--eval-timeout", type=int, default=10,
        help="Per-test-case evaluation timeout in seconds (default: 10)",
    )

    # Output options
    parser.add_argument(
        "--output", "-o", type=str, default="tests/live/livecodebench_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--save-raw", action="store_true",
        help="Save raw model responses for debugging",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def generate_openai(model, prompt, api_key, base_url, temperature, max_tokens, reasoning_effort=None):
    """Generate via OpenAI API."""
    import openai
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    is_reasoning = model.startswith(("o1", "o3", "gpt-5")) or reasoning_effort
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}

    if is_reasoning:
        kwargs["max_completion_tokens"] = max_tokens
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    tokens = response.usage.total_tokens if response.usage else 0
    return content, tokens


def generate_synthetic(model, prompt, api_key, base_url, max_tokens, reasoning_effort=None):
    """Generate via Synthetic OpenAI-compatible API."""
    import openai
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        # If reasoning_effort was rejected, retry without it
        error_str = str(e).lower()
        if reasoning_effort and ("reasoning" in error_str or "non-reasoning" in error_str):
            del kwargs["reasoning_effort"]
            response = client.chat.completions.create(**kwargs)
        else:
            raise

    content = response.choices[0].message.content or ""
    tokens = response.usage.total_tokens if response.usage else 0
    return content, tokens


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve API key
    if args.backend == "synthetic":
        api_key = args.api_key or os.environ.get("SYNTHETIC_API_KEY")
        base_url = args.base_url or "https://api.synthetic.new/openai/v1"
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        base_url = args.base_url

    if not api_key:
        key_env = "SYNTHETIC_API_KEY" if args.backend == "synthetic" else "OPENAI_API_KEY"
        logger.error(f"API key not provided. Set {key_env} or use --api-key")
        sys.exit(1)

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_log_dir = None
    if args.save_raw:
        raw_log_dir = output_dir / "raw_baseline"
        raw_log_dir.mkdir(exist_ok=True)

    # Load problems
    problems = load_problems(
        version=args.version,
        max_problems=args.problems,
        difficulty=args.difficulty,
        platform=args.platform,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    logger.info(f"Running {len(problems)} problems against {args.model} (direct {args.backend} API)")

    # Run timestamp
    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    model_safe = args.model.replace("/", "-").replace(":", "-")
    results_file = output_dir / f"lcb_baseline_{model_safe}_{run_id}.jsonl"
    summary_file = output_dir / f"lcb_baseline_{model_safe}_{run_id}_summary.json"

    # Run problems
    results = []
    total_passed = 0
    total_time = 0
    total_tokens = 0

    for i, problem in enumerate(problems, 1):
        logger.info(
            f"[{i}/{len(problems)}] {problem.question_id} "
            f"({problem.difficulty.value}/{problem.platform.value})..."
        )

        start = time.time()
        prompt = build_prompt(problem, include_tests=True)

        # Generate code
        try:
            if args.backend == "synthetic":
                raw_content, tokens = generate_synthetic(
                    args.model, prompt, api_key, base_url,
                    args.max_tokens, reasoning_effort=args.reasoning_effort,
                )
            else:
                raw_content, tokens = generate_openai(
                    args.model, prompt, api_key, base_url,
                    args.temperature, args.max_tokens,
                    reasoning_effort=args.reasoning_effort,
                )
            total_tokens += tokens
        except Exception as e:
            logger.error(f"API error: {e}")
            raw_content = ""

        elapsed = time.time() - start

        # Save raw response
        if raw_log_dir:
            raw_path = raw_log_dir / f"{problem.question_id}.txt"
            raw_path.write_text(raw_content)

        # Extract code
        code = extract_code(raw_content, problem)

        # Evaluate
        if code:
            eval_result = evaluate_solution(problem, code, timeout=args.eval_timeout)
        else:
            eval_result = {
                "passed": False, "num_tests": 0, "num_passed": 0,
                "metadata": {"error": "Code extraction failed"},
            }

        total_time += elapsed

        result = {
            "question_id": problem.question_id,
            "code": code,
            "passed": eval_result["passed"],
            "num_tests": eval_result["num_tests"],
            "num_passed": eval_result["num_passed"],
            "metadata": eval_result.get("metadata", {}),
            "time_seconds": round(elapsed, 1),
            "difficulty": problem.difficulty.value,
            "platform": problem.platform.value,
        }
        results.append(result)

        if result["passed"]:
            total_passed += 1

        status = "PASS" if result["passed"] else "FAIL"
        detail = result.get("error", f"{result['num_passed']}/{result['num_tests']} tests")
        if not code:
            detail = "extraction failed"
        logger.info(f"  [{status}] {problem.question_id}: {detail} ({elapsed:.1f}s)")

    # Write results JSONL
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    logger.info(f"Results written to {results_file}")

    # Compute stats
    by_difficulty = {}
    by_platform = {}
    for r in results:
        d = r["difficulty"]
        p = r["platform"]
        by_difficulty.setdefault(d, {"total": 0, "passed": 0})
        by_platform.setdefault(p, {"total": 0, "passed": 0})
        by_difficulty[d]["total"] += 1
        by_platform[p]["total"] += 1
        if r["passed"]:
            by_difficulty[d]["passed"] += 1
            by_platform[p]["passed"] += 1

    summary = {
        "run_id": run_id,
        "type": "baseline",
        "model": args.model,
        "backend": args.backend,
        "dataset_version": args.version,
        "temperature": args.temperature,
        "problems_attempted": len(results),
        "problems_passed": total_passed,
        "pass_rate": round(total_passed / max(len(results), 1), 4),
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_problem": round(total_time / max(len(results), 1), 2),
        "total_tokens": total_tokens,
        "by_difficulty": by_difficulty,
        "by_platform": by_platform,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written to {summary_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("LiveCodeBench Baseline Run Summary")
    print("=" * 60)
    print(f"Model:                 {args.model}")
    print(f"Backend:               {args.backend}")
    print(f"Problems attempted:    {len(results)}")
    print(f"Problems passed:       {total_passed}/{len(results)} ({summary['pass_rate']:.1%})")
    print(f"Total time:            {total_time:.1f}s")
    print(f"Avg time/problem:      {summary['avg_time_per_problem']:.1f}s")
    print(f"Total tokens:          {total_tokens}")
    print()
    print("By Difficulty:")
    for d, stats in sorted(by_difficulty.items()):
        rate = stats["passed"] / max(stats["total"], 1)
        print(f"  {d:8s}: {stats['passed']}/{stats['total']} ({rate:.1%})")
    print()
    print("By Platform:")
    for p, stats in sorted(by_platform.items()):
        rate = stats["passed"] / max(stats["total"], 1)
        print(f"  {p:12s}: {stats['passed']}/{stats['total']} ({rate:.1%})")
    print("=" * 60)


if __name__ == "__main__":
    main()
