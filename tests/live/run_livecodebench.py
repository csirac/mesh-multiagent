#!/usr/bin/env python3
"""
LiveCodeBench benchmark runner for mesh agents.

Runs LiveCodeBench problems through mesh agents and evaluates solutions
locally using sandboxed subprocess execution.

Usage:
    # Run 10 easy problems
    python -m tests.live.run_livecodebench --problems 10 --difficulty easy

    # Full run with specific agent
    python -m tests.live.run_livecodebench --agent agent:evalplus:glm47-v02 --problems 164

    # Agentic mode (includes public test cases in prompt)
    python -m tests.live.run_livecodebench --agent agent:evalplus:glm47-agentic --prompt-mode agentic

    # Filter by date (post-cutoff problems only)
    python -m tests.live.run_livecodebench --start-date 2024-07-01
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.live.livecodebench_provider import (
    load_problems,
    build_prompt,
    build_agentic_prompt,
    extract_code,
    evaluate_solution,
)

try:
    from mesh.api_client import MeshClient
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from mesh.api_client import MeshClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LiveCodeBench benchmark against mesh agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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

    # Agent configuration
    parser.add_argument(
        "--agent", type=str, default="agent:assistant:v02",
        help="Agent node ID to use",
    )
    parser.add_argument(
        "--ws-url", type=str,
        default=os.environ.get("MESH_WS_URL", ""),
        help="WebSocket URL for mesh router",
    )
    parser.add_argument(
        "--auth-token", type=str,
        default=os.environ.get("MESH_AUTH_TOKEN"),
        help="Mesh authentication token",
    )

    # Prompt mode
    parser.add_argument(
        "--prompt-mode", type=str, default="simple",
        choices=["simple", "agentic"],
        help="Prompt mode: simple (no tests) or agentic (includes public tests + tool use)",
    )

    # Execution options
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Timeout per problem in seconds (default: 300)",
    )
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
        help="Save raw agent responses for debugging",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


async def run_problem(
    client: MeshClient,
    agent: str,
    problem,
    prompt_mode: str,
    timeout: float,
    eval_timeout: int,
    raw_log_dir: Path | None = None,
) -> dict:
    """Run a single problem through the mesh agent and evaluate."""
    # Build prompt
    if prompt_mode == "agentic":
        prompt = build_agentic_prompt(problem)
    else:
        prompt = build_prompt(problem, include_tests=True)

    # Reset agent context
    await client.reset_context(
        agent,
        reason=f"Starting new problem: {problem.question_id}",
    )

    # Send to agent and wait for response
    response = await client.send(
        agent,
        prompt,
        wait_response=True,
        timeout=timeout,
    )

    if response is None:
        return {
            "question_id": problem.question_id,
            "code": "",
            "passed": False,
            "error": "No response from agent",
            "num_tests": 0,
            "num_passed": 0,
        }

    raw_content = response.content if isinstance(response.content, str) else str(response.content)

    # Save raw response
    if raw_log_dir:
        raw_path = raw_log_dir / f"{problem.question_id}.txt"
        raw_path.write_text(raw_content)

    # Extract code
    code = extract_code(raw_content, problem)

    if not code:
        return {
            "question_id": problem.question_id,
            "code": "",
            "passed": False,
            "error": "Code extraction failed",
            "num_tests": 0,
            "num_passed": 0,
        }

    # Evaluate
    eval_result = evaluate_solution(problem, code, timeout=eval_timeout)

    return {
        "question_id": problem.question_id,
        "code": code,
        "passed": eval_result["passed"],
        "num_tests": eval_result["num_tests"],
        "num_passed": eval_result["num_passed"],
        "metadata": eval_result.get("metadata", {}),
    }


async def main_async():
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

    # Load problems
    problems = load_problems(
        version=args.version,
        max_problems=args.problems,
        difficulty=args.difficulty,
        platform=args.platform,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    logger.info(f"Running {len(problems)} problems against {args.agent}")

    # Run timestamp
    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    results_file = output_dir / f"lcb_run_{run_id}.jsonl"
    summary_file = output_dir / f"lcb_run_{run_id}_summary.json"

    # Run problems
    results = []
    total_passed = 0
    total_time = 0

    async with MeshClient(
        nickname="lcb-runner-0",
        ws_url=args.ws_url,
        auth_token=args.auth_token,
    ) as client:
        for i, problem in enumerate(problems, 1):
            logger.info(
                f"[{i}/{len(problems)}] {problem.question_id} "
                f"({problem.difficulty.value}/{problem.platform.value})..."
            )

            start = datetime.now()
            result = await run_problem(
                client, args.agent, problem,
                args.prompt_mode, args.timeout, args.eval_timeout,
                raw_log_dir,
            )
            elapsed = (datetime.now() - start).total_seconds()
            total_time += elapsed

            result["time_seconds"] = round(elapsed, 1)
            result["difficulty"] = problem.difficulty.value
            result["platform"] = problem.platform.value
            results.append(result)

            if result["passed"]:
                total_passed += 1

            status = "PASS" if result["passed"] else "FAIL"
            detail = result.get("error", f"{result['num_passed']}/{result['num_tests']} tests")
            logger.info(f"  [{status}] {problem.question_id}: {detail} ({elapsed:.1f}s)")

    # Write results JSONL
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    logger.info(f"Results written to {results_file}")

    # Compute stats by difficulty and platform
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
        "agent": args.agent,
        "prompt_mode": args.prompt_mode,
        "dataset_version": args.version,
        "problems_attempted": len(results),
        "problems_passed": total_passed,
        "pass_rate": round(total_passed / max(len(results), 1), 4),
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_problem": round(total_time / max(len(results), 1), 2),
        "by_difficulty": by_difficulty,
        "by_platform": by_platform,
        "filters": {
            "difficulty": args.difficulty,
            "platform": args.platform,
            "start_date": args.start_date,
            "end_date": args.end_date,
        },
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written to {summary_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("LiveCodeBench Run Summary")
    print("=" * 60)
    print(f"Agent:                 {args.agent}")
    print(f"Prompt mode:           {args.prompt_mode}")
    print(f"Problems attempted:    {len(results)}")
    print(f"Problems passed:       {total_passed}/{len(results)} ({summary['pass_rate']:.1%})")
    print(f"Total time:            {total_time:.1f}s")
    print(f"Avg time/problem:      {summary['avg_time_per_problem']:.1f}s")
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


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
