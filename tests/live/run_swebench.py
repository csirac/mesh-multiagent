#!/usr/bin/env python3
"""
SWE-bench Pro benchmark runner for mesh agents.

Runs SWE-bench Pro tasks through mesh agents in agentic mode. The agent
clones the repo, explores code, and generates a git diff patch.

Results are saved in SWE-bench prediction format (JSONL) for later
evaluation with the SWE-bench eval harness.

Usage:
    # Run 5 easy Python problems as a pilot
    python -m tests.live.run_swebench --problems 5 --language python

    # Full Python run with specific agent
    python -m tests.live.run_swebench --agent agent:swebench:gpt-oss-agentic --language python --problems 266

    # Filter by repo
    python -m tests.live.run_swebench --repo ansible/ansible --problems 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.live.swebench_provider import (
    load_problems,
    build_agentic_prompt,
    build_simple_prompt,
    extract_patch,
    validate_patch,
    format_prediction,
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
        description="Run SWE-bench Pro benchmark against mesh agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Problem selection
    parser.add_argument(
        "--problems", "-n", type=int, default=5,
        help="Number of problems to run (default: 5)",
    )
    parser.add_argument(
        "--language", type=str, choices=["python", "js", "go", "ts"],
        help="Filter by language",
    )
    parser.add_argument(
        "--repo", type=str,
        help="Filter by repo (e.g., 'ansible/ansible')",
    )
    parser.add_argument(
        "--difficulty", type=str, choices=["easy", "medium", "hard"],
        help="Filter by estimated difficulty",
    )
    parser.add_argument(
        "--sort", type=str, default="patch_size",
        choices=["patch_size", "instance_id", "none"],
        help="Sort order before limiting (default: patch_size = easiest first)",
    )

    # Agent configuration
    parser.add_argument(
        "--agent", type=str, default="agent:swebench:swe-gpt-oss",
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
        "--prompt-mode", type=str, default="agentic",
        choices=["simple", "agentic"],
        help="Prompt mode: agentic (clone+tools, default) or simple (reasoning only)",
    )

    # Execution options
    parser.add_argument(
        "--timeout", type=float, default=600.0,
        help="Timeout per problem in seconds (default: 600 = 10 min)",
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Clean up /tmp/swebench_repo between problems",
    )

    # Output options
    parser.add_argument(
        "--output", "-o", type=str, default="tests/live/swebench_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--save-raw", action="store_true", default=True,
        help="Save raw agent responses for debugging (default: True)",
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
    cleanup: bool,
    raw_log_dir: Path | None = None,
) -> dict:
    """Run a single SWE-bench problem through the mesh agent."""
    # Clean up any previous repo checkout
    if cleanup:
        repo_dir = Path("/tmp/swebench_repo")
        if repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)

    # Build prompt
    if prompt_mode == "agentic":
        prompt = build_agentic_prompt(problem)
    else:
        prompt = build_simple_prompt(problem)

    # Reset agent context
    await client.reset_context(
        agent,
        reason=f"Starting SWE-bench task: {problem.instance_id}",
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
            "instance_id": problem.instance_id,
            "model_patch": "",
            "passed": False,
            "error": "No response from agent",
            "patch_valid": False,
            "validation": {"valid": False, "issues": ["No response"]},
        }

    raw_content = response.content if isinstance(response.content, str) else str(response.content)

    # Save raw response
    if raw_log_dir:
        safe_name = problem.instance_id.replace("/", "_").replace(":", "_")
        # Truncate long names
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
        raw_path = raw_log_dir / f"{safe_name}.txt"
        raw_path.write_text(raw_content)

    # Extract patch from response
    patch = extract_patch(raw_content)

    # Validate patch
    validation = validate_patch(patch, problem)

    return {
        "instance_id": problem.instance_id,
        "model_patch": patch,
        "passed": validation["valid"],
        "patch_valid": validation["valid"],
        "validation": validation,
        "files_changed": validation["files_changed"],
        "lines_added": validation["lines_added"],
        "lines_removed": validation["lines_removed"],
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
        run_ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        raw_log_dir = output_dir / f"raw_{run_ts}"
        raw_log_dir.mkdir(exist_ok=True)

    # Load problems
    problems = load_problems(
        max_problems=args.problems,
        language=args.language,
        repo=args.repo,
        difficulty=args.difficulty,
        sort_by=args.sort,
    )

    if not problems:
        logger.error("No problems found matching filters")
        return

    logger.info(f"Running {len(problems)} SWE-bench Pro problems against {args.agent}")
    logger.info(f"Prompt mode: {args.prompt_mode}, timeout: {args.timeout}s")

    # Run timestamp
    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    predictions_file = output_dir / f"swe_predictions_{run_id}.jsonl"
    results_file = output_dir / f"swe_run_{run_id}.jsonl"
    summary_file = output_dir / f"swe_run_{run_id}_summary.json"

    # Run problems
    results = []
    total_valid = 0
    total_time = 0

    async with MeshClient(
        nickname="swebench-runner-0",
        ws_url=args.ws_url,
        auth_token=args.auth_token,
    ) as client:
        for i, problem in enumerate(problems, 1):
            logger.info(
                f"[{i}/{len(problems)}] {problem.instance_id} "
                f"({problem.repo_language}/{problem.repo}) "
                f"~{problem.difficulty_estimate} ({problem.patch_file_count} files, "
                f"{problem.patch_line_count} lines in gold)..."
            )

            start = datetime.now()
            result = await run_problem(
                client, args.agent, problem,
                args.prompt_mode, args.timeout, args.cleanup,
                raw_log_dir,
            )
            elapsed = (datetime.now() - start).total_seconds()
            total_time += elapsed

            result["time_seconds"] = round(elapsed, 1)
            result["repo"] = problem.repo
            result["repo_language"] = problem.repo_language
            result["difficulty_estimate"] = problem.difficulty_estimate
            result["gold_patch_files"] = problem.patch_file_count
            result["gold_patch_lines"] = problem.patch_line_count
            results.append(result)

            if result["patch_valid"]:
                total_valid += 1

            status = "VALID" if result["patch_valid"] else "INVALID"
            detail = result.get("error", "")
            if not detail:
                v = result["validation"]
                detail = f"{len(v['files_changed'])} files, +{v['lines_added']}/-{v['lines_removed']}"
                if v.get("issues"):
                    detail += f" [{'; '.join(v['issues'])}]"

            logger.info(f"  [{status}] {problem.instance_id}: {detail} ({elapsed:.1f}s)")

    # Write SWE-bench predictions JSONL (for eval harness)
    with open(predictions_file, "w") as f:
        for result in results:
            pred = format_prediction(
                result["instance_id"],
                result["model_patch"],
                model_name=args.agent,
            )
            f.write(json.dumps(pred) + "\n")
    logger.info(f"Predictions written to {predictions_file}")

    # Write detailed results JSONL
    with open(results_file, "w") as f:
        for result in results:
            # Remove nested validation dict for cleaner JSONL
            row = {k: v for k, v in result.items() if k != "validation"}
            row["validation_issues"] = result["validation"].get("issues", [])
            f.write(json.dumps(row) + "\n")
    logger.info(f"Results written to {results_file}")

    # Compute stats
    by_language = {}
    by_repo = {}
    by_difficulty = {}
    for r in results:
        lang = r["repo_language"]
        repo = r["repo"]
        diff = r["difficulty_estimate"]

        by_language.setdefault(lang, {"total": 0, "valid": 0})
        by_repo.setdefault(repo, {"total": 0, "valid": 0})
        by_difficulty.setdefault(diff, {"total": 0, "valid": 0})

        by_language[lang]["total"] += 1
        by_repo[repo]["total"] += 1
        by_difficulty[diff]["total"] += 1

        if r["patch_valid"]:
            by_language[lang]["valid"] += 1
            by_repo[repo]["valid"] += 1
            by_difficulty[diff]["valid"] += 1

    summary = {
        "run_id": run_id,
        "agent": args.agent,
        "prompt_mode": args.prompt_mode,
        "problems_attempted": len(results),
        "patches_valid": total_valid,
        "valid_rate": round(total_valid / max(len(results), 1), 4),
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_problem": round(total_time / max(len(results), 1), 2),
        "by_language": by_language,
        "by_repo": by_repo,
        "by_difficulty": by_difficulty,
        "filters": {
            "language": args.language,
            "repo": args.repo,
            "difficulty": args.difficulty,
            "max_problems": args.problems,
        },
        "note": (
            "valid_rate measures patch format validity only. "
            "Actual resolve rate requires Docker-based evaluation with "
            "the SWE-bench eval harness."
        ),
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written to {summary_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SWE-bench Pro Run Summary")
    print("=" * 70)
    print(f"Agent:                 {args.agent}")
    print(f"Prompt mode:           {args.prompt_mode}")
    print(f"Problems attempted:    {len(results)}")
    print(f"Valid patches:         {total_valid}/{len(results)} ({summary['valid_rate']:.1%})")
    print(f"Total time:            {total_time:.1f}s")
    print(f"Avg time/problem:      {summary['avg_time_per_problem']:.1f}s")
    print()
    if by_difficulty:
        print("By Estimated Difficulty:")
        for d in ["easy", "medium", "hard"]:
            if d in by_difficulty:
                stats = by_difficulty[d]
                rate = stats["valid"] / max(stats["total"], 1)
                print(f"  {d:8s}: {stats['valid']}/{stats['total']} ({rate:.1%})")
        print()
    if by_language:
        print("By Language:")
        for lang, stats in sorted(by_language.items()):
            rate = stats["valid"] / max(stats["total"], 1)
            print(f"  {lang:8s}: {stats['valid']}/{stats['total']} ({rate:.1%})")
        print()
    if by_repo and len(by_repo) > 1:
        print("By Repository:")
        for repo, stats in sorted(by_repo.items()):
            rate = stats["valid"] / max(stats["total"], 1)
            print(f"  {repo:40s}: {stats['valid']}/{stats['total']} ({rate:.1%})")
        print()
    print("NOTE: Valid patches = well-formed git diff output.")
    print("      Actual resolve rate requires Docker-based SWE-bench evaluation.")
    print(f"\nPredictions file: {predictions_file}")
    print(f"(Use with SWE-bench eval harness for actual resolve rate)")
    print("=" * 70)


def convert_predictions_for_eval(predictions_jsonl: str, output_json: str, prefix: str = "mesh-agent"):
    """Convert our JSONL predictions to the JSON array format expected by SWE-bench Pro eval."""
    predictions = []
    with open(predictions_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pred = json.loads(line)
            predictions.append({
                "instance_id": pred["instance_id"],
                "model_patch": pred.get("model_patch", ""),
                "prefix": prefix,
            })

    with open(output_json, "w") as f:
        json.dump(predictions, f, indent=2)

    logger.info(f"Converted {len(predictions)} predictions to {output_json}")
    return predictions


def generate_dataset_csv(output_path: str = "swe_bench_pro_full.csv"):
    """Generate the dataset CSV needed by the SWE-bench Pro eval harness."""
    from datasets import load_dataset
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    df = ds.to_pandas()
    df.to_csv(output_path, index=False)
    logger.info(f"Wrote {len(df)} instances to {output_path}")
    return output_path


def main():
    # Check for subcommand-style usage
    if len(sys.argv) > 1 and sys.argv[1] == "convert":
        # Convert predictions for eval harness
        parser = argparse.ArgumentParser(description="Convert predictions to eval format")
        parser.add_argument("convert", help="subcommand")
        parser.add_argument("predictions_jsonl", help="Input JSONL file")
        parser.add_argument("--output", "-o", default=None, help="Output JSON file")
        parser.add_argument("--prefix", default="mesh-agent", help="Model prefix for eval")
        args = parser.parse_args()
        output = args.output or args.predictions_jsonl.replace(".jsonl", "_eval.json")
        convert_predictions_for_eval(args.predictions_jsonl, output, args.prefix)
    elif len(sys.argv) > 1 and sys.argv[1] == "gen-csv":
        # Generate dataset CSV
        parser = argparse.ArgumentParser(description="Generate dataset CSV")
        parser.add_argument("gen_csv", help="subcommand")
        parser.add_argument("--output", "-o", default="swe_bench_pro_full.csv")
        args = parser.parse_args()
        generate_dataset_csv(args.output)
    else:
        asyncio.run(main_async())


if __name__ == "__main__":
    main()
