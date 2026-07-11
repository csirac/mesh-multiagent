#!/usr/bin/env python3
"""
Run full HumanEval benchmark with live progress reporting.

Usage:
    # Run both baselines (direct + passthrough-simple)
    python tests/live/run_full_benchmark.py --baseline both

    # Run only direct decode
    python tests/live/run_full_benchmark.py --baseline direct

    # Run only passthrough simple
    python tests/live/run_full_benchmark.py --baseline passthrough

    # Run passthrough agentic (with tools)
    python tests/live/run_full_benchmark.py --baseline agentic

    # Run v02 (phase-flow-v02 controller)
    python tests/live/run_full_benchmark.py --baseline v02 --agent agent:evalplus:glm47-v02 --model "hf:zai-org/GLM-4.7" --timeout 180

    # Limit number of problems
    python tests/live/run_full_benchmark.py --baseline direct --problems 30

    # Use a different model
    python tests/live/run_full_benchmark.py --baseline direct --model gpt-4o

Results are printed live and appended to tests/live/evalplus_results/benchmark_log.txt
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evalplus.data import get_human_eval_plus
from tests.live.evalplus_provider import DirectOpenAIDecoder, MeshDecoder, SyntheticDecoder


def log(msg: str, log_file: Path):
    """Print and append to log file."""
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")


async def run_baseline(name: str, decoder, problems: dict, output_file: Path, log_file: Path):
    """Run a baseline and print progress with detailed failure logging."""
    results = []
    passed = 0
    failures = []  # Track failures with details
    total = len(problems)

    log(f"\n{'='*70}", log_file)
    log(f"Running: {name} ({total} problems)", log_file)
    log(f"Started: {datetime.now().isoformat()}", log_file)
    log(f"{'='*70}\n", log_file)

    for i, (task_id, problem) in enumerate(problems.items(), 1):
        prompt = problem["prompt"]
        entry_point = problem["entry_point"]

        # Get completion
        try:
            # Use codegen_async for MeshDecoder (async), codegen for DirectOpenAI (sync)
            if hasattr(decoder, 'codegen_async'):
                completions = await decoder.codegen_async(prompt, do_sample=False, num_samples=1, task_id=task_id)
            else:
                completions = decoder.codegen(prompt, do_sample=False, num_samples=1)
            completion = completions[0] if completions else ""
        except Exception as e:
            completion = ""
            log(f"[{i}/{total}] {task_id}: ERROR generating - {e}", log_file)
            results.append({"task_id": task_id, "completion": "", "error": str(e)})
            failures.append({
                "task_id": task_id,
                "error_type": "generation",
                "error": str(e),
                "completion": "",
            })
            continue

        # Test it
        test_code = prompt + completion + "\n" + problem["test"] + f"\ncheck({entry_point})"
        try:
            exec(test_code, {})
            passed += 1
            status = "PASS"
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            status = f"FAIL ({type(e).__name__})"
            failures.append({
                "task_id": task_id,
                "error_type": "test",
                "error": error_msg,
                "completion": completion,
            })

        # Save result
        results.append({"task_id": task_id, "completion": completion})

        # Print progress
        pct = 100 * passed / i
        log(f"[{i}/{total}] {task_id}: {status}  (Running: {pct:.1f}%)", log_file)

    # Save detailed results (JSONL)
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Log failure details
    log(f"\n{'-'*70}", log_file)
    log(f"FAILURE DETAILS for {name} ({len(failures)} failures)", log_file)
    log(f"{'-'*70}", log_file)
    for fail in failures:
        log(f"\n### {fail['task_id']} ({fail['error_type']} error)", log_file)
        log(f"Error: {fail['error']}", log_file)
        # Truncate long completions for readability
        comp = fail['completion']
        if len(comp) > 800:
            comp = comp[:800] + "\n... [truncated]"
        log(f"Completion:\n```python\n{comp}\n```", log_file)

    log(f"\n{name} Final: {passed}/{total} ({100*passed/total:.1f}%)", log_file)
    log(f"Finished: {datetime.now().isoformat()}", log_file)
    log(f"Results saved to: {output_file}\n", log_file)

    return passed, total, failures


async def main():
    parser = argparse.ArgumentParser(description="Run HumanEval benchmark")
    parser.add_argument(
        "--baseline",
        choices=["direct", "passthrough", "agentic", "v02", "both"],
        default="both",
        help="Which baseline(s) to run (default: both = direct + passthrough)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--problems",
        type=int,
        default=None,
        help="Number of problems to run (default: all 164)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per problem in seconds (default: 300)"
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "synthetic"],
        default="openai",
        help="API backend: openai (default) or synthetic (for Synthetic.ai models)"
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="medium",
        help="Reasoning effort: low, medium, high (default: medium)"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Agent node ID for passthrough/agentic baselines (e.g., agent:evalplus:glm47)"
    )
    args = parser.parse_args()

    # Setup paths
    results_dir = Path("tests/live/evalplus_results")
    results_dir.mkdir(exist_ok=True)
    log_file = results_dir / "benchmark_log.txt"
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # Log header
    log(f"\n{'#'*60}", log_file)
    log(f"# BENCHMARK RUN: {timestamp}", log_file)
    log(f"# Baseline: {args.baseline}, Backend: {args.backend}, Model: {args.model}", log_file)
    if args.reasoning_effort:
        log(f"# Reasoning Effort: {args.reasoning_effort}", log_file)
    log(f"{'#'*60}", log_file)

    # Load problems
    all_problems = get_human_eval_plus()
    if args.problems:
        # Take first N problems
        problems = dict(list(all_problems.items())[:args.problems])
    else:
        problems = all_problems
    log(f"Running {len(problems)} HumanEval+ problems", log_file)

    results = {}  # Store results for comparison

    # Run direct decode baseline
    if args.baseline in ("direct", "both"):
        if args.backend == "synthetic":
            direct_decoder = SyntheticDecoder(
                name=args.model,
                api_key=os.environ.get("SYNTHETIC_API_KEY"),
                prompt_mode="simple",
                reasoning_effort=args.reasoning_effort,
            )
            backend_label = "Synthetic"
        else:
            direct_decoder = DirectOpenAIDecoder(
                name=args.model,
                api_key=os.environ["OPENAI_API_KEY"],
                prompt_mode="simple",
                reasoning_effort=args.reasoning_effort,
            )
            backend_label = "OpenAI"

        direct_passed, direct_total, direct_failures = await run_baseline(
            f"Direct Decode [{backend_label}] ({args.model})",
            direct_decoder,
            problems,
            results_dir / f"full_direct_{args.backend}_{args.model.replace(':', '_').replace('/', '_')}_{timestamp}.jsonl",
            log_file,
        )
        results["direct"] = (direct_passed, direct_total, direct_failures)

    # Run passthrough simple baseline
    if args.baseline in ("passthrough", "both"):
        agent_id = args.agent or "agent:evalplus:simple"
        mesh_decoder = MeshDecoder(
            name=agent_id,
            ws_url=os.environ.get("MESH_WS_URL", ""),
            auth_token=os.environ["MESH_AUTH_TOKEN"],
            prompt_mode="simple",
            timeout=args.timeout,
        )
        # Extract agent nickname for display
        agent_nick = agent_id.split(":")[-1] if ":" in agent_id else agent_id
        mesh_passed, mesh_total, mesh_failures = await run_baseline(
            f"Passthrough Simple [{agent_nick}] ({args.model})",
            mesh_decoder,
            problems,
            results_dir / f"full_passthrough_{agent_nick}_{args.model.replace(':', '_').replace('/', '_')}_{timestamp}.jsonl",
            log_file,
        )
        results["passthrough"] = (mesh_passed, mesh_total, mesh_failures)

    # Run passthrough agentic baseline
    if args.baseline == "agentic":
        agent_id = args.agent or "agent:evalplus:agentic"
        # Create raw log directory for agentic runs
        agent_nick = agent_id.split(":")[-1] if ":" in agent_id else agent_id
        raw_log_dir = results_dir / f"raw_agentic_{agent_nick}_{timestamp}"
        raw_log_dir.mkdir(exist_ok=True)

        agentic_decoder = MeshDecoder(
            name=agent_id,
            ws_url=os.environ.get("MESH_WS_URL", ""),
            auth_token=os.environ["MESH_AUTH_TOKEN"],
            prompt_mode="agentic",
            timeout=args.timeout or 300,
            raw_log_dir=str(raw_log_dir),  # Log raw responses for debugging
        )
        agentic_passed, agentic_total, agentic_failures = await run_baseline(
            f"Passthrough Agentic [{agent_nick}] ({args.model})",
            agentic_decoder,
            problems,
            results_dir / f"full_agentic_{agent_nick}_{args.model.replace(':', '_').replace('/', '_')}_{timestamp}.jsonl",
            log_file,
        )
        results["agentic"] = (agentic_passed, agentic_total, agentic_failures)

    # Run v02 (phase-flow-v02 controller) baseline
    if args.baseline == "v02":
        agent_id = args.agent or "agent:evalplus:v02"
        agent_nick = agent_id.split(":")[-1] if ":" in agent_id else agent_id
        raw_log_dir = results_dir / f"raw_v02_{agent_nick}_{timestamp}"
        raw_log_dir.mkdir(exist_ok=True)

        v02_decoder = MeshDecoder(
            name=agent_id,
            ws_url=os.environ.get("MESH_WS_URL", ""),
            auth_token=os.environ["MESH_AUTH_TOKEN"],
            prompt_mode="agentic",
            timeout=args.timeout or 300,
            raw_log_dir=str(raw_log_dir),
        )
        v02_passed, v02_total, v02_failures = await run_baseline(
            f"V02 Phase-Flow [{agent_nick}] ({args.model})",
            v02_decoder,
            problems,
            results_dir / f"full_v02_{agent_nick}_{args.model.replace(':', '_').replace('/', '_')}_{timestamp}.jsonl",
            log_file,
        )
        results["v02"] = (v02_passed, v02_total, v02_failures)

    # Final summary
    log(f"\n{'='*70}", log_file)
    log(f"FINAL SUMMARY - {timestamp}", log_file)
    log(f"{'='*70}", log_file)

    for name, (passed, total, failures) in results.items():
        log(f"{name.capitalize():20s}: {passed}/{total} ({100*passed/total:.1f}%)", log_file)

    # If we have both direct and passthrough, show comparison
    if "direct" in results and "passthrough" in results:
        direct_passed, direct_total, direct_failures = results["direct"]
        mesh_passed, mesh_total, mesh_failures = results["passthrough"]

        direct_failed_ids = {f["task_id"] for f in direct_failures}
        mesh_failed_ids = {f["task_id"] for f in mesh_failures}
        both_failed = direct_failed_ids & mesh_failed_ids
        only_direct_failed = direct_failed_ids - mesh_failed_ids
        only_mesh_failed = mesh_failed_ids - direct_failed_ids

        log(f"\nDelta: {mesh_passed - direct_passed} problems", log_file)
        log(f"\nFAILURE BREAKDOWN:", log_file)
        log(f"- Both failed:             {len(both_failed)} problems", log_file)
        if both_failed:
            log(f"  {sorted(both_failed)}", log_file)
        log(f"- Only Direct failed:      {len(only_direct_failed)} problems", log_file)
        if only_direct_failed:
            log(f"  {sorted(only_direct_failed)}", log_file)
        log(f"- Only Passthrough failed: {len(only_mesh_failed)} problems", log_file)
        if only_mesh_failed:
            log(f"  {sorted(only_mesh_failed)}", log_file)

    log(f"{'='*70}\n", log_file)

    # Save structured summary JSON
    summary = {
        "timestamp": timestamp,
        "model": args.model,
        "backend": args.backend,
        "num_problems": len(problems),
        "baselines": {},
    }
    for name, (passed, total, failures) in results.items():
        summary["baselines"][name] = {
            "passed": passed,
            "total": total,
            "pct": round(100 * passed / total, 1),
            "failed": [f["task_id"] for f in failures],
        }

    summary_file = results_dir / f"summary_{args.model.replace(':', '_').replace('/', '_')}_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Summary saved to: {summary_file}", log_file)


if __name__ == "__main__":
    asyncio.run(main())
