#!/usr/bin/env python3
"""
Run an evaluator agent for a specific model.

This is started as a subprocess by the orchestrator.

Usage:
    python -m mesh.eval.run_evaluator --model gpt51
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Setup logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run model evaluator agent")
    parser.add_argument("--model", "-m", required=True, help="Model name (e.g., gpt51, deepseek)")
    parser.add_argument("--router-host", default="127.0.0.1", help="Eval router host")
    parser.add_argument("--router-port", type=int, default=9999, help="Eval router port")
    parser.add_argument("--sandbox-base", type=Path, default=Path("/tmp/mesh_eval"), help="Base sandbox directory")
    parser.add_argument("--max-iterations", type=int, default=50, help="Max tool iterations per task")
    args = parser.parse_args()

    logger.info(f"Starting evaluator for model: {args.model}")

    from mesh.eval.evaluator_agent import run_evaluator

    asyncio.run(run_evaluator(
        model_name=args.model,
        sandbox_base=args.sandbox_base,
        router_host=args.router_host,
        router_port=args.router_port,
        max_iterations=args.max_iterations,
    ))


if __name__ == "__main__":
    main()
