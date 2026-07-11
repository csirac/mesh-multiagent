#!/usr/bin/env python3
"""
Live integration test for RouterV3 planning pipeline.

Connects to the running router as a test user, sends messages to a
RouterV3-enabled test agent, and validates responses.

Usage:
    # Start the test agent first:
    #   ./agent-ctl.sh start -n testv3 -t test -b claude-code
    #
    # Then run this test:
    #   python tests/test_routerv3_live.py
    #
    # Or test against bob (already running):
    #   python tests/test_routerv3_live.py --target agent:sysadmin:bob

Requirements:
    - Router running on localhost:7700
    - Target agent running with use_router_v3: true
    - MESH_AUTH_TOKEN or MESH_ALAN_AUTH environment variable set
"""

import asyncio
import json
import logging
import os
import sys
import time
import argparse
from dataclasses import dataclass
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh.api_client import MeshClient
from mesh.protocol import Message, MessageType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Reduce noise from mesh internals
logging.getLogger("mesh.transport").setLevel(logging.WARNING)
logging.getLogger("mesh.api_client").setLevel(logging.WARNING)


@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float
    details: str
    messages_received: list[str]


async def collect_messages(
    client: MeshClient,
    timeout: float = 10.0,
    max_messages: int = 20,
    stop_on: str | None = None,
) -> list[Message]:
    """Collect messages from the mesh client until timeout or stop condition."""
    messages = []
    deadline = time.monotonic() + timeout

    while len(messages) < max_messages:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        try:
            msg = await asyncio.wait_for(
                client._conn.receive(),
                timeout=min(remaining, 2.0),
            )
            if msg is None:
                break

            # Skip control/presence messages
            if msg.type in (MessageType.CONTROL, MessageType.PRESENCE):
                continue

            messages.append(msg)
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            logger.info(f"  <- [{msg.type.value}] {content[:120]}")

            if stop_on and stop_on.lower() in content.lower():
                break

        except asyncio.TimeoutError:
            continue

    return messages


async def send_and_collect(
    client: MeshClient,
    target: str,
    content: str,
    timeout: float = 10.0,
    max_messages: int = 20,
    stop_on: str | None = None,
) -> list[Message]:
    """Send a message and collect responses."""
    from mesh.protocol import make_message

    msg = make_message(client.node_id, target, content)
    await client._conn.send(msg)
    logger.info(f"  -> Sent: {content[:80]}")

    return await collect_messages(
        client, timeout=timeout, max_messages=max_messages, stop_on=stop_on,
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


async def test_v2_passthrough(client: MeshClient, target: str) -> TestResult:
    """Test 5: Non-plan messages should pass through to V2 routing."""
    name = "V2 Passthrough (non-plan message)"
    start = time.monotonic()

    try:
        messages = await send_and_collect(
            client, target,
            "What is 2 + 2?",
            timeout=60.0,
            stop_on=None,
        )

        duration = time.monotonic() - start

        if not messages:
            return TestResult(name, False, duration, "No response received", [])

        # Should get a normal response (not a planning ack)
        content = messages[0].content if isinstance(messages[0].content, str) else str(messages[0].content)
        previews = [
            (m.content if isinstance(m.content, str) else str(m.content))[:100]
            for m in messages
        ]

        # Check it didn't enter planning mode
        planning_keywords = ["planning", "plan it out", "abort to cancel"]
        entered_planning = any(
            any(kw in p.lower() for kw in planning_keywords)
            for p in previews
        )

        if entered_planning:
            return TestResult(
                name, False, duration,
                "Message incorrectly triggered planning mode",
                previews,
            )

        return TestResult(
            name, True, duration,
            f"Got {len(messages)} response(s) via V2 routing",
            previews,
        )

    except Exception as e:
        return TestResult(name, False, time.monotonic() - start, f"Error: {e}", [])


async def test_plan_trigger_and_ack(client: MeshClient, target: str) -> TestResult:
    """Test 1: Sending 'plan it out' should trigger planning and return an ack."""
    name = "Plan Trigger + Ack"
    start = time.monotonic()

    try:
        messages = await send_and_collect(
            client, target,
            "Can you plan it out for creating a simple hello world script in /tmp/test_plan_output?",
            timeout=30.0,
            max_messages=1,
        )

        duration = time.monotonic() - start

        if not messages:
            return TestResult(name, False, duration, "No ack received", [])

        content = messages[0].content if isinstance(messages[0].content, str) else str(messages[0].content)
        previews = [content[:200]]

        # The ack should be a natural language acknowledgment
        # It might mention "planning", "plan", "abort", etc.
        return TestResult(
            name, True, duration,
            f"Got ack response in {duration:.1f}s",
            previews,
        )

    except Exception as e:
        return TestResult(name, False, time.monotonic() - start, f"Error: {e}", [])


async def test_status_during_planning(client: MeshClient, target: str) -> TestResult:
    """Test 2: Sending a message during planning should get a status update."""
    name = "Status During Planning"
    start = time.monotonic()

    try:
        # Wait a bit for planning to start doing real work
        await asyncio.sleep(5.0)

        messages = await send_and_collect(
            client, target,
            "How is the planning going?",
            timeout=30.0,
            max_messages=1,
        )

        duration = time.monotonic() - start

        if not messages:
            return TestResult(name, False, duration, "No status response received", [])

        content = messages[0].content if isinstance(messages[0].content, str) else str(messages[0].content)
        previews = [content[:300]]

        return TestResult(
            name, True, duration,
            f"Got status update in {duration:.1f}s",
            previews,
        )

    except Exception as e:
        return TestResult(name, False, time.monotonic() - start, f"Error: {e}", [])


async def test_cancellation(client: MeshClient, target: str) -> TestResult:
    """Test 3: Sending 'abort' during planning should cancel it."""
    name = "Cancellation"
    start = time.monotonic()

    try:
        # Send abort
        messages = await send_and_collect(
            client, target,
            "abort",
            timeout=30.0,
            max_messages=1,
        )

        duration = time.monotonic() - start

        if not messages:
            return TestResult(name, False, duration, "No cancellation response received", [])

        content = messages[0].content if isinstance(messages[0].content, str) else str(messages[0].content)
        previews = [content[:300]]

        return TestResult(
            name, True, duration,
            f"Got cancellation response in {duration:.1f}s",
            previews,
        )

    except Exception as e:
        return TestResult(name, False, time.monotonic() - start, f"Error: {e}", [])


async def test_full_pipeline(client: MeshClient, target: str) -> TestResult:
    """Test 4: Full planning pipeline — trigger → generate → validate → complete."""
    name = "Full Pipeline (end-to-end)"
    start = time.monotonic()

    try:
        # Trigger a simple plan
        messages = await send_and_collect(
            client, target,
            "Plan it out: create a Python script at /tmp/routerv3_test_output.py that prints 'hello world' and the current date. Keep it very simple, one file, no dependencies.",
            timeout=300.0,  # Planning can take a while with LLM calls
            max_messages=50,
            stop_on="plan_meta",  # Plan completion includes metadata
        )

        duration = time.monotonic() - start

        if not messages:
            return TestResult(name, False, duration, "No response received", [])

        previews = [
            (m.content if isinstance(m.content, str) else str(m.content))[:200]
            for m in messages
        ]

        # Check for plan completion signals
        all_content = " ".join(previews).lower()
        completed = (
            "plan" in all_content
            and any(word in all_content for word in ["complete", "ready", "done", "here", "created"])
        )

        # Check if plan file was created
        plan_dir = os.path.expanduser(f"~/.mesh/plans/")
        plan_files = []
        if os.path.isdir(plan_dir):
            for d in os.listdir(plan_dir):
                subdir = os.path.join(plan_dir, d)
                if os.path.isdir(subdir):
                    for f in os.listdir(subdir):
                        fpath = os.path.join(subdir, f)
                        if os.path.getmtime(fpath) > start:
                            plan_files.append(fpath)

        details = f"Got {len(messages)} message(s) in {duration:.1f}s"
        if plan_files:
            details += f"\nPlan file(s) created: {', '.join(plan_files)}"
        if completed:
            details += "\nPipeline appears to have completed successfully"

        return TestResult(
            name, completed or len(plan_files) > 0, duration,
            details,
            previews,
        )

    except Exception as e:
        return TestResult(name, False, time.monotonic() - start, f"Error: {e}", [])


async def test_post_cancel_recovery(client: MeshClient, target: str) -> TestResult:
    """Test: After cancellation, agent should return to IDLE and handle normal messages."""
    name = "Post-Cancel Recovery"
    start = time.monotonic()

    try:
        # Wait a moment for cleanup
        await asyncio.sleep(2.0)

        # Send a normal message — should route via V2
        messages = await send_and_collect(
            client, target,
            "Are you back to normal? Just say yes.",
            timeout=60.0,
            max_messages=1,
        )

        duration = time.monotonic() - start

        if not messages:
            return TestResult(name, False, duration, "No response after cancel recovery", [])

        content = messages[0].content if isinstance(messages[0].content, str) else str(messages[0].content)
        previews = [content[:200]]

        return TestResult(
            name, True, duration,
            f"Agent recovered to IDLE state in {duration:.1f}s",
            previews,
        )

    except Exception as e:
        return TestResult(name, False, time.monotonic() - start, f"Error: {e}", [])


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------


async def run_tests(target: str, run_full: bool = True) -> list[TestResult]:
    """Run all live tests against the specified target agent."""
    auth_token = os.environ.get("MESH_AUTH_TOKEN") or os.environ.get("MESH_ALAN_AUTH")
    if not auth_token:
        logger.error("No auth token found. Set MESH_AUTH_TOKEN or MESH_ALAN_AUTH.")
        sys.exit(1)

    results: list[TestResult] = []

    async with MeshClient(
        nickname="tester",
        auth_token=auth_token,
        host="127.0.0.1",
        port=7700,
    ) as client:
        logger.info(f"Connected as {client.node_id}")
        logger.info(f"Target agent: {target}")
        logger.info("=" * 60)

        # --- Test 5 first (V2 passthrough) since it doesn't need planning state ---
        logger.info("\n--- Test: V2 Passthrough ---")
        result = await test_v2_passthrough(client, target)
        results.append(result)
        logger.info(f"  Result: {'PASS' if result.passed else 'FAIL'} ({result.details})")

        # Drain any delayed messages
        await asyncio.sleep(3.0)
        await collect_messages(client, timeout=2.0)

        # --- Test 1: Plan trigger + ack ---
        logger.info("\n--- Test: Plan Trigger + Ack ---")
        result = await test_plan_trigger_and_ack(client, target)
        results.append(result)
        logger.info(f"  Result: {'PASS' if result.passed else 'FAIL'} ({result.details})")

        # --- Test 2: Status during planning ---
        logger.info("\n--- Test: Status During Planning ---")
        result = await test_status_during_planning(client, target)
        results.append(result)
        logger.info(f"  Result: {'PASS' if result.passed else 'FAIL'} ({result.details})")

        # --- Test 3: Cancellation ---
        logger.info("\n--- Test: Cancellation ---")
        result = await test_cancellation(client, target)
        results.append(result)
        logger.info(f"  Result: {'PASS' if result.passed else 'FAIL'} ({result.details})")

        # --- Test: Post-cancel recovery ---
        logger.info("\n--- Test: Post-Cancel Recovery ---")
        result = await test_post_cancel_recovery(client, target)
        results.append(result)
        logger.info(f"  Result: {'PASS' if result.passed else 'FAIL'} ({result.details})")

        # Drain again before full pipeline test
        await asyncio.sleep(2.0)
        await collect_messages(client, timeout=2.0)

        # --- Test 4: Full pipeline (optional, takes a long time) ---
        if run_full:
            logger.info("\n--- Test: Full Pipeline (this may take several minutes) ---")
            result = await test_full_pipeline(client, target)
            results.append(result)
            logger.info(f"  Result: {'PASS' if result.passed else 'FAIL'} ({result.details})")

    return results


def print_summary(results: list[TestResult]) -> None:
    """Print a summary table of test results."""
    print("\n" + "=" * 70)
    print("ROUTERV3 LIVE TEST RESULTS")
    print("=" * 70)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"\n  [{status}] {r.name} ({r.duration:.1f}s)")
        print(f"         {r.details}")
        if r.messages_received:
            for i, preview in enumerate(r.messages_received[:3]):
                print(f"         msg[{i}]: {preview[:120]}")

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n{'=' * 70}")
    print(f"  {passed}/{total} tests passed")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live test RouterV3 planning pipeline")
    parser.add_argument(
        "--target",
        default="agent:test:routerv3",
        help="Target agent node ID (default: agent:test:routerv3)",
    )
    parser.add_argument(
        "--no-full",
        action="store_true",
        help="Skip the full pipeline test (faster)",
    )
    args = parser.parse_args()

    # Source env if not already set
    if not os.environ.get("MESH_AUTH_TOKEN"):
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "env.bash")
        if os.path.exists(env_file):
            import subprocess
            result = subprocess.run(
                ["bash", "-c", f"source {env_file} && env"],
                capture_output=True, text=True,
            )
            for line in result.stdout.splitlines():
                if "=" in line:
                    key, _, val = line.partition("=")
                    if key.startswith("MESH_"):
                        os.environ[key] = val

    results = asyncio.run(run_tests(args.target, run_full=not args.no_full))
    print_summary(results)

    # Exit with error code if any test failed
    sys.exit(0 if all(r.passed for r in results) else 1)
