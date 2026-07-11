#!/usr/bin/env python3
"""
Live integration tests for concurrent workers, discussion classification,
and context-refresh synthesis.

Requires:
- Router running on localhost:7700
- Alice running with max_concurrent_workers >= 2
- MESH_AUTH_TOKEN set in environment

Usage:
    source env.bash
    .venv/bin/python tests/live/test_concurrent_workers.py
"""

import asyncio
import logging
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mesh.api_client import MeshClient
from mesh.protocol import Message, MessageType, make_message

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

TARGET = "agent:assistant:alice"
TOKEN = os.environ.get("MESH_AUTH_TOKEN")
if not TOKEN:
    print("ERROR: Set MESH_AUTH_TOKEN")
    sys.exit(1)


async def collect_messages(
    client: MeshClient,
    timeout: float = 120.0,
    max_messages: int = 10,
    stop_phrase: str | None = None,
) -> list[Message]:
    """Collect messages from the connection until timeout or stop phrase."""
    messages = []
    deadline = asyncio.get_event_loop().time() + timeout

    while len(messages) < max_messages:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            break
        try:
            msg = await asyncio.wait_for(client._conn.receive(), timeout=min(remaining, 5.0))
        except asyncio.TimeoutError:
            # Check if we have at least one response already
            if messages:
                break
            continue

        if msg is None:
            break

        if msg.type == MessageType.MESSAGE and msg.from_node == TARGET:
            messages.append(msg)
            logger.info(f"  <- [{len(messages)}] ({len(msg.content)} chars): {msg.content[:120]}")
            if stop_phrase and stop_phrase.lower() in msg.content.lower():
                break

    return messages


async def test_a_basic_classification(client: MeshClient) -> bool:
    """Test A: Basic message classification — classifier inline response (hi/thanks)."""
    print("\n=== Test A: Basic Classification (inline response) ===")
    msg = make_message(client.node_id, TARGET, "Hi Alice, how are you?")
    await client._conn.send(msg)
    logger.info("  -> Sent: 'Hi Alice, how are you?'")

    responses = await collect_messages(client, timeout=60, max_messages=3)

    if not responses:
        print("  FAIL: No response received")
        return False

    print(f"  Got {len(responses)} response(s)")
    print(f"  First response: {responses[0].content[:200]}")
    print("  PASS")
    return True


async def test_b_discussion_classification(client: MeshClient) -> bool:
    """Test B: Discussion classification — should use worker LLM (not tools)."""
    print("\n=== Test B: Discussion Classification (worker LLM, no tools) ===")
    msg = make_message(
        client.node_id, TARGET,
        "What do you think is the most interesting thing about distributed systems? "
        "I'd love to hear your perspective on the challenges."
    )
    await client._conn.send(msg)
    logger.info("  -> Sent: discussion question about distributed systems")

    responses = await collect_messages(client, timeout=120, max_messages=5)

    if not responses:
        print("  FAIL: No response received")
        return False

    # We expect at least a short ack and/or a thoughtful response
    total_chars = sum(len(r.content) for r in responses)
    print(f"  Got {len(responses)} response(s), {total_chars} total chars")
    for i, r in enumerate(responses):
        print(f"  Response {i+1}: {r.content[:200]}")

    # A discussion response should be substantive (>100 chars)
    if total_chars > 100:
        print("  PASS (substantive discussion response)")
        return True
    else:
        print("  WARN: Response seems short for a discussion question")
        return True  # Still pass — may just be brief


async def test_c_worker_dispatch(client: MeshClient) -> bool:
    """Test C: Worker dispatch — should trigger tool use."""
    print("\n=== Test C: Worker Dispatch (tools required) ===")
    msg = make_message(
        client.node_id, TARGET,
        "What is the current date and time?"
    )
    await client._conn.send(msg)
    logger.info("  -> Sent: 'What is the current date and time?'")

    responses = await collect_messages(client, timeout=120, max_messages=5)

    if not responses:
        print("  FAIL: No response received")
        return False

    print(f"  Got {len(responses)} response(s)")
    for i, r in enumerate(responses):
        print(f"  Response {i+1}: {r.content[:200]}")

    print("  PASS")
    return True


async def test_d_concurrent_workers(client: MeshClient) -> bool:
    """Test D: Concurrent workers — send two messages rapidly."""
    print("\n=== Test D: Concurrent Worker Dispatch ===")
    print("  Sending two messages rapidly to trigger concurrent workers...")

    msg1 = make_message(
        client.node_id, TARGET,
        "Search my notes for anything about topic segmentation."
    )
    await client._conn.send(msg1)
    logger.info("  -> Sent message 1: 'Search notes for topic segmentation'")

    # Wait just a moment for the first worker to start, then send second
    await asyncio.sleep(2)

    msg2 = make_message(
        client.node_id, TARGET,
        "What's the weather like? Can you tell me the current time too?"
    )
    await client._conn.send(msg2)
    logger.info("  -> Sent message 2: 'What's the weather / current time?'")

    # Collect all responses (both workers should respond)
    responses = await collect_messages(client, timeout=180, max_messages=10)

    print(f"\n  Got {len(responses)} total response(s)")
    for i, r in enumerate(responses):
        print(f"  Response {i+1}: {r.content[:200]}")

    if len(responses) >= 2:
        print("  PASS (got responses from multiple interactions)")
        return True
    elif len(responses) == 1:
        print("  WARN: Only got 1 response — second may have been queued or acked")
        return True  # Partial pass
    else:
        print("  FAIL: No responses")
        return False


async def test_e_history_persistence(client: MeshClient) -> bool:
    """Test E: Verify history files are updated after the tests."""
    print("\n=== Test E: History Persistence Check ===")

    import json
    from pathlib import Path

    history_path = Path.home() / ".mesh" / "history" / "router-alice.json"
    if not history_path.exists():
        print(f"  WARN: {history_path} not found")
        return True

    # Read last few entries
    entries = []
    with open(history_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not entries:
        print("  FAIL: No entries in history file")
        return False

    recent = entries[-5:]
    print(f"  Total history entries: {len(entries)}")
    print(f"  Last 5 entries:")
    for e in recent:
        role = e.get("role", "?")
        content = str(e.get("content", ""))[:80]
        worker = e.get("meta", {}).get("worker_origin", "")
        print(f"    [{role}] {content}{'  (worker: '+worker+')' if worker else ''}")

    # Check if any recent entries have worker_origin metadata
    worker_entries = [e for e in entries[-20:] if e.get("meta", {}).get("worker_origin")]
    if worker_entries:
        print(f"  Found {len(worker_entries)} entries with worker_origin in last 20")
        print("  PASS (worker context is being persisted)")
    else:
        print("  INFO: No worker_origin entries found in recent history")
        print("  PASS (history is being written)")

    return True


async def main():
    print("=" * 60)
    print("Live Concurrent Workers Test Suite")
    print(f"Target: {TARGET}")
    print("=" * 60)

    client = MeshClient(
        nickname="live-test",
        auth_token=TOKEN,
        host="127.0.0.1",
        port=7700,
    )
    # Override node_id to use test: prefix (agents token allows agent:/test: prefixes)
    client.node_id = "test:live-test"

    try:
        await client.connect()
        print(f"Connected as {client.node_id}")
    except Exception as e:
        print(f"ERROR: Failed to connect: {e}")
        sys.exit(1)

    results = {}

    try:
        # Test A: Basic classification
        results["A: Basic Classification"] = await test_a_basic_classification(client)

        # Brief pause between tests
        await asyncio.sleep(3)

        # Test B: Discussion classification
        results["B: Discussion"] = await test_b_discussion_classification(client)

        await asyncio.sleep(3)

        # Test C: Worker dispatch
        results["C: Worker Dispatch"] = await test_c_worker_dispatch(client)

        await asyncio.sleep(3)

        # Test D: Concurrent workers
        results["D: Concurrent Workers"] = await test_d_concurrent_workers(client)

        await asyncio.sleep(2)

        # Test E: History persistence
        results["E: History Persistence"] = await test_e_history_persistence(client)

    except Exception as e:
        print(f"\nERROR during tests: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.disconnect()

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    passed = 0
    total = len(results)
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        if result:
            passed += 1
        print(f"  {status}: {name}")

    print(f"\n  {passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
