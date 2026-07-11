#!/usr/bin/env python3
"""
RouterV2 LLM Classification Test Suite

Tests the LLM-based classification in RouterV2 with expanded test cases.
Backend: gpt-5.1 with reasoning_effort: medium
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass

# Add mesh to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mesh.api_client import MeshClient


@dataclass
class TestCase:
    """A test case for RouterV2 classification."""
    message: str
    expected_worker: bool   # True = needs worker, False = router can handle
    category: str           # Category for grouping results
    expected_response: bool = True  # False = should NOT respond (relevance filter)


# Comprehensive test cases
TEST_CASES = [
    # ==================== SIMPLE QUESTIONS (should NOT need worker) ====================
    TestCase("Hello!", False, "greeting"),
    TestCase("Hi there, how are you?", False, "greeting"),
    TestCase("Good morning!", False, "greeting"),
    TestCase("Thanks!", False, "thanks"),
    TestCase("Thank you for your help.", False, "thanks"),
    TestCase("That's perfect, thanks!", False, "thanks"),
    TestCase("What is the capital of France?", False, "factual"),
    TestCase("What is 2 + 2?", False, "factual"),
    TestCase("Who wrote Romeo and Juliet?", False, "factual"),
    TestCase("What year did World War II end?", False, "factual"),
    TestCase("What is the largest planet in our solar system?", False, "factual"),
    TestCase("What does HTTP stand for?", False, "factual"),
    TestCase("What is Python?", False, "factual"),
    TestCase("What's the difference between a list and a tuple?", False, "factual"),
    TestCase("Can you explain what an API is?", False, "explanation"),
    TestCase("What is machine learning in simple terms?", False, "explanation"),
    TestCase("How does a binary search work?", False, "explanation"),
    TestCase("Yes", False, "short"),
    TestCase("No", False, "short"),
    TestCase("OK", False, "short"),
    TestCase("Got it", False, "short"),

    # ==================== COMPLEX TASKS (SHOULD need worker) ====================
    # File operations
    TestCase("Read the file at /etc/passwd", True, "file-read"),
    TestCase("Show me the contents of mesh.yaml", True, "file-read"),
    TestCase("What's in the README?", True, "file-read"),
    TestCase("Can you check what's in my config file?", True, "file-read"),

    # Command execution
    TestCase("Run ls -la in the current directory", True, "bash"),
    TestCase("Check disk usage with df -h", True, "bash"),
    TestCase("Show me the running processes", True, "bash"),
    TestCase("What's my current directory?", True, "bash"),
    TestCase("Execute 'echo hello'", True, "bash"),

    # Multi-step / Research tasks
    TestCase("Analyze the codebase and tell me about the architecture", True, "research"),
    TestCase("Find all Python files that import asyncio", True, "research"),
    TestCase("Search for TODO comments in the code", True, "research"),
    TestCase("Compare the performance of these two functions", True, "research"),

    # Code writing
    TestCase("Write a Python function to calculate factorial", True, "code"),
    TestCase("Create a bash script to backup my home directory", True, "code"),
    TestCase("Implement a binary search tree in Python", True, "code"),
    TestCase("Help me refactor this function to be more efficient", True, "code"),

    # Status queries (edge case - depends on context)
    TestCase("What are you working on?", False, "status"),
    TestCase("Are you busy?", False, "status"),
    TestCase("How's the task going?", False, "status"),

    # ==================== NO RESPONSE EXPECTED (relevance filter) ====================
    # Channel messages not directed at the agent
    TestCase("Hey everyone, what's up?", False, "no-response", expected_response=False),
    TestCase("alice: can you check the logs?", False, "no-response", expected_response=False),
    TestCase("@alice look at this", False, "no-response", expected_response=False),
    TestCase("I just pushed some changes", False, "no-response", expected_response=False),
    TestCase("lol that's funny", False, "no-response", expected_response=False),

    # Other agent status updates
    TestCase("[agent:coder:alice] Starting work on task...", False, "no-response", expected_response=False),
    TestCase("[INFO] Build completed successfully", False, "no-response", expected_response=False),

    # ==================== DIRECT MENTIONS (should respond) ====================
    TestCase("Hey bob, what's up?", False, "direct-mention"),
    TestCase("@bob can you help?", False, "direct-mention"),
    TestCase("bob: check the logs please", True, "direct-mention"),
    TestCase("Bob, run the tests", True, "direct-mention"),
]


async def run_classification_test(
    client: MeshClient, agent_node: str, test_case: TestCase, channel_name: str | None = None
) -> dict:
    """Run a single classification test and return results."""
    start = time.time()

    # For no-response tests, use shorter timeout since we expect no reply
    timeout = 5.0 if not test_case.expected_response else 30.0

    # Determine target: channel for no-response tests, direct for others
    target = f"channel:{channel_name}" if (not test_case.expected_response and channel_name) else agent_node

    try:
        if not test_case.expected_response:
            # For no-response tests: send to channel without waiting, then listen
            # for any message from the agent within the timeout window
            await client.send(to=target, content=test_case.message, wait_response=False)
            # Listen for any response from the agent
            response = None
            deadline = time.time() + timeout
            while time.time() < deadline:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    msg = await asyncio.wait_for(client._conn.receive(), timeout=remaining)
                    if msg and hasattr(msg, 'from_node') and msg.from_node == agent_node:
                        response = msg
                        break
                    # Ignore other messages (echoes, presence, etc.)
                except asyncio.TimeoutError:
                    break
        else:
            # For response tests: send directly and wait for response
            response = await client.send(
                to=target,
                content=test_case.message,
                wait_response=True,
                timeout=timeout,
            )

        elapsed = time.time() - start

        response_content = ""
        got_response = False
        if response:
            got_response = True
            if isinstance(response.content, str):
                response_content = response.content
            elif isinstance(response.content, dict):
                response_content = response.content.get("text", str(response.content))

        # Determine if test passed
        # For no-response tests: success = no response received
        # For response tests: success = got a response
        if test_case.expected_response:
            passed = got_response
        else:
            passed = not got_response

        return {
            "message": test_case.message,
            "category": test_case.category,
            "expected_worker": test_case.expected_worker,
            "expected_response": test_case.expected_response,
            "got_response": got_response,
            "response_preview": response_content[:200] if response_content else "(no response)",
            "latency_ms": int(elapsed * 1000),
            "passed": passed,
            "success": True,  # Test ran without error
        }
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        # Timeout is expected for no-response tests
        if not test_case.expected_response:
            return {
                "message": test_case.message,
                "category": test_case.category,
                "expected_worker": test_case.expected_worker,
                "expected_response": test_case.expected_response,
                "got_response": False,
                "response_preview": "(timeout - expected)",
                "latency_ms": int(elapsed * 1000),
                "passed": True,  # No response expected, timeout is correct
                "success": True,
            }
        else:
            return {
                "message": test_case.message,
                "category": test_case.category,
                "expected_worker": test_case.expected_worker,
                "expected_response": test_case.expected_response,
                "got_response": False,
                "error": "timeout (expected response)",
                "passed": False,
                "success": False,
            }
    except Exception as e:
        return {
            "message": test_case.message,
            "category": test_case.category,
            "expected_worker": test_case.expected_worker,
            "expected_response": test_case.expected_response,
            "error": str(e),
            "passed": False,
            "success": False,
        }


async def main():
    # Load auth token
    token = os.environ.get("MESH_AUTH_TOKEN")
    if not token:
        token_file = os.path.expanduser("~/.config/mesh/auth_token")
        if os.path.exists(token_file):
            token = open(token_file).read().strip()

    if not token:
        print("ERROR: No auth token found. Set MESH_AUTH_TOKEN or create ~/.config/mesh/auth_token")
        return 1

    agent_node = "agent:test:routerv2-llm"
    test_channel = "test-routerv2"  # Channel for relevance testing

    print(f"=" * 70)
    print(f"RouterV2 LLM Classification Test Suite")
    print(f"Agent: {agent_node}")
    print(f"Test Channel: channel:{test_channel}")
    print(f"Backend: claude-code-opus")
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"=" * 70)
    print()

    results = []
    categories = {}

    # Reorder tests: run no-response (channel) tests first on a fresh connection
    # to avoid stale messages from direct tests polluting the results
    no_response_tests = [tc for tc in TEST_CASES if not tc.expected_response]
    response_tests = [tc for tc in TEST_CASES if tc.expected_response]
    ordered_tests = no_response_tests + response_tests

    async with MeshClient(
        nickname="test-runner",
        auth_token=token,
        host="127.0.0.1",
        port=7700,
    ) as client:
        # Create and join the test channel for relevance testing
        print(f"Setting up test channel: {test_channel}")
        created = await client.create_channel(test_channel, "RouterV2 test channel")
        if created:
            print(f"  Created channel:{test_channel}")
        else:
            print(f"  Channel already exists")

        joined = await client.join_channel(test_channel)
        if joined:
            print(f"  Joined channel:{test_channel}")
        else:
            print(f"  Already in channel:{test_channel}")

        # Reset agent context to start clean
        print(f"  Resetting agent context...")
        await client.send(to=agent_node, content="", wait_response=False)
        await asyncio.sleep(1.0)
        # Drain any stale messages
        try:
            while True:
                await asyncio.wait_for(client._conn.receive(), timeout=0.5)
        except (asyncio.TimeoutError, Exception):
            pass
        print()

        for i, test_case in enumerate(ordered_tests):
            print(f"[{i+1}/{len(ordered_tests)}] Testing: {test_case.message[:50]}...")
            result = await run_classification_test(client, agent_node, test_case, test_channel)
            results.append(result)

            # Track by category
            if test_case.category not in categories:
                categories[test_case.category] = {"total": 0, "passed": 0}
            categories[test_case.category]["total"] += 1
            if result.get("passed", False):
                categories[test_case.category]["passed"] += 1

            # Brief output
            status = "PASS" if result.get("passed", False) else "FAIL"
            latency = result.get("latency_ms", "?")
            resp_status = "responded" if result.get("got_response") else "no response"
            expected_resp = "expect response" if test_case.expected_response else "expect silence"
            print(f"       [{status}] {latency}ms - {resp_status} ({expected_resp})")

            # Reasonable delay between requests to avoid overwhelming the agent
            # and to let it complete processing before next request
            await asyncio.sleep(2.0)

    print()
    print("=" * 70)
    print("RESULTS BY CATEGORY")
    print("=" * 70)
    total_passed = 0
    total_tests = 0
    for cat, stats in sorted(categories.items()):
        rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        total_passed += stats["passed"]
        total_tests += stats["total"]
        print(f"  {cat:15s}: {stats['passed']}/{stats['total']} ({rate:.0f}%)")
    print(f"  {'TOTAL':15s}: {total_passed}/{total_tests} ({total_passed/total_tests*100:.0f}%)")

    print()
    print("=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    for r in results:
        status = "PASS" if r.get("passed") else "FAIL"
        resp = "got_resp" if r.get("got_response") else "no_resp"
        exp_resp = "exp_resp" if r.get("expected_response", True) else "exp_silent"
        if r.get("success"):
            print(f"[{status}] [{r['category']:12s}] {r['message'][:40]:40s} -> {r['latency_ms']:4d}ms ({resp}, {exp_resp})")
        else:
            print(f"[{status}] [{r['category']:12s}] {r['message'][:40]:40s} -> ERROR: {r.get('error', 'unknown')}")

    # Save full results
    results_file = "/tmp/routerv2_test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {results_file}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
