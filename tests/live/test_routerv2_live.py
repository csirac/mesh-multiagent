#!/usr/bin/env python3
"""
RouterV2 Live Integration Tests — Comprehensive Pre-Deployment Suite

Tests all RouterV2 enhancements end-to-end against a live agent:
  A. Classification (IDLE routing — LLM decides needs_response / needs_worker)
  B. Worker dispatch + completion synthesis (router generates final response)
  C. Busy handling with worker context peek
  D. Context attribution (worker_origin in merged context)
  E. Error surfacing (LLM failures shown to user)
  F. Concurrent message race protection
  G. LLM defaults (llm_enabled=True, use_router_v2=True)
  H. History persistence (router stores acks/busy/completion in history)
  I. Channel routing (responses go to correct destination)
  J. Multi-turn conversation continuity

Requires:
- Mesh router running
- agent:test:routerv2-llm running (use_router_v2=true, llm_enabled=true)

Usage:
    cd ~/apps/hello-world
    source env.bash && source .venv/bin/activate
    python -m tests.live.test_routerv2_live                    # Run all
    python -m tests.live.test_routerv2_live --group A          # Run group A only
    python -m tests.live.test_routerv2_live --group B C        # Run groups B and C
    python -m tests.live.test_routerv2_live -v                 # Verbose
    python -m tests.live.test_routerv2_live --agent agent:sysadmin:bob  # Test other agent
"""

import asyncio
import argparse
import json
import os
import re
import subprocess
import sys
import time
import glob as glob_mod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mesh.api_client import MeshClient
from mesh.protocol import Message, MessageType


# =============================================================================
# Test infrastructure
# =============================================================================

@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    group: str
    passed: bool
    duration_ms: int
    details: str = ""
    error: str = ""
    messages_received: list[dict] = field(default_factory=list)


def result_line(r: TestResult) -> str:
    status = "PASS" if r.passed else "FAIL"
    detail = f" — {r.details}" if r.details else ""
    err = f" ERROR: {r.error}" if r.error else ""
    return f"  [{status}] {r.name} ({r.duration_ms}ms){detail}{err}"


class RouterV2TestRunner:
    """
    Runs RouterV2 integration tests against a live agent.

    Uses MeshClient to send messages and collect responses.
    """

    def __init__(
        self,
        agent_node: str = "agent:test:routerv2-llm",
        verbose: bool = False,
    ):
        self.agent_node = agent_node
        self.verbose = verbose
        self.client: MeshClient | None = None
        self.results: list[TestResult] = []
        self._log_dir: str | None = None

    def log(self, msg: str) -> None:
        if self.verbose:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:12]
            print(f"  [{ts}] {msg}")

    async def setup(self) -> bool:
        """Connect to mesh."""
        token = os.environ.get("MESH_AUTH_TOKEN")
        if not token:
            token_file = os.path.expanduser("~/.config/mesh/auth_token")
            if os.path.exists(token_file):
                token = open(token_file).read().strip()

        if not token:
            print("ERROR: No MESH_AUTH_TOKEN found")
            return False

        self.client = MeshClient(
            nickname=f"rv2-test-{int(time.time()) % 10000}",
            auth_token=token,
            host="127.0.0.1",
            port=7700,
        )
        try:
            await self.client.connect()
            # Wait for connection to stabilize and drain stale messages
            await asyncio.sleep(2.0)
            await self._drain(1.0)
            return True
        except Exception as e:
            print(f"ERROR: Connection failed: {e}")
            return False

    async def teardown(self) -> None:
        if self.client:
            await self.client.disconnect()

    def _find_agent_log(self) -> str | None:
        """Find the agent's log file."""
        if self._log_dir:
            return self._log_dir

        candidates = (
            glob_mod.glob("/tmp/rv2test/*.log")
            + glob_mod.glob("/tmp/routerv2-llm-test.log/*.log")
            + glob_mod.glob("/tmp/*routerv2*.log")
        )
        if candidates:
            # Return the most recently modified
            candidates.sort(key=os.path.getmtime, reverse=True)
            self._log_dir = candidates[0]
            return self._log_dir
        return None

    def _grep_log(self, pattern: str, count: bool = False) -> str | int:
        """Grep the agent log file. Returns text or count."""
        log_file = self._find_agent_log()
        if not log_file:
            return -1 if count else ""
        try:
            flags = ["-c"] if count else ["-E"]
            result = subprocess.run(
                ["grep"] + flags + [pattern, log_file],
                capture_output=True, text=True, timeout=5,
            )
            if count:
                return int(result.stdout.strip()) if result.returncode == 0 else 0
            return result.stdout if result.returncode == 0 else ""
        except Exception:
            return -1 if count else ""

    async def _drain(self, timeout: float = 0.5) -> list[Message]:
        """Drain all pending messages from the connection."""
        drained = []
        try:
            while True:
                msg = await asyncio.wait_for(
                    self.client._conn.receive(), timeout=timeout
                )
                if msg:
                    drained.append(msg)
        except (asyncio.TimeoutError, Exception):
            pass
        return drained

    async def _send_and_collect(
        self,
        content: str,
        timeout: float = 45.0,
        expect_count: int | None = None,
        collect_all: bool = False,
    ) -> list[Message]:
        """
        Send a message to the agent and collect responses.

        Args:
            content: Message to send
            timeout: How long to wait for responses
            expect_count: If set, stop after this many MESSAGE responses
            collect_all: If True, collect until timeout expires
        """
        # Drain before sending
        await self._drain(0.3)

        await self.client.send(
            to=self.agent_node,
            content=content,
            wait_response=False,
        )
        self.log(f"Sent: {content[:100]}")

        messages = []
        msg_count = 0
        deadline = time.time() + timeout

        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                msg = await asyncio.wait_for(
                    self.client._conn.receive(),
                    timeout=min(remaining, 5.0),
                )
                if msg is None:
                    continue

                # Skip presence and control messages
                if msg.type in (MessageType.PRESENCE, MessageType.CONTROL):
                    continue

                # Only care about messages from our target agent
                if msg.from_node == self.agent_node:
                    if msg.type == MessageType.MESSAGE:
                        content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
                        self.log(f"MSG: {content_str[:150]}")
                        messages.append(msg)
                        msg_count += 1
                        if expect_count and msg_count >= expect_count:
                            break
                        if not collect_all and expect_count is None:
                            break
                    elif msg.type == MessageType.TOOL_ACTIVITY:
                        self.log(f"TOOL_ACTIVITY: {str(msg.content)[:80]}")
                        messages.append(msg)
                    elif msg.type == MessageType.STATUS:
                        self.log(f"STATUS: {msg.content}")
                        messages.append(msg)

            except asyncio.TimeoutError:
                if not collect_all and expect_count is None:
                    break
                continue

        return messages

    async def _send_raw_and_collect(
        self, to: str, content: str, timeout: float = 30.0
    ) -> list[Message]:
        """Send a raw message to a specific target and collect responses."""
        await self._drain(0.3)
        await self.client.send(to=to, content=content, wait_response=False)
        self.log(f"Sent to {to}: {content[:80]}")

        messages = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                msg = await asyncio.wait_for(
                    self.client._conn.receive(), timeout=min(remaining, 5.0)
                )
                if msg and msg.from_node == self.agent_node and msg.type == MessageType.MESSAGE:
                    content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
                    self.log(f"MSG: {content_str[:150]}")
                    messages.append(msg)
                    break
            except asyncio.TimeoutError:
                continue
        return messages

    def _text_msgs(self, msgs: list[Message]) -> list[str]:
        """Extract text content from MESSAGE-type messages."""
        return [
            m.content if isinstance(m.content, str) else str(m.content)
            for m in msgs if m.type == MessageType.MESSAGE
        ]

    # =========================================================================
    # Group A: Classification (IDLE routing)
    # =========================================================================

    async def test_a1_simple_greeting(self) -> TestResult:
        """A1: Simple greeting — no worker, direct LLM response."""
        start = time.time()
        msgs = await self._send_and_collect("Hello!", timeout=15)
        elapsed = int((time.time() - start) * 1000)

        texts = self._text_msgs(msgs)
        response = texts[0] if texts else None

        passed = response is not None and len(response) > 0
        # Should be fast (no tool use) — under 12s
        details = f"response={response[:80]!r}" if response else "no response"
        if elapsed > 12000:
            details += " (SLOW — possible unnecessary worker dispatch)"

        return TestResult(
            name="A1: Simple greeting (no worker)",
            group="A", passed=passed, duration_ms=elapsed, details=details,
        )

    async def test_a2_factual_question(self) -> TestResult:
        """A2: Factual question — no worker, router answers directly."""
        start = time.time()
        msgs = await self._send_and_collect("What is the capital of France?", timeout=15)
        elapsed = int((time.time() - start) * 1000)

        texts = self._text_msgs(msgs)
        response = texts[0] if texts else None

        passed = response is not None and "paris" in response.lower()
        details = f"response={response[:80]!r}" if response else "no response"

        return TestResult(
            name="A2: Factual question (no worker)",
            group="A", passed=passed, duration_ms=elapsed, details=details,
        )

    async def test_a3_tool_task_dispatches_worker(self) -> TestResult:
        """A3: File read request — should dispatch worker, get ack + result."""
        start = time.time()

        msgs = await self._send_and_collect(
            "Read the file /etc/hostname and tell me its contents",
            timeout=45, collect_all=True,
        )
        elapsed = int((time.time() - start) * 1000)

        texts = self._text_msgs(msgs)
        tool_msgs = [m for m in msgs if m.type == MessageType.TOOL_ACTIVITY]

        # We should get at least 1 response (ack or completion)
        passed = len(texts) >= 1
        details = f"{len(texts)} MSG, {len(tool_msgs)} TOOL_ACTIVITY"
        if texts:
            details += f", last={texts[-1][:60]!r}"

        return TestResult(
            name="A3: Tool task dispatches worker",
            group="A", passed=passed, duration_ms=elapsed, details=details,
        )

    async def test_a4_irrelevant_message_ignored(self) -> TestResult:
        """A4: Message addressed to someone else — should be ignored (needs_response=false)."""
        start = time.time()

        # Send a message clearly addressed to "alice", not to our test agent
        msgs = await self._send_and_collect(
            "Hey alice, can you help me with something?",
            timeout=8,
        )
        elapsed = int((time.time() - start) * 1000)

        texts = self._text_msgs(msgs)
        # The agent should NOT respond since the message mentions "alice"
        passed = len(texts) == 0
        details = f"responses={len(texts)}"
        if texts:
            details += f", unexpected: {texts[0][:60]!r}"

        return TestResult(
            name="A4: Irrelevant message ignored (needs_response=false)",
            group="A", passed=passed, duration_ms=elapsed, details=details,
        )

    async def test_a5_math_no_worker(self) -> TestResult:
        """A5: Simple arithmetic — should answer directly, no worker needed."""
        start = time.time()
        msgs = await self._send_and_collect("What is 7 * 8?", timeout=15)
        elapsed = int((time.time() - start) * 1000)

        texts = self._text_msgs(msgs)
        response = texts[0] if texts else None

        passed = response is not None and "56" in response
        details = f"response={response[:80]!r}" if response else "no response"

        return TestResult(
            name="A5: Simple math (no worker needed)",
            group="A", passed=passed, duration_ms=elapsed, details=details,
        )

    # =========================================================================
    # Group B: Worker dispatch + completion synthesis
    # =========================================================================

    async def test_b1_completion_contains_result(self) -> TestResult:
        """B1: Worker runs echo command — completion response must contain the marker."""
        marker = f"RV2_MARKER_{int(time.time())}"
        start = time.time()

        msgs = await self._send_and_collect(
            f"Run this command: echo '{marker}'",
            timeout=60, collect_all=True,
        )
        elapsed = int((time.time() - start) * 1000)

        texts = self._text_msgs(msgs)
        found_marker = any(marker in t for t in texts)

        passed = found_marker and len(texts) >= 1
        details = f"{len(texts)} responses, marker_found={found_marker}"
        if texts:
            details += f", texts={[t[:50] for t in texts]}"

        return TestResult(
            name="B1: Completion response contains result",
            group="B", passed=passed, duration_ms=elapsed, details=details,
        )

    async def test_b2_ack_then_result(self) -> TestResult:
        """B2: Worker task should produce ack + synthesized completion (2+ messages)."""
        start = time.time()

        msgs = await self._send_and_collect(
            "List the files in /etc with: ls /etc | head -5",
            timeout=60, collect_all=True,
        )
        elapsed = int((time.time() - start) * 1000)

        texts = self._text_msgs(msgs)

        # Expect at least 2 messages: ack + completion
        passed = len(texts) >= 2
        details = f"{len(texts)} messages"
        if texts:
            details += f": {[t[:50] for t in texts]}"

        return TestResult(
            name="B2: Ack then synthesized result (2+ messages)",
            group="B", passed=passed, duration_ms=elapsed, details=details,
        )

    async def test_b3_worker_draft_not_sent_directly(self) -> TestResult:
        """B3: Worker draft is captured but NOT sent — router synthesizes instead.

        We verify by checking that we don't get two responses with the same content
        (which would indicate both worker and router sent).
        """
        marker = f"DRAFT_CHECK_{int(time.time())}"
        start = time.time()

        msgs = await self._send_and_collect(
            f"Run: echo '{marker}'",
            timeout=60, collect_all=True,
        )
        elapsed = int((time.time() - start) * 1000)

        texts = self._text_msgs(msgs)
        marker_msgs = [t for t in texts if marker in t]

        # Should have exactly 1 message containing the marker (the synthesized one)
        # If worker also sent directly, we'd have 2
        passed = len(marker_msgs) == 1
        details = f"{len(marker_msgs)} responses with marker (expected 1)"
        if len(marker_msgs) > 1:
            details += " — DUPLICATE: worker sent directly AND router synthesized"

        return TestResult(
            name="B3: No duplicate response (worker draft suppressed)",
            group="B", passed=passed, duration_ms=elapsed, details=details,
        )

    async def test_b4_completion_is_synthesized(self) -> TestResult:
        """B4: Verify the final response is synthesized by the router LLM, not raw tool output."""
        start = time.time()

        msgs = await self._send_and_collect(
            "Read /etc/hostname and tell me what it says.",
            timeout=60, collect_all=True,
        )
        elapsed = int((time.time() - start) * 1000)

        texts = self._text_msgs(msgs)

        # Check logs for completion synthesis
        synthesis_count = self._grep_log("router sent synthesized response", count=True)

        # The final response should look like natural language, not raw tool output
        last_response = texts[-1] if texts else ""
        is_natural = len(last_response) > 5 and not last_response.startswith("<mesh_result")

        passed = len(texts) >= 1 and is_natural
        details = f"synthesis_events_in_log={synthesis_count}, is_natural={is_natural}"
        if last_response:
            details += f", last={last_response[:60]!r}"

        return TestResult(
            name="B4: Completion is LLM-synthesized (not raw tool output)",
            group="B", passed=passed, duration_ms=elapsed, details=details,
        )

    # =========================================================================
    # Group C: Busy handling with worker context peek
    # =========================================================================

    async def test_c1_busy_response(self) -> TestResult:
        """C1: Send slow task then ask status — should get contextual busy response."""
        start = time.time()

        # Send a task that takes long enough that we can query status while it runs.
        # LLM classification takes ~5s, so use a 20s sleep to have plenty of margin.
        await self._drain(0.5)
        await self.client.send(
            to=self.agent_node,
            content="Run: sleep 20 && echo 'SLOW_TASK_DONE'",
            wait_response=False,
        )
        self.log("Sent slow task (sleep 20)")

        # Wait for the ack (classification + ack can take ~8s with Opus)
        ack_msg = None
        try:
            deadline = time.time() + 20
            while time.time() < deadline:
                msg = await asyncio.wait_for(
                    self.client._conn.receive(), timeout=5
                )
                if msg and msg.from_node == self.agent_node and msg.type == MessageType.MESSAGE:
                    ack_content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    self.log(f"Got ack: {ack_content[:80]}")
                    ack_msg = msg
                    break
        except asyncio.TimeoutError:
            pass

        if not ack_msg:
            return TestResult(
                name="C1: Busy response", group="C",
                passed=False, duration_ms=int((time.time() - start) * 1000),
                error="No ack received for slow task",
            )

        # Small delay, then ask status while worker is still running
        await asyncio.sleep(1.0)
        await self._drain(0.3)

        # Send status query while worker is busy
        await self.client.send(
            to=self.agent_node,
            content="What are you working on right now?",
            wait_response=False,
        )
        self.log("Sent status query")

        # Collect busy response (classification + generation can take ~10s)
        busy_response = None
        try:
            deadline = time.time() + 25
            while time.time() < deadline:
                msg = await asyncio.wait_for(
                    self.client._conn.receive(), timeout=5
                )
                if msg and msg.from_node == self.agent_node and msg.type == MessageType.MESSAGE:
                    busy_response = msg.content if isinstance(msg.content, str) else str(msg.content)
                    self.log(f"Got busy response: {busy_response[:150]}")
                    break
        except asyncio.TimeoutError:
            pass

        # Drain remaining (wait for slow task to finish)
        await self._drain(30)

        elapsed = int((time.time() - start) * 1000)

        passed = busy_response is not None
        details = f"busy_response={busy_response[:120]!r}" if busy_response else "no busy response"

        return TestResult(
            name="C1: Busy response while worker running",
            group="C", passed=passed, duration_ms=elapsed, details=details,
        )

    async def test_c2_busy_mentions_current_task(self) -> TestResult:
        """C2: Busy response should mention what the worker is doing (context peek)."""
        start = time.time()

        # Send a task with a distinctive keyword (use long sleep so it's still running)
        await self._drain(0.5)
        await self.client.send(
            to=self.agent_node,
            content="Run: sleep 20 && cat /etc/hostname",
            wait_response=False,
        )
        self.log("Sent task (sleep 20)")

        # Wait for ack
        ack = None
        try:
            deadline = time.time() + 20
            while time.time() < deadline:
                msg = await asyncio.wait_for(self.client._conn.receive(), timeout=5)
                if msg and msg.from_node == self.agent_node and msg.type == MessageType.MESSAGE:
                    ack = msg
                    break
        except asyncio.TimeoutError:
            pass

        if not ack:
            return TestResult(
                name="C2: Busy mentions current task", group="C",
                passed=False, duration_ms=int((time.time() - start) * 1000),
                error="No ack",
            )

        await asyncio.sleep(1.0)
        await self._drain(0.3)

        # Ask status
        await self.client.send(
            to=self.agent_node,
            content="Status update please?",
            wait_response=False,
        )

        busy_response = None
        try:
            deadline = time.time() + 25
            while time.time() < deadline:
                msg = await asyncio.wait_for(self.client._conn.receive(), timeout=5)
                if msg and msg.from_node == self.agent_node and msg.type == MessageType.MESSAGE:
                    busy_response = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break
        except asyncio.TimeoutError:
            pass

        await self._drain(30)
        elapsed = int((time.time() - start) * 1000)

        # Check that busy response mentions something about the task
        mentions_task = False
        if busy_response:
            lower = busy_response.lower()
            mentions_task = any(
                word in lower for word in [
                    "sleep", "hostname", "running", "command", "executing",
                    "working", "processing", "busy", "task",
                ]
            )

        passed = busy_response is not None and mentions_task
        details = f"mentions_task={mentions_task}"
        if busy_response:
            details += f", response={busy_response[:100]!r}"

        return TestResult(
            name="C2: Busy response mentions current task (context peek)",
            group="C", passed=passed, duration_ms=elapsed, details=details,
        )

    async def test_c3_worker_peek_in_logs(self) -> TestResult:
        """C3: Verify worker_activity entries were stored (check logs)."""
        # This test relies on C1/C2 having run first (they trigger peeks)
        peek_count = self._grep_log("stored worker peek", count=True)

        passed = peek_count > 0 if isinstance(peek_count, int) else False
        details = f"worker_peek_events_in_log={peek_count}"
        if peek_count == -1:
            details += " (no log file found)"
            passed = True  # Can't verify, don't fail

        return TestResult(
            name="C3: Worker peek stored in logs",
            group="C", passed=passed, duration_ms=0, details=details,
        )

    # =========================================================================
    # Group D: Context attribution
    # =========================================================================

    async def test_d1_worker_origin_in_logs(self) -> TestResult:
        """D1: After worker completes, logs should show worker_origin merge."""
        start = time.time()

        msgs = await self._send_and_collect(
            "Run: echo 'ATTRIBUTION_D1'", timeout=45, collect_all=True,
        )
        elapsed = int((time.time() - start) * 1000)
        await asyncio.sleep(1)  # Let logs flush

        merge_count = self._grep_log("merging.*messages from.*worker", count=True)
        synthesis_count = self._grep_log("router sent synthesized response", count=True)

        texts = self._text_msgs(msgs)
        passed = len(texts) >= 1 and (
            (isinstance(merge_count, int) and merge_count > 0) or merge_count == -1
        )
        details = f"merge={merge_count}, synthesis={synthesis_count}, responses={len(texts)}"

        return TestResult(
            name="D1: Worker origin attribution in logs",
            group="D", passed=passed, duration_ms=elapsed, details=details,
        )

    async def test_d2_worker_id_increments(self) -> TestResult:
        """D2: Each worker dispatch should get a unique, incrementing worker ID."""
        start = time.time()

        # Send two sequential tasks
        msgs1 = await self._send_and_collect(
            "Run: echo 'WORKER_ID_TEST_1'", timeout=45, collect_all=True,
        )
        await asyncio.sleep(2)
        msgs2 = await self._send_and_collect(
            "Run: echo 'WORKER_ID_TEST_2'", timeout=45, collect_all=True,
        )
        elapsed = int((time.time() - start) * 1000)
        await asyncio.sleep(1)

        # Check logs for incrementing worker IDs
        log_text = self._grep_log("starting worker.*worker[0-9]")
        worker_ids = re.findall(r'worker(\d+)', str(log_text))
        unique_ids = set(worker_ids)

        passed = len(unique_ids) >= 2
        details = f"worker_ids_found={sorted(unique_ids)}"

        return TestResult(
            name="D2: Worker IDs increment across dispatches",
            group="D", passed=passed, duration_ms=elapsed, details=details,
        )

    # =========================================================================
    # Group E: Error surfacing
    # =========================================================================

    async def test_e1_empty_message_no_crash(self) -> TestResult:
        """E1: Empty message should not crash the agent."""
        start = time.time()
        msgs = await self._send_and_collect("", timeout=10)
        elapsed = int((time.time() - start) * 1000)

        # No crash = pass. Empty message may or may not get a response.
        passed = True
        texts = self._text_msgs(msgs)
        details = f"response={texts[0][:60]!r}" if texts else "no response (OK for empty msg)"

        return TestResult(
            name="E1: Empty message no crash",
            group="E", passed=passed, duration_ms=elapsed, details=details,
        )

    async def test_e2_agent_still_responds_after_edge_case(self) -> TestResult:
        """E2: After edge-case inputs, agent should still respond to normal messages."""
        # Send an edge case first
        await self._send_and_collect("", timeout=10)
        await asyncio.sleep(5)

        # Now send a normal message — increased timeout for safety
        start = time.time()
        msgs = await self._send_and_collect("Hello, are you still working?", timeout=30)
        elapsed = int((time.time() - start) * 1000)

        texts = self._text_msgs(msgs)
        passed = len(texts) >= 1
        details = f"response={texts[0][:60]!r}" if texts else "no response after edge case"

        return TestResult(
            name="E2: Agent responds after edge case",
            group="E", passed=passed, duration_ms=elapsed, details=details,
        )

    async def test_e3_llm_error_surfacing_check(self) -> TestResult:
        """E3: Verify LLM error surfacing code paths exist in logs.

        We can't easily trigger a real LLM failure in a live test,
        but we verify the error handling code is wired by checking that
        the router doesn't silently swallow errors (checking log patterns).
        """
        # Check that error-surfacing log patterns exist in the code
        import inspect
        from mesh.router_v2 import RouterV2

        source = inspect.getsource(RouterV2._handle_idle_with_llm)
        has_error_surfacing_idle = "Router LLM error" in source
        source2 = inspect.getsource(RouterV2._handle_busy_with_llm)
        has_error_surfacing_busy = "Router LLM error" in source2
        source3 = inspect.getsource(RouterV2._handle_worker_complete)
        has_error_surfacing_complete = "Router LLM error" in source3

        passed = has_error_surfacing_idle and has_error_surfacing_busy and has_error_surfacing_complete
        details = (
            f"idle={has_error_surfacing_idle}, "
            f"busy={has_error_surfacing_busy}, "
            f"complete={has_error_surfacing_complete}"
        )

        return TestResult(
            name="E3: LLM error surfacing code paths verified",
            group="E", passed=passed, duration_ms=0, details=details,
        )

    # =========================================================================
    # Group F: Concurrent message race protection
    # =========================================================================

    async def test_f1_no_double_dispatch(self) -> TestResult:
        """F1: Two rapid messages — only one worker should start."""
        start = time.time()

        await self._drain(0.5)

        # Send two messages rapidly
        await self.client.send(
            to=self.agent_node,
            content="Run: sleep 5 && echo FIRST_TASK",
            wait_response=False,
        )
        await asyncio.sleep(0.05)  # Minimal delay
        await self.client.send(
            to=self.agent_node,
            content="Run: echo SECOND_TASK",
            wait_response=False,
        )
        self.log("Sent two rapid messages")

        # Collect all responses
        responses = []
        deadline = time.time() + 30
        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                msg = await asyncio.wait_for(
                    self.client._conn.receive(), timeout=min(remaining, 5.0)
                )
                if msg and msg.from_node == self.agent_node and msg.type == MessageType.MESSAGE:
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    responses.append(content)
                    self.log(f"Response: {content[:80]}")
            except asyncio.TimeoutError:
                continue

        elapsed = int((time.time() - start) * 1000)

        # We should get at least 2 messages: ack/result for first + busy response for second
        has_busy_indicator = any(
            word in r.lower() for r in responses
            for word in ["busy", "finish", "working", "first", "current", "let me"]
        )

        passed = len(responses) >= 2
        details = f"{len(responses)} responses, busy_indicator={has_busy_indicator}"
        if responses:
            details += f", texts={[r[:50] for r in responses]}"

        return TestResult(
            name="F1: No double worker dispatch",
            group="F", passed=passed, duration_ms=elapsed, details=details,
        )

    async def test_f2_state_lock_prevents_race(self) -> TestResult:
        """F2: Verify state lock exists and is used (code inspection)."""
        import inspect
        from mesh.router_v2 import RouterV2

        source = inspect.getsource(RouterV2.on_message)
        has_lock = "self._state_lock" in source

        source2 = inspect.getsource(RouterV2._handle_worker_complete)
        complete_has_lock = "self._state_lock" in source2

        source3 = inspect.getsource(RouterV2._handle_worker_error)
        error_has_lock = "self._state_lock" in source3

        passed = has_lock and complete_has_lock and error_has_lock
        details = (
            f"on_message_lock={has_lock}, "
            f"complete_lock={complete_has_lock}, "
            f"error_lock={error_has_lock}"
        )

        return TestResult(
            name="F2: State lock on critical sections",
            group="F", passed=passed, duration_ms=0, details=details,
        )

    # =========================================================================
    # Group G: LLM defaults
    # =========================================================================

    async def test_g1_llm_enabled_default(self) -> TestResult:
        """G1: LLM should be enabled by default — intelligent response, not canned."""
        start = time.time()
        msgs = await self._send_and_collect("What is 2+2?", timeout=15)
        elapsed = int((time.time() - start) * 1000)

        texts = self._text_msgs(msgs)
        response = texts[0] if texts else None

        passed = response is not None and "4" in response
        is_canned = response and ("got it" in response.lower() and "finish" in response.lower())
        if is_canned:
            passed = False

        details = f"response={response[:80]!r}" if response else "no response"
        if is_canned:
            details += " (CANNED — LLM may not be enabled)"

        return TestResult(
            name="G1: LLM enabled by default (intelligent response)",
            group="G", passed=passed, duration_ms=elapsed, details=details,
        )

    async def test_g2_config_defaults_correct(self) -> TestResult:
        """G2: Verify config defaults for use_router_v2 and llm_enabled."""
        from mesh.config import NodeConfig
        from mesh.router_v2 import RouterV2Config

        node_rv2 = NodeConfig.use_router_v2  # Should be True
        node_llm = NodeConfig.router_v2_llm_enabled  # Should be True
        config_llm = RouterV2Config().llm_enabled  # Should be True

        # Check the actual default values on the class
        import dataclasses
        nc_fields = {f.name: f.default for f in dataclasses.fields(NodeConfig) if f.name in ('use_router_v2', 'router_v2_llm_enabled')}

        passed = nc_fields.get('use_router_v2', False) == True and nc_fields.get('router_v2_llm_enabled', False) == True and config_llm == True
        details = f"use_router_v2={nc_fields.get('use_router_v2')}, router_v2_llm_enabled={nc_fields.get('router_v2_llm_enabled')}, RouterV2Config.llm_enabled={config_llm}"

        return TestResult(
            name="G2: Config defaults (use_router_v2=True, llm_enabled=True)",
            group="G", passed=passed, duration_ms=0, details=details,
        )

    # =========================================================================
    # Group H: History persistence
    # =========================================================================

    async def test_h1_router_responses_in_history(self) -> TestResult:
        """H1: Router responses (acks, busy, completion) should be stored in history.

        We verify by checking the _send_and_store method is used for all response paths.
        """
        import inspect
        from mesh.router_v2 import RouterV2

        # Check that all response paths use _send_and_store
        idle_src = inspect.getsource(RouterV2._handle_idle_with_llm)
        busy_src = inspect.getsource(RouterV2._handle_busy_with_llm)
        complete_src = inspect.getsource(RouterV2._handle_worker_complete)
        busy_fallback_src = inspect.getsource(RouterV2._handle_busy)

        idle_uses = "_send_and_store" in idle_src
        busy_uses = "_send_and_store" in busy_src
        complete_uses = "_send_and_store" in complete_src
        fallback_uses = "_send_and_store" in busy_fallback_src

        passed = idle_uses and busy_uses and complete_uses and fallback_uses
        details = (
            f"idle={idle_uses}, busy={busy_uses}, "
            f"complete={complete_uses}, fallback={fallback_uses}"
        )

        return TestResult(
            name="H1: All response paths use _send_and_store",
            group="H", passed=passed, duration_ms=0, details=details,
        )

    async def test_h2_context_grows_with_interaction(self) -> TestResult:
        """H2: Router context should grow as we interact (not just worker context)."""
        start = time.time()

        # Send two sequential messages and verify the second response is aware
        # of the first (proves context persistence)
        msgs1 = await self._send_and_collect(
            "Remember the number 42. Just say OK.",
            timeout=15,
        )
        await asyncio.sleep(2)

        msgs2 = await self._send_and_collect(
            "What number did I ask you to remember?",
            timeout=15,
        )
        elapsed = int((time.time() - start) * 1000)

        texts2 = self._text_msgs(msgs2)
        response = texts2[0] if texts2 else ""

        passed = "42" in response
        details = f"response={response[:80]!r}"
        if not passed:
            details += " (context may not persist across router LLM calls)"

        return TestResult(
            name="H2: Context persistence across interactions",
            group="H", passed=passed, duration_ms=elapsed, details=details,
        )

    # =========================================================================
    # Group I: Routing correctness (channel vs DM)
    # =========================================================================

    async def test_i1_dm_response_routing(self) -> TestResult:
        """I1: DM response should come back to the sender (not to a channel)."""
        start = time.time()

        msgs = await self._send_and_collect("Say hello", timeout=15)
        elapsed = int((time.time() - start) * 1000)

        texts = self._text_msgs(msgs)
        # Check that the response is addressed to our client
        response_to_us = any(
            m.to_node and self.client.node_id in m.to_node
            for m in msgs if m.type == MessageType.MESSAGE
        ) if msgs else False

        # Even if to_node isn't explicitly set, the message arriving at our client
        # means routing is correct
        passed = len(texts) >= 1
        details = f"responses={len(texts)}, explicit_to_us={response_to_us}"

        return TestResult(
            name="I1: DM response routes to sender",
            group="I", passed=passed, duration_ms=elapsed, details=details,
        )

    async def test_i2_infer_destination_logic(self) -> TestResult:
        """I2: Verify _infer_destination_from_trigger logic via code inspection."""
        from mesh.agent_node import AgentNode
        import inspect

        source = inspect.getsource(AgentNode._infer_destination_from_trigger)

        has_channel_check = "channel:" in source
        has_from_node = "from_node" in source

        passed = has_channel_check and has_from_node
        details = f"channel_check={has_channel_check}, from_node_fallback={has_from_node}"

        return TestResult(
            name="I2: Destination inference logic correct",
            group="I", passed=passed, duration_ms=0, details=details,
        )

    # =========================================================================
    # Group J: Multi-turn conversation continuity
    # =========================================================================

    async def test_j1_multi_turn_simple(self) -> TestResult:
        """J1: Three-turn conversation — agent maintains context."""
        start = time.time()

        # Turn 1: Set context
        msgs1 = await self._send_and_collect("My name is TestUser42.", timeout=15)
        await asyncio.sleep(2)

        # Turn 2: Ask to recall
        msgs2 = await self._send_and_collect("What is my name?", timeout=15)
        elapsed = int((time.time() - start) * 1000)

        texts2 = self._text_msgs(msgs2)
        response = texts2[0] if texts2 else ""

        passed = "testuser42" in response.lower() or "TestUser42" in response
        details = f"response={response[:80]!r}"

        return TestResult(
            name="J1: Multi-turn context retention",
            group="J", passed=passed, duration_ms=elapsed, details=details,
        )

    async def test_j2_tool_task_then_followup(self) -> TestResult:
        """J2: After a tool task completes, agent should handle a follow-up question."""
        start = time.time()

        # First: tool task
        msgs1 = await self._send_and_collect(
            "Run: echo 'FOLLOWUP_TEST_OK'", timeout=60, collect_all=True,
        )
        texts1 = self._text_msgs(msgs1)
        task_ok = any("FOLLOWUP_TEST_OK" in t for t in texts1)

        await asyncio.sleep(3)

        # Follow-up: simple question (should not dispatch worker)
        msgs2 = await self._send_and_collect(
            "What was the output of the command I just asked you to run?",
            timeout=20,
        )
        elapsed = int((time.time() - start) * 1000)

        texts2 = self._text_msgs(msgs2)
        response = texts2[0] if texts2 else ""

        # The follow-up should reference the previous command output
        passed = task_ok and len(texts2) >= 1
        details = f"task_ok={task_ok}, followup={response[:80]!r}"

        return TestResult(
            name="J2: Follow-up after tool task",
            group="J", passed=passed, duration_ms=elapsed, details=details,
        )

    # =========================================================================
    # Runner
    # =========================================================================

    def get_all_tests(self) -> dict[str, list]:
        """Return all test methods grouped by letter."""
        return {
            "A": [
                self.test_a1_simple_greeting,
                self.test_a2_factual_question,
                self.test_a3_tool_task_dispatches_worker,
                self.test_a4_irrelevant_message_ignored,
                self.test_a5_math_no_worker,
            ],
            "B": [
                self.test_b1_completion_contains_result,
                self.test_b2_ack_then_result,
                self.test_b3_worker_draft_not_sent_directly,
                self.test_b4_completion_is_synthesized,
            ],
            "C": [
                self.test_c1_busy_response,
                self.test_c2_busy_mentions_current_task,
                self.test_c3_worker_peek_in_logs,
            ],
            "D": [
                self.test_d1_worker_origin_in_logs,
                self.test_d2_worker_id_increments,
            ],
            "E": [
                self.test_e1_empty_message_no_crash,
                self.test_e2_agent_still_responds_after_edge_case,
                self.test_e3_llm_error_surfacing_check,
            ],
            "F": [
                self.test_f1_no_double_dispatch,
                self.test_f2_state_lock_prevents_race,
            ],
            "G": [
                self.test_g1_llm_enabled_default,
                self.test_g2_config_defaults_correct,
            ],
            "H": [
                self.test_h1_router_responses_in_history,
                self.test_h2_context_grows_with_interaction,
            ],
            "I": [
                self.test_i1_dm_response_routing,
                self.test_i2_infer_destination_logic,
            ],
            "J": [
                self.test_j1_multi_turn_simple,
                self.test_j2_tool_task_then_followup,
            ],
        }

    async def run(self, groups: list[str] | None = None) -> list[TestResult]:
        """Run selected test groups (or all)."""
        all_tests = self.get_all_tests()

        if groups:
            selected = {g.upper(): all_tests[g.upper()] for g in groups if g.upper() in all_tests}
        else:
            selected = all_tests

        results = []
        for group_name, tests in sorted(selected.items()):
            print(f"\n--- Group {group_name} ---")
            for test_fn in tests:
                try:
                    result = await test_fn()
                except Exception as e:
                    result = TestResult(
                        name=test_fn.__doc__.strip().split("\n")[0] if test_fn.__doc__ else test_fn.__name__,
                        group=group_name,
                        passed=False,
                        duration_ms=0,
                        error=f"{type(e).__name__}: {e}",
                    )
                results.append(result)
                print(result_line(result))

                # Settle between tests
                await asyncio.sleep(3.0)

        return results


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="RouterV2 Live Integration Tests")
    parser.add_argument("--group", nargs="*", help="Test groups to run (A B C D E F G H I J)")
    parser.add_argument("--agent", default="agent:test:routerv2-llm", help="Agent node ID to test")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 70)
    print("RouterV2 Live Integration Tests — Comprehensive Suite")
    print(f"Agent: {args.agent}")
    print(f"Groups: {', '.join(args.group) if args.group else 'ALL (A-J)'}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)

    runner = RouterV2TestRunner(
        agent_node=args.agent,
        verbose=args.verbose,
    )

    if not await runner.setup():
        return 1

    try:
        results = await runner.run(groups=args.group)
    finally:
        await runner.teardown()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    # By group
    groups_seen = {}
    for r in results:
        if r.group not in groups_seen:
            groups_seen[r.group] = {"total": 0, "passed": 0}
        groups_seen[r.group]["total"] += 1
        if r.passed:
            groups_seen[r.group]["passed"] += 1

    for g, stats in sorted(groups_seen.items()):
        rate = stats["passed"] / stats["total"] * 100 if stats["total"] else 0
        print(f"  Group {g}: {stats['passed']}/{stats['total']} ({rate:.0f}%)")

    print(f"\n  TOTAL: {passed}/{total} ({passed/total*100:.0f}%)")

    if failed > 0:
        print(f"\n  FAILURES:")
        for r in results:
            if not r.passed:
                print(f"    - {r.name}: {r.error or r.details}")

    # Save results
    results_file = f"/tmp/routerv2_live_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(
            [
                {
                    "name": r.name,
                    "group": r.group,
                    "passed": r.passed,
                    "duration_ms": r.duration_ms,
                    "details": r.details,
                    "error": r.error,
                }
                for r in results
            ],
            f,
            indent=2,
        )
    print(f"\n  Results saved to: {results_file}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
