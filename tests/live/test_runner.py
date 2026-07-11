#!/usr/bin/env python3
"""
Live Test Runner for v0.2 Controller.

Runs test scenarios from scenarios.yaml against a live v0.2 agent,
using isolated fixture environments.

Usage:
    python -m tests.live.test_runner                    # Run all tests
    python -m tests.live.test_runner --smoke            # Run smoke test subset
    python -m tests.live.test_runner --category simple  # Run category
    python -m tests.live.test_runner --scenario T1      # Run specific scenario
    python -m tests.live.test_runner --list             # List available scenarios
    python -m tests.live.test_runner -v                 # Verbose output
"""

import asyncio
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# Add parent to path for mesh imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mesh.protocol import Message, MessageType, ControlAction
from mesh.transport import connect, connect_ws
from mesh.api_client import MeshClient

from .models import (
    Grade,
    TestScenario,
    TestResult,
    TestRunReport,
    ToolCall,
    PhaseTransition,
)
from .fixtures import test_environment, verify_fixtures_available, DEFAULT_FIXTURE_REPO
from .grader import grade_test, format_grade_report


# =============================================================================
# Configuration
# =============================================================================

# Connection configuration - supports remote testing via WebSocket
# For remote testing, use: --ws wss://your-host.example.com/mesh/ws
ROUTER_HOST = os.environ.get("MESH_HOST", "127.0.0.1")
ROUTER_PORT = int(os.environ.get("MESH_PORT", "7700"))
ROUTER_USE_TLS = os.environ.get("MESH_TLS", "").lower() in ("1", "true", "yes")
ROUTER_WS_URL = os.environ.get("MESH_WS_URL", "")  # e.g., wss://your-host.example.com/mesh/ws
AUTH_TOKEN = os.environ.get("MESH_AUTH_TOKEN", "")

SCENARIOS_PATH = Path(__file__).parent / "scenarios.yaml"
RESULTS_DIR = Path(__file__).parent / "results"

# Smoke test scenarios (quick validation)
SMOKE_SCENARIOS = ["T1", "T2", "S1", "M1", "E1"]


# =============================================================================
# Agent Management
# =============================================================================

class AgentManager:
    """
    Manages agent lifecycle for testing.

    Can start/stop agents programmatically using the mesh API.
    """

    def __init__(
        self,
        ws_url: str | None = None,
        host: str = "127.0.0.1",
        port: int = 7700,
        auth_token: str | None = None,
    ):
        self.ws_url = ws_url or ROUTER_WS_URL
        self.host = host
        self.port = port
        self.auth_token = auth_token or AUTH_TOKEN
        self._client: MeshClient | None = None
        self._managed_agents: list[str] = []  # node_ids we started

    async def connect(self) -> bool:
        """Connect to mesh for agent management."""
        self._client = MeshClient(
            nickname="test-manager",
            auth_token=self.auth_token,
            ws_url=self.ws_url,
            host=self.host,
            port=self.port,
        )
        try:
            await self._client.connect()
            return True
        except Exception as e:
            print(f"✗ Agent manager connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from mesh."""
        if self._client:
            await self._client.disconnect()
            self._client = None

    async def list_agents(self) -> dict:
        """List configured and connected agents."""
        if not self._client:
            return {"configured": [], "connected": []}
        return await self._client.list_agents()

    async def start_agent(
        self,
        agent_type: str,
        nickname: str,
        backend: str | None = None,
        controller: str | None = None,
        effort: str | None = None,
        fresh: bool = True,
    ) -> dict:
        """
        Start an agent for testing.

        Args:
            agent_type: Agent type (e.g., "assistant")
            nickname: Unique nickname
            backend: LLM backend
            controller: Controller mode (passthrough, task-fsm-v0, phase-flow-v02)
            effort: Effort level for v0.2 (low, medium, high)
            fresh: Start without history

        Returns:
            Dict with success, node_id, pid, error
        """
        if not self._client:
            return {"success": False, "error": "Not connected"}

        result = await self._client.start_agent(
            agent_type=agent_type,
            nickname=nickname,
            backend=backend,
            controller=controller,
            effort=effort,
            fresh=fresh,
        )

        if result.get("success") and result.get("node_id"):
            self._managed_agents.append(result["node_id"])

        return result

    async def stop_agent(self, target: str, reason: str = "test cleanup") -> bool:
        """Stop a managed agent."""
        if not self._client:
            return False

        success = await self._client.stop_agent(target, reason)
        if success and target in self._managed_agents:
            self._managed_agents.remove(target)
        return success

    async def ensure_agent_running(
        self,
        agent_type: str,
        nickname: str,
        backend: str | None = None,
        controller: str | None = None,
        effort: str | None = None,
    ) -> tuple[bool, str]:
        """
        Ensure an agent is running, starting it if needed.

        Returns:
            (success, node_id)
        """
        if not self._client:
            return False, ""

        # Check if already running
        agents = await self.list_agents()
        target_node = f"agent:{agent_type}:{nickname}"

        for connected in agents.get("connected", []):
            # connected can be either a string (node_id) or a dict with node_id
            if isinstance(connected, str):
                if connected == target_node:
                    return True, target_node
            elif isinstance(connected, dict):
                if connected.get("node_id") == target_node:
                    return True, target_node

        # Not running, start it
        result = await self.start_agent(
            agent_type=agent_type,
            nickname=nickname,
            backend=backend,
            controller=controller,
            effort=effort,
            fresh=True,
        )

        if result.get("success"):
            # Wait a moment for agent to initialize
            await asyncio.sleep(2.0)
            return True, result.get("node_id", target_node)

        return False, ""

    async def cleanup_all(self) -> None:
        """Stop all agents we started."""
        for node_id in list(self._managed_agents):
            await self.stop_agent(node_id, "test run complete")
        self._managed_agents.clear()


# =============================================================================
# Scenario Loader
# =============================================================================

def load_scenarios(path: Path = SCENARIOS_PATH) -> tuple[dict, list[TestScenario]]:
    """Load test scenarios from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    defaults = data.get("defaults", {})
    scenarios = []

    for s_data in data.get("scenarios", []):
        scenario = TestScenario.from_dict(s_data, defaults)
        scenarios.append(scenario)

    return defaults, scenarios


def filter_scenarios(
    scenarios: list[TestScenario],
    category: str | None = None,
    scenario_id: str | None = None,
    smoke: bool = False,
) -> list[TestScenario]:
    """Filter scenarios based on criteria."""
    if scenario_id:
        return [s for s in scenarios if s.id == scenario_id]
    if smoke:
        return [s for s in scenarios if s.id in SMOKE_SCENARIOS]
    if category:
        return [s for s in scenarios if s.category == category]
    return scenarios


# =============================================================================
# Test Client
# =============================================================================

class TestRunner:
    """Runs live tests against a v0.2 agent."""

    def __init__(
        self,
        agent_node: str = "agent:assistant:v02",
        fixture_repo: str = DEFAULT_FIXTURE_REPO,
        verbose: bool = False,
        agent_manager: AgentManager | None = None,
        auto_start_agent: bool = False,
        default_controller: str | None = None,
        default_effort: str | None = None,
        default_backend: str | None = None,
    ):
        self.agent_node = agent_node
        self.fixture_repo = fixture_repo
        self.verbose = verbose
        self.conn = None
        self.test_user = f"user:test-{int(time.time())}"
        self.agent_manager = agent_manager
        self.auto_start_agent = auto_start_agent
        self.default_controller = default_controller
        self.default_effort = default_effort
        self.default_backend = default_backend
        self._started_agent = False

    def log(self, msg: str) -> None:
        """Print debug message if verbose."""
        if self.verbose:
            print(f"  [DEBUG] {msg}")

    async def setup(self) -> bool:
        """Connect to the mesh and optionally start the target agent."""
        # Auto-start agent if requested
        if self.auto_start_agent and self.agent_manager:
            # Parse agent node to get type and nickname
            # agent:assistant:v02 -> assistant, v02
            parts = self.agent_node.split(":")
            if len(parts) >= 3:
                agent_type = parts[1]
                nickname = parts[2]
            else:
                agent_type = "assistant"
                nickname = self.agent_node.split(":")[-1]

            self.log(f"Starting agent {agent_type}:{nickname}...")
            success, node_id = await self.agent_manager.ensure_agent_running(
                agent_type=agent_type,
                nickname=nickname,
                backend=self.default_backend,
                controller=self.default_controller,
                effort=self.default_effort,
            )
            if success:
                self.log(f"Agent running: {node_id}")
                self._started_agent = True
            else:
                print(f"✗ Failed to start agent {self.agent_node}")
                return False

        # Connect to the mesh (TCP or WebSocket)
        if ROUTER_WS_URL:
            self.log(f"Connecting via WebSocket to {ROUTER_WS_URL}")
            self.conn = await connect_ws(ROUTER_WS_URL)
        else:
            self.log(f"Connecting via TCP to {ROUTER_HOST}:{ROUTER_PORT}")
            self.conn = await connect(
                ROUTER_HOST,
                ROUTER_PORT,
                use_tls=ROUTER_USE_TLS,
                server_hostname=ROUTER_HOST if ROUTER_USE_TLS else None,
            )

        # Register with router
        register_msg = Message(
            type=MessageType.CONTROL,
            from_node=self.test_user,
            to_node="router",
            content={
                "action": ControlAction.REGISTER.value,
                "node_id": self.test_user,
                "type": "user",
                "auth_token": AUTH_TOKEN,
            },
        )
        await self.conn.send(register_msg)

        # Wait for ack
        response = await asyncio.wait_for(self.conn.receive(), timeout=5.0)
        if response:
            content = response.content if isinstance(response.content, dict) else {}
            if content.get("action") == "ack":
                self.log(f"Connected as {self.test_user}")
                return True

        print("✗ Registration failed")
        return False

    async def teardown(self) -> None:
        """Disconnect from the mesh and optionally stop managed agents."""
        if self.conn:
            await self.conn.close()
            self.log("Disconnected from mesh")

    async def send_and_collect(
        self,
        content: str,
        timeout: float = 30.0,
    ) -> tuple[list[Message], TestResult]:
        """
        Send a message and collect all responses.

        Returns:
            (messages, partial TestResult with phases/tools populated)
        """
        messages: list[Message] = []
        result = TestResult(scenario_id="", category="")

        # Send message
        msg = Message(
            type=MessageType.MESSAGE,
            from_node=self.test_user,
            to_node=self.agent_node,
            content=content,
        )
        await self.conn.send(msg)
        self.log(f"Sent: {content[:50]}...")

        # Collect responses
        start = time.time()
        got_final = False

        while (time.time() - start) < timeout and not got_final:
            try:
                response = await asyncio.wait_for(self.conn.receive(), timeout=10.0)
                if not response:
                    continue

                # Only process messages from our target agent
                if response.from_node != self.agent_node:
                    self.log(f"Ignoring message from {response.from_node}")
                    continue

                messages.append(response)
                self.log(f"Received: type={response.type.value}")

                # Extract phase transitions from content
                content_str = str(response.content)
                phase_match = re.search(
                    r'\[PHASE:\s*(\w+)\]|\[(\w+)\]|phase[=:]\s*(\w+)',
                    content_str,
                    re.IGNORECASE,
                )
                if phase_match:
                    phase = (
                        phase_match.group(1)
                        or phase_match.group(2)
                        or phase_match.group(3)
                    )
                    if phase.upper() in ["INFO", "PLAN", "EXECUTE", "VALIDATE", "DOCUMENT", "DONE"]:
                        phase_upper = phase.upper()
                        if phase_upper not in result.phases_triggered:
                            result.phases_triggered.append(phase_upper)
                            result.phase_transitions.append(
                                PhaseTransition(
                                    phase=phase_upper,
                                    timestamp_ms=(time.time() - start) * 1000,
                                )
                            )

                # Extract tool calls from TOOL_ACTIVITY messages
                if response.type == MessageType.TOOL_ACTIVITY:
                    tool_data = response.content if isinstance(response.content, dict) else {}
                    # Protocol uses "tool_name" and "tool_call" event_type
                    tool_name = tool_data.get("tool_name", tool_data.get("tool", tool_data.get("name", "")))
                    if tool_name and tool_data.get("event_type") == "tool_call":
                        result.tools_called.append(ToolCall(
                            name=tool_name,
                            params=tool_data.get("data", {}).get("args", {}),
                            result="",  # Result comes in separate tool_result message
                        ))

                # Check if this is the final response
                if response.type == MessageType.MESSAGE and isinstance(response.content, str):
                    result.response_text = response.content
                    # Wait a moment for any follow-up messages
                    await asyncio.sleep(0.5)
                    got_final = True

            except asyncio.TimeoutError:
                self.log(f"Waiting... ({time.time() - start:.0f}s elapsed)")

        result.total_duration_seconds = time.time() - start
        result.llm_calls = max(1, len(result.phases_triggered))  # Estimate
        # Token counting would need LLM response metadata

        return messages, result

    async def run_scenario(self, scenario: TestScenario, use_fixtures: bool = True) -> TestResult:
        """Run a single test scenario."""
        print(f"  [{scenario.id}] {scenario.category}: {scenario.input[:40]}...", end=" ", flush=True)

        result = TestResult(
            scenario_id=scenario.id,
            category=scenario.category,
        )

        start = time.time()

        try:
            if use_fixtures and scenario.fixture_branch not in ["main", ""]:
                # Use fixture environment for branch-specific tests
                async with test_environment(
                    scenario_id=scenario.id,
                    fixture_repo=self.fixture_repo,
                    branch=scenario.fixture_branch,
                ) as env:
                    self.log(f"Fixture environment: {env.work_dir}")
                    # Prepend fixture context to prompt
                    fixture_prompt = (
                        f"[Context: Working directory is {env.work_dir}. "
                        f"Set your working directory there first.]\n\n"
                        f"{scenario.input}"
                    )
                    _, result = await self.send_and_collect(
                        content=fixture_prompt,
                        timeout=scenario.timeout_seconds,
                    )
            else:
                # Simple tests don't need fixtures
                _, result = await self.send_and_collect(
                    content=scenario.input,
                    timeout=scenario.timeout_seconds,
                )

            result.scenario_id = scenario.id
            result.category = scenario.category

        except Exception as e:
            result.has_errors = True
            result.error_message = str(e)
            import traceback
            self.log(traceback.format_exc())

        result.total_duration_seconds = time.time() - start

        # Grade the result
        result.grade, result.grade_issues = grade_test(result, scenario.expected)

        # Print status
        status_char = {"PASS": "✓", "PARTIAL": "○", "FAIL": "✗"}[result.grade.value]
        print(f"{status_char} {result.grade.value} ({result.total_duration_seconds:.1f}s)")

        if self.verbose and result.grade_issues:
            for issue in result.grade_issues:
                print(f"      - {issue}")

        return result

    async def run_all(
        self,
        scenarios: list[TestScenario],
        use_fixtures: bool = True,
    ) -> list[TestResult]:
        """Run all scenarios."""
        results: list[TestResult] = []

        if not await self.setup():
            print("Failed to connect to mesh")
            return results

        try:
            for scenario in scenarios:
                result = await self.run_scenario(scenario, use_fixtures=use_fixtures)
                results.append(result)
                # Small delay between tests
                await asyncio.sleep(1.0)
        finally:
            await self.teardown()

        return results


# =============================================================================
# Reporting
# =============================================================================

def print_summary(report: TestRunReport) -> None:
    """Print test run summary."""
    print("\n" + "=" * 70)
    print(f"Live Test Results: {report.passed}/{report.total_tests} passed")
    print(f"  PASS: {report.passed} | PARTIAL: {report.partial} | FAIL: {report.failed}")
    print(f"  Duration: {report.total_duration_seconds:.1f}s | LLM calls: {report.total_llm_calls}")
    print("=" * 70)

    # Group by category
    by_category: dict[str, list[TestResult]] = {}
    for r in report.results:
        by_category.setdefault(r.category, []).append(r)

    for category, results in by_category.items():
        cat_passed = sum(1 for r in results if r.grade == Grade.PASS)
        cat_partial = sum(1 for r in results if r.grade == Grade.PARTIAL)
        print(f"\n{category.upper()} ({cat_passed}/{len(results)} pass, {cat_partial} partial):")
        for r in results:
            print(format_grade_report(r))

    print("\n" + "=" * 70)


def save_report(report: TestRunReport) -> Path:
    """Save report to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"run-{report.run_id}.json"
    path = RESULTS_DIR / filename

    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    return path


# =============================================================================
# Main
# =============================================================================

async def main() -> int:
    global ROUTER_HOST, ROUTER_PORT, ROUTER_USE_TLS, ROUTER_WS_URL

    parser = argparse.ArgumentParser(description="Live test runner for v0.2 controller")
    parser.add_argument("--agent", default="agent:assistant:v02",
                        help="Agent node to test (default: agent:assistant:v02)")
    parser.add_argument("--category", choices=["trivial", "simple", "moderate", "complex", "edge"],
                        help="Filter tests by category")
    parser.add_argument("--scenario", "-s", help="Run specific scenario by ID")
    parser.add_argument("--smoke", action="store_true",
                        help="Run smoke test subset only")
    parser.add_argument("--no-fixtures", action="store_true",
                        help="Skip fixture environment setup")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--list", "-l", action="store_true", help="List available scenarios")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")

    # Remote connection options
    parser.add_argument("--host", help="Router host (default: MESH_HOST env or 127.0.0.1)")
    parser.add_argument("--port", type=int, help="Router port (default: MESH_PORT env or 7700)")
    parser.add_argument("--tls", action="store_true", help="Use TLS for TCP connection")
    parser.add_argument("--ws", help="WebSocket URL for remote testing (e.g., wss://your-host.example.com/mesh/ws)")
    parser.add_argument("--fixture-repo", help="Fixture repo path/URL (default: FIXTURE_REPO env or local test-fixtures.git)")

    # Agent management options
    parser.add_argument("--auto-start", action="store_true", default=True,
                        help="Automatically start the target agent before tests (default: True)")
    parser.add_argument("--no-auto-start", action="store_true",
                        help="Disable auto-start of agents")
    parser.add_argument("--controller", choices=["passthrough", "task-fsm-v0", "phase-flow-v02"],
                        help="Controller mode for auto-started agents")
    parser.add_argument("--effort", choices=["low", "medium", "high"],
                        help="Effort level for v0.2 controller")
    parser.add_argument("--backend", help="LLM backend for auto-started agents")
    parser.add_argument("--list-agents", action="store_true",
                        help="List available and connected agents")
    args = parser.parse_args()

    # Override connection settings from CLI args
    if args.ws:
        ROUTER_WS_URL = args.ws
    if args.host:
        ROUTER_HOST = args.host
    if args.port:
        ROUTER_PORT = args.port
    if args.tls:
        ROUTER_USE_TLS = True

    # Show connection info
    if args.verbose:
        if ROUTER_WS_URL:
            print(f"Connecting via WebSocket to {ROUTER_WS_URL}")
        else:
            tls_str = " (TLS)" if ROUTER_USE_TLS else ""
            print(f"Connecting via TCP to {ROUTER_HOST}:{ROUTER_PORT}{tls_str}")

    # Handle --no-auto-start override
    auto_start = args.auto_start and not args.no_auto_start

    # Create agent manager if needed
    agent_manager = None
    if auto_start or args.list_agents:
        agent_manager = AgentManager(
            ws_url=ROUTER_WS_URL,
            host=ROUTER_HOST,
            port=ROUTER_PORT,
            auth_token=AUTH_TOKEN,
        )
        if not await agent_manager.connect():
            print("Failed to connect agent manager")
            return 1

    # Handle --list-agents
    if args.list_agents:
        agents = await agent_manager.list_agents()
        print("\nConfigured agents:")
        for cfg in agents.get("configured", []):
            if isinstance(cfg, dict):
                print(f"  {cfg.get('node_id', cfg.get('agent_type', '?'))}")
                if cfg.get("controller"):
                    print(f"    controller: {cfg['controller']}")
            else:
                print(f"  {cfg}")
        print("\nConnected agents:")
        for conn in agents.get("connected", []):
            print(f"  {conn}")
        await agent_manager.disconnect()
        return 0

    # Load scenarios
    try:
        defaults, all_scenarios = load_scenarios()
    except FileNotFoundError:
        print(f"Scenarios file not found: {SCENARIOS_PATH}")
        return 1

    if args.list:
        print("Available scenarios:")
        for s in all_scenarios:
            print(f"  {s.id}: [{s.category}] {s.input[:50]}...")
        return 0

    # Filter scenarios
    scenarios = filter_scenarios(
        all_scenarios,
        category=args.category,
        scenario_id=args.scenario,
        smoke=args.smoke,
    )

    if not scenarios:
        print("No scenarios match the filter")
        return 1

    # Determine fixture repo (CLI > env > defaults > hardcoded)
    fixture_repo = (
        args.fixture_repo
        or os.environ.get("FIXTURE_REPO")
        or defaults.get("fixture_repo")
        or DEFAULT_FIXTURE_REPO
    )

    # Verify fixtures available (if needed)
    if not args.no_fixtures:
        if not await verify_fixtures_available(fixture_repo):
            print(f"Warning: Fixture repo not accessible at {fixture_repo}, running without fixtures")
            print(f"  Hint: For remote testing, use: --fixture-repo git@your-host.example.com:/path/to/test-fixtures.git")
            args.no_fixtures = True

    # Run tests
    print(f"\nRunning {len(scenarios)} test scenarios against {args.agent}")
    if args.controller:
        print(f"  Controller: {args.controller}")
    if args.effort:
        print(f"  Effort: {args.effort}")
    print("-" * 70)

    runner = TestRunner(
        agent_node=args.agent,
        fixture_repo=fixture_repo,
        verbose=args.verbose,
        agent_manager=agent_manager,
        auto_start_agent=auto_start,
        default_controller=args.controller,
        default_effort=args.effort,
        default_backend=args.backend,
    )

    try:
        results = await runner.run_all(scenarios, use_fixtures=not args.no_fixtures)
    finally:
        # Cleanup agent manager
        if agent_manager:
            if auto_start:
                await agent_manager.cleanup_all()
            await agent_manager.disconnect()

    # Build report
    report = TestRunReport(
        run_id=datetime.now().strftime("%Y%m%d-%H%M%S"),
        timestamp=datetime.now().isoformat(),
        results=results,
    )
    report.compute_aggregates()

    # Output
    print_summary(report)

    if args.save:
        path = save_report(report)
        print(f"\nReport saved to: {path}")

    # Exit code: 0 if all pass, 1 if any failures
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
