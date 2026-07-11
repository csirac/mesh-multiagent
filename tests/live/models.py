"""
Data models for the live testing framework.

Defines the core structures for test definitions, results, and grading.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Grade(Enum):
    """Test result grade."""
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"


@dataclass
class TestExpectation:
    """Expected behavior for a test scenario."""
    min_phases: list[str] = field(default_factory=list)  # Required phases
    max_phases: int = 5  # Phase budget
    no_errors: bool = True  # Whether errors are acceptable
    tools_required: list[str] = field(default_factory=list)  # Must be called
    tools_forbidden: list[str] = field(default_factory=list)  # Must not be called
    response_contains: list[str] = field(default_factory=list)  # Keywords in response
    response_excludes: list[str] = field(default_factory=list)  # Keywords not in response
    max_duration_seconds: int = 60
    max_tokens: int = 10000
    max_llm_calls: int = 5

    @classmethod
    def from_dict(cls, data: dict) -> "TestExpectation":
        """Create from YAML dict."""
        return cls(
            min_phases=data.get("min_phases", []),
            max_phases=data.get("max_phases", 5),
            no_errors=data.get("no_errors", True),
            tools_required=data.get("tools_required", []),
            tools_forbidden=data.get("tools_forbidden", []),
            response_contains=data.get("response_contains", []),
            response_excludes=data.get("response_excludes", []),
            max_duration_seconds=data.get("max_duration_seconds", 60),
            max_tokens=data.get("max_tokens", 10000),
            max_llm_calls=data.get("max_llm_calls", 5),
        )


@dataclass
class TestScenario:
    """A single test scenario definition."""
    id: str
    category: str
    input: str
    expected: TestExpectation
    fixture_branch: str = "main"
    agent: str = "agent:assistant:v02"
    timeout_seconds: int = 120
    # Agent configuration (optional per-scenario overrides)
    controller: str | None = None  # passthrough, task-fsm-v0, phase-flow-v02
    effort: str | None = None  # low, medium, high (for v0.2)
    backend: str | None = None  # LLM backend override

    @classmethod
    def from_dict(cls, data: dict, defaults: dict) -> "TestScenario":
        """Create from YAML dict with defaults applied."""
        return cls(
            id=data["id"],
            category=data["category"],
            input=data["input"],
            expected=TestExpectation.from_dict(data.get("expected", {})),
            fixture_branch=data.get("fixture_branch", defaults.get("fixture_branch", "main")),
            agent=data.get("agent", defaults.get("agent", "agent:assistant:v02")),
            timeout_seconds=data.get("timeout_seconds", defaults.get("timeout_seconds", 120)),
            controller=data.get("controller", defaults.get("controller")),
            effort=data.get("effort", defaults.get("effort")),
            backend=data.get("backend", defaults.get("backend")),
        )


@dataclass
class ToolCall:
    """Record of a tool invocation."""
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    duration_ms: float = 0.0


@dataclass
class PhaseTransition:
    """Record of a phase transition."""
    phase: str
    timestamp_ms: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Complete result of running a test scenario."""
    scenario_id: str
    category: str

    # Execution results
    phases_triggered: list[str] = field(default_factory=list)
    phase_transitions: list[PhaseTransition] = field(default_factory=list)
    tools_called: list[ToolCall] = field(default_factory=list)
    response_text: str = ""

    # Error tracking
    has_errors: bool = False
    error_message: str | None = None

    # Metrics
    total_duration_seconds: float = 0.0
    total_tokens: int = 0
    llm_calls: int = 0

    # Grading
    grade: Grade = Grade.FAIL
    grade_issues: list[str] = field(default_factory=list)

    def get_tool_names(self) -> list[str]:
        """Get unique tool names called."""
        return list(set(tc.name for tc in self.tools_called))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "phases_triggered": self.phases_triggered,
            "tools_called": [{"name": tc.name, "params": tc.params} for tc in self.tools_called],
            "response_text": self.response_text[:500] if len(self.response_text) > 500 else self.response_text,
            "has_errors": self.has_errors,
            "error_message": self.error_message,
            "total_duration_seconds": self.total_duration_seconds,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "grade": self.grade.value,
            "grade_issues": self.grade_issues,
        }


@dataclass
class TestRunReport:
    """Summary of a complete test run."""
    run_id: str
    timestamp: str
    results: list[TestResult] = field(default_factory=list)

    # Aggregates
    total_tests: int = 0
    passed: int = 0
    partial: int = 0
    failed: int = 0

    # Metrics
    total_duration_seconds: float = 0.0
    total_tokens: int = 0
    total_llm_calls: int = 0

    def compute_aggregates(self) -> None:
        """Compute aggregate stats from results."""
        self.total_tests = len(self.results)
        self.passed = sum(1 for r in self.results if r.grade == Grade.PASS)
        self.partial = sum(1 for r in self.results if r.grade == Grade.PARTIAL)
        self.failed = sum(1 for r in self.results if r.grade == Grade.FAIL)
        self.total_duration_seconds = sum(r.total_duration_seconds for r in self.results)
        self.total_tokens = sum(r.total_tokens for r in self.results)
        self.total_llm_calls = sum(r.llm_calls for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "summary": {
                "total": self.total_tests,
                "passed": self.passed,
                "partial": self.partial,
                "failed": self.failed,
                "pass_rate": round(self.passed / self.total_tests, 3) if self.total_tests > 0 else 0,
            },
            "metrics": {
                "total_duration_seconds": round(self.total_duration_seconds, 2),
                "total_tokens": self.total_tokens,
                "total_llm_calls": self.total_llm_calls,
            },
            "results": [r.to_dict() for r in self.results],
        }
