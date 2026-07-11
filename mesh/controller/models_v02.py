"""
Data models for the v0.2 stateless phase-flow controller.

This module defines the core data structures for the v0.2 controller:
- FlowPhase: Phases in the message processing flow
- FlowState: Ephemeral state during a single message processing
- InfoAssessment: LLM-scored assessment of information needs and complexity
- PlanV02: Execution plan with quality score
- ValidationResult: Result of the validation phase
- FlowMetrics: Optional metrics tracking for monitoring

Unlike v0.1, this controller is STATELESS between messages.
No task persistence, no task IDs - conversation history provides continuity.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class FlowPhase(str, Enum):
    """
    Phases in the v0.2 message processing flow.

    Each message goes through a subset of these phases based on
    complexity routing. The flow is ephemeral - no state persists
    between messages.
    """
    # Assessment phase - always first
    INFO = "info"                    # Assess info needs and complexity

    # Optional planning phase (MODERATE/HIGH complexity)
    PLAN = "plan"                    # Generate execution plan

    # Execution phase - always present
    EXECUTE = "execute"              # Execute the response/plan

    # Optional validation phase (MODERATE/HIGH complexity)
    VALIDATE = "validate"            # Verify execution results

    # Optional documentation phase (HIGH complexity only)
    DOCUMENT = "document"            # Update docs/state

    # Terminal phases
    DONE = "done"                    # Flow completed successfully
    FAILED = "failed"                # Flow aborted due to error


class ComplexityLevel(str, Enum):
    """
    Complexity classification for routing.

    Determines which phases are executed:
    - LOW: INFO -> EXECUTE -> DONE
    - MODERATE: INFO -> PLAN -> EXECUTE -> VALIDATE -> DONE
    - HIGH: INFO -> PLAN -> EXECUTE -> VALIDATE -> DOCUMENT -> DONE
    """
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass
class InfoAssessment:
    """
    LLM-scored assessment from the INFO phase.

    All scores are 0.0 to 1.0 where:
    - 0.0 = definitely not needed / definitely simple
    - 1.0 = definitely needed / definitely complex
    """
    # Complexity assessment (0-1)
    complexity: float = 0.5          # Overall task complexity

    # Information needs (0-1 each)
    need_clarification: float = 0.0  # Need to ask user clarifying questions
    need_web: float = 0.0            # Need to search the web
    need_literature: float = 0.0     # Need to search academic literature
    need_project_files: float = 0.0  # Need to search project files

    # Optional context from assessment
    clarification_questions: list[str] | None = None  # Questions to ask user
    web_search_intent: str | None = None              # What to search for
    literature_search_intent: str | None = None       # Academic search query
    project_files_intent: str | None = None           # What to look for in project

    # Parsing metadata
    raw_output: str = ""             # Original output from LLM (for debugging)
    parsed_successfully: bool = True # Whether XML parsed successfully

    def max_info_score(self) -> float:
        """Return the highest info need score (excluding clarification)."""
        return max(self.need_web, self.need_literature, self.need_project_files)

    def any_info_needed(self, threshold: float) -> bool:
        """Check if any info gathering is needed above threshold."""
        return (
            self.need_web >= threshold or
            self.need_literature >= threshold or
            self.need_project_files >= threshold
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/debugging."""
        return {
            "complexity": self.complexity,
            "need_clarification": self.need_clarification,
            "need_web": self.need_web,
            "need_literature": self.need_literature,
            "need_project_files": self.need_project_files,
            "clarification_questions": self.clarification_questions,
            "web_search_intent": self.web_search_intent,
            "parsed_successfully": self.parsed_successfully,
        }


@dataclass
class PlanStepV02:
    """A step in a v0.2 execution plan."""
    number: int                      # Step number (1-indexed)
    description: str                 # What this step does
    estimated_turns: int = 1         # Expected LLM turns to complete

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "description": self.description,
            "estimated_turns": self.estimated_turns,
        }


@dataclass
class PlanV02:
    """
    Execution plan from the PLAN phase.

    Plans are generated for MODERATE and HIGH complexity tasks.
    """
    steps: list[PlanStepV02] = field(default_factory=list)
    quality_score: float = 0.0       # LLM-assessed plan quality (0-1)
    revision_count: int = 0          # How many times this plan was revised

    # Optional fields from XML
    rollback_strategy: str | None = None       # How to undo if needed
    complexity_reassessment: str | None = None # If complexity changed

    # Parsing metadata
    raw_output: str = ""
    parsed_successfully: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "quality_score": self.quality_score,
            "revision_count": self.revision_count,
            "rollback_strategy": self.rollback_strategy,
            "parsed_successfully": self.parsed_successfully,
        }


@dataclass
class ValidationResult:
    """
    Result from the VALIDATE phase.

    Validates that execution achieved the intended outcome.
    """
    task_accomplished: float = 0.0   # Score 0-1: was the goal achieved?
    verified: float = 0.0            # Score 0-1: has result been verified?
    issues: list[str] = field(default_factory=list)  # Any issues found
    can_fix_without_replan: bool = False  # Can issues be fixed without replanning?
    fix_actions: str | None = None   # What to do to fix issues

    # Parsing metadata
    raw_output: str = ""
    parsed_successfully: bool = True

    def is_successful(self, accomplished_threshold: float = 0.8, verified_threshold: float = 0.7) -> bool:
        """Check if validation passed based on thresholds."""
        return self.task_accomplished >= accomplished_threshold and self.verified >= verified_threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_accomplished": self.task_accomplished,
            "verified": self.verified,
            "issues": self.issues,
            "can_fix_without_replan": self.can_fix_without_replan,
            "fix_actions": self.fix_actions,
            "parsed_successfully": self.parsed_successfully,
        }


@dataclass
class FlowMetrics:
    """
    Optional metrics tracking for monitoring and debugging.

    Tracks resource usage during a single message flow.
    """
    llm_calls: int = 0               # Total LLM API calls
    total_input_tokens: int = 0      # Total input tokens
    total_output_tokens: int = 0     # Total output tokens
    phases_executed: list[str] = field(default_factory=list)
    start_time: str = ""             # ISO timestamp
    end_time: str = ""               # ISO timestamp
    duration_ms: int = 0             # Total duration in milliseconds

    # Per-phase timing (phase -> ms)
    phase_durations: dict[str, int] = field(default_factory=dict)

    # Iteration counts
    info_iterations: int = 0         # How many INFO loops
    plan_iterations: int = 0         # How many PLAN revisions

    # Error tracking
    errors: list[str] = field(default_factory=list)
    xml_parse_failures: int = 0      # Count of XML parsing failures

    def __post_init__(self):
        if not self.start_time:
            self.start_time = datetime.utcnow().isoformat() + "Z"

    def record_phase(self, phase: str, duration_ms: int) -> None:
        """Record completion of a phase."""
        self.phases_executed.append(phase)
        self.phase_durations[phase] = duration_ms

    def record_llm_call(self, input_tokens: int, output_tokens: int) -> None:
        """Record an LLM API call."""
        self.llm_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def record_error(self, error: str) -> None:
        """Record an error."""
        self.errors.append(error)

    def finalize(self) -> None:
        """Mark the flow as complete and calculate duration."""
        self.end_time = datetime.utcnow().isoformat() + "Z"
        # Duration calculated from phase durations
        self.duration_ms = sum(self.phase_durations.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "phases_executed": self.phases_executed,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "phase_durations": self.phase_durations,
            "info_iterations": self.info_iterations,
            "plan_iterations": self.plan_iterations,
            "errors": self.errors,
            "xml_parse_failures": self.xml_parse_failures,
        }


@dataclass
class FlowState:
    """
    Ephemeral state during a single message processing flow.

    This is NOT persisted between messages. Each message starts
    with a fresh FlowState that tracks progress through the phases.
    """
    # Current phase
    phase: FlowPhase = FlowPhase.INFO

    # Complexity routing (set after INFO phase)
    complexity: ComplexityLevel | None = None

    # Phase outputs (populated as phases complete)
    info_assessment: InfoAssessment | None = None
    plan: PlanV02 | None = None
    validation: ValidationResult | None = None

    # Iteration counts (for loop limits)
    info_iterations: int = 0
    plan_iterations: int = 0

    # Gathered information (from INFO phase tools)
    gathered_info: dict[str, str] = field(default_factory=dict)
    # Keys: "web", "literature", "project" -> search results

    # Optional metrics
    metrics: FlowMetrics | None = None

    # Error state
    error_message: str = ""          # Set if flow fails

    # Original message (for context in failure reports)
    original_message: str = ""

    def is_terminal(self) -> bool:
        """Check if flow is in a terminal state."""
        return self.phase in (FlowPhase.DONE, FlowPhase.FAILED)

    def fail(self, message: str) -> None:
        """Transition to FAILED state with error message."""
        self.phase = FlowPhase.FAILED
        self.error_message = message
        if self.metrics:
            self.metrics.record_error(message)

    def complete(self) -> None:
        """Transition to DONE state."""
        self.phase = FlowPhase.DONE
        if self.metrics:
            self.metrics.finalize()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/debugging."""
        return {
            "phase": self.phase.value,
            "complexity": self.complexity.value if self.complexity else None,
            "info_assessment": self.info_assessment.to_dict() if self.info_assessment else None,
            "plan": self.plan.to_dict() if self.plan else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "info_iterations": self.info_iterations,
            "plan_iterations": self.plan_iterations,
            "gathered_info_keys": list(self.gathered_info.keys()),
            "error_message": self.error_message,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }
