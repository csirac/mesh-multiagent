"""
Observability module for v0.2 phase-flow controller.

Provides real-time visibility into phase transitions, assessments,
and tool activity during message processing. Events are streamed
to users as they occur, enabling:

1. Visibility into the flow - User sees which phase is active
2. Tool call transparency - Every tool invocation is visible
3. Debugging support - Full trace available when issues occur

This module defines:
- PhaseEvent: Data class for phase-related events
- ObservabilityEmitter: Protocol for emitting events
- LoggingObserver: Default implementation that logs events
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .models_v02 import FlowPhase, FlowState, InfoAssessment, PlanV02, ValidationResult


class PhaseEventType(str, Enum):
    """Types of observability events emitted during phase flow."""

    # Phase transitions
    PHASE_START = "phase_start"     # Entering a new phase
    PHASE_END = "phase_end"         # Completing a phase

    # Assessment events
    INFO_ASSESSMENT = "info_assessment"  # INFO phase assessment complete
    PLAN_CREATED = "plan_created"        # PLAN phase produced a plan
    PLAN_REVISION = "plan_revision"      # Plan being revised
    VALIDATION_RESULT = "validation_result"  # VALIDATE phase result

    # Info gathering
    INFO_GATHERING_START = "info_gathering_start"
    INFO_GATHERING_RESULT = "info_gathering_result"

    # Tool activity (forwarded from agent)
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    # Flow events
    FLOW_START = "flow_start"       # New message flow starting
    FLOW_COMPLETE = "flow_complete"  # Flow completed successfully
    FLOW_FAILED = "flow_failed"      # Flow aborted with error

    # Clarification
    CLARIFICATION_NEEDED = "clarification_needed"


@dataclass
class PhaseEvent:
    """
    An observability event from the phase-flow controller.

    These events are emitted in real-time and can be:
    - Logged for debugging
    - Streamed to users for visibility
    - Collected for metrics
    """

    event_type: PhaseEventType
    timestamp: str = ""  # ISO format

    # Phase context
    phase: FlowPhase | None = None
    previous_phase: FlowPhase | None = None

    # Event-specific data
    data: dict[str, Any] = field(default_factory=dict)

    # Human-readable message
    message: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/transmission."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "phase": self.phase.value if self.phase else None,
            "previous_phase": self.previous_phase.value if self.previous_phase else None,
            "data": self.data,
            "message": self.message,
        }

    def format_for_user(self) -> str:
        """
        Format the event for display to the user.

        Returns a human-readable string suitable for streaming.
        """
        if self.event_type == PhaseEventType.PHASE_START:
            return f"[PHASE: {self.phase.value.upper()}]" if self.phase else "[PHASE: UNKNOWN]"

        elif self.event_type == PhaseEventType.FLOW_START:
            return "[FLOW: Starting new message processing]"

        elif self.event_type == PhaseEventType.FLOW_COMPLETE:
            phases = self.data.get("phases_executed", [])
            return f"[FLOW: Complete - phases: {' → '.join(phases)}]"

        elif self.event_type == PhaseEventType.FLOW_FAILED:
            error = self.data.get("error", "Unknown error")
            return f"[FLOW: Failed - {error}]"

        elif self.event_type == PhaseEventType.INFO_ASSESSMENT:
            complexity = self.data.get("complexity", 0)
            return f"[INFO: Complexity={complexity:.2f}]"

        elif self.event_type == PhaseEventType.PLAN_CREATED:
            steps = self.data.get("step_count", 0)
            quality = self.data.get("quality", 0)
            return f"[PLAN: {steps} steps, quality={quality:.2f}]"

        elif self.event_type == PhaseEventType.PLAN_REVISION:
            iteration = self.data.get("iteration", 0)
            return f"[PLAN: Revision {iteration}]"

        elif self.event_type == PhaseEventType.VALIDATION_RESULT:
            accomplished = self.data.get("accomplished", 0)
            verified = self.data.get("verified", 0)
            issues = self.data.get("issue_count", 0)
            status = "PASSED" if accomplished >= 0.8 and verified >= 0.7 else "ISSUES"
            return f"[VALIDATE: {status} - accomplished={accomplished:.2f}, verified={verified:.2f}, issues={issues}]"

        elif self.event_type == PhaseEventType.CLARIFICATION_NEEDED:
            questions = self.data.get("questions", [])
            return f"[CLARIFICATION: {len(questions)} question(s)]"

        elif self.event_type == PhaseEventType.TOOL_CALL:
            tool = self.data.get("tool", "unknown")
            return f"[TOOL: Calling {tool}]"

        elif self.event_type == PhaseEventType.TOOL_RESULT:
            tool = self.data.get("tool", "unknown")
            success = self.data.get("success", True)
            status = "OK" if success else "FAILED"
            return f"[TOOL: {tool} - {status}]"

        else:
            return self.message or f"[{self.event_type.value}]"


@runtime_checkable
class ObservabilityEmitter(Protocol):
    """
    Protocol for observability event emitters.

    Implementations can log events, stream to users, collect metrics, etc.
    The controller calls emit() for each event; the emitter decides what to do.
    """

    async def emit(self, event: PhaseEvent) -> None:
        """
        Emit an observability event.

        Args:
            event: The event to emit
        """
        ...


class LoggingObserver:
    """
    Default observer that logs events.

    Useful for debugging and as a reference implementation.
    """

    def __init__(self, logger_name: str = "mesh.controller.observability"):
        import logging
        self.logger = logging.getLogger(logger_name)

    async def emit(self, event: PhaseEvent) -> None:
        """Log the event at appropriate level."""
        formatted = event.format_for_user()

        if event.event_type in (PhaseEventType.FLOW_FAILED,):
            self.logger.error(formatted)
        elif event.event_type in (PhaseEventType.PHASE_START, PhaseEventType.FLOW_START, PhaseEventType.FLOW_COMPLETE):
            self.logger.info(formatted)
        else:
            self.logger.debug(formatted)


class CompositeObserver:
    """
    Observer that forwards events to multiple observers.

    Useful for combining logging with user streaming.
    """

    def __init__(self, observers: list[ObservabilityEmitter] | None = None):
        self.observers: list[ObservabilityEmitter] = observers or []

    def add(self, observer: ObservabilityEmitter) -> None:
        """Add an observer."""
        self.observers.append(observer)

    async def emit(self, event: PhaseEvent) -> None:
        """Forward event to all observers."""
        for observer in self.observers:
            await observer.emit(event)


class StreamingObserver:
    """
    Observer that streams phase events to users via a callback.

    This is the primary observer for production use. It converts
    phase events to user-friendly messages and sends them via
    the provided callback.
    """

    def __init__(
        self,
        callback: "Callable[[str], Awaitable[None]]",
        include_tool_events: bool = False,
    ):
        """
        Initialize the streaming observer.

        Args:
            callback: Async function that sends a message to the user.
                     Signature: async def send(message: str) -> None
            include_tool_events: If True, also stream TOOL_CALL/TOOL_RESULT events.
                                Defaults to False since agent_node streams these separately.
        """
        from typing import Callable, Awaitable
        self.callback = callback
        self.include_tool_events = include_tool_events

    async def emit(self, event: PhaseEvent) -> None:
        """Stream the event to the user if it's user-visible."""
        # Skip tool events unless explicitly enabled
        if event.event_type in (PhaseEventType.TOOL_CALL, PhaseEventType.TOOL_RESULT):
            if not self.include_tool_events:
                return

        # Skip PHASE_END events (less noise for user)
        if event.event_type == PhaseEventType.PHASE_END:
            return

        # Format and send
        formatted = event.format_for_user()
        await self.callback(formatted)


class CollectingObserver:
    """
    Observer that collects events for later inspection.

    Useful for testing and debugging.
    """

    def __init__(self):
        self.events: list[PhaseEvent] = []

    async def emit(self, event: PhaseEvent) -> None:
        """Collect the event."""
        self.events.append(event)

    def clear(self) -> None:
        """Clear collected events."""
        self.events.clear()

    def get_events(self, event_type: PhaseEventType | None = None) -> list[PhaseEvent]:
        """Get collected events, optionally filtered by type."""
        if event_type is None:
            return list(self.events)
        return [e for e in self.events if e.event_type == event_type]

    def get_phase_transitions(self) -> list[str]:
        """Get list of phases that were started."""
        return [
            e.phase.value for e in self.events
            if e.event_type == PhaseEventType.PHASE_START and e.phase
        ]


# Factory functions for common observer configurations


def create_logging_observer(verbose: bool = False) -> LoggingObserver:
    """Create a logging observer with appropriate settings."""
    return LoggingObserver()


def create_collecting_observer() -> CollectingObserver:
    """Create a collecting observer for testing."""
    return CollectingObserver()


# Helper to create events


def make_phase_start_event(phase: FlowPhase, previous: FlowPhase | None = None) -> PhaseEvent:
    """Create a PHASE_START event."""
    return PhaseEvent(
        event_type=PhaseEventType.PHASE_START,
        phase=phase,
        previous_phase=previous,
        message=f"Entering {phase.value} phase",
    )


def make_flow_start_event(message_preview: str = "") -> PhaseEvent:
    """Create a FLOW_START event."""
    return PhaseEvent(
        event_type=PhaseEventType.FLOW_START,
        data={"message_preview": message_preview[:100] if message_preview else ""},
        message="Starting new message flow",
    )


def make_flow_complete_event(flow: FlowState) -> PhaseEvent:
    """Create a FLOW_COMPLETE event from final flow state."""
    phases = flow.metrics.phases_executed if flow.metrics else []
    return PhaseEvent(
        event_type=PhaseEventType.FLOW_COMPLETE,
        phase=flow.phase,
        data={
            "phases_executed": phases,
            "complexity": flow.complexity.value if flow.complexity else None,
            "metrics": flow.metrics.to_dict() if flow.metrics else None,
        },
        message="Flow completed successfully",
    )


def make_flow_failed_event(flow: FlowState, error: str) -> PhaseEvent:
    """Create a FLOW_FAILED event."""
    return PhaseEvent(
        event_type=PhaseEventType.FLOW_FAILED,
        phase=FlowPhase.FAILED,
        data={
            "error": error,
            "phases_executed": flow.metrics.phases_executed if flow.metrics else [],
        },
        message=f"Flow failed: {error}",
    )


def make_info_assessment_event(assessment: InfoAssessment) -> PhaseEvent:
    """Create an INFO_ASSESSMENT event."""
    return PhaseEvent(
        event_type=PhaseEventType.INFO_ASSESSMENT,
        phase=FlowPhase.INFO,
        data={
            "complexity": assessment.complexity,
            "need_clarification": assessment.need_clarification,
            "need_web": assessment.need_web,
            "need_literature": assessment.need_literature,
            "need_project_files": assessment.need_project_files,
        },
        message=f"Assessment complete: complexity={assessment.complexity:.2f}",
    )


def make_plan_created_event(plan: PlanV02) -> PhaseEvent:
    """Create a PLAN_CREATED event."""
    return PhaseEvent(
        event_type=PhaseEventType.PLAN_CREATED,
        phase=FlowPhase.PLAN,
        data={
            "step_count": len(plan.steps),
            "quality": plan.quality_score,
            "steps": [s.description for s in plan.steps],
        },
        message=f"Plan created: {len(plan.steps)} steps, quality={plan.quality_score:.2f}",
    )


def make_plan_revision_event(plan: PlanV02, iteration: int) -> PhaseEvent:
    """Create a PLAN_REVISION event."""
    return PhaseEvent(
        event_type=PhaseEventType.PLAN_REVISION,
        phase=FlowPhase.PLAN,
        data={
            "iteration": iteration,
            "quality": plan.quality_score,
        },
        message=f"Plan revision {iteration}: quality={plan.quality_score:.2f}",
    )


def make_validation_event(validation: ValidationResult) -> PhaseEvent:
    """Create a VALIDATION_RESULT event."""
    return PhaseEvent(
        event_type=PhaseEventType.VALIDATION_RESULT,
        phase=FlowPhase.VALIDATE,
        data={
            "accomplished": validation.task_accomplished,
            "verified": validation.verified,
            "issue_count": len(validation.issues),
            "issues": validation.issues,
            "can_fix": validation.can_fix_without_replan,
        },
        message=f"Validation: accomplished={validation.task_accomplished:.2f}, verified={validation.verified:.2f}",
    )


def make_clarification_event(questions: list[str]) -> PhaseEvent:
    """Create a CLARIFICATION_NEEDED event."""
    return PhaseEvent(
        event_type=PhaseEventType.CLARIFICATION_NEEDED,
        phase=FlowPhase.INFO,
        data={"questions": questions},
        message=f"Clarification needed: {len(questions)} question(s)",
    )
