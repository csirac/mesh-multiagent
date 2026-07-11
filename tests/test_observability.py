"""
Tests for the v0.2 controller observability module.

Tests cover:
- PhaseEvent creation and formatting
- Observer implementations (Logging, Collecting, Composite)
- Event factory functions
- Integration with FlowState
"""

import pytest
from datetime import datetime

from mesh.controller.observability import (
    PhaseEventType,
    PhaseEvent,
    ObservabilityEmitter,
    LoggingObserver,
    CollectingObserver,
    CompositeObserver,
    create_logging_observer,
    create_collecting_observer,
    make_phase_start_event,
    make_flow_start_event,
    make_flow_complete_event,
    make_flow_failed_event,
    make_info_assessment_event,
    make_plan_created_event,
    make_plan_revision_event,
    make_validation_event,
    make_clarification_event,
)
from mesh.controller.models_v02 import (
    FlowPhase,
    FlowState,
    FlowMetrics,
    ComplexityLevel,
    InfoAssessment,
    PlanV02,
    PlanStepV02,
    ValidationResult,
)


# =============================================================================
# PhaseEventType tests
# =============================================================================


class TestPhaseEventType:
    """Tests for PhaseEventType enum."""

    def test_phase_transition_types(self):
        """Test phase transition event types exist."""
        assert PhaseEventType.PHASE_START.value == "phase_start"
        assert PhaseEventType.PHASE_END.value == "phase_end"

    def test_flow_event_types(self):
        """Test flow-level event types exist."""
        assert PhaseEventType.FLOW_START.value == "flow_start"
        assert PhaseEventType.FLOW_COMPLETE.value == "flow_complete"
        assert PhaseEventType.FLOW_FAILED.value == "flow_failed"

    def test_assessment_event_types(self):
        """Test assessment-related event types exist."""
        assert PhaseEventType.INFO_ASSESSMENT.value == "info_assessment"
        assert PhaseEventType.PLAN_CREATED.value == "plan_created"
        assert PhaseEventType.PLAN_REVISION.value == "plan_revision"
        assert PhaseEventType.VALIDATION_RESULT.value == "validation_result"

    def test_tool_event_types(self):
        """Test tool activity event types exist."""
        assert PhaseEventType.TOOL_CALL.value == "tool_call"
        assert PhaseEventType.TOOL_RESULT.value == "tool_result"


# =============================================================================
# PhaseEvent tests
# =============================================================================


class TestPhaseEvent:
    """Tests for PhaseEvent data class."""

    def test_basic_creation(self):
        """Test basic event creation."""
        event = PhaseEvent(
            event_type=PhaseEventType.PHASE_START,
            phase=FlowPhase.INFO,
            message="Starting INFO phase",
        )
        assert event.event_type == PhaseEventType.PHASE_START
        assert event.phase == FlowPhase.INFO
        assert event.message == "Starting INFO phase"
        assert event.timestamp  # Auto-generated

    def test_timestamp_auto_generation(self):
        """Test timestamp is auto-generated if not provided."""
        event = PhaseEvent(event_type=PhaseEventType.FLOW_START)
        assert event.timestamp
        assert "Z" in event.timestamp  # ISO format with Z suffix

    def test_timestamp_preserved_if_provided(self):
        """Test provided timestamp is preserved."""
        ts = "2026-02-04T12:00:00Z"
        event = PhaseEvent(event_type=PhaseEventType.FLOW_START, timestamp=ts)
        assert event.timestamp == ts

    def test_to_dict(self):
        """Test serialization to dict."""
        event = PhaseEvent(
            event_type=PhaseEventType.PHASE_START,
            phase=FlowPhase.PLAN,
            previous_phase=FlowPhase.INFO,
            data={"quality": 0.85},
            message="Entering PLAN phase",
        )
        d = event.to_dict()
        assert d["event_type"] == "phase_start"
        assert d["phase"] == "plan"
        assert d["previous_phase"] == "info"
        assert d["data"] == {"quality": 0.85}
        assert d["message"] == "Entering PLAN phase"

    def test_to_dict_with_none_phases(self):
        """Test serialization handles None phases."""
        event = PhaseEvent(event_type=PhaseEventType.FLOW_START)
        d = event.to_dict()
        assert d["phase"] is None
        assert d["previous_phase"] is None


class TestPhaseEventFormatting:
    """Tests for PhaseEvent.format_for_user()."""

    def test_format_phase_start(self):
        """Test PHASE_START formatting."""
        event = PhaseEvent(
            event_type=PhaseEventType.PHASE_START,
            phase=FlowPhase.INFO,
        )
        assert event.format_for_user() == "[PHASE: INFO]"

    def test_format_flow_start(self):
        """Test FLOW_START formatting."""
        event = PhaseEvent(event_type=PhaseEventType.FLOW_START)
        assert event.format_for_user() == "[FLOW: Starting new message processing]"

    def test_format_flow_complete(self):
        """Test FLOW_COMPLETE formatting."""
        event = PhaseEvent(
            event_type=PhaseEventType.FLOW_COMPLETE,
            data={"phases_executed": ["info", "plan", "execute"]},
        )
        assert "info → plan → execute" in event.format_for_user()

    def test_format_flow_failed(self):
        """Test FLOW_FAILED formatting."""
        event = PhaseEvent(
            event_type=PhaseEventType.FLOW_FAILED,
            data={"error": "Validation failed"},
        )
        formatted = event.format_for_user()
        assert "Failed" in formatted
        assert "Validation failed" in formatted

    def test_format_info_assessment(self):
        """Test INFO_ASSESSMENT formatting."""
        event = PhaseEvent(
            event_type=PhaseEventType.INFO_ASSESSMENT,
            data={"complexity": 0.65},
        )
        assert "0.65" in event.format_for_user()

    def test_format_plan_created(self):
        """Test PLAN_CREATED formatting."""
        event = PhaseEvent(
            event_type=PhaseEventType.PLAN_CREATED,
            data={"step_count": 5, "quality": 0.85},
        )
        formatted = event.format_for_user()
        assert "5 steps" in formatted
        assert "0.85" in formatted

    def test_format_validation_passed(self):
        """Test VALIDATION_RESULT formatting (passed)."""
        event = PhaseEvent(
            event_type=PhaseEventType.VALIDATION_RESULT,
            data={"accomplished": 0.9, "verified": 0.85, "issue_count": 0},
        )
        formatted = event.format_for_user()
        assert "PASSED" in formatted
        assert "0.9" in formatted

    def test_format_validation_issues(self):
        """Test VALIDATION_RESULT formatting (with issues)."""
        event = PhaseEvent(
            event_type=PhaseEventType.VALIDATION_RESULT,
            data={"accomplished": 0.5, "verified": 0.4, "issue_count": 2},
        )
        formatted = event.format_for_user()
        assert "ISSUES" in formatted
        assert "0.5" in formatted

    def test_format_tool_call(self):
        """Test TOOL_CALL formatting."""
        event = PhaseEvent(
            event_type=PhaseEventType.TOOL_CALL,
            data={"tool": "file_read"},
        )
        assert "[TOOL: Calling file_read]" == event.format_for_user()

    def test_format_tool_result_success(self):
        """Test TOOL_RESULT formatting (success)."""
        event = PhaseEvent(
            event_type=PhaseEventType.TOOL_RESULT,
            data={"tool": "file_read", "success": True},
        )
        assert "[TOOL: file_read - OK]" == event.format_for_user()

    def test_format_tool_result_failure(self):
        """Test TOOL_RESULT formatting (failure)."""
        event = PhaseEvent(
            event_type=PhaseEventType.TOOL_RESULT,
            data={"tool": "file_read", "success": False},
        )
        assert "[TOOL: file_read - FAILED]" == event.format_for_user()


# =============================================================================
# Observer implementation tests
# =============================================================================


class TestCollectingObserver:
    """Tests for CollectingObserver."""

    @pytest.mark.asyncio
    async def test_collect_events(self):
        """Test events are collected."""
        observer = CollectingObserver()
        event1 = PhaseEvent(event_type=PhaseEventType.FLOW_START)
        event2 = PhaseEvent(event_type=PhaseEventType.PHASE_START, phase=FlowPhase.INFO)

        await observer.emit(event1)
        await observer.emit(event2)

        assert len(observer.events) == 2
        assert observer.events[0].event_type == PhaseEventType.FLOW_START
        assert observer.events[1].event_type == PhaseEventType.PHASE_START

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing collected events."""
        observer = CollectingObserver()
        await observer.emit(PhaseEvent(event_type=PhaseEventType.FLOW_START))
        assert len(observer.events) == 1

        observer.clear()
        assert len(observer.events) == 0

    @pytest.mark.asyncio
    async def test_get_events_filtered(self):
        """Test filtering events by type."""
        observer = CollectingObserver()
        await observer.emit(PhaseEvent(event_type=PhaseEventType.FLOW_START))
        await observer.emit(PhaseEvent(event_type=PhaseEventType.PHASE_START, phase=FlowPhase.INFO))
        await observer.emit(PhaseEvent(event_type=PhaseEventType.PHASE_START, phase=FlowPhase.PLAN))
        await observer.emit(PhaseEvent(event_type=PhaseEventType.FLOW_COMPLETE))

        phase_starts = observer.get_events(PhaseEventType.PHASE_START)
        assert len(phase_starts) == 2

        all_events = observer.get_events()
        assert len(all_events) == 4

    @pytest.mark.asyncio
    async def test_get_phase_transitions(self):
        """Test getting list of phase transitions."""
        observer = CollectingObserver()
        await observer.emit(PhaseEvent(event_type=PhaseEventType.FLOW_START))
        await observer.emit(PhaseEvent(event_type=PhaseEventType.PHASE_START, phase=FlowPhase.INFO))
        await observer.emit(PhaseEvent(event_type=PhaseEventType.PHASE_START, phase=FlowPhase.PLAN))
        await observer.emit(PhaseEvent(event_type=PhaseEventType.PHASE_START, phase=FlowPhase.EXECUTE))

        transitions = observer.get_phase_transitions()
        assert transitions == ["info", "plan", "execute"]


class TestCompositeObserver:
    """Tests for CompositeObserver."""

    @pytest.mark.asyncio
    async def test_forwards_to_all(self):
        """Test events are forwarded to all observers."""
        obs1 = CollectingObserver()
        obs2 = CollectingObserver()
        composite = CompositeObserver([obs1, obs2])

        event = PhaseEvent(event_type=PhaseEventType.FLOW_START)
        await composite.emit(event)

        assert len(obs1.events) == 1
        assert len(obs2.events) == 1

    @pytest.mark.asyncio
    async def test_add_observer(self):
        """Test adding observers dynamically."""
        composite = CompositeObserver()
        obs = CollectingObserver()
        composite.add(obs)

        await composite.emit(PhaseEvent(event_type=PhaseEventType.FLOW_START))
        assert len(obs.events) == 1


class TestLoggingObserver:
    """Tests for LoggingObserver."""

    @pytest.mark.asyncio
    async def test_does_not_crash(self):
        """Test logging observer doesn't crash on emit."""
        observer = LoggingObserver()

        # Should not raise
        await observer.emit(PhaseEvent(event_type=PhaseEventType.FLOW_START))
        await observer.emit(PhaseEvent(event_type=PhaseEventType.PHASE_START, phase=FlowPhase.INFO))
        await observer.emit(PhaseEvent(event_type=PhaseEventType.FLOW_FAILED, data={"error": "test"}))


# =============================================================================
# Event factory function tests
# =============================================================================


class TestEventFactories:
    """Tests for event factory functions."""

    def test_make_phase_start_event(self):
        """Test make_phase_start_event."""
        event = make_phase_start_event(FlowPhase.PLAN, FlowPhase.INFO)
        assert event.event_type == PhaseEventType.PHASE_START
        assert event.phase == FlowPhase.PLAN
        assert event.previous_phase == FlowPhase.INFO
        assert "Entering plan phase" in event.message

    def test_make_flow_start_event(self):
        """Test make_flow_start_event."""
        event = make_flow_start_event("Hello world")
        assert event.event_type == PhaseEventType.FLOW_START
        assert event.data["message_preview"] == "Hello world"

    def test_make_flow_start_event_truncates(self):
        """Test message preview is truncated."""
        long_msg = "x" * 200
        event = make_flow_start_event(long_msg)
        assert len(event.data["message_preview"]) == 100

    def test_make_flow_complete_event(self):
        """Test make_flow_complete_event."""
        flow = FlowState(
            phase=FlowPhase.DONE,
            complexity=ComplexityLevel.MODERATE,
            metrics=FlowMetrics(),
        )
        flow.metrics.phases_executed = ["info", "plan", "execute", "validate"]
        flow.complete()

        event = make_flow_complete_event(flow)
        assert event.event_type == PhaseEventType.FLOW_COMPLETE
        assert event.phase == FlowPhase.DONE
        assert event.data["phases_executed"] == ["info", "plan", "execute", "validate"]
        assert event.data["complexity"] == "moderate"

    def test_make_flow_failed_event(self):
        """Test make_flow_failed_event."""
        flow = FlowState(phase=FlowPhase.VALIDATE)
        event = make_flow_failed_event(flow, "Validation failed: test error")
        assert event.event_type == PhaseEventType.FLOW_FAILED
        assert event.phase == FlowPhase.FAILED
        assert event.data["error"] == "Validation failed: test error"

    def test_make_info_assessment_event(self):
        """Test make_info_assessment_event."""
        assessment = InfoAssessment(
            complexity=0.65,
            need_clarification=0.1,
            need_web=0.8,
            need_literature=0.0,
            need_project_files=0.5,
        )
        event = make_info_assessment_event(assessment)
        assert event.event_type == PhaseEventType.INFO_ASSESSMENT
        assert event.phase == FlowPhase.INFO
        assert event.data["complexity"] == 0.65
        assert event.data["need_web"] == 0.8

    def test_make_plan_created_event(self):
        """Test make_plan_created_event."""
        plan = PlanV02(
            steps=[
                PlanStepV02(number=1, description="Read file"),
                PlanStepV02(number=2, description="Edit file"),
            ],
            quality_score=0.85,
        )
        event = make_plan_created_event(plan)
        assert event.event_type == PhaseEventType.PLAN_CREATED
        assert event.phase == FlowPhase.PLAN
        assert event.data["step_count"] == 2
        assert event.data["quality"] == 0.85
        assert "Read file" in event.data["steps"]

    def test_make_plan_revision_event(self):
        """Test make_plan_revision_event."""
        plan = PlanV02(quality_score=0.6)
        event = make_plan_revision_event(plan, iteration=2)
        assert event.event_type == PhaseEventType.PLAN_REVISION
        assert event.data["iteration"] == 2
        assert event.data["quality"] == 0.6

    def test_make_validation_event(self):
        """Test make_validation_event."""
        validation = ValidationResult(
            task_accomplished=0.9,
            verified=0.85,
            issues=["minor warning"],
            can_fix_without_replan=True,
        )
        event = make_validation_event(validation)
        assert event.event_type == PhaseEventType.VALIDATION_RESULT
        assert event.phase == FlowPhase.VALIDATE
        assert event.data["accomplished"] == 0.9
        assert event.data["verified"] == 0.85
        assert event.data["issue_count"] == 1
        assert event.data["can_fix"] is True

    def test_make_clarification_event(self):
        """Test make_clarification_event."""
        questions = ["What file?", "Which format?"]
        event = make_clarification_event(questions)
        assert event.event_type == PhaseEventType.CLARIFICATION_NEEDED
        assert event.phase == FlowPhase.INFO
        assert event.data["questions"] == questions


# =============================================================================
# Protocol compliance tests
# =============================================================================


class TestProtocolCompliance:
    """Tests verifying protocol compliance."""

    def test_collecting_observer_is_emitter(self):
        """Test CollectingObserver implements ObservabilityEmitter."""
        observer = CollectingObserver()
        assert isinstance(observer, ObservabilityEmitter)

    def test_composite_observer_is_emitter(self):
        """Test CompositeObserver implements ObservabilityEmitter."""
        observer = CompositeObserver()
        assert isinstance(observer, ObservabilityEmitter)

    def test_logging_observer_is_emitter(self):
        """Test LoggingObserver implements ObservabilityEmitter."""
        observer = LoggingObserver()
        assert isinstance(observer, ObservabilityEmitter)


# =============================================================================
# Factory function tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for observer factory functions."""

    def test_create_logging_observer(self):
        """Test create_logging_observer."""
        observer = create_logging_observer()
        assert isinstance(observer, LoggingObserver)

    def test_create_collecting_observer(self):
        """Test create_collecting_observer."""
        observer = create_collecting_observer()
        assert isinstance(observer, CollectingObserver)
