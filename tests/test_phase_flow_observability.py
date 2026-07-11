"""
Tests for PhaseFlowController observability integration.

Verifies that the controller emits the correct observability events
at each phase transition and for key decisions.
"""

import pytest
from dataclasses import dataclass
from typing import Any

from mesh.controller.phase_flow import PhaseFlowController
from mesh.controller.base import ControllerContext
from mesh.controller.models_v02 import FlowPhase, ComplexityLevel
from mesh.controller.observability import (
    CollectingObserver,
    PhaseEventType,
)
from mesh.config import ControllerConfigV02


@dataclass
class MockMessage:
    """Mock message for testing."""
    content: str


def make_context() -> ControllerContext:
    """Create a test context."""
    return ControllerContext(
        cwd="/test",
        history=[],
        agent_id="test-agent",
    )


# XML format that matches the phase detector's expected format
# (detector expects <tag>value</tag>, NOT <tag score="value">)
def make_assessment_xml(complexity: float, clarification: float = 0.0, web: float = 0.0,
                        literature: float = 0.0, project_files: float = 0.0) -> str:
    """Create a properly formatted INFO assessment XML."""
    return f"""<assessment>
<complexity>{complexity}</complexity>
<need_clarification>{clarification}</need_clarification>
<need_web>{web}</need_web>
<need_literature>{literature}</need_literature>
<need_project_files>{project_files}</need_project_files>
</assessment>"""


def make_plan_xml(steps: list[str], quality: float = 0.9, rollback: str = "N/A") -> str:
    """Create a properly formatted PLAN XML."""
    steps_xml = "\n".join(f"<step>{s}</step>" for s in steps)
    return f"""<plan>
<quality>{quality}</quality>
<steps>
{steps_xml}
</steps>
<rollback>{rollback}</rollback>
</plan>"""


def make_validation_xml(accomplished: float, verified: float, issues: list[str] | None = None,
                        can_fix: bool = False, fix_actions: str | None = None) -> str:
    """Create a properly formatted VALIDATION XML."""
    issues_xml = ""
    if issues:
        issues_items = "\n".join(f"<issue>{i}</issue>" for i in issues)
        issues_xml = f"<issues>{issues_items}</issues>"
    else:
        issues_xml = "<issues></issues>"

    fix_xml = f"<fix_actions>{fix_actions}</fix_actions>" if fix_actions else ""

    return f"""<validation>
<task_accomplished>{accomplished}</task_accomplished>
<verified>{verified}</verified>
{issues_xml}
<can_fix_without_replan>{'true' if can_fix else 'false'}</can_fix_without_replan>
{fix_xml}
</validation>"""


# =============================================================================
# Basic event emission tests
# =============================================================================


class TestFlowStartEvents:
    """Tests for flow start event emission."""

    @pytest.mark.asyncio
    async def test_emits_flow_start(self):
        """Test FLOW_START event is emitted on new message."""
        observer = CollectingObserver()
        controller = PhaseFlowController(observer=observer)

        msg = MockMessage(content="test message")
        await controller.on_message(msg, make_context())

        flow_starts = observer.get_events(PhaseEventType.FLOW_START)
        assert len(flow_starts) == 1
        assert flow_starts[0].data["message_preview"] == "test message"

    @pytest.mark.asyncio
    async def test_emits_info_phase_start(self):
        """Test PHASE_START for INFO is emitted on new message."""
        observer = CollectingObserver()
        controller = PhaseFlowController(observer=observer)

        msg = MockMessage(content="test message")
        await controller.on_message(msg, make_context())

        phase_starts = observer.get_events(PhaseEventType.PHASE_START)
        assert len(phase_starts) == 1
        assert phase_starts[0].phase == FlowPhase.INFO


class TestInfoPhaseEvents:
    """Tests for INFO phase event emission."""

    @pytest.mark.asyncio
    async def test_emits_info_assessment(self):
        """Test INFO_ASSESSMENT event is emitted after parsing assessment."""
        observer = CollectingObserver()
        controller = PhaseFlowController(observer=observer)

        # Start flow
        msg = MockMessage(content="test message")
        await controller.on_message(msg, make_context())
        observer.clear()

        # Simulate LLM response with assessment (low clarification to avoid clarification path)
        await controller.on_llm_response(make_assessment_xml(0.3, 0.0, 0.0, 0.0, 0.0), [], make_context())

        assessments = observer.get_events(PhaseEventType.INFO_ASSESSMENT)
        assert len(assessments) == 1
        assert assessments[0].data["complexity"] == 0.3

    @pytest.mark.asyncio
    async def test_emits_clarification_event(self):
        """Test CLARIFICATION_NEEDED event when clarification is needed."""
        config = ControllerConfigV02()
        observer = CollectingObserver()
        controller = PhaseFlowController(config=config, observer=observer)

        # Start flow
        msg = MockMessage(content="test message")
        await controller.on_message(msg, make_context())
        observer.clear()

        # Simulate high clarification score (0.9 > default threshold 0.3)
        await controller.on_llm_response(make_assessment_xml(0.5, 0.9, 0.0, 0.0, 0.0), [], make_context())

        clarifications = observer.get_events(PhaseEventType.CLARIFICATION_NEEDED)
        assert len(clarifications) == 1


class TestComplexityRoutingEvents:
    """Tests for complexity-based routing events."""

    @pytest.mark.asyncio
    async def test_low_complexity_emits_execute_phase_start(self):
        """Test low complexity routes to EXECUTE and emits phase start."""
        observer = CollectingObserver()
        controller = PhaseFlowController(observer=observer)

        # Start flow
        msg = MockMessage(content="test message")
        await controller.on_message(msg, make_context())

        # Low complexity assessment (< 0.3 default threshold)
        await controller.on_llm_response(make_assessment_xml(0.1), [], make_context())

        # Should have INFO + EXECUTE phase starts
        transitions = observer.get_phase_transitions()
        assert FlowPhase.INFO.value in transitions
        assert FlowPhase.EXECUTE.value in transitions

    @pytest.mark.asyncio
    async def test_moderate_complexity_emits_plan_phase_start(self):
        """Test moderate complexity routes to PLAN and emits phase start."""
        observer = CollectingObserver()
        controller = PhaseFlowController(observer=observer)

        # Start flow
        msg = MockMessage(content="test message")
        await controller.on_message(msg, make_context())

        # Moderate complexity (0.3 <= x < 0.7)
        await controller.on_llm_response(make_assessment_xml(0.5), [], make_context())

        transitions = observer.get_phase_transitions()
        assert FlowPhase.INFO.value in transitions
        assert FlowPhase.PLAN.value in transitions


class TestPlanPhaseEvents:
    """Tests for PLAN phase event emission."""

    @pytest.mark.asyncio
    async def test_emits_plan_created(self):
        """Test PLAN_CREATED event is emitted after parsing plan."""
        observer = CollectingObserver()
        controller = PhaseFlowController(observer=observer)

        # Setup: get to PLAN phase
        msg = MockMessage(content="test message")
        await controller.on_message(msg, make_context())
        await controller.on_llm_response(make_assessment_xml(0.5), [], make_context())

        # PLAN phase response
        await controller.on_llm_response(
            make_plan_xml(["Read file", "Edit file"], quality=0.85),
            [], make_context()
        )

        plan_events = observer.get_events(PhaseEventType.PLAN_CREATED)
        assert len(plan_events) == 1
        assert plan_events[0].data["step_count"] == 2
        assert plan_events[0].data["quality"] == 0.85

    @pytest.mark.asyncio
    async def test_emits_plan_revision_on_low_quality(self):
        """Test PLAN_REVISION event when plan quality is too low."""
        config = ControllerConfigV02()
        observer = CollectingObserver()
        controller = PhaseFlowController(config=config, observer=observer)

        # Setup: get to PLAN phase
        msg = MockMessage(content="test message")
        await controller.on_message(msg, make_context())
        await controller.on_llm_response(make_assessment_xml(0.5), [], make_context())

        # Low quality plan (< 0.8 default threshold)
        await controller.on_llm_response(
            make_plan_xml(["Do something"], quality=0.5),
            [], make_context()
        )

        revision_events = observer.get_events(PhaseEventType.PLAN_REVISION)
        assert len(revision_events) == 1
        assert revision_events[0].data["iteration"] == 1


class TestValidatePhaseEvents:
    """Tests for VALIDATE phase event emission."""

    @pytest.mark.asyncio
    async def test_emits_validation_result(self):
        """Test VALIDATION_RESULT event is emitted."""
        observer = CollectingObserver()
        controller = PhaseFlowController(observer=observer)

        # Setup: get to VALIDATE phase through moderate complexity flow
        msg = MockMessage(content="test message")
        await controller.on_message(msg, make_context())

        # INFO phase - moderate complexity
        await controller.on_llm_response(make_assessment_xml(0.5), [], make_context())

        # PLAN phase
        await controller.on_llm_response(make_plan_xml(["Do task"], quality=0.9), [], make_context())

        # EXECUTE phase (no tool calls)
        await controller.on_llm_response("Execution complete.", [], make_context())

        # VALIDATE phase
        await controller.on_llm_response(
            make_validation_xml(0.95, 0.9),
            [], make_context()
        )

        validation_events = observer.get_events(PhaseEventType.VALIDATION_RESULT)
        assert len(validation_events) == 1
        assert validation_events[0].data["accomplished"] == 0.95
        assert validation_events[0].data["verified"] == 0.9


class TestFlowCompleteEvents:
    """Tests for flow completion event emission."""

    @pytest.mark.asyncio
    async def test_emits_flow_complete_on_success(self):
        """Test FLOW_COMPLETE event on successful flow."""
        observer = CollectingObserver()
        controller = PhaseFlowController(observer=observer)

        # Simple low complexity flow
        msg = MockMessage(content="test message")
        await controller.on_message(msg, make_context())

        # Low complexity assessment
        await controller.on_llm_response(make_assessment_xml(0.1), [], make_context())

        # EXECUTE phase completes
        await controller.on_llm_response("Done!", [], make_context())

        complete_events = observer.get_events(PhaseEventType.FLOW_COMPLETE)
        assert len(complete_events) == 1

    @pytest.mark.asyncio
    async def test_emits_flow_failed_on_validation_failure(self):
        """Test FLOW_FAILED event when validation fails without fix."""
        observer = CollectingObserver()
        controller = PhaseFlowController(observer=observer)

        # Setup: get to VALIDATE phase
        msg = MockMessage(content="test message")
        await controller.on_message(msg, make_context())
        await controller.on_llm_response(make_assessment_xml(0.5), [], make_context())
        await controller.on_llm_response(make_plan_xml(["Do task"], quality=0.9), [], make_context())
        await controller.on_llm_response("Executed.", [], make_context())

        # Failed validation with no fix possible
        await controller.on_llm_response(
            make_validation_xml(0.2, 0.1, issues=["Major error"], can_fix=False),
            [], make_context()
        )

        failed_events = observer.get_events(PhaseEventType.FLOW_FAILED)
        assert len(failed_events) == 1
        assert "Validation failed" in failed_events[0].data["error"]


class TestObserverConfiguration:
    """Tests for observer configuration."""

    @pytest.mark.asyncio
    async def test_set_observer_runtime(self):
        """Test observer can be changed at runtime."""
        controller = PhaseFlowController()

        observer1 = CollectingObserver()
        controller.set_observer(observer1)

        msg = MockMessage(content="message 1")
        await controller.on_message(msg, make_context())
        assert len(observer1.events) > 0

        # Switch observer
        observer2 = CollectingObserver()
        controller.set_observer(observer2)

        msg2 = MockMessage(content="message 2")
        await controller.on_message(msg2, make_context())
        assert len(observer2.events) > 0
        # observer1 should not have new events
        count1 = len(observer1.events)

        await controller.on_message(MockMessage(content="message 3"), make_context())
        assert len(observer1.events) == count1  # unchanged

    @pytest.mark.asyncio
    async def test_default_logging_observer(self):
        """Test default observer is LoggingObserver (no crash)."""
        controller = PhaseFlowController()

        # Should not crash with default observer
        msg = MockMessage(content="test")
        await controller.on_message(msg, make_context())

        await controller.on_llm_response(make_assessment_xml(0.1), [], make_context())


class TestFullFlowEventSequence:
    """Tests for complete flow event sequences."""

    @pytest.mark.asyncio
    async def test_low_complexity_flow_events(self):
        """Test event sequence for low complexity flow."""
        observer = CollectingObserver()
        controller = PhaseFlowController(observer=observer)

        # Execute full low complexity flow
        msg = MockMessage(content="simple question")
        await controller.on_message(msg, make_context())
        await controller.on_llm_response(make_assessment_xml(0.1), [], make_context())
        await controller.on_llm_response("Here's the answer.", [], make_context())

        # Expected: FLOW_START, PHASE_START(INFO), INFO_ASSESSMENT, PHASE_START(EXECUTE), FLOW_COMPLETE
        event_types = [e.event_type for e in observer.events]

        assert PhaseEventType.FLOW_START in event_types
        assert PhaseEventType.INFO_ASSESSMENT in event_types
        assert PhaseEventType.FLOW_COMPLETE in event_types

        # Check phase transitions: INFO -> EXECUTE
        transitions = observer.get_phase_transitions()
        assert transitions == [FlowPhase.INFO.value, FlowPhase.EXECUTE.value]

    @pytest.mark.asyncio
    async def test_moderate_complexity_flow_events(self):
        """Test event sequence for moderate complexity flow."""
        observer = CollectingObserver()
        controller = PhaseFlowController(observer=observer)

        # Execute full moderate complexity flow
        msg = MockMessage(content="moderate task")
        await controller.on_message(msg, make_context())
        await controller.on_llm_response(make_assessment_xml(0.5), [], make_context())
        await controller.on_llm_response(make_plan_xml(["Step 1"], quality=0.9), [], make_context())
        await controller.on_llm_response("Executed.", [], make_context())
        await controller.on_llm_response(make_validation_xml(1.0, 1.0), [], make_context())

        # Expected transitions: INFO -> PLAN -> EXECUTE -> VALIDATE
        transitions = observer.get_phase_transitions()
        assert transitions == [
            FlowPhase.INFO.value,
            FlowPhase.PLAN.value,
            FlowPhase.EXECUTE.value,
            FlowPhase.VALIDATE.value,
        ]

        # Should have key events
        event_types = [e.event_type for e in observer.events]
        assert PhaseEventType.FLOW_START in event_types
        assert PhaseEventType.INFO_ASSESSMENT in event_types
        assert PhaseEventType.PLAN_CREATED in event_types
        assert PhaseEventType.VALIDATION_RESULT in event_types
        assert PhaseEventType.FLOW_COMPLETE in event_types
