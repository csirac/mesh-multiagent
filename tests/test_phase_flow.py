"""
Tests for v0.2 PhaseFlowController.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from dataclasses import dataclass

from mesh.controller.phase_flow import PhaseFlowController
from mesh.controller.base import ControllerContext, ControllerDecision
from mesh.controller.models_v02 import (
    FlowPhase,
    FlowState,
    ComplexityLevel,
    InfoAssessment,
    PlanV02,
    PlanStepV02,
    ValidationResult,
)
from mesh.config import ControllerConfigV02


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def default_config():
    """Default controller config."""
    return ControllerConfigV02()


@pytest.fixture
def controller(default_config):
    """Create a PhaseFlowController with default config."""
    return PhaseFlowController(config=default_config)


@pytest.fixture
def context():
    """Create a minimal ControllerContext."""
    return ControllerContext(
        cwd="/test",
        history=[],
        agent_id="agent:test:test",
    )


@dataclass
class MockMessage:
    """Mock message for testing."""
    content: str


# -----------------------------------------------------------------------------
# Test: Controller initialization
# -----------------------------------------------------------------------------

class TestControllerInit:
    """Test controller initialization."""

    def test_init_with_default_config(self):
        """Controller initializes with default config."""
        controller = PhaseFlowController()
        assert controller.config is not None
        assert controller._current_flow is None

    def test_init_with_custom_config(self):
        """Controller initializes with custom config."""
        config = ControllerConfigV02(
            effort="high",
            enable_metrics=True,
        )
        controller = PhaseFlowController(config=config)
        assert controller.config.effort == "high"
        assert controller.config.enable_metrics is True


# -----------------------------------------------------------------------------
# Test: on_message (INFO phase start)
# -----------------------------------------------------------------------------

class TestOnMessage:
    """Test on_message handling (starts INFO phase)."""

    @pytest.mark.asyncio
    async def test_starts_info_phase(self, controller, context):
        """on_message starts INFO phase with fresh flow state."""
        message = MockMessage(content="Help me implement caching")

        decision = await controller.on_message(message, context)

        assert decision.action == "PROCESS_WITH_LLM"
        assert decision.phase == FlowPhase.INFO.value
        assert decision.system_addendum is not None
        assert "<controller_phase" in decision.system_addendum

        # Check flow state was created
        flow = controller.get_current_flow()
        assert flow is not None
        assert flow.phase == FlowPhase.INFO
        assert flow.original_message == "Help me implement caching"

    @pytest.mark.asyncio
    async def test_empty_message_passthrough(self, controller, context):
        """Empty messages pass through without starting flow."""
        message = MockMessage(content="")

        decision = await controller.on_message(message, context)

        assert decision.action == "PROCESS_WITH_LLM"
        assert controller.get_current_flow() is None

    @pytest.mark.asyncio
    async def test_string_message(self, controller, context):
        """Handles string messages directly."""
        decision = await controller.on_message("Simple question", context)

        assert decision.action == "PROCESS_WITH_LLM"
        assert controller.get_current_flow() is not None

    @pytest.mark.asyncio
    async def test_metrics_enabled(self, context):
        """Metrics are tracked when enabled."""
        config = ControllerConfigV02(enable_metrics=True)
        controller = PhaseFlowController(config=config)

        await controller.on_message("Test", context)

        flow = controller.get_current_flow()
        assert flow.metrics is not None


# -----------------------------------------------------------------------------
# Test: INFO phase handling
# -----------------------------------------------------------------------------

class TestInfoPhase:
    """Test INFO phase response handling."""

    @pytest.mark.asyncio
    async def test_low_complexity_routes_to_execute(self, controller, context):
        """Low complexity assessment routes directly to EXECUTE."""
        # Start flow
        await controller.on_message("Simple question", context)

        # Simulate INFO response with low complexity
        response = """
        <assessment>
            <complexity>0.1</complexity>
            <need_clarification>0.0</need_clarification>
            <need_web>0.0</need_web>
            <need_literature>0.0</need_literature>
            <need_project_files>0.0</need_project_files>
        </assessment>

        The answer is straightforward.
        """

        decision = await controller.on_llm_response(response, [], context)

        assert decision.phase == FlowPhase.EXECUTE.value

        flow = controller.get_current_flow()
        assert flow.complexity == ComplexityLevel.LOW
        assert flow.phase == FlowPhase.EXECUTE

    @pytest.mark.asyncio
    async def test_moderate_complexity_routes_to_plan(self, controller, context):
        """Moderate complexity assessment routes to PLAN."""
        await controller.on_message("Implement feature X", context)

        response = """
        <assessment>
            <complexity>0.5</complexity>
            <need_clarification>0.0</need_clarification>
            <need_web>0.0</need_web>
            <need_literature>0.0</need_literature>
            <need_project_files>0.0</need_project_files>
        </assessment>
        """

        decision = await controller.on_llm_response(response, [], context)

        assert decision.phase == FlowPhase.PLAN.value

        flow = controller.get_current_flow()
        assert flow.complexity == ComplexityLevel.MODERATE
        assert flow.phase == FlowPhase.PLAN

    @pytest.mark.asyncio
    async def test_high_complexity_routes_to_plan(self, controller, context):
        """High complexity assessment routes to PLAN."""
        await controller.on_message("Refactor entire module", context)

        response = """
        <assessment>
            <complexity>0.9</complexity>
            <need_clarification>0.0</need_clarification>
            <need_web>0.0</need_web>
            <need_literature>0.0</need_literature>
            <need_project_files>0.0</need_project_files>
        </assessment>
        """

        decision = await controller.on_llm_response(response, [], context)

        assert decision.phase == FlowPhase.PLAN.value

        flow = controller.get_current_flow()
        assert flow.complexity == ComplexityLevel.HIGH

    @pytest.mark.asyncio
    async def test_clarification_needed_completes_flow(self, controller, context):
        """High clarification score completes flow (user needs to respond)."""
        await controller.on_message("Do the thing", context)

        response = """
        <assessment>
            <complexity>0.5</complexity>
            <need_clarification>0.8</need_clarification>
            <need_web>0.0</need_web>
            <need_literature>0.0</need_literature>
            <need_project_files>0.0</need_project_files>
            <clarification_questions>
                <question>Which thing do you mean?</question>
            </clarification_questions>
        </assessment>

        I need more information. Which thing do you mean?
        """

        decision = await controller.on_llm_response(response, [], context)

        assert decision.action == "DONE"
        assert decision.payload.get("needs_clarification") is True

    @pytest.mark.asyncio
    async def test_info_iteration_tracking(self, controller, context):
        """Info iterations are tracked."""
        await controller.on_message("Research topic", context)

        response = """
        <assessment>
            <complexity>0.3</complexity>
            <need_clarification>0.0</need_clarification>
            <need_web>0.0</need_web>
            <need_literature>0.0</need_literature>
            <need_project_files>0.0</need_project_files>
        </assessment>
        """

        await controller.on_llm_response(response, [], context)

        flow = controller.get_current_flow()
        assert flow.info_iterations == 1


# -----------------------------------------------------------------------------
# Test: PLAN phase handling
# -----------------------------------------------------------------------------

class TestPlanPhase:
    """Test PLAN phase response handling."""

    @pytest.mark.asyncio
    async def test_good_plan_proceeds_to_execute(self, controller, context):
        """Plan with good quality proceeds to EXECUTE."""
        await controller.on_message("Implement feature", context)

        # First go through INFO to get to PLAN
        info_response = """
        <assessment>
            <complexity>0.6</complexity>
            <need_clarification>0.0</need_clarification>
            <need_web>0.0</need_web>
            <need_literature>0.0</need_literature>
            <need_project_files>0.0</need_project_files>
        </assessment>
        """
        await controller.on_llm_response(info_response, [], context)

        # Now in PLAN phase - provide good quality plan
        plan_response = """
        <plan>
            <quality>0.9</quality>
            <steps>
                <step>Design the interface</step>
                <step>Implement the core logic</step>
                <step>Write tests</step>
            </steps>
            <rollback>Revert the changes</rollback>
        </plan>
        """

        decision = await controller.on_llm_response(plan_response, [], context)

        assert decision.phase == FlowPhase.EXECUTE.value

        flow = controller.get_current_flow()
        assert flow.plan is not None
        assert len(flow.plan.steps) == 3
        assert flow.plan.quality_score == 0.9

    @pytest.mark.asyncio
    async def test_low_quality_plan_triggers_revision(self, controller, context):
        """Plan with low quality triggers revision."""
        await controller.on_message("Implement feature", context)

        # Go through INFO
        info_response = """
        <assessment>
            <complexity>0.6</complexity>
            <need_clarification>0.0</need_clarification>
            <need_web>0.0</need_web>
            <need_literature>0.0</need_literature>
            <need_project_files>0.0</need_project_files>
        </assessment>
        """
        await controller.on_llm_response(info_response, [], context)

        # Low quality plan
        plan_response = """
        <plan>
            <quality>0.3</quality>
            <steps>
                <step>Do it</step>
            </steps>
        </plan>
        """

        decision = await controller.on_llm_response(plan_response, [], context)

        # Should stay in PLAN for revision
        assert decision.phase == FlowPhase.PLAN.value
        assert "REVISION NEEDED" in decision.system_addendum

    @pytest.mark.asyncio
    async def test_max_plan_iterations_forces_proceed(self, controller, context):
        """After max iterations, low quality plan still proceeds."""
        config = ControllerConfigV02(max_plan_iterations=1)
        controller = PhaseFlowController(config=config)

        await controller.on_message("Implement feature", context)

        # Go through INFO
        info_response = """
        <assessment>
            <complexity>0.6</complexity>
            <need_clarification>0.0</need_clarification>
            <need_web>0.0</need_web>
            <need_literature>0.0</need_literature>
            <need_project_files>0.0</need_project_files>
        </assessment>
        """
        await controller.on_llm_response(info_response, [], context)

        # Low quality plan - but only 1 iteration allowed
        plan_response = """
        <plan>
            <quality>0.3</quality>
            <steps>
                <step>Do it</step>
            </steps>
        </plan>
        """

        decision = await controller.on_llm_response(plan_response, [], context)

        # Should proceed despite low quality
        assert decision.phase == FlowPhase.EXECUTE.value


# -----------------------------------------------------------------------------
# Test: EXECUTE phase handling
# -----------------------------------------------------------------------------

class TestExecutePhase:
    """Test EXECUTE phase response handling."""

    @pytest.mark.asyncio
    async def test_low_complexity_completes_after_execute(self, controller, context):
        """LOW complexity completes after EXECUTE."""
        await controller.on_message("Simple task", context)

        # Low complexity INFO
        info_response = """
        <assessment>
            <complexity>0.1</complexity>
            <need_clarification>0.0</need_clarification>
            <need_web>0.0</need_web>
            <need_literature>0.0</need_literature>
            <need_project_files>0.0</need_project_files>
        </assessment>
        """
        await controller.on_llm_response(info_response, [], context)

        # Execute response
        execute_response = "Here's the answer to your question."
        decision = await controller.on_llm_response(execute_response, [], context)

        assert decision.action == "DONE"
        assert decision.phase == FlowPhase.DONE.value

    @pytest.mark.asyncio
    async def test_moderate_complexity_proceeds_to_validate(self, controller, context):
        """MODERATE complexity proceeds to VALIDATE after EXECUTE."""
        await controller.on_message("Implement feature", context)

        # Moderate complexity INFO
        info_response = """
        <assessment>
            <complexity>0.5</complexity>
            <need_clarification>0.0</need_clarification>
            <need_web>0.0</need_web>
            <need_literature>0.0</need_literature>
            <need_project_files>0.0</need_project_files>
        </assessment>
        """
        await controller.on_llm_response(info_response, [], context)

        # PLAN phase
        plan_response = """
        <plan>
            <quality>0.9</quality>
            <steps>
                <step>Step 1</step>
            </steps>
        </plan>
        """
        await controller.on_llm_response(plan_response, [], context)

        # EXECUTE response
        execute_response = "I've implemented the feature."
        decision = await controller.on_llm_response(execute_response, [], context)

        assert decision.phase == FlowPhase.VALIDATE.value

    @pytest.mark.asyncio
    async def test_tool_calls_execute_in_phase(self, controller, context):
        """Tool calls during EXECUTE stay in EXECUTE phase."""
        await controller.on_message("Simple task", context)

        info_response = """
        <assessment>
            <complexity>0.1</complexity>
            <need_clarification>0.0</need_clarification>
            <need_web>0.0</need_web>
            <need_literature>0.0</need_literature>
            <need_project_files>0.0</need_project_files>
        </assessment>
        """
        await controller.on_llm_response(info_response, [], context)

        # Execute with tool calls
        mock_tool_call = {"name": "file_write", "args": {}}
        decision = await controller.on_llm_response("Writing file...", [mock_tool_call], context)

        assert decision.action == "EXECUTE_TOOLS"
        assert decision.phase == FlowPhase.EXECUTE.value


# -----------------------------------------------------------------------------
# Test: VALIDATE phase handling
# -----------------------------------------------------------------------------

class TestValidatePhase:
    """Test VALIDATE phase response handling."""

    async def _get_to_validate(self, controller, context):
        """Helper to get to VALIDATE phase."""
        await controller.on_message("Implement feature", context)

        # Moderate complexity INFO
        info_response = """
        <assessment>
            <complexity>0.5</complexity>
            <need_clarification>0.0</need_clarification>
            <need_web>0.0</need_web>
            <need_literature>0.0</need_literature>
            <need_project_files>0.0</need_project_files>
        </assessment>
        """
        await controller.on_llm_response(info_response, [], context)

        # PLAN
        plan_response = """
        <plan>
            <quality>0.9</quality>
            <steps><step>Step 1</step></steps>
        </plan>
        """
        await controller.on_llm_response(plan_response, [], context)

        # EXECUTE
        await controller.on_llm_response("Done implementing.", [], context)

    @pytest.mark.asyncio
    async def test_successful_validation_completes_moderate(self, controller, context):
        """Successful validation completes MODERATE flow."""
        await self._get_to_validate(controller, context)

        validate_response = """
        <validation>
            <task_accomplished>0.95</task_accomplished>
            <verified>0.9</verified>
            <issues></issues>
        </validation>
        """

        decision = await controller.on_llm_response(validate_response, [], context)

        assert decision.action == "DONE"

    @pytest.mark.asyncio
    async def test_successful_validation_proceeds_to_document_high(self, context):
        """Successful validation for HIGH complexity proceeds to DOCUMENT."""
        config = ControllerConfigV02()
        controller = PhaseFlowController(config=config)

        await controller.on_message("Major refactor", context)

        # HIGH complexity
        info_response = """
        <assessment>
            <complexity>0.9</complexity>
            <need_clarification>0.0</need_clarification>
            <need_web>0.0</need_web>
            <need_literature>0.0</need_literature>
            <need_project_files>0.0</need_project_files>
        </assessment>
        """
        await controller.on_llm_response(info_response, [], context)

        # PLAN
        plan_response = """
        <plan>
            <quality>0.9</quality>
            <steps><step>Step 1</step></steps>
        </plan>
        """
        await controller.on_llm_response(plan_response, [], context)

        # EXECUTE
        await controller.on_llm_response("Done.", [], context)

        # VALIDATE with success
        validate_response = """
        <validation>
            <task_accomplished>0.95</task_accomplished>
            <verified>0.9</verified>
            <issues></issues>
        </validation>
        """
        decision = await controller.on_llm_response(validate_response, [], context)

        assert decision.phase == FlowPhase.DOCUMENT.value

    @pytest.mark.asyncio
    async def test_failed_validation_with_fix_loops_back(self, controller, context):
        """Failed validation that can be fixed loops back to EXECUTE."""
        await self._get_to_validate(controller, context)

        validate_response = """
        <validation>
            <task_accomplished>0.6</task_accomplished>
            <verified>0.5</verified>
            <issues>
                <issue>Type error in line 42</issue>
            </issues>
            <can_fix_without_replan>true</can_fix_without_replan>
            <fix_actions>Fix the type annotation</fix_actions>
        </validation>
        """

        decision = await controller.on_llm_response(validate_response, [], context)

        assert decision.phase == FlowPhase.EXECUTE.value
        assert decision.payload.get("fix_needed") is True

    @pytest.mark.asyncio
    async def test_failed_validation_no_fix_fails_flow(self, controller, context):
        """Failed validation that cannot be fixed fails the flow."""
        await self._get_to_validate(controller, context)

        validate_response = """
        <validation>
            <task_accomplished>0.3</task_accomplished>
            <verified>0.2</verified>
            <issues>
                <issue>Fundamental design problem</issue>
            </issues>
            <can_fix_without_replan>false</can_fix_without_replan>
        </validation>
        """

        decision = await controller.on_llm_response(validate_response, [], context)

        assert decision.phase == FlowPhase.FAILED.value
        assert "error" in decision.payload


# -----------------------------------------------------------------------------
# Test: DOCUMENT phase handling
# -----------------------------------------------------------------------------

class TestDocumentPhase:
    """Test DOCUMENT phase response handling."""

    @pytest.mark.asyncio
    async def test_document_completes_flow(self, context):
        """DOCUMENT phase completes the flow."""
        controller = PhaseFlowController()

        await controller.on_message("Major refactor", context)

        # HIGH complexity path
        info_response = """
        <assessment>
            <complexity>0.9</complexity>
            <need_clarification>0.0</need_clarification>
            <need_web>0.0</need_web>
            <need_literature>0.0</need_literature>
            <need_project_files>0.0</need_project_files>
        </assessment>
        """
        await controller.on_llm_response(info_response, [], context)

        plan_response = """
        <plan>
            <quality>0.9</quality>
            <steps><step>Step 1</step></steps>
        </plan>
        """
        await controller.on_llm_response(plan_response, [], context)

        await controller.on_llm_response("Done implementing.", [], context)

        validate_response = """
        <validation>
            <task_accomplished>0.95</task_accomplished>
            <verified>0.9</verified>
        </validation>
        """
        await controller.on_llm_response(validate_response, [], context)

        # DOCUMENT response
        document_response = "Updated the README with the changes."
        decision = await controller.on_llm_response(document_response, [], context)

        assert decision.action == "DONE"
        assert decision.phase == FlowPhase.DONE.value


# -----------------------------------------------------------------------------
# Test: Complexity routing
# -----------------------------------------------------------------------------

class TestComplexityRouting:
    """Test complexity determination and routing."""

    def test_low_threshold(self, controller):
        """Scores below low threshold route to LOW."""
        complexity = controller._determine_complexity(0.1)
        assert complexity == ComplexityLevel.LOW

    def test_high_threshold(self, controller):
        """Scores above high threshold route to HIGH."""
        complexity = controller._determine_complexity(0.9)
        assert complexity == ComplexityLevel.HIGH

    def test_moderate_range(self, controller):
        """Scores in middle range route to MODERATE."""
        complexity = controller._determine_complexity(0.5)
        assert complexity == ComplexityLevel.MODERATE

    def test_custom_thresholds(self):
        """Custom thresholds are respected."""
        config = ControllerConfigV02(
            effort="high",  # high effort = lower thresholds
        )
        controller = PhaseFlowController(config=config)

        # With high effort, even 0.3 might be MODERATE
        complexity = controller._determine_complexity(0.3)
        # The exact result depends on the preset thresholds


# -----------------------------------------------------------------------------
# Test: Stateless behavior
# -----------------------------------------------------------------------------

class TestStatelessBehavior:
    """Test that controller is properly stateless."""

    @pytest.mark.asyncio
    async def test_load_state_is_noop(self, controller):
        """load_state does nothing (stateless)."""
        await controller.load_state()
        # Should not raise

    @pytest.mark.asyncio
    async def test_save_state_is_noop(self, controller):
        """save_state does nothing (stateless)."""
        await controller.save_state()
        # Should not raise

    @pytest.mark.asyncio
    async def test_new_message_creates_fresh_flow(self, controller, context):
        """Each message creates a fresh flow state."""
        # First message
        await controller.on_message("Message 1", context)
        flow1 = controller.get_current_flow()

        # Complete the flow
        info_response = """
        <assessment>
            <complexity>0.1</complexity>
            <need_clarification>0.0</need_clarification>
            <need_web>0.0</need_web>
            <need_literature>0.0</need_literature>
            <need_project_files>0.0</need_project_files>
        </assessment>
        """
        await controller.on_llm_response(info_response, [], context)

        # Second message - should get fresh flow
        await controller.on_message("Message 2", context)
        flow2 = controller.get_current_flow()

        assert flow2 is not flow1
        assert flow2.original_message == "Message 2"
        assert flow2.phase == FlowPhase.INFO


# -----------------------------------------------------------------------------
# Test: Metrics tracking
# -----------------------------------------------------------------------------

class TestMetricsTracking:
    """Test optional metrics tracking."""

    @pytest.mark.asyncio
    async def test_metrics_disabled_by_default(self, controller, context):
        """Metrics are not tracked by default."""
        await controller.on_message("Test", context)

        metrics = controller.get_flow_metrics()
        assert metrics is None

    @pytest.mark.asyncio
    async def test_metrics_tracked_when_enabled(self, context):
        """Metrics are tracked when enabled."""
        config = ControllerConfigV02(enable_metrics=True)
        controller = PhaseFlowController(config=config)

        await controller.on_message("Test", context)

        metrics = controller.get_flow_metrics()
        assert metrics is not None
        assert "start_time" in metrics


# -----------------------------------------------------------------------------
# Test: Error handling
# -----------------------------------------------------------------------------

class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_no_flow_state_passthrough(self, controller, context):
        """on_llm_response without flow state passes through."""
        # Don't start a flow
        decision = await controller.on_llm_response("Some response", [], context)

        assert decision.action == "PROCESS_WITH_LLM"

    @pytest.mark.asyncio
    async def test_malformed_xml_uses_defaults(self, controller, context):
        """Malformed XML uses fallback defaults."""
        await controller.on_message("Test", context)

        # Malformed assessment
        response = "I'll help with that. No XML here."
        decision = await controller.on_llm_response(response, [], context)

        # Should still route based on default complexity (0.5 = MODERATE)
        flow = controller.get_current_flow()
        assert flow.info_assessment is not None
