"""
Tests for v0.2 controller models.
"""

import pytest
from mesh.config import EffortPreset, get_effort_threshold, ControllerConfigV02
from mesh.controller.models_v02 import (
    FlowPhase, FlowState, InfoAssessment, PlanV02,
    PlanStepV02, ValidationResult, FlowMetrics, ComplexityLevel
)


class TestEffortPresets:
    """Tests for effort presets and thresholds."""

    def test_preset_values(self):
        """All presets have correct string values."""
        assert EffortPreset.HIGH.value == "high"
        assert EffortPreset.MEDIUM.value == "medium"
        assert EffortPreset.LOW.value == "low"

    def test_high_effort_has_lower_thresholds(self):
        """High effort should have lower thresholds (more phases trigger)."""
        high_info = get_effort_threshold(EffortPreset.HIGH, "info")
        medium_info = get_effort_threshold(EffortPreset.MEDIUM, "info")
        low_info = get_effort_threshold(EffortPreset.LOW, "info")

        assert high_info < medium_info < low_info

    def test_complexity_thresholds_make_sense(self):
        """Low complexity cutoff should be less than high complexity cutoff."""
        for preset in EffortPreset:
            low = get_effort_threshold(preset, "complexity_low")
            high = get_effort_threshold(preset, "complexity_high")
            assert low < high, f"Preset {preset}: low={low} should be < high={high}"

    def test_invalid_threshold_name_raises(self):
        """Invalid threshold name should raise KeyError."""
        with pytest.raises(KeyError):
            get_effort_threshold(EffortPreset.MEDIUM, "nonexistent")


class TestControllerConfigV02:
    """Tests for ControllerConfigV02."""

    def test_defaults(self):
        """Default values are set correctly."""
        cfg = ControllerConfigV02()
        assert cfg.mode == "passthrough"
        assert cfg.effort == "medium"
        assert cfg.max_info_iterations == 3
        assert cfg.max_plan_iterations == 3
        assert cfg.enable_metrics is False
        assert cfg.stream_phase_updates is True

    def test_get_effort_preset(self):
        """get_effort_preset returns correct enum."""
        cfg = ControllerConfigV02(effort="high")
        assert cfg.get_effort_preset() == EffortPreset.HIGH

    def test_get_threshold_from_preset(self):
        """get_threshold uses effort preset when no override."""
        cfg = ControllerConfigV02(effort="high")
        assert cfg.get_threshold("info") == 0.2  # HIGH preset value

    def test_get_threshold_with_override(self):
        """get_threshold uses override when set."""
        cfg = ControllerConfigV02(effort="high", info_threshold=0.99)
        assert cfg.get_threshold("info") == 0.99  # Override value

    def test_get_threshold_partial_override(self):
        """Non-overridden thresholds still use preset."""
        cfg = ControllerConfigV02(effort="high", info_threshold=0.99)
        # info is overridden, but complexity_low should use preset
        assert cfg.get_threshold("info") == 0.99
        assert cfg.get_threshold("complexity_low") == 0.2  # HIGH preset value


class TestFlowPhase:
    """Tests for FlowPhase enum."""

    def test_all_phases_exist(self):
        """All expected phases exist."""
        phases = [p.value for p in FlowPhase]
        assert "info" in phases
        assert "plan" in phases
        assert "execute" in phases
        assert "validate" in phases
        assert "document" in phases
        assert "done" in phases
        assert "failed" in phases


class TestComplexityLevel:
    """Tests for ComplexityLevel enum."""

    def test_all_levels_exist(self):
        """All expected levels exist."""
        assert ComplexityLevel.LOW.value == "low"
        assert ComplexityLevel.MODERATE.value == "moderate"
        assert ComplexityLevel.HIGH.value == "high"


class TestInfoAssessment:
    """Tests for InfoAssessment."""

    def test_defaults(self):
        """Default values are zero/empty."""
        assessment = InfoAssessment()
        assert assessment.need_clarification == 0.0
        assert assessment.need_web == 0.0
        assert assessment.need_literature == 0.0
        assert assessment.need_project_files == 0.0
        assert assessment.complexity == 0.5
        assert assessment.clarification_questions is None
        assert assessment.parsed_successfully is True

    def test_max_info_score(self):
        """max_info_score returns highest of web/literature/project."""
        assessment = InfoAssessment(need_web=0.3, need_literature=0.8, need_project_files=0.5)
        assert assessment.max_info_score() == 0.8

    def test_max_info_score_excludes_clarification(self):
        """max_info_score does not include clarification."""
        assessment = InfoAssessment(
            need_clarification=0.99,
            need_web=0.3,
            need_literature=0.1,
            need_project_files=0.2
        )
        assert assessment.max_info_score() == 0.3

    def test_any_info_needed(self):
        """any_info_needed checks if any score exceeds threshold."""
        assessment = InfoAssessment(need_web=0.5, need_literature=0.2, need_project_files=0.1)
        assert assessment.any_info_needed(0.3) is True  # web > 0.3
        assert assessment.any_info_needed(0.6) is False  # nothing > 0.6

    def test_to_dict(self):
        """to_dict serializes correctly."""
        assessment = InfoAssessment(complexity=0.7, need_web=0.4)
        d = assessment.to_dict()
        assert d["complexity"] == 0.7
        assert d["need_web"] == 0.4
        assert "raw_output" not in d  # raw_output not in to_dict


class TestPlanV02:
    """Tests for PlanV02."""

    def test_defaults(self):
        """Default values are set."""
        plan = PlanV02()
        assert plan.steps == []
        assert plan.quality_score == 0.0
        assert plan.revision_count == 0

    def test_with_steps(self):
        """Plan with steps works."""
        steps = [
            PlanStepV02(number=1, description="First step"),
            PlanStepV02(number=2, description="Second step", estimated_turns=2),
        ]
        plan = PlanV02(steps=steps, quality_score=0.8)
        assert len(plan.steps) == 2
        assert plan.quality_score == 0.8

    def test_to_dict(self):
        """to_dict serializes steps."""
        steps = [PlanStepV02(number=1, description="Test")]
        plan = PlanV02(steps=steps)
        d = plan.to_dict()
        assert len(d["steps"]) == 1
        assert d["steps"][0]["description"] == "Test"


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_defaults(self):
        """Default values are zero/empty."""
        result = ValidationResult()
        assert result.task_accomplished == 0.0
        assert result.verified == 0.0
        assert result.issues == []
        assert result.can_fix_without_replan is False

    def test_with_issues(self):
        """Can create result with issues."""
        result = ValidationResult(
            task_accomplished=0.5,
            verified=0.3,
            issues=["Test failed", "Missing file"],
            can_fix_without_replan=True
        )
        assert result.task_accomplished == 0.5
        assert len(result.issues) == 2
        assert result.can_fix_without_replan is True

    def test_is_successful(self):
        """is_successful helper method."""
        # Successful validation
        success = ValidationResult(task_accomplished=0.9, verified=0.8)
        assert success.is_successful() is True

        # Below accomplished threshold
        fail_accomplished = ValidationResult(task_accomplished=0.7, verified=0.9)
        assert fail_accomplished.is_successful() is False

        # Below verified threshold
        fail_verified = ValidationResult(task_accomplished=0.9, verified=0.5)
        assert fail_verified.is_successful() is False


class TestFlowMetrics:
    """Tests for FlowMetrics."""

    def test_defaults(self):
        """Default values are zero."""
        metrics = FlowMetrics()
        assert metrics.llm_calls == 0
        assert metrics.total_input_tokens == 0
        assert metrics.phases_executed == []
        assert metrics.start_time != ""  # Auto-set

    def test_record_llm_call(self):
        """record_llm_call increments counters."""
        metrics = FlowMetrics()
        metrics.record_llm_call(100, 50)
        metrics.record_llm_call(200, 75)
        assert metrics.llm_calls == 2
        assert metrics.total_input_tokens == 300
        assert metrics.total_output_tokens == 125

    def test_record_phase(self):
        """record_phase tracks phases and timing."""
        metrics = FlowMetrics()
        metrics.record_phase("INFO", 1000)
        metrics.record_phase("PLAN", 2000)
        assert metrics.phases_executed == ["INFO", "PLAN"]
        assert metrics.phase_durations["INFO"] == 1000
        assert metrics.phase_durations["PLAN"] == 2000

    def test_record_error(self):
        """record_error appends to errors list."""
        metrics = FlowMetrics()
        metrics.record_error("Something went wrong")
        assert "Something went wrong" in metrics.errors

    def test_finalize(self):
        """finalize sets end_time and calculates duration."""
        metrics = FlowMetrics()
        metrics.record_phase("INFO", 1000)
        metrics.record_phase("EXECUTE", 2000)
        metrics.finalize()
        assert metrics.end_time != ""
        assert metrics.duration_ms == 3000


class TestFlowState:
    """Tests for FlowState."""

    def test_defaults(self):
        """Default state starts in INFO phase."""
        state = FlowState()
        assert state.phase == FlowPhase.INFO
        assert state.complexity is None
        assert state.info_assessment is None
        assert state.plan is None
        assert state.validation is None
        assert state.info_iterations == 0
        assert state.plan_iterations == 0

    def test_is_terminal(self):
        """is_terminal returns True for DONE and FAILED."""
        state = FlowState()
        assert state.is_terminal() is False

        state.phase = FlowPhase.EXECUTE
        assert state.is_terminal() is False

        state.phase = FlowPhase.DONE
        assert state.is_terminal() is True

        state.phase = FlowPhase.FAILED
        assert state.is_terminal() is True

    def test_fail(self):
        """fail transitions to FAILED with message."""
        state = FlowState()
        state.metrics = FlowMetrics()
        state.fail("Something went wrong")

        assert state.phase == FlowPhase.FAILED
        assert state.error_message == "Something went wrong"
        assert "Something went wrong" in state.metrics.errors

    def test_complete(self):
        """complete transitions to DONE."""
        state = FlowState()
        state.metrics = FlowMetrics()
        state.complete()

        assert state.phase == FlowPhase.DONE
        assert state.metrics.end_time != ""

    def test_to_dict(self):
        """to_dict serializes state."""
        state = FlowState()
        state.complexity = ComplexityLevel.MODERATE
        state.info_assessment = InfoAssessment(complexity=0.6)

        d = state.to_dict()
        assert d["phase"] == "info"
        assert d["complexity"] == "moderate"
        assert d["info_assessment"]["complexity"] == 0.6
