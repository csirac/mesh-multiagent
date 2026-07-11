"""
Unit tests for mesh/controller/phase_detector.py
"""

import pytest
from mesh.controller.phase_detector import PhaseDetector
from mesh.controller.models import TaskPhase


class MockToolCall:
    """Mock tool call for testing."""
    def __init__(self, name: str):
        self.name = name


class TestPhaseDetector:
    """Test the PhaseDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a PhaseDetector instance."""
        return PhaseDetector()

    # --- Clarification Detection ---

    def test_detect_clarification_need_to_know(self, detector):
        """Test detection of clarification requests."""
        response = "I need to know which database you want to use."
        phase = detector.detect_phase(
            response=response,
            tool_calls=[],
            current_phase=TaskPhase.PLANNING,
        )
        assert phase == TaskPhase.NEEDS_CLARIFICATION

    def test_detect_clarification_please_clarify(self, detector):
        """Test detection of 'please clarify' pattern."""
        response = "Please clarify what authentication method you prefer."
        phase = detector.detect_phase(
            response=response,
            tool_calls=[],
            current_phase=TaskPhase.PLANNING,
        )
        assert phase == TaskPhase.NEEDS_CLARIFICATION

    def test_detect_clarification_not_sure(self, detector):
        """Test detection of uncertainty patterns."""
        response = "I'm not sure which approach to take here."
        phase = detector.detect_phase(
            response=response,
            tool_calls=[],
            current_phase=TaskPhase.PLANNING,
        )
        assert phase == TaskPhase.NEEDS_CLARIFICATION

    def test_no_clarification_during_execution(self, detector):
        """Test that clarification patterns during execution don't trigger transition."""
        response = "I need to know the current state, so I'll read the file."
        phase = detector.detect_phase(
            response=response,
            tool_calls=[],
            current_phase=TaskPhase.EXECUTING,
        )
        assert phase is None  # Don't transition to NEEDS_CLARIFICATION

    # --- Completion Detection ---

    def test_detect_completion_task_complete(self, detector):
        """Test detection of 'task complete' pattern."""
        response = "The task is complete! All tests are passing."
        phase = detector.detect_phase(
            response=response,
            tool_calls=[],
            current_phase=TaskPhase.EXECUTING,
        )
        assert phase == TaskPhase.DONE

    def test_detect_completion_successfully_completed(self, detector):
        """Test detection of 'successfully completed' pattern."""
        response = "I've successfully completed the implementation."
        phase = detector.detect_phase(
            response=response,
            tool_calls=[],
            current_phase=TaskPhase.EXECUTING,
        )
        assert phase == TaskPhase.DONE

    def test_detect_completion_all_done(self, detector):
        """Test detection of 'all done' pattern."""
        response = "All done! The feature is ready to use."
        phase = detector.detect_phase(
            response=response,
            tool_calls=[],
            current_phase=TaskPhase.EXECUTING,
        )
        assert phase == TaskPhase.DONE

    # --- Blocking Detection ---

    def test_detect_blocked_waiting_for(self, detector):
        """Test detection of blocking conditions."""
        response = "I'm blocked on this task - waiting for API access."
        phase = detector.detect_phase(
            response=response,
            tool_calls=[],
            current_phase=TaskPhase.EXECUTING,
        )
        assert phase == TaskPhase.BLOCKED

    def test_detect_blocked_requires_approval(self, detector):
        """Test detection of approval requirements."""
        response = "This change requires approval before I can proceed."
        phase = detector.detect_phase(
            response=response,
            tool_calls=[],
            current_phase=TaskPhase.EXECUTING,
        )
        assert phase == TaskPhase.BLOCKED

    def test_detect_blocked_cannot_proceed(self, detector):
        """Test detection of 'cannot proceed' pattern."""
        response = "I cannot proceed without database credentials."
        phase = detector.detect_phase(
            response=response,
            tool_calls=[],
            current_phase=TaskPhase.PLANNING,
        )
        assert phase == TaskPhase.BLOCKED

    # --- Tool Call Analysis ---

    def test_detect_executing_from_file_write(self, detector):
        """Test transition to EXECUTING when file writes are detected."""
        response = "Creating the new module."
        tool_calls = [MockToolCall("file_write")]
        phase = detector.detect_phase(
            response=response,
            tool_calls=tool_calls,
            current_phase=TaskPhase.PLANNING,
        )
        assert phase == TaskPhase.EXECUTING

    def test_detect_executing_from_file_create(self, detector):
        """Test transition to EXECUTING from file_create."""
        response = "Adding the configuration file."
        tool_calls = [MockToolCall("file_create")]
        phase = detector.detect_phase(
            response=response,
            tool_calls=tool_calls,
            current_phase=TaskPhase.PLANNING,
        )
        assert phase == TaskPhase.EXECUTING

    def test_detect_executing_from_file_edit(self, detector):
        """Test transition to EXECUTING from file_edit."""
        response = "Updating the function."
        tool_calls = [MockToolCall("file_edit")]
        phase = detector.detect_phase(
            response=response,
            tool_calls=tool_calls,
            current_phase=TaskPhase.PLANNING,
        )
        assert phase == TaskPhase.EXECUTING

    def test_detect_executing_from_bash(self, detector):
        """Test transition to EXECUTING from bash execution."""
        response = "Running the tests."
        tool_calls = [MockToolCall("bash_exec")]
        phase = detector.detect_phase(
            response=response,
            tool_calls=tool_calls,
            current_phase=TaskPhase.PLANNING,
        )
        assert phase == TaskPhase.EXECUTING

    def test_no_transition_when_already_executing(self, detector):
        """Test that file writes during execution don't re-transition."""
        response = "Updating another file."
        tool_calls = [MockToolCall("file_write")]
        phase = detector.detect_phase(
            response=response,
            tool_calls=tool_calls,
            current_phase=TaskPhase.EXECUTING,
        )
        assert phase is None  # No transition needed

    # --- Priority Testing ---

    def test_blocking_takes_priority_over_completion(self, detector):
        """Test that BLOCKED takes priority over DONE."""
        response = "Task complete, but blocked on deployment approval."
        phase = detector.detect_phase(
            response=response,
            tool_calls=[],
            current_phase=TaskPhase.EXECUTING,
        )
        assert phase == TaskPhase.BLOCKED

    def test_completion_takes_priority_over_clarification(self, detector):
        """Test that DONE takes priority over NEEDS_CLARIFICATION."""
        response = "Task is complete! I need to know if you want me to continue."
        phase = detector.detect_phase(
            response=response,
            tool_calls=[],
            current_phase=TaskPhase.PLANNING,
        )
        # DONE should win due to earlier check
        assert phase == TaskPhase.DONE

    # --- No Transition Cases ---

    def test_no_transition_on_normal_response(self, detector):
        """Test that normal responses don't trigger transitions."""
        response = "Here's how the authentication flow works..."
        phase = detector.detect_phase(
            response=response,
            tool_calls=[],
            current_phase=TaskPhase.PLANNING,
        )
        assert phase is None

    def test_no_transition_on_file_read(self, detector):
        """Test that file reads don't trigger EXECUTING."""
        response = "Reading the config file."
        tool_calls = [MockToolCall("file_read")]
        phase = detector.detect_phase(
            response=response,
            tool_calls=tool_calls,
            current_phase=TaskPhase.PLANNING,
        )
        assert phase is None


class TestPlanStepExtraction:
    """Test plan step extraction."""

    @pytest.fixture
    def detector(self):
        """Create a PhaseDetector instance."""
        return PhaseDetector()

    def test_extract_numbered_steps(self, detector):
        """Test extraction of numbered steps with periods."""
        response = """
Here's my plan:

1. Create the database schema
2. Implement the API endpoints
3. Write unit tests
4. Deploy to staging
"""
        steps = detector.extract_plan_steps(response)
        assert len(steps) == 4
        assert "Create the database schema" in steps
        assert "Implement the API endpoints" in steps
        assert "Write unit tests" in steps
        assert "Deploy to staging" in steps

    def test_extract_numbered_steps_with_parens(self, detector):
        """Test extraction of numbered steps with parentheses."""
        response = """
Plan:

1) Set up the environment
2) Install dependencies
3) Run the build
"""
        steps = detector.extract_plan_steps(response)
        assert len(steps) == 3
        assert "Set up the environment" in steps

    def test_extract_bullet_points(self, detector):
        """Test extraction of bullet point steps."""
        response = """
I'll do the following:

- Add authentication middleware
- Update the routes
- Test the endpoints
"""
        steps = detector.extract_plan_steps(response)
        assert len(steps) == 3
        assert "Add authentication middleware" in steps
        assert "Update the routes" in steps

    def test_extract_asterisk_bullets(self, detector):
        """Test extraction of asterisk bullet points."""
        response = """
Steps:

* Create new file
* Update existing code
* Run tests
"""
        steps = detector.extract_plan_steps(response)
        assert len(steps) == 3
        assert "Create new file" in steps

    def test_no_steps_in_prose(self, detector):
        """Test that prose without list format returns empty."""
        response = "I'll create the file and then update the code and run tests."
        steps = detector.extract_plan_steps(response)
        assert len(steps) == 0

    def test_extract_mixed_formatting(self, detector):
        """Test that numbered format is preferred over bullets."""
        response = """
1. First step
2. Second step

- Some bullet
- Another bullet
"""
        steps = detector.extract_plan_steps(response)
        # Should extract numbered steps
        assert len(steps) == 2
        assert "First step" in steps
        assert "Second step" in steps
