"""
Integration tests for Phase 4: Automatic phase transitions.
"""

import pytest
from mesh.controller import (
    TaskFSMController,
    Task,
    TaskPhase,
    PlanStep,
    StepStatus,
)
from mesh.controller.base import ControllerContext
from mesh.config import ControllerConfig
import tempfile
import os


class MockToolCall:
    """Mock tool call for testing."""
    def __init__(self, name: str):
        self.name = name


class MockToolCallWithArgs:
    """Mock tool call with name and arguments (for Phase 5)."""
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = arguments


class MockMessage:
    """Mock message for testing."""
    def __init__(self, content: str):
        self.content = content


class TestPhase4Integration:
    """Integration tests for Phase 4 automatic phase transitions."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def controller(self, temp_dir):
        """Create a controller with temp storage."""
        config = ControllerConfig(
            mode="task-fsm-v0",
            tasks_path=os.path.join(temp_dir, "tasks.json"),
            config_path=os.path.join(temp_dir, "config.json"),
        )
        return TaskFSMController(config)

    @pytest.fixture
    def context(self):
        """Create a mock ControllerContext."""
        return ControllerContext(
            cwd="/test",
            history=[],
            agent_id="test-agent",
            message=MockMessage("test"),
        )

    # --- Phase Transition Tests ---

    @pytest.mark.asyncio
    async def test_planning_to_executing_on_file_write(self, controller, context):
        """Phase 5: File writes now trigger WAITING_APPROVAL instead of EXECUTING."""
        # Create a task in PLANNING phase
        task = controller.create_task(
            title="Test task",
            description="Test phase transitions",
        )
        controller.set_active_task(task.id)
        assert task.phase == TaskPhase.PLANNING

        # Simulate LLM response with file write
        response = "Creating the implementation file."
        # Phase 5: file_write tool needs arguments for edit interceptor
        tool_calls = [MockToolCallWithArgs("file_write", {"path": "/tmp/test.py", "content": "test"})]

        decision = await controller.on_llm_response(response, tool_calls, context)

        # Phase 5: File writes trigger WAITING_APPROVAL
        task = controller.get_active_task()
        assert task.phase == TaskPhase.WAITING_APPROVAL
        assert decision.action == "WAITING_APPROVAL"
        assert len(task.pending_edits) == 1

    @pytest.mark.asyncio
    async def test_planning_to_needs_clarification(self, controller, context):
        """Test transition to NEEDS_CLARIFICATION when LLM asks questions."""
        task = controller.create_task(title="Test task")
        controller.set_active_task(task.id)

        response = "I need to know which database you want to use."
        tool_calls = []

        decision = await controller.on_llm_response(response, tool_calls, context)

        task = controller.get_active_task()
        assert task.phase == TaskPhase.NEEDS_CLARIFICATION
        assert decision.phase == TaskPhase.NEEDS_CLARIFICATION.value

    @pytest.mark.asyncio
    async def test_executing_to_done(self, controller, context):
        """Test transition to DONE when LLM signals completion."""
        task = controller.create_task(title="Test task")
        task.phase = TaskPhase.EXECUTING
        controller.set_active_task(task.id)

        response = "The task is complete! All tests are passing."
        tool_calls = []

        decision = await controller.on_llm_response(response, tool_calls, context)

        task = controller.get_active_task()
        assert task.phase == TaskPhase.DONE
        assert decision.phase == TaskPhase.DONE.value

    @pytest.mark.asyncio
    async def test_transition_to_blocked(self, controller, context):
        """Test transition to BLOCKED when LLM identifies blockers."""
        task = controller.create_task(title="Test task")
        controller.set_active_task(task.id)

        response = "I'm blocked on this - waiting for API credentials."
        tool_calls = []

        decision = await controller.on_llm_response(response, tool_calls, context)

        task = controller.get_active_task()
        assert task.phase == TaskPhase.BLOCKED
        assert decision.phase == TaskPhase.BLOCKED.value

    # --- Plan Extraction Tests ---

    @pytest.mark.asyncio
    async def test_extract_plan_during_planning(self, controller, context):
        """Test that plan steps are extracted during PLANNING phase."""
        task = controller.create_task(title="Test task")
        controller.set_active_task(task.id)
        assert len(task.plan) == 0

        response = """
Here's my plan:

1. Create database schema
2. Implement API endpoints
3. Write tests
"""
        tool_calls = [MockToolCall("file_write")]  # Will trigger transition

        await controller.on_llm_response(response, tool_calls, context)

        task = controller.get_active_task()
        # Plan should be extracted
        assert len(task.plan) == 3
        assert task.plan[0].description == "Create database schema"
        assert task.plan[0].status == StepStatus.IN_PROGRESS  # First step starts immediately
        assert task.plan[1].description == "Implement API endpoints"
        assert task.plan[2].description == "Write tests"
        # Phase should transition to EXECUTING
        assert task.phase == TaskPhase.WAITING_APPROVAL

    @pytest.mark.asyncio
    async def test_no_plan_extraction_if_already_exists(self, controller, context):
        """Test that existing plans are not overwritten."""
        task = controller.create_task(title="Test task")
        # Add existing plan
        task.plan = [PlanStep(id="step-1", description="Existing step")]
        controller.set_active_task(task.id)

        response = """
1. New step 1
2. New step 2
"""
        tool_calls = [MockToolCall("file_write")]

        await controller.on_llm_response(response, tool_calls, context)

        task = controller.get_active_task()
        # Original plan should be preserved
        assert len(task.plan) == 1
        assert task.plan[0].description == "Existing step"

    # --- No Active Task Tests ---

    @pytest.mark.asyncio
    async def test_no_transition_without_active_task(self, controller, context):
        """Test that phase detection doesn't crash without active task."""
        response = "The task is complete!"
        tool_calls = []

        # Should not raise an error
        decision = await controller.on_llm_response(response, tool_calls, context)

        assert decision.action == "DONE"
        assert decision.task_id is None
        assert decision.phase is None

    # --- Timestamp Updates ---

    @pytest.mark.asyncio
    async def test_updated_at_timestamp_on_transition(self, controller, context):
        """Test that task.updated_at is updated on phase transitions."""
        task = controller.create_task(title="Test task")
        controller.set_active_task(task.id)
        original_updated_at = task.updated_at

        # Wait a tiny bit to ensure timestamp changes
        import asyncio
        await asyncio.sleep(0.01)

        response = "I need to know which approach to use."
        await controller.on_llm_response(response, [], context)

        task = controller.get_active_task()
        assert task.updated_at != original_updated_at

    # --- Phase Priority Tests ---

    @pytest.mark.asyncio
    async def test_blocked_takes_priority(self, controller, context):
        """Test that BLOCKED signal takes priority over other signals."""
        task = controller.create_task(title="Test task")
        task.phase = TaskPhase.EXECUTING
        controller.set_active_task(task.id)

        # Response has both DONE and BLOCKED signals
        response = "Task is done, but I'm blocked on deployment approval."
        await controller.on_llm_response(response, [], context)

        task = controller.get_active_task()
        # BLOCKED should win
        assert task.phase == TaskPhase.BLOCKED

    # --- Multiple Tool Calls ---

    @pytest.mark.asyncio
    async def test_multiple_file_operations(self, controller, context):
        """Test transition with multiple file operations."""
        task = controller.create_task(title="Test task")
        controller.set_active_task(task.id)

        response = "Updating multiple files."
        tool_calls = [
            MockToolCall("file_read"),
            MockToolCall("file_write"),
            MockToolCall("file_edit"),
        ]

        await controller.on_llm_response(response, tool_calls, context)

        task = controller.get_active_task()
        # Should transition due to file writes
        assert task.phase == TaskPhase.WAITING_APPROVAL

    # --- Persistence After Transition ---

    @pytest.mark.asyncio
    async def test_phase_persists_after_transition(self, controller, context):
        """Test that phase transitions are saved to disk."""
        # Load state first (required before save)
        await controller.load_state()

        task = controller.create_task(title="Test task")
        controller.set_active_task(task.id)

        response = "Creating the file."
        tool_calls = [MockToolCall("file_create")]

        await controller.on_llm_response(response, tool_calls, context)

        # Save state
        await controller.save_state()

        # Create new controller instance to test loading
        new_controller = TaskFSMController(controller.config)
        await new_controller.load_state()

        loaded_task = new_controller.get_active_task()
        assert loaded_task is not None
        assert loaded_task.phase == TaskPhase.WAITING_APPROVAL
        assert loaded_task.id == task.id
