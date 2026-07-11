"""
Integration tests for Phase 6: User Commands (task reopen, plan/step management).
"""

import pytest
from mesh.controller.task_fsm import TaskFSMController
from mesh.controller.models import Task, TaskPhase, PlanStep, StepStatus
from mesh.config import ControllerConfig
import tempfile
import os


@pytest.fixture
def controller():
    """Create a controller with temp storage."""
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "test-tasks.json")
    config = ControllerConfig(tasks_path=temp_file)
    controller = TaskFSMController(config=config)
    yield controller


class TestTaskReopenCommand:
    """Test /task reopen command."""

    @pytest.mark.asyncio
    async def test_reopen_done_task(self, controller):
        """Should reopen a DONE task back to PLANNING phase."""
        # Create and complete a task
        task = controller.create_task("Test task", "Test description")
        controller.complete_task(task.id)
        assert controller._get_task_by_id(task.id).phase == TaskPhase.DONE

        # Reopen it
        result = await controller.handle_command("task", ["reopen", task.id])
        assert "reopened" in result.lower()

        # Verify phase changed
        task = controller._get_task_by_id(task.id)
        assert task.phase == TaskPhase.PLANNING

    @pytest.mark.asyncio
    async def test_reopen_nonexistent_task(self, controller):
        """Should fail gracefully for nonexistent task."""
        result = await controller.handle_command("task", ["reopen", "task-99999-999"])
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_reopen_nondone_task(self, controller):
        """Should reject reopening a task that's not DONE."""
        task = controller.create_task("Active task", "Still working")
        task.phase = TaskPhase.EXECUTING

        result = await controller.handle_command("task", ["reopen", task.id])
        assert "not done" in result.lower()

    @pytest.mark.asyncio
    async def test_reopen_missing_id(self, controller):
        """Should show usage when ID is missing."""
        result = await controller.handle_command("task", ["reopen"])
        assert "usage" in result.lower()


class TestPlanCommands:
    """Test /plan management commands."""

    @pytest.mark.asyncio
    async def test_plan_show_empty(self, controller):
        """Should show no plan steps when plan is empty."""
        task = controller.create_task("Test task", "Description")
        controller.set_active_task(task.id)

        result = await controller.handle_command("plan", [])
        assert "no plan steps" in result.lower()

    @pytest.mark.asyncio
    async def test_plan_show_with_steps(self, controller):
        """Should display plan steps."""
        task = controller.create_task("Test task", "Description")
        task.plan = [
            PlanStep(id="step-1", description="Step 1", status=StepStatus.PENDING),
            PlanStep(id="step-2", description="Step 2", status=StepStatus.COMPLETED),
        ]
        controller.set_active_task(task.id)

        result = await controller.handle_command("plan", [])
        assert "step 1" in result.lower()
        assert "step 2" in result.lower()

    @pytest.mark.asyncio
    async def test_plan_add(self, controller):
        """Should add a new plan step."""
        task = controller.create_task("Test task", "Description")
        controller.set_active_task(task.id)

        result = await controller.handle_command("plan", ["add", "Write", "tests"])
        assert "added step" in result.lower()
        assert "write tests" in result.lower()

        assert len(task.plan) == 1
        assert task.plan[0].description == "Write tests"
        assert task.plan[0].status == StepStatus.PENDING

    @pytest.mark.asyncio
    async def test_plan_edit(self, controller):
        """Should edit an existing plan step."""
        task = controller.create_task("Test task", "Description")
        task.plan = [PlanStep(id="step-1", description="Old description", status=StepStatus.PENDING)]
        controller.set_active_task(task.id)

        result = await controller.handle_command("plan", ["edit", "1", "New", "description"])
        assert "updated step 1" in result.lower()

        assert task.plan[0].description == "New description"

    @pytest.mark.asyncio
    async def test_plan_edit_invalid_step(self, controller):
        """Should reject editing nonexistent step."""
        task = controller.create_task("Test task", "Description")
        controller.set_active_task(task.id)

        result = await controller.handle_command("plan", ["edit", "5", "Invalid"])
        assert "out of range" in result.lower()

    @pytest.mark.asyncio
    async def test_plan_delete(self, controller):
        """Should delete a plan step."""
        task = controller.create_task("Test task", "Description")
        task.plan = [
            PlanStep(id="step-1", description="Step 1", status=StepStatus.PENDING),
            PlanStep(id="step-2", description="Step 2", status=StepStatus.PENDING),
        ]
        controller.set_active_task(task.id)

        result = await controller.handle_command("plan", ["delete", "1"])
        assert "deleted step 1" in result.lower()

        assert len(task.plan) == 1
        assert task.plan[0].description == "Step 2"

    @pytest.mark.asyncio
    async def test_plan_reorder(self, controller):
        """Should reorder plan steps."""
        task = controller.create_task("Test task", "Description")
        task.plan = [
            PlanStep(id="step-1", description="First", status=StepStatus.PENDING),
            PlanStep(id="step-2", description="Second", status=StepStatus.PENDING),
            PlanStep(id="step-3", description="Third", status=StepStatus.PENDING),
        ]
        controller.set_active_task(task.id)

        result = await controller.handle_command("plan", ["reorder", "1", "3"])
        assert "moved step 1 to position 3" in result.lower()

        # First should now be at position 3
        assert task.plan[2].description == "First"
        assert task.plan[0].description == "Second"

    @pytest.mark.asyncio
    async def test_plan_no_active_task(self, controller):
        """Should fail when no active task."""
        result = await controller.handle_command("plan", ["add", "Test"])
        assert "no active task" in result.lower()


class TestStepCommands:
    """Test /step manipulation commands."""

    @pytest.mark.asyncio
    async def test_step_done(self, controller):
        """Should mark step as completed."""
        task = controller.create_task("Test task", "Description")
        task.plan = [PlanStep(id="step-1", description="Test step", status=StepStatus.PENDING)]
        controller.set_active_task(task.id)

        result = await controller.handle_command("step", ["done", "1"])
        assert "marked step 1 as completed" in result.lower()

        assert task.plan[0].status == StepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_step_skip(self, controller):
        """Should mark step as skipped."""
        task = controller.create_task("Test task", "Description")
        task.plan = [PlanStep(id="step-1", description="Test step", status=StepStatus.PENDING)]
        controller.set_active_task(task.id)

        result = await controller.handle_command("step", ["skip", "1"])
        assert "marked step 1 as skipped" in result.lower()

        assert task.plan[0].status == StepStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_step_block(self, controller):
        """Should mark step as blocked."""
        task = controller.create_task("Test task", "Description")
        task.plan = [PlanStep(id="step-1", description="Test step", status=StepStatus.PENDING)]
        controller.set_active_task(task.id)

        result = await controller.handle_command("step", ["block", "1", "Waiting", "for", "approval"])
        assert "marked step 1 as blocked" in result.lower()

        assert task.plan[0].status == StepStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_step_invalid_number(self, controller):
        """Should reject invalid step number."""
        task = controller.create_task("Test task", "Description")
        task.plan = [PlanStep(id="step-1", description="Test step", status=StepStatus.PENDING)]
        controller.set_active_task(task.id)

        result = await controller.handle_command("step", ["done", "99"])
        assert "out of range" in result.lower()

    @pytest.mark.asyncio
    async def test_step_no_plan(self, controller):
        """Should fail when task has no plan."""
        task = controller.create_task("Test task", "Description")
        controller.set_active_task(task.id)

        result = await controller.handle_command("step", ["done", "1"])
        assert "no plan steps" in result.lower()

    @pytest.mark.asyncio
    async def test_step_no_active_task(self, controller):
        """Should fail when no active task."""
        result = await controller.handle_command("step", ["done", "1"])
        assert "no active task" in result.lower()


class TestPhase6Persistence:
    """Test that Phase 6 operations persist correctly."""

    @pytest.mark.asyncio
    async def test_plan_changes_persist(self, controller):
        """Plan modifications should persist across save/load."""
        await controller.load_state()  # Initialize state tracking
        task = controller.create_task("Test task", "Description")
        controller.set_active_task(task.id)
        await controller.handle_command("plan", ["add", "Step", "1"])
        await controller.handle_command("plan", ["add", "Step", "2"])

        # Save
        await controller.save_state()

        # Create new controller with same file
        new_config = ControllerConfig(tasks_path=controller._persistence.tasks_path)
        new_controller = TaskFSMController(config=new_config)
        await new_controller.load_state()

        # Verify plan persisted
        loaded_task = new_controller._get_task_by_id(task.id)
        assert len(loaded_task.plan) == 2
        assert loaded_task.plan[0].description == "Step 1"
        assert loaded_task.plan[1].description == "Step 2"

    @pytest.mark.asyncio
    async def test_step_status_persists(self, controller):
        """Step status changes should persist."""
        await controller.load_state()  # Initialize state tracking
        task = controller.create_task("Test task", "Description")
        task.plan = [PlanStep(id="step-1", description="Test", status=StepStatus.PENDING)]
        controller.set_active_task(task.id)
        await controller.handle_command("step", ["done", "1"])

        await controller.save_state()

        new_config = ControllerConfig(tasks_path=controller._persistence.tasks_path)
        new_controller = TaskFSMController(config=new_config)
        await new_controller.load_state()

        loaded_task = new_controller._get_task_by_id(task.id)
        assert loaded_task.plan[0].status == StepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_reopened_task_persists(self, controller):
        """Reopened task phase should persist."""
        await controller.load_state()  # Initialize state tracking
        task = controller.create_task("Test task", "Description")
        controller.complete_task(task.id)
        await controller.handle_command("task", ["reopen", task.id])

        await controller.save_state()

        new_config = ControllerConfig(tasks_path=controller._persistence.tasks_path)
        new_controller = TaskFSMController(config=new_config)
        await new_controller.load_state()

        loaded_task = new_controller._get_task_by_id(task.id)
        assert loaded_task.phase == TaskPhase.PLANNING
