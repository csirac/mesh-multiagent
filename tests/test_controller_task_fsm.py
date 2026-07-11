"""
Unit tests for mesh/controller/task_fsm.py
"""

import os
import pytest
import tempfile

from mesh.controller.task_fsm import TaskFSMController
from mesh.controller.base import ControllerContext
from mesh.controller.models import Task, TaskPhase, PlanStep, StepStatus
from mesh.config import ControllerConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def controller(temp_dir):
    """Create a TaskFSMController with temp paths."""
    config = ControllerConfig(
        mode="task-fsm-v0",
        tasks_path=os.path.join(temp_dir, "tasks.json"),
        config_path=os.path.join(temp_dir, "config.json"),
    )
    return TaskFSMController(config)


@pytest.fixture
def context():
    """Create a minimal ControllerContext."""
    return ControllerContext(
        cwd="/home/test",
        history=[],
    )


class TestTaskFSMControllerInit:
    """Test controller initialization."""

    def test_init_with_config(self, controller):
        """Should initialize with provided config."""
        assert controller.config is not None
        assert controller._tasks == []
        assert controller._active_task_id is None
        assert controller._state_loaded is False

    def test_init_without_config(self, temp_dir):
        """Should use defaults when config is None."""
        controller = TaskFSMController(config=None)
        assert controller.config is not None
        # Default paths
        assert "tasks.json" in str(controller._persistence.tasks_path)


class TestLoadAndSaveState:
    """Test state persistence."""

    @pytest.mark.asyncio
    async def test_load_state_empty(self, controller):
        """Should load empty state when no files exist."""
        await controller.load_state()
        assert controller._tasks == []
        assert controller._active_task_id is None
        assert controller._state_loaded is True

    @pytest.mark.asyncio
    async def test_save_and_load_roundtrip(self, controller):
        """Should save and load tasks correctly."""
        # Load (creates empty state)
        await controller.load_state()

        # Create a task
        task = controller.create_task(
            title="Test task",
            description="A test",
            original_request="Do something"
        )
        controller.set_active_task(task.id)

        # Save
        await controller.save_state()

        # Create new controller with same paths
        controller2 = TaskFSMController(controller.config)
        await controller2.load_state()

        # Verify state was restored
        assert len(controller2._tasks) == 1
        assert controller2._tasks[0].id == task.id
        assert controller2._tasks[0].title == "Test task"
        assert controller2._active_task_id == task.id

    @pytest.mark.asyncio
    async def test_skip_save_if_not_loaded(self, controller, caplog):
        """Should skip save if state wasn't loaded first."""
        await controller.save_state()
        assert "Skipping save" in caplog.text


class TestTaskManagement:
    """Test task CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_task(self, controller):
        """Should create a new task with generated ID."""
        await controller.load_state()

        task = controller.create_task(
            title="New task",
            description="Description",
            original_request="User message"
        )

        assert task.id.startswith("task-")
        assert task.title == "New task"
        assert task.description == "Description"
        assert task.original_request == "User message"
        assert task.phase == TaskPhase.PLANNING
        assert task in controller._tasks

    @pytest.mark.asyncio
    async def test_set_active_task(self, controller):
        """Should activate a task by ID."""
        await controller.load_state()

        task1 = controller.create_task(title="Task 1")
        task2 = controller.create_task(title="Task 2")

        # Activate task2
        result = controller.set_active_task(task2.id)
        assert result is True
        assert controller._active_task_id == task2.id
        assert controller.get_active_task() == task2

        # Clear active task
        result = controller.set_active_task(None)
        assert result is True
        assert controller._active_task_id is None

    @pytest.mark.asyncio
    async def test_set_active_task_not_found(self, controller):
        """Should return False if task not found."""
        await controller.load_state()

        result = controller.set_active_task("nonexistent-task")
        assert result is False

    @pytest.mark.asyncio
    async def test_complete_task(self, controller):
        """Should mark task as done."""
        await controller.load_state()

        task = controller.create_task(title="To complete")
        controller.set_active_task(task.id)

        result = controller.complete_task(task.id)
        assert result is True
        assert task.phase == TaskPhase.DONE
        assert task.completed_at != ""
        assert controller._active_task_id is None  # Cleared

    @pytest.mark.asyncio
    async def test_complete_task_not_found(self, controller):
        """Should return False if task not found."""
        await controller.load_state()

        result = controller.complete_task("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_tasks(self, controller):
        """Should return all tasks."""
        await controller.load_state()

        controller.create_task(title="Task 1")
        controller.create_task(title="Task 2")

        tasks = controller.get_tasks()
        assert len(tasks) == 2


class TestOnMessage:
    """Test message routing (v0 passthrough behavior)."""

    @pytest.mark.asyncio
    async def test_on_message_passthrough(self, controller, context):
        """v0 should pass through to LLM."""
        await controller.load_state()

        # Create a mock message
        class MockMessage:
            content = "Hello"

        decision = await controller.on_message(MockMessage(), context)
        assert decision.action == "PROCESS_WITH_LLM"
        assert decision.payload["message"].content == "Hello"

    @pytest.mark.asyncio
    async def test_on_message_with_active_task(self, controller, context):
        """Should route to active task when appropriate (Phase 3)."""
        from unittest.mock import AsyncMock, patch
        await controller.load_state()

        task = controller.create_task(title="Active task")
        controller.set_active_task(task.id)

        class MockMessage:
            content = "Hello"

        # Mock router to return ROUTE_TO_TASK
        mock_classification = {
            "action": "ROUTE_TO_TASK",
            "task_id": task.id,
            "confidence": 0.95,
            "reasoning": "Continuing active task"
        }

        with patch.object(controller._router, 'classify_message', new=AsyncMock(return_value=mock_classification)):
            decision = await controller.on_message(MockMessage(), context)

        assert decision.task_id == task.id


class TestOnLLMResponse:
    """Test LLM response handling (v0 passthrough behavior)."""

    @pytest.mark.asyncio
    async def test_on_llm_response_no_tools(self, controller, context):
        """Should return DONE when no tool calls."""
        await controller.load_state()

        decision = await controller.on_llm_response(
            response="Hello!",
            tool_calls=[],
            context=context
        )
        assert decision.action == "DONE"
        assert decision.payload["response"] == "Hello!"

    @pytest.mark.asyncio
    async def test_on_llm_response_with_tools(self, controller, context):
        """Should return EXECUTE_TOOLS when tool calls present."""
        await controller.load_state()

        mock_tool_calls = [{"name": "some_tool", "args": {}}]
        decision = await controller.on_llm_response(
            response="Let me help",
            tool_calls=mock_tool_calls,
            context=context
        )
        assert decision.action == "EXECUTE_TOOLS"
        assert decision.payload["tool_calls"] == mock_tool_calls


class TestCommandHandling:
    """Test controller commands."""

    @pytest.mark.asyncio
    async def test_tasks_command_empty(self, controller):
        """Should show no active tasks message."""
        await controller.load_state()

        result = await controller.handle_command("tasks", [])
        assert "No active tasks" in result

    @pytest.mark.asyncio
    async def test_tasks_command_with_tasks(self, controller):
        """Should list active tasks."""
        await controller.load_state()

        t1 = controller.create_task(title="First task")
        t2 = controller.create_task(title="Second task")
        controller.set_active_task(t1.id)

        result = await controller.handle_command("tasks", [])
        assert "2 task(s)" in result
        assert "First task" in result
        assert "Second task" in result
        assert "→" in result  # Active task marker

    @pytest.mark.asyncio
    async def test_tasks_command_all(self, controller):
        """Should include completed tasks with --all."""
        await controller.load_state()

        t1 = controller.create_task(title="Active")
        t2 = controller.create_task(title="Done")
        controller.complete_task(t2.id)

        # Without --all
        result = await controller.handle_command("tasks", [])
        assert "Done" not in result

        # With --all
        result = await controller.handle_command("tasks", ["--all"])
        assert "Done" in result
        assert "including completed" in result

    @pytest.mark.asyncio
    async def test_task_command_no_active(self, controller):
        """Should show message when no active task."""
        await controller.load_state()

        result = await controller.handle_command("task", [])
        assert "No active task" in result

    @pytest.mark.asyncio
    async def test_task_command_show_active(self, controller):
        """Should show active task details."""
        await controller.load_state()

        task = controller.create_task(
            title="My task",
            description="Detailed description"
        )
        task.plan = [
            PlanStep(id="s1", description="Step 1", status=StepStatus.COMPLETED),
            PlanStep(id="s2", description="Step 2", status=StepStatus.PENDING),
        ]
        controller.set_active_task(task.id)

        result = await controller.handle_command("task", [])
        assert "My task" in result
        assert task.id in result
        assert "planning" in result.lower()
        assert "2 steps" in result

    @pytest.mark.asyncio
    async def test_task_command_switch(self, controller):
        """Should switch to specified task."""
        await controller.load_state()

        t1 = controller.create_task(title="Task 1")
        t2 = controller.create_task(title="Task 2")

        result = await controller.handle_command("task", [t2.id])
        assert "Switched to task" in result
        assert controller._active_task_id == t2.id

    @pytest.mark.asyncio
    async def test_task_command_done(self, controller):
        """Should mark task as done."""
        await controller.load_state()

        task = controller.create_task(title="To complete")
        controller.set_active_task(task.id)

        result = await controller.handle_command("task", ["done"])
        assert "marked as done" in result
        assert task.phase == TaskPhase.DONE

    @pytest.mark.asyncio
    async def test_task_command_done_with_id(self, controller):
        """Should mark specific task as done."""
        await controller.load_state()

        task = controller.create_task(title="To complete")

        result = await controller.handle_command("task", ["done", task.id])
        assert "marked as done" in result
        assert task.phase == TaskPhase.DONE

    @pytest.mark.asyncio
    async def test_task_command_delete(self, controller):
        """Should delete a task."""
        await controller.load_state()

        task = controller.create_task(title="To delete")
        task_id = task.id

        result = await controller.handle_command("task", ["delete", task_id])
        assert "deleted" in result
        assert len(controller._tasks) == 0

    @pytest.mark.asyncio
    async def test_task_command_delete_clears_active(self, controller):
        """Should clear active task if deleted."""
        await controller.load_state()

        task = controller.create_task(title="Active then deleted")
        controller.set_active_task(task.id)

        await controller.handle_command("task", ["delete", task.id])
        assert controller._active_task_id is None

    @pytest.mark.asyncio
    async def test_unknown_command_returns_none(self, controller):
        """Should return None for unknown commands."""
        await controller.load_state()

        result = await controller.handle_command("unknown", [])
        assert result is None

    @pytest.mark.asyncio
    async def test_phase5_commands_without_task(self, controller):
        """Phase 5 commands should return appropriate error with no active task."""
        await controller.load_state()

        for cmd in ["approve", "reject", "diff"]:
            result = await controller.handle_command(cmd, [])
            assert "No active task" in result or "No pending edits" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
