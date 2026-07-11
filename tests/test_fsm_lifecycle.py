"""
Level 1: Deterministic FSM Lifecycle Tests

This module tests the complete task lifecycle through the phase state machine
using mocked components. These tests are:
- Fast and deterministic (no LLM calls)
- CI-ready (no external dependencies)
- Cover the full task lifecycle: creation → planning → executing → done

Test scenarios:
1. Simple lifecycle (create → plan → execute → complete)
2. Clarification loop (plan → clarify → answer → plan → execute)
3. Blocked task (execute → blocked → unblock → execute)
4. Task switching (task A → pause → task B → resume A)
5. Edit approval flow (execute → waiting_approval → approve → execute)
6. Multi-step plan progression
"""

import pytest
import tempfile
import os
from unittest.mock import AsyncMock, patch, MagicMock

from mesh.controller import (
    TaskFSMController,
    Task,
    TaskPhase,
    PlanStep,
    StepStatus,
)
from mesh.controller.base import ControllerContext, ControllerDecision
from mesh.config import ControllerConfig


class MockMessage:
    """Mock message for testing."""
    def __init__(self, content: str, role: str = "user"):
        self.content = content
        self.role = role


class MockToolCall:
    """Mock tool call for testing."""
    def __init__(self, name: str, arguments: dict | None = None):
        self.name = name
        self.arguments = arguments or {}


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def controller(temp_dir):
    """Create a controller with temp storage and mocked RouterLLM."""
    config = ControllerConfig(
        mode="task-fsm-v0",
        tasks_path=os.path.join(temp_dir, "tasks.json"),
        config_path=os.path.join(temp_dir, "config.json"),
    )
    ctrl = TaskFSMController(config)
    # Mock the router to avoid LLM calls
    ctrl._router.classify = AsyncMock()
    return ctrl


@pytest.fixture
def context():
    """Create a mock ControllerContext."""
    return ControllerContext(
        cwd="/test",
        history=[],
        agent_id="test-agent",
        message=MockMessage("test"),
    )


# ============================================================================
# Scenario 1: Simple Lifecycle
# ============================================================================

class TestSimpleLifecycle:
    """Test the basic happy path: create → plan → execute → complete."""

    @pytest.mark.asyncio
    async def test_task_creation_sets_planning_phase(self, controller):
        """New tasks start in PLANNING phase."""
        task = controller.create_task(
            title="Implement feature X",
            description="Add feature X to the codebase"
        )

        assert task.phase == TaskPhase.PLANNING
        assert task.title == "Implement feature X"
        assert task.id.startswith("task-")

    @pytest.mark.asyncio
    async def test_planning_to_executing_via_file_write(self, controller, context):
        """File writes during planning trigger transition to WAITING_APPROVAL."""
        task = controller.create_task(title="Test task")
        controller.set_active_task(task.id)
        assert task.phase == TaskPhase.PLANNING

        # Simulate LLM response with file write
        response = "I'll create the implementation file."
        tool_calls = [MockToolCall("file_write", {"path": "/tmp/test.py", "content": "test"})]

        decision = await controller.on_llm_response(response, tool_calls, context)

        task = controller.get_active_task()
        assert task.phase == TaskPhase.WAITING_APPROVAL
        assert decision.action == "WAITING_APPROVAL"

    @pytest.mark.asyncio
    async def test_approval_to_executing(self, controller, context):
        """Approving edits transitions from WAITING_APPROVAL to EXECUTING."""
        task = controller.create_task(title="Test task")
        controller.set_active_task(task.id)

        # Trigger the edit proposal flow
        response = "Creating the file."
        tool_calls = [MockToolCall("file_write", {"path": "/tmp/test.py", "content": "test"})]
        await controller.on_llm_response(response, tool_calls, context)

        assert task.phase == TaskPhase.WAITING_APPROVAL
        assert len(task.pending_edits) > 0

        # Mock file operations for approval
        with patch("builtins.open", MagicMock()), \
             patch("os.makedirs"), \
             patch("pathlib.Path.exists", return_value=False):
            result = await controller.handle_command("approve", [])

        assert task.phase == TaskPhase.EXECUTING
        assert "Applied" in result

    @pytest.mark.asyncio
    async def test_executing_to_done_on_completion(self, controller, context):
        """Completion signals transition from EXECUTING to DONE."""
        task = controller.create_task(title="Test task")
        task.phase = TaskPhase.EXECUTING
        controller.set_active_task(task.id)

        response = "The implementation is complete! All tests pass."
        tool_calls = []

        decision = await controller.on_llm_response(response, tool_calls, context)

        task = controller.get_active_task()
        assert task.phase == TaskPhase.DONE
        assert decision.phase == TaskPhase.DONE.value

    @pytest.mark.asyncio
    async def test_full_lifecycle_flow(self, controller, context):
        """Test the complete lifecycle from creation to completion."""
        await controller.load_state()

        # Step 1: Create task
        task = controller.create_task(
            title="Add authentication",
            description="Implement user login"
        )
        controller.set_active_task(task.id)
        assert task.phase == TaskPhase.PLANNING

        # Step 2: LLM plans and starts writing
        response = """
Here's my plan:
1. Create auth middleware
2. Add login endpoint
3. Write tests

Creating the middleware now.
"""
        tool_calls = [MockToolCall("file_write", {"path": "/tmp/auth.py", "content": "auth"})]
        await controller.on_llm_response(response, tool_calls, context)

        task = controller.get_active_task()
        assert task.phase == TaskPhase.WAITING_APPROVAL
        assert len(task.plan) == 3  # Plan extracted

        # Step 3: Approve the edit
        with patch("builtins.open", MagicMock()), \
             patch("os.makedirs"), \
             patch("pathlib.Path.exists", return_value=False):
            await controller.handle_command("approve", [])

        assert task.phase == TaskPhase.EXECUTING

        # Step 4: LLM completes
        response = "All done! The authentication system is complete and tests pass."
        await controller.on_llm_response(response, [], context)

        assert task.phase == TaskPhase.DONE

        # Verify persistence
        await controller.save_state()

        new_controller = TaskFSMController(controller.config)
        await new_controller.load_state()
        # Find the task by iterating (no public get_by_id method)
        loaded_task = None
        for t in new_controller.get_tasks():
            if t.id == task.id:
                loaded_task = t
                break

        assert loaded_task is not None
        assert loaded_task.phase == TaskPhase.DONE
        assert len(loaded_task.plan) == 3


# ============================================================================
# Scenario 2: Clarification Loop
# ============================================================================

class TestClarificationLoop:
    """Test the clarification flow: ask question → user answers → continue."""

    @pytest.mark.asyncio
    async def test_planning_to_needs_clarification(self, controller, context):
        """Questions during planning trigger NEEDS_CLARIFICATION."""
        task = controller.create_task(title="Implement database")
        controller.set_active_task(task.id)

        response = "Before I proceed, I need to know: which database do you want to use?"
        await controller.on_llm_response(response, [], context)

        task = controller.get_active_task()
        assert task.phase == TaskPhase.NEEDS_CLARIFICATION

    @pytest.mark.asyncio
    async def test_clarification_to_planning_on_file_write(self, controller, context):
        """After clarification, file writes move to WAITING_APPROVAL."""
        task = controller.create_task(title="Test task")
        task.phase = TaskPhase.NEEDS_CLARIFICATION
        controller.set_active_task(task.id)

        # User provided answer, LLM now proceeds
        response = "Got it, using PostgreSQL. Creating the schema."
        tool_calls = [MockToolCall("file_write", {"path": "/tmp/schema.sql", "content": "CREATE TABLE..."})]

        await controller.on_llm_response(response, tool_calls, context)

        task = controller.get_active_task()
        # Goes to WAITING_APPROVAL since there's a file write
        assert task.phase == TaskPhase.WAITING_APPROVAL

    @pytest.mark.asyncio
    async def test_multiple_clarification_rounds(self, controller, context):
        """Multiple rounds of clarification before proceeding."""
        task = controller.create_task(title="Complex feature")
        controller.set_active_task(task.id)

        # Round 1: First question
        response = "I need to know: should this use REST or GraphQL?"
        await controller.on_llm_response(response, [], context)
        assert task.phase == TaskPhase.NEEDS_CLARIFICATION

        # Round 2: User answers "REST", but LLM has another question
        # Note: In real flow, on_message would be called first, but we're testing phase detector
        response = "REST it is. One more thing - what authentication method?"
        await controller.on_llm_response(response, [], context)
        # Still needs clarification
        assert task.phase == TaskPhase.NEEDS_CLARIFICATION

        # Round 3: Now LLM has enough info
        response = "Perfect. JWT auth with REST API. Creating implementation."
        tool_calls = [MockToolCall("file_write", {"path": "/tmp/api.py", "content": "..."})]
        await controller.on_llm_response(response, tool_calls, context)

        assert task.phase == TaskPhase.WAITING_APPROVAL


# ============================================================================
# Scenario 3: Blocked Task
# ============================================================================

class TestBlockedTask:
    """Test blocking flow: task becomes blocked and then unblocked."""

    @pytest.mark.asyncio
    async def test_executing_to_blocked(self, controller, context):
        """External dependencies cause BLOCKED transition."""
        task = controller.create_task(title="Deploy to prod")
        task.phase = TaskPhase.EXECUTING
        controller.set_active_task(task.id)

        response = "I'm blocked on this - waiting for the API credentials from the team."
        await controller.on_llm_response(response, [], context)

        task = controller.get_active_task()
        assert task.phase == TaskPhase.BLOCKED

    @pytest.mark.asyncio
    async def test_planning_to_blocked(self, controller, context):
        """Blocking can happen from any phase."""
        task = controller.create_task(title="Setup CI")
        controller.set_active_task(task.id)
        assert task.phase == TaskPhase.PLANNING

        response = "Cannot proceed - requires approval from the security team first."
        await controller.on_llm_response(response, [], context)

        task = controller.get_active_task()
        assert task.phase == TaskPhase.BLOCKED

    @pytest.mark.asyncio
    async def test_blocked_priority_over_done(self, controller, context):
        """BLOCKED takes priority over completion signals."""
        task = controller.create_task(title="Test priority")
        task.phase = TaskPhase.EXECUTING
        controller.set_active_task(task.id)

        # Response has both completion and blocking signals
        response = "The code is complete, but blocked on deployment - waiting for approval."
        await controller.on_llm_response(response, [], context)

        task = controller.get_active_task()
        # BLOCKED should win
        assert task.phase == TaskPhase.BLOCKED


# ============================================================================
# Scenario 4: Task Switching
# ============================================================================

class TestTaskSwitching:
    """Test switching between multiple tasks."""

    @pytest.mark.asyncio
    async def test_switch_active_task(self, controller):
        """Switching active task preserves both tasks' states."""
        # Create two tasks
        task_a = controller.create_task(title="Feature A")
        task_b = controller.create_task(title="Feature B")

        # Work on task A
        controller.set_active_task(task_a.id)
        task_a.phase = TaskPhase.EXECUTING

        # Switch to task B
        controller.set_active_task(task_b.id)
        assert controller.get_active_task().id == task_b.id
        assert task_b.phase == TaskPhase.PLANNING

        # Task A state preserved - find it in the task list
        task_a_check = None
        for t in controller.get_tasks():
            if t.id == task_a.id:
                task_a_check = t
                break
        assert task_a_check is not None
        assert task_a_check.phase == TaskPhase.EXECUTING

    @pytest.mark.asyncio
    async def test_list_all_tasks(self, controller):
        """Can list all tasks regardless of phase."""
        task_a = controller.create_task(title="Task A")
        task_b = controller.create_task(title="Task B")
        task_c = controller.create_task(title="Task C")

        task_a.phase = TaskPhase.DONE
        task_b.phase = TaskPhase.EXECUTING
        task_c.phase = TaskPhase.PLANNING

        all_tasks = controller.get_tasks()
        assert len(all_tasks) == 3

        active_tasks = [t for t in all_tasks if t.is_active()]
        assert len(active_tasks) == 2  # B and C

    @pytest.mark.asyncio
    async def test_resume_task_by_id(self, controller, context):
        """Can resume a specific task by ID."""
        task_a = controller.create_task(title="Feature A")
        task_b = controller.create_task(title="Feature B")

        controller.set_active_task(task_a.id)
        task_a.phase = TaskPhase.EXECUTING
        task_a.plan = [
            PlanStep(id="step-1", description="First step", status=StepStatus.COMPLETED),
            PlanStep(id="step-2", description="Second step", status=StepStatus.IN_PROGRESS),
        ]

        # Switch away
        controller.set_active_task(task_b.id)

        # Resume task A
        controller.set_active_task(task_a.id)

        task = controller.get_active_task()
        assert task.id == task_a.id
        assert task.phase == TaskPhase.EXECUTING
        assert task.plan[1].status == StepStatus.IN_PROGRESS


# ============================================================================
# Scenario 5: Edit Approval Flow
# ============================================================================

class TestEditApprovalFlow:
    """Test the edit proposal and approval workflow."""

    @pytest.mark.asyncio
    async def test_file_write_creates_proposal(self, controller, context):
        """File write creates an edit proposal."""
        task = controller.create_task(title="Write config")
        controller.set_active_task(task.id)

        response = "Creating the configuration file."
        tool_calls = [MockToolCall("file_write", {
            "path": "/tmp/config.yaml",
            "content": "key: value"
        })]

        await controller.on_llm_response(response, tool_calls, context)

        task = controller.get_active_task()
        assert task.phase == TaskPhase.WAITING_APPROVAL
        assert len(task.pending_edits) == 1
        assert task.pending_edits[0].file_path == "/tmp/config.yaml"

    @pytest.mark.asyncio
    async def test_reject_proposal(self, controller, context):
        """Rejecting proposal removes it and returns to EXECUTING."""
        task = controller.create_task(title="Write file")
        controller.set_active_task(task.id)

        tool_calls = [MockToolCall("file_write", {"path": "/tmp/test.py", "content": "..."})]
        await controller.on_llm_response("Creating file.", tool_calls, context)

        assert task.phase == TaskPhase.WAITING_APPROVAL
        assert len(task.pending_edits) == 1

        result = await controller.handle_command("reject", [])

        assert task.phase == TaskPhase.EXECUTING
        assert len(task.pending_edits) == 0
        assert "Rejected" in result

    @pytest.mark.asyncio
    async def test_diff_command_shows_proposal(self, controller, context):
        """diff command shows the pending edit."""
        task = controller.create_task(title="Write file")
        controller.set_active_task(task.id)

        tool_calls = [MockToolCall("file_write", {
            "path": "/tmp/test.py",
            "content": "print('hello')"
        })]
        await controller.on_llm_response("Creating file.", tool_calls, context)

        result = await controller.handle_command("diff", [])

        assert "/tmp/test.py" in result
        assert "print('hello')" in result

    @pytest.mark.asyncio
    async def test_multiple_edits_single_approval(self, controller, context):
        """Multiple file writes create multiple proposals."""
        task = controller.create_task(title="Multi-file change")
        controller.set_active_task(task.id)

        tool_calls = [
            MockToolCall("file_write", {"path": "/tmp/a.py", "content": "a"}),
            MockToolCall("file_create", {"path": "/tmp/b.py", "content": "b"}),
            MockToolCall("file_edit", {"path": "/tmp/c.py", "old_string": "x", "new_string": "y"}),
        ]
        await controller.on_llm_response("Creating files.", tool_calls, context)

        task = controller.get_active_task()
        assert task.phase == TaskPhase.WAITING_APPROVAL
        assert len(task.pending_edits) == 3


# ============================================================================
# Scenario 6: Multi-Step Plan Progression
# ============================================================================

class TestMultiStepPlan:
    """Test progressing through multiple plan steps."""

    @pytest.mark.asyncio
    async def test_plan_extraction(self, controller, context):
        """Plan steps are extracted from numbered lists when transitioning from PLANNING."""
        task = controller.create_task(title="Build API")
        controller.set_active_task(task.id)

        response = """
Here's my plan for the API:

1. Create the database models
2. Implement the REST endpoints
3. Add authentication middleware
4. Write integration tests
5. Update documentation

Let me start with the database models.
"""
        # File write triggers the transition from PLANNING, which extracts the plan
        tool_calls = [MockToolCall("file_write", {"path": "/tmp/models.py", "content": "..."})]
        await controller.on_llm_response(response, tool_calls, context)

        task = controller.get_active_task()
        # Plan is extracted during the PLANNING → WAITING_APPROVAL transition
        assert len(task.plan) == 5
        assert task.plan[0].description == "Create the database models"
        assert task.plan[4].description == "Update documentation"

    @pytest.mark.asyncio
    async def test_step_done_command(self, controller):
        """Marking steps as done via command."""
        task = controller.create_task(title="Multi-step task")
        task.plan = [
            PlanStep(id="step-1", description="Step one"),
            PlanStep(id="step-2", description="Step two"),
            PlanStep(id="step-3", description="Step three"),
        ]
        controller.set_active_task(task.id)

        result = await controller.handle_command("step", ["done", "1"])

        assert task.plan[0].status == StepStatus.COMPLETED
        assert "completed" in result.lower()

    @pytest.mark.asyncio
    async def test_step_skip_command(self, controller):
        """Skipping steps via command."""
        task = controller.create_task(title="Task with skippable step")
        task.plan = [
            PlanStep(id="step-1", description="Essential"),
            PlanStep(id="step-2", description="Optional optimization"),
        ]
        controller.set_active_task(task.id)

        result = await controller.handle_command("step", ["skip", "2"])

        assert task.plan[1].status == StepStatus.SKIPPED
        assert "skipped" in result.lower()

    @pytest.mark.asyncio
    async def test_plan_add_command(self, controller):
        """Adding steps via command."""
        task = controller.create_task(title="Extensible task")
        task.plan = [PlanStep(id="step-1", description="Initial step")]
        controller.set_active_task(task.id)

        result = await controller.handle_command("plan", ["add", "New step from user"])

        assert len(task.plan) == 2
        assert task.plan[1].description == "New step from user"
        assert "Added" in result

    @pytest.mark.asyncio
    async def test_plan_reorder_command(self, controller):
        """Reordering steps via command."""
        task = controller.create_task(title="Reorder task")
        task.plan = [
            PlanStep(id="step-1", description="First"),
            PlanStep(id="step-2", description="Second"),
            PlanStep(id="step-3", description="Third"),
        ]
        controller.set_active_task(task.id)

        result = await controller.handle_command("plan", ["reorder", "3", "1"])

        assert task.plan[0].description == "Third"
        assert task.plan[1].description == "First"
        assert task.plan[2].description == "Second"


# ============================================================================
# Persistence Tests
# ============================================================================

class TestPersistence:
    """Test state persistence across controller restarts."""

    @pytest.mark.asyncio
    async def test_save_and_load_task_state(self, temp_dir):
        """Tasks survive save/load cycle."""
        config = ControllerConfig(
            mode="task-fsm-v0",
            tasks_path=os.path.join(temp_dir, "tasks.json"),
            config_path=os.path.join(temp_dir, "config.json"),
        )

        # Create and save
        ctrl1 = TaskFSMController(config)
        ctrl1._router.classify = AsyncMock()
        await ctrl1.load_state()

        task = ctrl1.create_task(title="Persistent task")
        task.phase = TaskPhase.EXECUTING
        task.plan = [
            PlanStep(id="step-1", description="Done", status=StepStatus.COMPLETED),
            PlanStep(id="step-2", description="In progress", status=StepStatus.IN_PROGRESS),
        ]
        ctrl1.set_active_task(task.id)
        await ctrl1.save_state()

        # Load in new controller
        ctrl2 = TaskFSMController(config)
        ctrl2._router.classify = AsyncMock()
        await ctrl2.load_state()

        loaded_task = ctrl2.get_active_task()
        assert loaded_task is not None
        assert loaded_task.title == "Persistent task"
        assert loaded_task.phase == TaskPhase.EXECUTING
        assert len(loaded_task.plan) == 2
        assert loaded_task.plan[0].status == StepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_pending_edits_persist(self, temp_dir, context):
        """Pending edits survive save/load cycle."""
        config = ControllerConfig(
            mode="task-fsm-v0",
            tasks_path=os.path.join(temp_dir, "tasks.json"),
            config_path=os.path.join(temp_dir, "config.json"),
        )

        ctrl1 = TaskFSMController(config)
        ctrl1._router.classify = AsyncMock()
        await ctrl1.load_state()

        task = ctrl1.create_task(title="Edit task")
        ctrl1.set_active_task(task.id)

        tool_calls = [MockToolCall("file_write", {"path": "/tmp/test.py", "content": "test"})]
        await ctrl1.on_llm_response("Writing file.", tool_calls, context)

        assert task.phase == TaskPhase.WAITING_APPROVAL
        assert len(task.pending_edits) == 1

        await ctrl1.save_state()

        # Load in new controller
        ctrl2 = TaskFSMController(config)
        ctrl2._router.classify = AsyncMock()
        await ctrl2.load_state()

        loaded_task = ctrl2.get_active_task()
        assert loaded_task.phase == TaskPhase.WAITING_APPROVAL
        assert len(loaded_task.pending_edits) == 1
        assert loaded_task.pending_edits[0].file_path == "/tmp/test.py"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_no_active_task(self, controller, context):
        """Operations without active task don't crash."""
        decision = await controller.on_llm_response("Some response", [], context)

        # Should return a decision but no task transitions
        assert decision is not None
        assert decision.task_id is None

    @pytest.mark.asyncio
    async def test_command_without_active_task(self, controller):
        """Commands without active task return helpful error."""
        result = await controller.handle_command("plan", ["add", "step"])
        assert "no active task" in result.lower()

    @pytest.mark.asyncio
    async def test_invalid_task_id(self, controller):
        """Setting invalid task ID is handled gracefully."""
        controller.set_active_task("nonexistent-task-id")
        task = controller.get_active_task()
        assert task is None

    @pytest.mark.asyncio
    async def test_empty_plan_commands(self, controller):
        """Plan commands with empty plan are handled."""
        task = controller.create_task(title="Empty plan task")
        controller.set_active_task(task.id)

        result = await controller.handle_command("plan", [])
        assert "no plan steps" in result.lower() or "no steps" in result.lower() or "empty" in result.lower()

    @pytest.mark.asyncio
    async def test_step_command_out_of_range(self, controller):
        """Step commands with invalid index are handled."""
        task = controller.create_task(title="Small plan")
        task.plan = [PlanStep(id="step-1", description="Only step")]
        controller.set_active_task(task.id)

        result = await controller.handle_command("step", ["done", "5"])
        assert "invalid" in result.lower() or "range" in result.lower() or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_reopen_non_done_task(self, controller):
        """Reopening a task that isn't DONE returns error."""
        task = controller.create_task(title="Active task")
        task.phase = TaskPhase.EXECUTING
        controller.set_active_task(task.id)

        result = await controller.handle_command("task", ["reopen", task.id])
        assert "not done" in result.lower() or "only done" in result.lower()
