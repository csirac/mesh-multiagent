"""
Integration tests for Phase 5: Edit Approval
"""

import pytest
import tempfile
import os
from mesh.controller import TaskFSMController, ControllerContext, TaskPhase
from mesh.controller.task_fsm import TaskFSMController as TaskFSM


class MockMessage:
    """Mock message for testing."""
    def __init__(self, content):
        self.content = content


class MockToolCall:
    """Mock tool call for testing."""
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


@pytest.mark.asyncio
class TestPhase5Integration:
    """Integration tests for edit approval flow."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def controller(self, temp_dir):
        """Create controller with temp storage."""
        from mesh.config import ControllerConfig

        tasks_path = os.path.join(temp_dir, "tasks.json")
        config_path = os.path.join(temp_dir, "config.json")

        config = ControllerConfig(
            mode="task-fsm-v0",
            tasks_path=tasks_path,
            config_path=config_path,
            router_model="gpt-4o-mini",
        )

        controller = TaskFSM(config)
        return controller

    async def test_edit_interception_file_write(self, controller, temp_dir):
        """Test that file_write is intercepted and creates proposal."""
        await controller.load_state()

        # Create a task
        task = controller.create_task("Test edit interception", "Testing Phase 5")
        controller.set_active_task(task.id)
        task.phase = TaskPhase.EXECUTING

        # Create test file
        test_file = os.path.join(temp_dir, "test.py")
        with open(test_file, "w") as f:
            f.write("old content")

        # Simulate LLM response with file_write
        tool_calls = [
            MockToolCall("file_write", {
                "path": test_file,
                "content": "new content",
            })
        ]

        ctx = ControllerContext(
            cwd=temp_dir,
            history=[],
            agent_id="test-agent",
            message=MockMessage("write the file"),
        )

        decision = await controller.on_llm_response(
            response="Writing the file now.",
            tool_calls=tool_calls,
            context=ctx,
        )

        # Should transition to WAITING_APPROVAL
        assert decision.action == "WAITING_APPROVAL"
        assert task.phase == TaskPhase.WAITING_APPROVAL
        assert len(task.pending_edits) == 1

        # Check proposal
        proposal = task.pending_edits[0]
        assert proposal.file_path == test_file
        assert proposal.old_content == "old content"
        assert proposal.new_content == "new content"
        assert proposal.approved is None

    async def test_edit_interception_multiple_files(self, controller, temp_dir):
        """Test interception of multiple file writes."""
        await controller.load_state()

        task = controller.create_task("Multiple edits", "")
        controller.set_active_task(task.id)
        task.phase = TaskPhase.EXECUTING

        # Create test files
        file1 = os.path.join(temp_dir, "file1.py")
        file2 = os.path.join(temp_dir, "file2.py")
        with open(file1, "w") as f:
            f.write("content 1")
        with open(file2, "w") as f:
            f.write("content 2")

        # Simulate multiple writes
        tool_calls = [
            MockToolCall("file_write", {"path": file1, "content": "new 1"}),
            MockToolCall("file_write", {"path": file2, "content": "new 2"}),
        ]

        ctx = ControllerContext(cwd=temp_dir, history=[], agent_id="test", message=MockMessage("edit"))

        decision = await controller.on_llm_response("", tool_calls, ctx)

        assert decision.action == "WAITING_APPROVAL"
        assert len(task.pending_edits) == 2

    async def test_approve_command(self, controller, temp_dir):
        """Test /approve command applies edits."""
        await controller.load_state()

        task = controller.create_task("Approve test", "")
        controller.set_active_task(task.id)
        task.phase = TaskPhase.WAITING_APPROVAL

        # Create test file
        test_file = os.path.join(temp_dir, "approve_test.py")
        with open(test_file, "w") as f:
            f.write("original")

        # Add pending edit
        from mesh.controller.models import EditProposal
        proposal = EditProposal(
            id="edit-001",
            task_id=task.id,
            file_path=test_file,
            old_content="original",
            new_content="modified",
            description="Test edit",
        )
        task.pending_edits.append(proposal)

        # Approve
        result = await controller.handle_command("approve", [])

        assert "Applied 1 edit(s)" in result
        assert task.phase == TaskPhase.EXECUTING
        assert len(task.pending_edits) == 0

        # Verify file was modified
        with open(test_file, "r") as f:
            content = f.read()
        assert content == "modified"

    async def test_reject_command(self, controller, temp_dir):
        """Test /reject command cancels edits."""
        await controller.load_state()

        task = controller.create_task("Reject test", "")
        controller.set_active_task(task.id)
        task.phase = TaskPhase.WAITING_APPROVAL

        # Create test file
        test_file = os.path.join(temp_dir, "reject_test.py")
        with open(test_file, "w") as f:
            f.write("original")

        # Add pending edit
        from mesh.controller.models import EditProposal
        proposal = EditProposal(
            id="edit-002",
            task_id=task.id,
            file_path=test_file,
            old_content="original",
            new_content="modified",
            description="Test edit",
        )
        task.pending_edits.append(proposal)

        # Reject
        result = await controller.handle_command("reject", [])

        assert "Rejected 1 pending edit(s)" in result
        assert task.phase == TaskPhase.EXECUTING
        assert len(task.pending_edits) == 0

        # Verify file was NOT modified
        with open(test_file, "r") as f:
            content = f.read()
        assert content == "original"

    async def test_diff_command(self, controller, temp_dir):
        """Test /diff command shows proposals."""
        await controller.load_state()

        task = controller.create_task("Diff test", "")
        controller.set_active_task(task.id)

        # Add pending edit
        from mesh.controller.models import EditProposal
        proposal = EditProposal(
            id="edit-003",
            task_id=task.id,
            file_path="/tmp/diff_test.py",
            old_content="old",
            new_content="new",
            diff="--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new",
            description="Diff test",
        )
        task.pending_edits.append(proposal)

        # Show diff
        result = await controller.handle_command("diff", [])

        assert "## Edit Proposal:" in result
        assert "edit-003" in result
        assert "/tmp/diff_test.py" in result
        assert "```diff" in result

    async def test_approve_no_pending(self, controller):
        """Test /approve with no pending edits."""
        await controller.load_state()

        task = controller.create_task("No edits", "")
        controller.set_active_task(task.id)

        result = await controller.handle_command("approve", [])
        assert "No pending edits" in result

    async def test_approve_wrong_phase(self, controller):
        """Test /approve when not in WAITING_APPROVAL phase."""
        await controller.load_state()

        task = controller.create_task("Wrong phase", "")
        controller.set_active_task(task.id)
        task.phase = TaskPhase.EXECUTING

        # Add pending edit (shouldn't happen in reality, but test the guard)
        from mesh.controller.models import EditProposal
        task.pending_edits.append(EditProposal(
            id="edit-004",
            task_id=task.id,
            file_path="/tmp/test.py",
            old_content="",
            new_content="",
            description="",
        ))

        result = await controller.handle_command("approve", [])
        assert "not in WAITING_APPROVAL phase" in result

    async def test_edit_with_no_active_task(self, controller, temp_dir):
        """Test edit interception with no active task."""
        await controller.load_state()

        # No active task
        tool_calls = [
            MockToolCall("file_write", {
                "path": "/tmp/test.py",
                "content": "test",
            })
        ]

        ctx = ControllerContext(cwd=temp_dir, history=[], agent_id="test", message=MockMessage("edit"))

        decision = await controller.on_llm_response("", tool_calls, ctx)

        # Should pass through to EXECUTE_TOOLS (no interception without task)
        assert decision.action == "EXECUTE_TOOLS"

    async def test_persistence_with_pending_edits(self, controller, temp_dir):
        """Test that pending edits persist across save/load."""
        await controller.load_state()

        task = controller.create_task("Persistence test", "")
        controller.set_active_task(task.id)
        task.phase = TaskPhase.WAITING_APPROVAL

        # Add pending edit
        from mesh.controller.models import EditProposal
        proposal = EditProposal(
            id="edit-005",
            task_id=task.id,
            file_path="/tmp/persist.py",
            old_content="old",
            new_content="new",
            diff="...",
            description="Persist test",
        )
        task.pending_edits.append(proposal)

        # Save
        await controller.save_state()

        # Create new controller instance and load
        from mesh.config import ControllerConfig
        config2 = ControllerConfig(
            mode="task-fsm-v0",
            tasks_path=controller._persistence.tasks_path,
            config_path=controller._persistence.config_path,
        )
        controller2 = TaskFSM(config2)
        await controller2.load_state()

        # Verify pending edit was restored
        task2 = controller2.get_active_task()
        assert task2 is not None
        assert task2.phase == TaskPhase.WAITING_APPROVAL
        assert len(task2.pending_edits) == 1
        assert task2.pending_edits[0].id == "edit-005"
