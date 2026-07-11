"""
Unit tests for mesh/controller/models.py
"""

import pytest
from datetime import datetime
from mesh.controller.models import (
    Task,
    TaskPhase,
    PlanStep,
    StepStatus,
    Resource,
    EditProposal,
)


class TestTaskPhase:
    """Test TaskPhase enum."""

    def test_all_phases_are_strings(self):
        """All phases should be string values for JSON serialization."""
        for phase in TaskPhase:
            assert isinstance(phase.value, str)

    def test_phase_values_are_unique(self):
        """All phase values should be unique."""
        values = [p.value for p in TaskPhase]
        assert len(values) == len(set(values))


class TestStepStatus:
    """Test StepStatus enum."""

    def test_all_statuses_are_strings(self):
        for status in StepStatus:
            assert isinstance(status.value, str)


class TestResource:
    """Test Resource dataclass."""

    def test_default_values(self):
        """Resource should have sensible defaults."""
        r = Resource(path="/some/file.py")
        assert r.path == "/some/file.py"
        assert r.resource_type == "file"
        assert r.description == ""
        assert r.discovered_at != ""  # Auto-populated

    def test_discovered_at_auto_populated(self):
        """discovered_at should be auto-populated with ISO timestamp."""
        r = Resource(path="/some/file.py")
        # Should be a valid ISO timestamp ending in Z
        assert r.discovered_at.endswith("Z")
        # Should parse as datetime
        datetime.fromisoformat(r.discovered_at.replace("Z", "+00:00"))

    def test_custom_discovered_at(self):
        """Custom discovered_at should not be overwritten."""
        r = Resource(path="/some/file.py", discovered_at="2026-01-01T00:00:00Z")
        assert r.discovered_at == "2026-01-01T00:00:00Z"


class TestPlanStep:
    """Test PlanStep dataclass."""

    def test_default_values(self):
        """PlanStep should have sensible defaults."""
        step = PlanStep(id="step-1", description="Do something")
        assert step.id == "step-1"
        assert step.description == "Do something"
        assert step.status == StepStatus.PENDING
        assert step.depends_on == []

    def test_to_dict(self):
        """Test serialization to dict."""
        step = PlanStep(
            id="step-1",
            description="Do something",
            status=StepStatus.IN_PROGRESS,
            intent="EDIT_FILE",
            target="/some/file.py",
        )
        d = step.to_dict()
        assert d["id"] == "step-1"
        assert d["description"] == "Do something"
        assert d["status"] == "in_progress"
        assert d["intent"] == "EDIT_FILE"
        assert d["target"] == "/some/file.py"

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "id": "step-2",
            "description": "Run tests",
            "status": "completed",
            "intent": "RUN_TESTS",
            "target": "pytest",
            "output": "All tests passed",
            "error": "",
            "started_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T00:01:00Z",
            "depends_on": ["step-1"],
        }
        step = PlanStep.from_dict(d)
        assert step.id == "step-2"
        assert step.status == StepStatus.COMPLETED
        assert step.output == "All tests passed"
        assert step.depends_on == ["step-1"]

    def test_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        original = PlanStep(
            id="step-1",
            description="Test",
            status=StepStatus.FAILED,
            error="Something broke",
            depends_on=["step-0"],
        )
        restored = PlanStep.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.description == original.description
        assert restored.status == original.status
        assert restored.error == original.error
        assert restored.depends_on == original.depends_on


class TestEditProposal:
    """Test EditProposal dataclass."""

    def test_default_values(self):
        """EditProposal should have sensible defaults."""
        edit = EditProposal(
            id="edit-1",
            task_id="task-001",
            file_path="/some/file.py",
            old_content="old",
            new_content="new",
        )
        assert edit.approved is None
        assert edit.created_at.endswith("Z")

    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        original = EditProposal(
            id="edit-1",
            task_id="task-001",
            file_path="/some/file.py",
            old_content="old code",
            new_content="new code",
            diff="--- a\n+++ b\n@@ -1 +1 @@\n-old code\n+new code",
            description="Fix bug",
        )
        restored = EditProposal.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.task_id == original.task_id
        assert restored.file_path == original.file_path
        assert restored.old_content == original.old_content
        assert restored.new_content == original.new_content
        assert restored.diff == original.diff
        assert restored.description == original.description


class TestTask:
    """Test Task dataclass."""

    def test_default_values(self):
        """Task should have sensible defaults."""
        task = Task(id="task-001", title="Test task")
        assert task.id == "task-001"
        assert task.title == "Test task"
        assert task.phase == TaskPhase.PLANNING
        assert task.priority == 0
        assert task.plan == []
        assert task.resources == []
        assert task.pending_edits == []
        assert task.created_at.endswith("Z")
        assert task.updated_at == task.created_at

    def test_touch(self):
        """touch() should update updated_at."""
        task = Task(id="task-001", title="Test")
        original_updated = task.updated_at
        import time
        time.sleep(0.01)  # Small delay to ensure different timestamp
        task.touch()
        assert task.updated_at != original_updated

    def test_is_active(self):
        """is_active() should return True for non-terminal phases."""
        task = Task(id="task-001", title="Test")

        task.phase = TaskPhase.PLANNING
        assert task.is_active() is True

        task.phase = TaskPhase.EXECUTING
        assert task.is_active() is True

        task.phase = TaskPhase.DONE
        assert task.is_active() is False

        task.phase = TaskPhase.BLOCKED
        assert task.is_active() is False

    def test_get_current_step_empty(self):
        """get_current_step() should return None if no plan."""
        task = Task(id="task-001", title="Test")
        assert task.get_current_step() is None

    def test_get_current_step(self):
        """get_current_step() should return first pending/in_progress step."""
        task = Task(id="task-001", title="Test")
        task.plan = [
            PlanStep(id="step-1", description="Done", status=StepStatus.COMPLETED),
            PlanStep(id="step-2", description="In progress", status=StepStatus.IN_PROGRESS),
            PlanStep(id="step-3", description="Pending", status=StepStatus.PENDING),
        ]
        current = task.get_current_step()
        assert current is not None
        assert current.id == "step-2"

    def test_get_step_by_id(self):
        """get_step_by_id() should find step by ID."""
        task = Task(id="task-001", title="Test")
        task.plan = [
            PlanStep(id="step-1", description="First"),
            PlanStep(id="step-2", description="Second"),
        ]
        step = task.get_step_by_id("step-2")
        assert step is not None
        assert step.description == "Second"

        assert task.get_step_by_id("step-99") is None

    def test_to_dict(self):
        """Test serialization to dict."""
        task = Task(
            id="task-001",
            title="Test task",
            description="A test",
            phase=TaskPhase.EXECUTING,
            priority=5,
            original_request="Do something",
            tags=["test", "important"],
        )
        task.plan = [PlanStep(id="step-1", description="Step one")]
        task.resources = [Resource(path="/some/file.py", description="Main file")]

        d = task.to_dict()
        assert d["id"] == "task-001"
        assert d["title"] == "Test task"
        assert d["phase"] == "executing"
        assert d["priority"] == 5
        assert d["tags"] == ["test", "important"]
        assert len(d["plan"]) == 1
        assert d["plan"][0]["id"] == "step-1"
        assert len(d["resources"]) == 1
        assert d["resources"][0]["path"] == "/some/file.py"

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "id": "task-002",
            "title": "Restored task",
            "description": "From JSON",
            "phase": "waiting_approval",
            "priority": 3,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T01:00:00Z",
            "completed_at": "",
            "original_request": "Fix bug",
            "conversation_ids": ["conv-1"],
            "tags": ["bug"],
            "plan": [
                {"id": "step-1", "description": "Fix", "status": "completed"}
            ],
            "resources": [
                {"path": "/file.py", "resource_type": "file", "description": "Bug file"}
            ],
            "pending_edits": [],
        }
        task = Task.from_dict(d)
        assert task.id == "task-002"
        assert task.title == "Restored task"
        assert task.phase == TaskPhase.WAITING_APPROVAL
        assert task.priority == 3
        assert len(task.plan) == 1
        assert task.plan[0].status == StepStatus.COMPLETED
        assert len(task.resources) == 1
        assert task.resources[0].path == "/file.py"

    def test_roundtrip(self):
        """Test full to_dict -> from_dict roundtrip with nested objects."""
        original = Task(
            id="task-003",
            title="Complex task",
            description="Has everything",
            phase=TaskPhase.EXECUTING,
            original_request="Build feature X",
            tags=["feature", "v2"],
        )
        original.plan = [
            PlanStep(id="s1", description="Plan", status=StepStatus.COMPLETED),
            PlanStep(id="s2", description="Build", status=StepStatus.IN_PROGRESS, depends_on=["s1"]),
            PlanStep(id="s3", description="Test", status=StepStatus.PENDING, depends_on=["s2"]),
        ]
        original.resources = [
            Resource(path="/src/main.py", description="Main source"),
            Resource(path="/tests/test_main.py", description="Tests"),
        ]
        original.pending_edits = [
            EditProposal(
                id="edit-1",
                task_id="task-003",
                file_path="/src/main.py",
                old_content="def foo(): pass",
                new_content="def foo(): return 42",
            )
        ]

        restored = Task.from_dict(original.to_dict())

        assert restored.id == original.id
        assert restored.title == original.title
        assert restored.phase == original.phase
        assert len(restored.plan) == 3
        assert restored.plan[1].depends_on == ["s1"]
        assert len(restored.resources) == 2
        assert len(restored.pending_edits) == 1
        assert restored.pending_edits[0].new_content == "def foo(): return 42"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
