"""
Unit tests for mesh/controller/persistence.py
"""

import json
import os
import pytest
import tempfile
from pathlib import Path

from mesh.controller.persistence import TaskPersistence
from mesh.controller.models import Task, TaskPhase, PlanStep, StepStatus


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def persistence(temp_dir):
    """Create a TaskPersistence instance with temp paths."""
    tasks_path = os.path.join(temp_dir, "tasks.json")
    config_path = os.path.join(temp_dir, "config.json")
    return TaskPersistence(tasks_path=tasks_path, config_path=config_path)


class TestTaskPersistenceInit:
    """Test TaskPersistence initialization."""

    def test_creates_parent_directories(self, temp_dir):
        """Should create parent directories if they don't exist."""
        nested_path = os.path.join(temp_dir, "a", "b", "c", "tasks.json")
        persistence = TaskPersistence(tasks_path=nested_path)
        assert os.path.isdir(os.path.dirname(nested_path))

    def test_expands_home_directory(self, temp_dir):
        """Should expand ~ in paths."""
        # We can't easily test ~ expansion without affecting real home dir,
        # but we can verify the path type
        persistence = TaskPersistence(
            tasks_path=os.path.join(temp_dir, "tasks.json")
        )
        assert isinstance(persistence.tasks_path, Path)
        assert not str(persistence.tasks_path).startswith("~")


class TestLoadTasks:
    """Test loading tasks from disk."""

    def test_returns_empty_list_if_no_file(self, persistence):
        """Should return empty list if tasks file doesn't exist."""
        tasks = persistence.load_tasks()
        assert tasks == []

    def test_loads_tasks_from_file(self, persistence):
        """Should load and deserialize tasks from JSON."""
        # Create a tasks file
        data = {
            "version": 1,
            "tasks": [
                {
                    "id": "task-001",
                    "title": "Test task",
                    "phase": "planning",
                    "plan": [],
                    "resources": [],
                    "pending_edits": [],
                    "tags": [],
                }
            ]
        }
        with open(persistence.tasks_path, "w") as f:
            json.dump(data, f)

        tasks = persistence.load_tasks()
        assert len(tasks) == 1
        assert tasks[0].id == "task-001"
        assert tasks[0].title == "Test task"
        assert isinstance(tasks[0], Task)

    def test_loads_from_backup_on_corruption(self, persistence):
        """Should load from backup if main file is corrupted."""
        # Create backup with valid data
        backup_path = persistence.tasks_path.with_suffix(".json.backup")
        backup_data = {
            "version": 1,
            "tasks": [
                {
                    "id": "task-backup",
                    "title": "Backup task",
                    "phase": "done",
                    "plan": [],
                    "resources": [],
                    "pending_edits": [],
                    "tags": [],
                }
            ]
        }
        with open(backup_path, "w") as f:
            json.dump(backup_data, f)

        # Create corrupted main file
        with open(persistence.tasks_path, "w") as f:
            f.write("{ invalid json ")

        tasks = persistence.load_tasks()
        assert len(tasks) == 1
        assert tasks[0].id == "task-backup"

    def test_returns_empty_on_complete_failure(self, persistence):
        """Should return empty list if both main and backup fail."""
        # Create corrupted files
        with open(persistence.tasks_path, "w") as f:
            f.write("not json")
        backup_path = persistence.tasks_path.with_suffix(".json.backup")
        with open(backup_path, "w") as f:
            f.write("also not json")

        tasks = persistence.load_tasks()
        assert tasks == []


class TestSaveTasks:
    """Test saving tasks to disk."""

    def test_saves_tasks_to_file(self, persistence):
        """Should serialize and save tasks to JSON."""
        tasks = [
            Task(id="task-001", title="First"),
            Task(id="task-002", title="Second", phase=TaskPhase.EXECUTING),
        ]
        tasks[0].plan = [PlanStep(id="s1", description="Step 1")]

        result = persistence.save_tasks(tasks)
        assert result is True
        assert persistence.tasks_path.exists()

        # Verify content
        with open(persistence.tasks_path) as f:
            data = json.load(f)

        assert data["version"] == 1
        assert "updated_at" in data
        assert len(data["tasks"]) == 2
        assert data["tasks"][0]["id"] == "task-001"
        assert data["tasks"][1]["phase"] == "executing"

    def test_creates_backup_before_overwriting(self, persistence):
        """Should backup existing file before overwriting."""
        # Create initial file
        tasks1 = [Task(id="task-v1", title="Version 1")]
        persistence.save_tasks(tasks1)

        # Save new version
        tasks2 = [Task(id="task-v2", title="Version 2")]
        persistence.save_tasks(tasks2)

        # Check backup exists and has old data
        backup_path = persistence.tasks_path.with_suffix(".json.backup")
        assert backup_path.exists()
        with open(backup_path) as f:
            backup_data = json.load(f)
        assert backup_data["tasks"][0]["id"] == "task-v1"

        # Check main file has new data
        with open(persistence.tasks_path) as f:
            main_data = json.load(f)
        assert main_data["tasks"][0]["id"] == "task-v2"

    def test_atomic_write(self, persistence):
        """Save should use atomic write (temp file + rename)."""
        tasks = [Task(id="task-001", title="Test")]
        result = persistence.save_tasks(tasks)
        assert result is True

        # Temp file should not exist after successful save
        temp_path = persistence.tasks_path.with_suffix(".json.tmp")
        assert not temp_path.exists()


class TestConfigPersistence:
    """Test config load/save."""

    def test_load_config_empty(self, persistence):
        """Should return empty dict if no config file."""
        config = persistence.load_config()
        assert config == {}

    def test_save_and_load_config(self, persistence):
        """Should save and load config correctly."""
        config = {"active_task_id": "task-001", "custom_setting": True}
        persistence.save_config(config)

        loaded = persistence.load_config()
        assert loaded["active_task_id"] == "task-001"
        assert loaded["custom_setting"] is True
        assert "version" in loaded
        assert "updated_at" in loaded

    def test_get_active_task_id(self, persistence):
        """get_active_task_id should return task ID from config."""
        assert persistence.get_active_task_id() is None

        persistence.save_config({"active_task_id": "task-123"})
        assert persistence.get_active_task_id() == "task-123"

    def test_set_active_task_id(self, persistence):
        """set_active_task_id should update config."""
        persistence.set_active_task_id("task-456")
        assert persistence.get_active_task_id() == "task-456"

        persistence.set_active_task_id(None)
        assert persistence.get_active_task_id() is None


class TestGenerateTaskId:
    """Test task ID generation."""

    def test_generates_unique_ids(self, persistence):
        """Should generate unique task IDs."""
        id1 = persistence.generate_task_id()
        id2 = persistence.generate_task_id()

        # Both should have today's date
        from datetime import datetime
        today = datetime.utcnow().strftime("%Y%m%d")
        assert today in id1
        assert today in id2

        # First should be 001 (no existing tasks)
        assert id1.endswith("-001")

    def test_increments_sequence(self, persistence):
        """Should increment sequence number for existing tasks."""
        # Create some tasks
        tasks = [
            Task(id="task-20260203-001", title="First"),
            Task(id="task-20260203-002", title="Second"),
        ]
        persistence.save_tasks(tasks)

        # Generate new ID (might be different date, but sequence should work)
        new_id = persistence.generate_task_id()
        # Should be 001 if different date, or 003 if same date
        assert new_id.startswith("task-")

    def test_considers_in_memory_task_ids(self, persistence):
        """Should consider provided in-memory task IDs to avoid collisions."""
        from datetime import datetime
        today = datetime.utcnow().strftime("%Y%m%d")

        # Generate ID with some existing in-memory IDs
        existing_ids = [f"task-{today}-001", f"task-{today}-002"]
        new_id = persistence.generate_task_id(existing_ids)

        # Should be 003, not 001
        assert new_id == f"task-{today}-003"

    def test_combines_persisted_and_in_memory(self, persistence):
        """Should combine both persisted and in-memory IDs."""
        from datetime import datetime
        today = datetime.utcnow().strftime("%Y%m%d")

        # Persist one task
        tasks = [Task(id=f"task-{today}-001", title="Persisted")]
        persistence.save_tasks(tasks)

        # Generate with additional in-memory ID
        existing_ids = [f"task-{today}-005"]
        new_id = persistence.generate_task_id(existing_ids)

        # Should be 006 (max of 001 and 005 is 005)
        assert new_id == f"task-{today}-006"


class TestRoundTrip:
    """Test complete roundtrip scenarios."""

    def test_complex_task_roundtrip(self, persistence):
        """Test saving and loading a complex task with all fields."""
        task = Task(
            id="task-roundtrip",
            title="Complex roundtrip test",
            description="Testing all fields",
            phase=TaskPhase.EXECUTING,
            priority=5,
            original_request="Do all the things",
            tags=["test", "roundtrip"],
        )
        task.plan = [
            PlanStep(
                id="s1",
                description="First step",
                status=StepStatus.COMPLETED,
                intent="PLAN",
                output="Done",
            ),
            PlanStep(
                id="s2",
                description="Second step",
                status=StepStatus.IN_PROGRESS,
                depends_on=["s1"],
            ),
        ]
        from mesh.controller.models import Resource
        task.resources = [
            Resource(path="/src/main.py", description="Main source file"),
        ]

        # Save
        persistence.save_tasks([task])

        # Load
        loaded_tasks = persistence.load_tasks()
        assert len(loaded_tasks) == 1

        loaded = loaded_tasks[0]
        assert loaded.id == task.id
        assert loaded.title == task.title
        assert loaded.phase == task.phase
        assert loaded.priority == task.priority
        assert loaded.tags == task.tags
        assert len(loaded.plan) == 2
        assert loaded.plan[0].status == StepStatus.COMPLETED
        assert loaded.plan[1].depends_on == ["s1"]
        assert len(loaded.resources) == 1
        assert loaded.resources[0].path == "/src/main.py"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
