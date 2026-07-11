"""
Persistence layer for task state.

Manages saving and loading of tasks, resources, and controller configuration
to ~/log/assistant/ directory. Uses atomic writes to prevent corruption.
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import Task

logger = logging.getLogger(__name__)


class TaskPersistence:
    """
    Handles persistence of task state to disk.

    Storage structure:
        ~/log/assistant/
        ├── tasks.json           # All tasks (active and completed)
        ├── tasks.json.backup    # Backup before each write
        └── config.json          # Controller configuration
    """

    def __init__(
        self,
        tasks_path: str = "~/log/assistant/tasks.json",
        config_path: str = "~/log/assistant/config.json",
    ):
        """
        Initialize persistence layer.

        Args:
            tasks_path: Path to tasks JSON file
            config_path: Path to controller config JSON file
        """
        from ..paths import resolve_path
        self.tasks_path = Path(resolve_path(tasks_path))
        self.config_path = Path(resolve_path(config_path))

        # Ensure parent directories exist
        self.tasks_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

    def load_tasks(self) -> list[Task]:
        """
        Load all tasks from disk.

        Returns:
            List of Task objects, or empty list if file doesn't exist
        """
        if not self.tasks_path.exists():
            logger.info(f"No tasks file at {self.tasks_path}, starting fresh")
            return []

        try:
            with open(self.tasks_path, "r") as f:
                data = json.load(f)

            tasks = [Task.from_dict(task_data) for task_data in data.get("tasks", [])]
            logger.info(f"Loaded {len(tasks)} tasks from {self.tasks_path}")
            return tasks

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tasks file: {e}")
            # Try backup
            backup_path = self.tasks_path.with_suffix(".json.backup")
            if backup_path.exists():
                logger.info("Attempting to load from backup")
                try:
                    with open(backup_path, "r") as f:
                        data = json.load(f)
                    tasks = [Task.from_dict(task_data) for task_data in data.get("tasks", [])]
                    logger.info(f"Loaded {len(tasks)} tasks from backup")
                    return tasks
                except Exception as e2:
                    logger.error(f"Backup also failed: {e2}")
            return []

        except Exception as e:
            logger.error(f"Failed to load tasks: {e}")
            return []

    def save_tasks(self, tasks: list[Task]) -> bool:
        """
        Save all tasks to disk atomically.

        Creates a backup before overwriting, writes to a temp file,
        then atomically renames to the target path.

        Args:
            tasks: List of Task objects to save

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create backup of existing file
            if self.tasks_path.exists():
                backup_path = self.tasks_path.with_suffix(".json.backup")
                shutil.copy2(self.tasks_path, backup_path)

            # Prepare data
            data = {
                "version": 1,
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "tasks": [task.to_dict() for task in tasks],
            }

            # Write to temp file
            temp_path = self.tasks_path.with_suffix(".json.tmp")
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)

            # Atomic rename
            temp_path.rename(self.tasks_path)

            logger.info(f"Saved {len(tasks)} tasks to {self.tasks_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save tasks: {e}")
            return False

    def load_config(self) -> dict[str, Any]:
        """
        Load controller configuration from disk.

        Returns:
            Config dict, or empty dict if file doesn't exist
        """
        if not self.config_path.exists():
            return {}

        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def save_config(self, config: dict[str, Any]) -> bool:
        """
        Save controller configuration to disk.

        Args:
            config: Configuration dict

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create backup
            if self.config_path.exists():
                backup_path = self.config_path.with_suffix(".json.backup")
                shutil.copy2(self.config_path, backup_path)

            data = {
                "version": 1,
                "updated_at": datetime.utcnow().isoformat() + "Z",
                **config,
            }

            temp_path = self.config_path.with_suffix(".json.tmp")
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)

            temp_path.rename(self.config_path)

            logger.info(f"Saved config to {self.config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False

    def get_active_task_id(self) -> str | None:
        """
        Get the ID of the currently active task from config.

        Returns:
            Task ID string, or None if no active task
        """
        config = self.load_config()
        return config.get("active_task_id")

    def set_active_task_id(self, task_id: str | None) -> bool:
        """
        Set the currently active task in config.

        Args:
            task_id: Task ID to set as active, or None to clear

        Returns:
            True if successful
        """
        config = self.load_config()
        if task_id:
            config["active_task_id"] = task_id
        else:
            config.pop("active_task_id", None)
        return self.save_config(config)

    def generate_task_id(self, existing_task_ids: list[str] | None = None) -> str:
        """
        Generate a unique task ID.

        Format: task-YYYYMMDD-NNN where NNN is a sequence number.

        Args:
            existing_task_ids: Optional list of task IDs to consider
                               (in addition to persisted tasks). Use this
                               to pass in-memory tasks not yet saved.

        Returns:
            Unique task ID string
        """
        today = datetime.utcnow().strftime("%Y%m%d")
        prefix = f"task-{today}-"

        # Combine persisted tasks and any provided in-memory task IDs
        tasks = self.load_tasks()
        all_ids = [t.id for t in tasks if t.id.startswith(prefix)]
        if existing_task_ids:
            all_ids.extend(tid for tid in existing_task_ids if tid.startswith(prefix))

        if not all_ids:
            return f"{prefix}001"

        # Find highest sequence number
        max_seq = 0
        for task_id in all_ids:
            try:
                seq = int(task_id.replace(prefix, ""))
                max_seq = max(max_seq, seq)
            except ValueError:
                continue

        return f"{prefix}{max_seq + 1:03d}"
