"""
Data models for task tracking and workflow management.

This module defines the core data structures used by the TaskFSMController:
- Task: A unit of work with phases, plans, and resources
- Resource: A file or artifact bound to a task
- PlanStep: A step in a task's execution plan
- EditProposal: A pending code edit awaiting user approval
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskPhase(str, Enum):
    """
    Phases a task can be in.

    The FSM transitions between these phases based on:
    - LLM outputs (e.g., requesting clarification)
    - User actions (e.g., approving edits)
    - Controller decisions (e.g., moving to execution)
    """
    # Initial/routing phases
    NEEDS_CLARIFICATION = "needs_clarification"  # Waiting for user to clarify
    PLANNING = "planning"                        # Building execution plan

    # Execution phases
    EXECUTING = "executing"                      # Running plan steps
    WAITING_APPROVAL = "waiting_approval"        # Edit proposal pending

    # Terminal phases
    DONE = "done"                                # Task completed
    BLOCKED = "blocked"                          # Task blocked on external dependency


class StepStatus(str, Enum):
    """Status of a plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"      # Step blocked on external dependency
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Resource:
    """
    A file or artifact bound to a task.

    Resources are discovered during planning and tracked during execution.
    They help the controller understand what files are relevant to a task.
    """
    path: str                          # Absolute path to the resource
    resource_type: str = "file"        # "file", "directory", "url", "note"
    description: str = ""              # Brief description of relevance
    discovered_at: str = ""            # ISO timestamp when discovered
    last_modified: str = ""            # ISO timestamp of last modification

    # Content summary (optional, for context building)
    summary: str = ""                  # LLM-generated summary of content

    def __post_init__(self):
        if not self.discovered_at:
            self.discovered_at = datetime.utcnow().isoformat() + "Z"


@dataclass
class PlanStep:
    """
    A step in a task's execution plan.

    Plans are built during the PLANNING phase and executed during EXECUTING.
    Steps can have dependencies and may spawn sub-steps.
    """
    id: str                            # Unique step ID (e.g., "step-1", "step-1.1")
    description: str                   # What this step does
    status: StepStatus = StepStatus.PENDING

    # Execution details
    intent: str = ""                   # High-level intent (e.g., "EDIT_FILE", "RUN_TESTS")
    target: str = ""                   # Target resource (file path, command, etc.)

    # Results
    output: str = ""                   # Output/result of execution
    error: str = ""                    # Error message if failed

    # Timing
    started_at: str = ""               # ISO timestamp
    completed_at: str = ""             # ISO timestamp

    # Dependencies
    depends_on: list[str] = field(default_factory=list)  # IDs of prerequisite steps

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "intent": self.intent,
            "target": self.target,
            "output": self.output,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "depends_on": self.depends_on,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanStep":
        """Deserialize from dict."""
        return cls(
            id=data["id"],
            description=data["description"],
            status=StepStatus(data.get("status", "pending")),
            intent=data.get("intent", ""),
            target=data.get("target", ""),
            output=data.get("output", ""),
            error=data.get("error", ""),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
            depends_on=data.get("depends_on", []),
        )


@dataclass
class EditProposal:
    """
    A pending code edit awaiting user approval.

    When the controller intercepts a file edit, it creates an EditProposal
    and transitions the task to WAITING_APPROVAL. The user can then
    approve, reject, or modify the edit.
    """
    id: str                            # Unique proposal ID
    task_id: str                       # Task this edit belongs to
    file_path: str                     # File being edited

    # Edit content
    old_content: str                   # Original file content
    new_content: str                   # Proposed new content
    diff: str = ""                     # Unified diff (for display)

    # Metadata
    description: str = ""              # Why this edit is being made
    created_at: str = ""               # ISO timestamp

    # Status
    approved: bool | None = None       # None=pending, True=approved, False=rejected
    approved_at: str = ""              # ISO timestamp of approval/rejection

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "file_path": self.file_path,
            "old_content": self.old_content,
            "new_content": self.new_content,
            "diff": self.diff,
            "description": self.description,
            "created_at": self.created_at,
            "approved": self.approved,
            "approved_at": self.approved_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EditProposal":
        """Deserialize from dict."""
        return cls(
            id=data["id"],
            task_id=data["task_id"],
            file_path=data["file_path"],
            old_content=data["old_content"],
            new_content=data["new_content"],
            diff=data.get("diff", ""),
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            approved=data.get("approved"),
            approved_at=data.get("approved_at", ""),
        )


@dataclass
class Task:
    """
    A unit of work tracked by the controller.

    Tasks represent user requests that require multi-step execution.
    They have:
    - A lifecycle (phases)
    - An execution plan (steps)
    - Bound resources (files, artifacts)
    - History of interactions
    """
    id: str                            # Unique task ID (e.g., "task-20260203-001")
    title: str                         # Short title (from first user message or LLM)
    description: str = ""              # Longer description of the task

    # State
    phase: TaskPhase = TaskPhase.PLANNING
    priority: int = 0                  # Higher = more important

    # Timing
    created_at: str = ""               # ISO timestamp
    updated_at: str = ""               # ISO timestamp of last update
    completed_at: str = ""             # ISO timestamp if DONE

    # Plan and resources
    plan: list[PlanStep] = field(default_factory=list)
    resources: list[Resource] = field(default_factory=list)

    # Pending edits (if in WAITING_APPROVAL)
    pending_edits: list[EditProposal] = field(default_factory=list)

    # Current step tracking
    current_step_id: str | None = None  # ID of the step currently being executed

    # Context
    original_request: str = ""         # The original user message that created this task
    conversation_ids: list[str] = field(default_factory=list)  # Related conversation IDs

    # Chronological log of task activity
    notes: list[str] = field(default_factory=list)  # Log entries with timestamps

    # Tags for organization
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if not self.updated_at:
            self.updated_at = self.created_at

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow().isoformat() + "Z"

    def get_current_step(self) -> PlanStep | None:
        """Get the current step being executed.

        If current_step_id is set, returns that step.
        Otherwise returns the first pending or in-progress step.
        """
        # If explicit current step set, use that
        if self.current_step_id:
            for step in self.plan:
                if step.id == self.current_step_id:
                    return step

        # Fallback: first pending or in-progress step
        for step in self.plan:
            if step.status in (StepStatus.PENDING, StepStatus.IN_PROGRESS):
                return step
        return None

    def advance_to_next_step(self) -> PlanStep | None:
        """Advance to the next pending step. Returns the new current step or None if done."""
        for step in self.plan:
            if step.status == StepStatus.PENDING:
                self.current_step_id = step.id
                step.status = StepStatus.IN_PROGRESS
                step.started_at = datetime.utcnow().isoformat() + "Z"
                return step
        # No pending steps left
        self.current_step_id = None
        return None

    def complete_current_step(self) -> bool:
        """Mark the current step as completed and advance. Returns True if there are more steps."""
        current = self.get_current_step()
        if current:
            current.status = StepStatus.COMPLETED
            current.completed_at = datetime.utcnow().isoformat() + "Z"
            next_step = self.advance_to_next_step()
            return next_step is not None
        return False

    def add_note(self, message: str) -> None:
        """Add a timestamped note to the task log."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        self.notes.append(f"[{timestamp}] {message}")

    def get_step_by_id(self, step_id: str) -> PlanStep | None:
        """Find a step by ID."""
        for step in self.plan:
            if step.id == step_id:
                return step
        return None

    def is_active(self) -> bool:
        """Check if task is in an active (non-terminal) phase."""
        return self.phase not in (TaskPhase.DONE, TaskPhase.BLOCKED)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "phase": self.phase.value,
            "priority": self.priority,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "plan": [step.to_dict() for step in self.plan],
            "resources": [
                {
                    "path": r.path,
                    "resource_type": r.resource_type,
                    "description": r.description,
                    "discovered_at": r.discovered_at,
                    "last_modified": r.last_modified,
                    "summary": r.summary,
                }
                for r in self.resources
            ],
            "pending_edits": [edit.to_dict() for edit in self.pending_edits],
            "current_step_id": self.current_step_id,
            "original_request": self.original_request,
            "conversation_ids": self.conversation_ids,
            "notes": self.notes,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        """Deserialize from dict."""
        task = cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            phase=TaskPhase(data.get("phase", "planning")),
            priority=data.get("priority", 0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            completed_at=data.get("completed_at", ""),
            original_request=data.get("original_request", ""),
            conversation_ids=data.get("conversation_ids", []),
            notes=data.get("notes", []),
            tags=data.get("tags", []),
        )
        task.current_step_id = data.get("current_step_id")

        # Parse plan steps
        task.plan = [
            PlanStep.from_dict(step_data)
            for step_data in data.get("plan", [])
        ]

        # Parse resources
        task.resources = [
            Resource(
                path=r["path"],
                resource_type=r.get("resource_type", "file"),
                description=r.get("description", ""),
                discovered_at=r.get("discovered_at", ""),
                last_modified=r.get("last_modified", ""),
                summary=r.get("summary", ""),
            )
            for r in data.get("resources", [])
        ]

        # Parse pending edits
        task.pending_edits = [
            EditProposal.from_dict(edit_data)
            for edit_data in data.get("pending_edits", [])
        ]

        return task
