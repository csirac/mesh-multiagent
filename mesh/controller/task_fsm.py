"""
Task FSM Controller - v0 implementation.

This controller implements task-based routing and workflow management:
- Routes messages to existing tasks or creates new ones
- Tracks task phases (NEEDS_CLARIFICATION, PLANNING, EXECUTING, etc.)
- Manages edit proposals with user approval
- Persists state to ~/log/assistant/

Phase 2 complete: Data models and persistence integrated.
"""

import logging
from datetime import datetime
from typing import Any, TYPE_CHECKING

from .base import BaseController, ControllerDecision, ControllerContext
from .models import Task, TaskPhase, PlanStep, StepStatus, Resource, EditProposal
from .persistence import TaskPersistence
from .router import RouterLLM
from .phase_detector import PhaseDetector
from .edit_interceptor import EditInterceptor

if TYPE_CHECKING:
    from ..config import ControllerConfig

logger = logging.getLogger(__name__)


class TaskFSMController(BaseController):
    """
    Task-based controller with FSM-driven workflow management.

    Implementation status:
    - Phase 1: ✓ Basic structure and skeleton
    - Phase 2: ✓ Data models and persistence
    - Phase 3: ✓ Router LLM for intelligent message classification
    - Phase 4: ✓ Automatic phase transitions based on LLM output
    - Phase 5: Pending - Edit proposal system
    - Phase 6: Pending - User commands

    Current behavior (v0 with router):
    - Uses LLM to classify messages (CREATE_TASK / ROUTE_TO_TASK / DIRECT_ANSWER)
    - Automatically creates tasks when user requests help
    - Routes messages to active tasks when appropriate
    - Persists task state to ~/log/assistant/tasks.json
    """

    def __init__(self, config: "ControllerConfig | None" = None):
        """
        Initialize the task FSM controller.

        Args:
            config: Controller configuration from mesh/config.py.
                    If None, uses default paths.
        """
        # Import here to avoid circular imports
        from ..config import ControllerConfig as ConfigClass

        if config is None:
            config = ConfigClass()

        self.config = config

        # Initialize persistence layer
        self._persistence = TaskPersistence(
            tasks_path=config.tasks_path,
            config_path=config.config_path,
        )

        # Initialize router LLM
        self._router = RouterLLM(
            model=config.router_model,
            backend=config.router_backend,
        )

        # Initialize phase detector
        self._phase_detector = PhaseDetector()

        # Initialize edit interceptor
        self._edit_interceptor = EditInterceptor()

        # Task state (populated in load_state)
        self._tasks: list[Task] = []
        self._active_task_id: str | None = None

        # State loaded flag
        self._state_loaded = False

    async def on_message(
        self, message: Any, context: ControllerContext
    ) -> ControllerDecision:
        """
        Route an incoming message using the router LLM.

        Phase 3 behavior:
        - Classifies message as CREATE_TASK / ROUTE_TO_TASK / DIRECT_ANSWER
        - Creates new tasks when appropriate
        - Routes to active tasks when relevant
        - Falls back to direct LLM processing for simple questions

        Args:
            message: The incoming Message object
            context: ControllerContext with cwd, history, etc.

        Returns:
            ControllerDecision indicating next action
        """
        # Extract message content
        message_text = self._extract_message_text(message)
        if not message_text:
            # No text content, pass through
            return ControllerDecision(
                action="PROCESS_WITH_LLM",
                payload={"message": message},
                task_id=self._active_task_id,
            )

        # Build active task context for router
        active_task_dict = None
        if self._active_task_id:
            task = self.get_active_task()
            if task:
                active_task_dict = {
                    "id": task.id,
                    "title": task.title,
                    "phase": task.phase.value,
                }

        # Build recent messages context
        recent_messages = []
        if context.history:
            for msg in context.history[-5:]:  # Last 5 messages
                recent_messages.append({
                    "role": getattr(msg, "role", "user"),
                    "content": self._extract_message_text(msg),
                })

        # Call router LLM to classify
        try:
            classification = await self._router.classify_message(
                message=message_text,
                active_task=active_task_dict,
                recent_messages=recent_messages,
            )
            logger.info(
                f"Router classification: {classification['action']} "
                f"(confidence={classification['confidence']:.2f})"
            )
        except Exception as e:
            logger.error(f"Router LLM failed: {e}")
            # Fallback to direct answer on error
            classification = {
                "action": "DIRECT_ANSWER",
                "confidence": 0.0,
                "reasoning": f"Router error: {e}",
            }

        # Handle classification
        action = classification["action"]

        if action == "CREATE_TASK":
            # Create a new task
            task = self.create_task(
                title=classification["task_title"],
                description=classification.get("task_description", ""),
                original_request=message_text,
            )
            self.set_active_task(task.id)

            # Generate phase-specific context (uses PLANNING instructions)
            system_addendum = self._generate_system_addendum(task)

            return ControllerDecision(
                action="PROCESS_WITH_LLM",
                payload={"message": message},
                task_id=task.id,
                phase=task.phase.value,
                system_addendum=system_addendum,
            )

        elif action == "ROUTE_TO_TASK":
            # Route to active task
            task_id = classification.get("task_id") or self._active_task_id
            task = self._get_task_by_id(task_id) if task_id else None
            if task_id:
                logger.info(f"Routing message to task {task_id}")

            # Generate phase-specific context for the LLM
            system_addendum = self._generate_system_addendum(task) if task else None

            return ControllerDecision(
                action="PROCESS_WITH_LLM",
                payload={"message": message},
                task_id=task_id,
                phase=task.phase.value if task else None,
                system_addendum=system_addendum,
            )

        else:  # DIRECT_ANSWER
            # No task needed, just answer directly
            return ControllerDecision(
                action="PROCESS_WITH_LLM",
                payload={"message": message},
                task_id=None,  # Explicitly no task context
            )

    def _extract_message_text(self, message: Any) -> str:
        """Extract text content from a message object."""
        # Try common message formats
        if hasattr(message, "content"):
            content = message.content
            if isinstance(content, str):
                return content.strip()
            elif isinstance(content, dict) and "text" in content:
                return content["text"].strip()

        # Try dict access
        if isinstance(message, dict):
            if "content" in message:
                content = message["content"]
                if isinstance(content, str):
                    return content.strip()
                elif isinstance(content, dict) and "text" in content:
                    return content["text"].strip()
            if "text" in message:
                return message["text"].strip()

        # Last resort: str() conversion
        return str(message).strip()

    async def on_llm_response(
        self,
        response: str,
        tool_calls: list[Any],
        context: ControllerContext,
    ) -> ControllerDecision:
        """
        Handle LLM response.

        Phase 4 behavior:
        - Detect phase transitions from LLM output
        - Update task phase automatically
        - Extract plan steps from PLANNING phase responses

        Args:
            response: The LLM's text response
            tool_calls: List of tool calls from the LLM
            context: Current context

        Returns:
            ControllerDecision for next action
        """
        # Phase 4: Automatic phase transitions
        task = self.get_active_task()
        if task:
            # Phase 11: Check if LLM is requesting clarification
            # This allows the LLM to naturally ask for more info and trigger phase transition
            if self._phase_detector.detect_clarification_request(response):
                if task.phase not in (TaskPhase.NEEDS_CLARIFICATION, TaskPhase.DONE):
                    logger.info(f"Task {task.id}: LLM requested clarification, transitioning to NEEDS_CLARIFICATION")
                    task.phase = TaskPhase.NEEDS_CLARIFICATION
                    task.add_note("LLM requested clarification from user")
                    task.touch()

            # If in PLANNING phase, check if LLM output contains a plan
            # This should trigger plan extraction AND transition to EXECUTING
            if task.phase == TaskPhase.PLANNING and not task.plan:
                plan_steps = self._phase_detector.extract_plan_steps(response)
                if plan_steps:
                    logger.info(f"Detected plan in LLM output with {len(plan_steps)} steps")
                    for i, step_desc in enumerate(plan_steps, start=1):
                        step = PlanStep(
                            id=f"step-{i}",
                            description=step_desc,
                            status=StepStatus.PENDING,
                        )
                        task.plan.append(step)
                    # Initialize current step to first step
                    task.current_step_id = task.plan[0].id if task.plan else None
                    if task.plan:
                        task.plan[0].status = StepStatus.IN_PROGRESS
                    task.add_note(f"Plan created with {len(plan_steps)} steps")

                    # Transition to EXECUTING now that we have a plan
                    logger.info(f"Task {task.id}: PLANNING → EXECUTING (plan extracted)")
                    task.phase = TaskPhase.EXECUTING
                    task.add_note("Phase transition: planning → executing (plan captured)")
                    task.touch()

            # Detect new phase from LLM output (pattern-based)
            new_phase = self._phase_detector.detect_phase(
                response=response,
                tool_calls=tool_calls,
                current_phase=task.phase,
            )

            if new_phase and new_phase != task.phase:
                logger.info(f"Task {task.id}: {task.phase.value} → {new_phase.value}")
                task.phase = new_phase
                task.add_note(f"Phase transition: {task.phase.value} → {new_phase.value}")
                task.touch()

            # Phase 11: Detect step completion and auto-advance
            if task.phase == TaskPhase.EXECUTING and task.plan:
                if self._phase_detector.detect_step_completion(response, tool_calls):
                    current_step = task.get_current_step()
                    if current_step and current_step.status == StepStatus.IN_PROGRESS:
                        step_num = task.plan.index(current_step) + 1
                        logger.info(f"Auto-completing step {step_num}: {current_step.description}")
                        has_more = task.complete_current_step()
                        task.add_note(f"Completed step {step_num}: {current_step.description[:50]}")
                        task.touch()

                        if not has_more:
                            # All steps done - suggest reviewing
                            logger.info(f"All steps complete, task may be ready for review")
                            task.add_note("All plan steps completed")

        # Phase 5: Intercept file edits for approval (if enabled)
        if task and tool_calls and self.config.require_edit_approval:
            file_writes = self._edit_interceptor.detect_file_writes(tool_calls)
            if file_writes:
                logger.info(f"Detected {len(file_writes)} file write(s), creating edit proposals")

                # Create proposals for each file write
                for fw in file_writes:
                    # Read current file content if it exists (using aiofiles for async I/O)
                    import os
                    old_content = ""
                    if os.path.exists(fw["file_path"]):
                        try:
                            import aiofiles
                            async with aiofiles.open(fw["file_path"], "r") as f:
                                old_content = await f.read()
                        except ImportError:
                            # Fallback to sync I/O if aiofiles not installed
                            try:
                                with open(fw["file_path"], "r") as f:
                                    old_content = f.read()
                            except Exception as e:
                                logger.warning(f"Failed to read {fw['file_path']}: {e}")
                        except Exception as e:
                            logger.warning(f"Failed to read {fw['file_path']}: {e}")

                    # Create proposal
                    proposal = self._edit_interceptor.create_proposal(
                        task_id=task.id,
                        file_path=fw["file_path"],
                        tool_name=fw["tool_name"],
                        arguments=fw["arguments"],
                        old_content=old_content,
                    )

                    # Add to task
                    task.pending_edits.append(proposal)
                    task.touch()

                # Transition to WAITING_APPROVAL
                logger.info(f"Task {task.id}: {task.phase.value} → waiting_approval")
                task.phase = TaskPhase.WAITING_APPROVAL
                task.touch()

                # Return decision to block tool execution
                return ControllerDecision(
                    action="WAITING_APPROVAL",
                    payload={
                        "proposals": [p.to_dict() for p in task.pending_edits],
                        "message": f"{len(task.pending_edits)} edit(s) require approval. Use /approve to apply, /reject to cancel, or /diff to review.",
                    },
                    task_id=task.id,
                    phase=task.phase.value,
                )

        if tool_calls:
            return ControllerDecision(
                action="EXECUTE_TOOLS",
                payload={"tool_calls": tool_calls},
                task_id=self._active_task_id,
                phase=task.phase.value if task else None,
            )
        return ControllerDecision(
            action="DONE",
            payload={"response": response},
            task_id=self._active_task_id,
            phase=task.phase.value if task else None,
        )

    async def load_state(self) -> None:
        """
        Load persisted state from ~/log/assistant/.

        Loads tasks from tasks.json and active task ID from config.json.
        """
        self._tasks = self._persistence.load_tasks()
        self._active_task_id = self._persistence.get_active_task_id()
        self._state_loaded = True

        active_count = sum(1 for t in self._tasks if t.is_active())
        logger.info(
            f"Loaded {len(self._tasks)} tasks ({active_count} active), "
            f"active_task_id={self._active_task_id}"
        )

    async def save_state(self) -> None:
        """
        Persist state to ~/log/assistant/.

        Saves tasks to tasks.json and updates active task ID in config.json.
        """
        if not self._state_loaded:
            logger.warning("Skipping save - state not loaded yet")
            return

        self._persistence.save_tasks(self._tasks)
        self._persistence.set_active_task_id(self._active_task_id)
        logger.debug(f"Saved {len(self._tasks)} tasks")

    def get_active_task(self) -> Task | None:
        """Get the currently active task."""
        if not self._active_task_id:
            return None
        return self._get_task_by_id(self._active_task_id)

    def get_tasks(self) -> list[Task]:
        """Get all tasks."""
        return self._tasks

    def _get_task_by_id(self, task_id: str) -> Task | None:
        """Find a task by ID."""
        for task in self._tasks:
            if task.id == task_id:
                return task
        return None

    # --- Task Management Methods (for Phase 3+) ---

    def create_task(self, title: str, description: str = "", original_request: str = "") -> Task:
        """
        Create a new task.

        Args:
            title: Short title for the task
            description: Longer description
            original_request: The user message that triggered this task

        Returns:
            The newly created Task
        """
        # Pass in-memory task IDs to avoid collisions
        existing_ids = [t.id for t in self._tasks]
        task_id = self._persistence.generate_task_id(existing_ids)
        task = Task(
            id=task_id,
            title=title,
            description=description,
            original_request=original_request,
            phase=TaskPhase.PLANNING,
        )
        self._tasks.append(task)
        logger.info(f"Created task {task_id}: {title}")
        return task

    def set_active_task(self, task_id: str | None) -> bool:
        """
        Set the active task.

        Args:
            task_id: Task ID to activate, or None to clear

        Returns:
            True if task was found and activated
        """
        if task_id is None:
            self._active_task_id = None
            return True

        task = self._get_task_by_id(task_id)
        if task:
            self._active_task_id = task_id
            logger.info(f"Activated task {task_id}")
            return True

        logger.warning(f"Task {task_id} not found")
        return False

    def complete_task(self, task_id: str) -> bool:
        """
        Mark a task as done.

        Args:
            task_id: Task ID to complete

        Returns:
            True if task was found and completed
        """
        task = self._get_task_by_id(task_id)
        if not task:
            return False

        task.phase = TaskPhase.DONE
        task.touch()
        task.completed_at = datetime.utcnow().isoformat() + "Z"

        # Clear active task if this was it
        if self._active_task_id == task_id:
            self._active_task_id = None

        logger.info(f"Completed task {task_id}")
        return True

    # --- Command Handling ---

    async def handle_command(self, command: str, args: list[str]) -> str | None:
        """
        Handle controller-specific commands.

        Commands:
        - /tasks: List all non-DONE tasks
        - /tasks --all: List all tasks including completed
        - /task <id>: Switch to task
        - /task done [id]: Mark task as done
        - /task delete <id>: Delete task
        - /approve: Approve pending edit (Phase 5)
        - /reject: Reject pending edit (Phase 5)
        - /diff: Show pending edit diff (Phase 5)

        Args:
            command: Command name (without /)
            args: Command arguments

        Returns:
            Response string, or None if not a controller command
        """
        if command == "tasks":
            return self._handle_tasks_command(args)

        if command == "task":
            return self._handle_task_command(args)

        if command == "approve":
            return self._handle_approve_command(args)

        if command == "reject":
            return self._handle_reject_command(args)

        if command == "diff":
            return self._handle_diff_command(args)

        if command == "plan":
            return self._handle_plan_command(args)

        if command == "step":
            return self._handle_step_command(args)

        return None

    def _handle_tasks_command(self, args: list[str]) -> str:
        """Handle /tasks command."""
        show_all = "--all" in args or "-a" in args

        if show_all:
            tasks = self._tasks
        else:
            tasks = [t for t in self._tasks if t.is_active()]

        if not tasks:
            if show_all:
                return "No tasks."
            return "No active tasks. Use `/tasks --all` to see completed tasks."

        lines = []
        for task in tasks:
            active_marker = "→ " if task.id == self._active_task_id else "  "
            phase_marker = task.phase.value[:4].upper()
            lines.append(f"{active_marker}[{phase_marker}] {task.id}: {task.title}")

        header = f"{len(tasks)} task(s)" + (" (including completed)" if show_all else "")
        return header + "\n" + "\n".join(lines)

    def _handle_task_command(self, args: list[str]) -> str:
        """Handle /task subcommands."""
        if not args:
            # Show current active task
            task = self.get_active_task()
            if task:
                return self._format_task_detail(task)
            return "No active task. Use `/task <id>` to switch to a task."

        subcommand = args[0]

        # /task done [id] - mark task as done
        if subcommand == "done":
            task_id = args[1] if len(args) > 1 else self._active_task_id
            if not task_id:
                return "No task specified and no active task."
            if self.complete_task(task_id):
                return f"Task {task_id} marked as done."
            return f"Task {task_id} not found."

        # /task delete <id>
        if subcommand == "delete":
            if len(args) < 2:
                return "Usage: /task delete <id>"
            task_id = args[1]
            task = self._get_task_by_id(task_id)
            if not task:
                return f"Task {task_id} not found."
            self._tasks.remove(task)
            if self._active_task_id == task_id:
                self._active_task_id = None
            return f"Task {task_id} deleted."

        # /task reopen <id>
        if subcommand == "reopen":
            if len(args) < 2:
                return "Usage: /task reopen <id>"
            task_id = args[1]
            task = self._get_task_by_id(task_id)
            if not task:
                return f"Task {task_id} not found."
            if task.phase != TaskPhase.DONE:
                return f"Task {task_id} is not done (current phase: {task.phase.value})."
            task.phase = TaskPhase.PLANNING
            task.updated_at = datetime.utcnow().isoformat() + "Z"
            return f"Task {task_id} reopened. Current phase: {task.phase.value}"

        # /task <id> - switch to task
        task_id = subcommand
        if self.set_active_task(task_id):
            task = self.get_active_task()
            return f"Switched to task {task_id}.\n" + self._format_task_detail(task)
        return f"Task {task_id} not found."

    def _format_task_detail(self, task: Task) -> str:
        """Format detailed task information."""
        lines = [
            f"**{task.title}**",
            f"ID: {task.id}",
            f"Phase: {task.phase.value}",
            f"Created: {task.created_at}",
        ]

        if task.description:
            lines.append(f"Description: {task.description}")

        if task.plan:
            lines.append(f"Plan: {len(task.plan)} steps")
            for step in task.plan[:3]:  # Show first 3 steps
                status_icon = {"pending": "○", "in_progress": "◐", "completed": "●", "blocked": "⊗", "failed": "✗", "skipped": "⊘"}.get(step.status.value, "?")
                lines.append(f"  {status_icon} {step.description}")
            if len(task.plan) > 3:
                lines.append(f"  ... and {len(task.plan) - 3} more steps")

        if task.resources:
            lines.append(f"Resources: {len(task.resources)} files")

        if task.pending_edits:
            lines.append(f"Pending edits: {len(task.pending_edits)} (use /approve or /reject)")

        return "\n".join(lines)

    def _handle_plan_command(self, args: list[str]) -> str:
        """Handle /plan subcommands."""
        task = self.get_active_task()
        if not task:
            return "No active task. Use `/task <id>` to switch to a task."

        if not args:
            # Show plan
            if not task.plan:
                return "No plan steps yet."
            lines = [f"Plan for {task.id}:"]
            for i, step in enumerate(task.plan, 1):
                status_icon = {"pending": "○", "in_progress": "◐", "completed": "●", "blocked": "⊗", "failed": "✗", "skipped": "⊘"}.get(step.status.value, "?")
                lines.append(f"  {i}. {status_icon} {step.description}")
            return "\n".join(lines)

        subcommand = args[0]

        # /plan add <description>
        if subcommand == "add":
            if len(args) < 2:
                return "Usage: /plan add <description>"
            description = " ".join(args[1:])
            step_id = f"step-{len(task.plan) + 1}"
            new_step = PlanStep(id=step_id, description=description, status=StepStatus.PENDING)
            task.plan.append(new_step)
            task.updated_at = datetime.utcnow().isoformat() + "Z"
            return f"Added step {len(task.plan)}: {description}"

        # /plan edit <N> <new description>
        if subcommand == "edit":
            if len(args) < 3:
                return "Usage: /plan edit <N> <new description>"
            try:
                step_num = int(args[1])
                if step_num < 1 or step_num > len(task.plan):
                    return f"Step {step_num} out of range (1-{len(task.plan)})."
                new_description = " ".join(args[2:])
                task.plan[step_num - 1].description = new_description
                task.updated_at = datetime.utcnow().isoformat() + "Z"
                return f"Updated step {step_num}: {new_description}"
            except ValueError:
                return "Step number must be an integer."

        # /plan delete <N>
        if subcommand == "delete":
            if len(args) < 2:
                return "Usage: /plan delete <N>"
            try:
                step_num = int(args[1])
                if step_num < 1 or step_num > len(task.plan):
                    return f"Step {step_num} out of range (1-{len(task.plan)})."
                deleted = task.plan.pop(step_num - 1)
                task.updated_at = datetime.utcnow().isoformat() + "Z"
                return f"Deleted step {step_num}: {deleted.description}"
            except ValueError:
                return "Step number must be an integer."

        # /plan reorder <N> <M>
        if subcommand == "reorder":
            if len(args) < 3:
                return "Usage: /plan reorder <from> <to>"
            try:
                from_num = int(args[1])
                to_num = int(args[2])
                if from_num < 1 or from_num > len(task.plan):
                    return f"Step {from_num} out of range (1-{len(task.plan)})."
                if to_num < 1 or to_num > len(task.plan):
                    return f"Position {to_num} out of range (1-{len(task.plan)})."
                step = task.plan.pop(from_num - 1)
                task.plan.insert(to_num - 1, step)
                task.updated_at = datetime.utcnow().isoformat() + "Z"
                return f"Moved step {from_num} to position {to_num}."
            except ValueError:
                return "Step numbers must be integers."

        return f"Unknown /plan subcommand: {subcommand}"

    def _handle_step_command(self, args: list[str]) -> str:
        """Handle /step subcommands."""
        task = self.get_active_task()
        if not task:
            return "No active task. Use `/task <id>` to switch to a task."

        if not task.plan:
            return "No plan steps yet."

        if not args:
            return "Usage: /step done|block|skip <N>"

        subcommand = args[0]

        # /step done <N>
        if subcommand == "done":
            if len(args) < 2:
                return "Usage: /step done <N>"
            try:
                step_num = int(args[1])
                if step_num < 1 or step_num > len(task.plan):
                    return f"Step {step_num} out of range (1-{len(task.plan)})."
                task.plan[step_num - 1].status = StepStatus.COMPLETED
                task.updated_at = datetime.utcnow().isoformat() + "Z"
                return f"Marked step {step_num} as completed."
            except ValueError:
                return "Step number must be an integer."

        # /step block <N> [reason]
        if subcommand == "block":
            if len(args) < 2:
                return "Usage: /step block <N> [reason]"
            try:
                step_num = int(args[1])
                if step_num < 1 or step_num > len(task.plan):
                    return f"Step {step_num} out of range (1-{len(task.plan)})."
                task.plan[step_num - 1].status = StepStatus.BLOCKED
                task.updated_at = datetime.utcnow().isoformat() + "Z"
                reason = " ".join(args[2:]) if len(args) > 2 else "No reason provided"
                return f"Marked step {step_num} as blocked: {reason}"
            except ValueError:
                return "Step number must be an integer."

        # /step skip <N>
        if subcommand == "skip":
            if len(args) < 2:
                return "Usage: /step skip <N>"
            try:
                step_num = int(args[1])
                if step_num < 1 or step_num > len(task.plan):
                    return f"Step {step_num} out of range (1-{len(task.plan)})."
                task.plan[step_num - 1].status = StepStatus.SKIPPED
                task.updated_at = datetime.utcnow().isoformat() + "Z"
                return f"Marked step {step_num} as skipped."
            except ValueError:
                return "Step number must be an integer."

        return f"Unknown /step subcommand: {subcommand}"

    # --- Phase 5: Edit Approval Handlers ---

    def _handle_approve_command(self, args: list[str]) -> str:
        """Handle /approve command to apply pending edits."""
        task = self.get_active_task()
        if not task:
            return "No active task."

        if not task.pending_edits:
            return "No pending edits to approve."

        if task.phase != TaskPhase.WAITING_APPROVAL:
            return f"Task is not in WAITING_APPROVAL phase (current: {task.phase.value})"

        # Apply all pending edits
        results = []
        for proposal in task.pending_edits:
            try:
                # Write the new content to file
                with open(proposal.file_path, "w") as f:
                    f.write(proposal.new_content)

                # Mark as approved
                proposal.approved = True
                proposal.approved_at = datetime.utcnow().isoformat() + "Z"

                results.append(f"✓ Applied edit to {proposal.file_path}")
                logger.info(f"Applied edit {proposal.id} to {proposal.file_path}")

            except Exception as e:
                results.append(f"✗ Failed to apply edit to {proposal.file_path}: {e}")
                logger.error(f"Failed to apply edit {proposal.id}: {e}")
                proposal.approved = False
                proposal.approved_at = datetime.utcnow().isoformat() + "Z"

        # Clear pending edits and transition back to EXECUTING
        task.pending_edits.clear()
        task.phase = TaskPhase.EXECUTING
        task.touch()

        summary = "\n".join(results)
        return f"Applied {len(results)} edit(s):\n{summary}\n\nTask resumed in EXECUTING phase."

    def _handle_reject_command(self, args: list[str]) -> str:
        """Handle /reject command to cancel pending edits."""
        task = self.get_active_task()
        if not task:
            return "No active task."

        if not task.pending_edits:
            return "No pending edits to reject."

        if task.phase != TaskPhase.WAITING_APPROVAL:
            return f"Task is not in WAITING_APPROVAL phase (current: {task.phase.value})"

        # Mark all as rejected
        count = len(task.pending_edits)
        for proposal in task.pending_edits:
            proposal.approved = False
            proposal.approved_at = datetime.utcnow().isoformat() + "Z"

        # Clear pending edits and transition back to EXECUTING
        task.pending_edits.clear()
        task.phase = TaskPhase.EXECUTING
        task.touch()

        return f"Rejected {count} pending edit(s). Task resumed in EXECUTING phase."

    def _handle_diff_command(self, args: list[str]) -> str:
        """Handle /diff command to show pending edit diffs."""
        task = self.get_active_task()
        if not task:
            return "No active task."

        if not task.pending_edits:
            return "No pending edits to show."

        # Format all proposals
        sections = []
        for i, proposal in enumerate(task.pending_edits, start=1):
            formatted = self._edit_interceptor.format_proposal_for_display(proposal)
            sections.append(f"## Edit {i}/{len(task.pending_edits)}\n{formatted}")

        return "\n\n".join(sections)

    # --- Phase 11: System Addendum Generation ---

    def _generate_system_addendum(self, task: Task) -> str:
        """
        Generate phase-specific context to inject into the LLM system prompt.

        This provides the LLM with structured awareness of:
        - Current task and phase
        - Plan steps and progress
        - What actions are appropriate for this phase

        Args:
            task: The active task

        Returns:
            System addendum string to prepend to messages
        """
        lines = [
            f"Task: {task.title} (ID: {task.id})",
            f"Phase: {task.phase.value.upper()}",
        ]

        # Phase-specific guidance
        if task.phase == TaskPhase.NEEDS_CLARIFICATION:
            lines.append("")
            lines.append("The user's request needs clarification before proceeding.")
            lines.append("Ask 1-2 specific questions to understand their intent.")
            lines.append("Once clarified, you can proceed with planning.")

        elif task.phase == TaskPhase.PLANNING:
            lines.append("")
            lines.append("Create a concrete execution plan with 3-7 numbered steps.")
            lines.append("Each step should be completable in a single turn.")
            lines.append("Focus on what needs to be done, not how long it takes.")

        elif task.phase == TaskPhase.EXECUTING:
            # Show current step context
            current_step = task.get_current_step()
            if current_step:
                step_num = task.plan.index(current_step) + 1 if current_step in task.plan else "?"
                lines.append("")
                lines.append(f"Current step ({step_num}/{len(task.plan)}): {current_step.description}")
            elif task.plan:
                # Show plan summary
                completed = sum(1 for s in task.plan if s.status == StepStatus.COMPLETED)
                lines.append("")
                lines.append(f"Plan progress: {completed}/{len(task.plan)} steps completed")
                remaining = [s for s in task.plan if s.status == StepStatus.PENDING]
                if remaining:
                    lines.append(f"Next: {remaining[0].description}")
            else:
                lines.append("")
                lines.append("Execute the user's request.")

            # Instruct LLM to signal step completion
            lines.append("")
            lines.append("When you complete the current step, emit [STEP_DONE] on its own line.")

        elif task.phase == TaskPhase.WAITING_APPROVAL:
            lines.append("")
            lines.append("Waiting for user to approve pending file edits.")
            lines.append("Do not propose new edits until current ones are resolved.")
            lines.append("User can: /approve, /reject, or /diff to review.")

        elif task.phase == TaskPhase.BLOCKED:
            lines.append("")
            lines.append("Task is blocked on an external dependency.")
            lines.append("Help the user understand what's needed to unblock.")

        elif task.phase == TaskPhase.DONE:
            lines.append("")
            lines.append("Task is marked as complete.")
            lines.append("If user has follow-up requests, a new task may be created.")

        # Add recent notes if any
        if task.notes and len(task.notes) <= 3:
            lines.append("")
            lines.append("Recent activity:")
            for note in task.notes[-3:]:
                lines.append(f"  - {note}")

        return "\n".join(lines)
