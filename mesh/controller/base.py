"""
Base controller interface for message routing and task management.

Controllers intercept messages between users and the LLM, allowing for:
- Task routing and tracking
- Workflow management (phases, plans)
- Edit proposal and approval flows
- Resource binding and discovery
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ControllerDecision:
    """
    Represents what the controller wants the agent to do next.

    This is the output of a controller's decision-making process.
    The AgentNode interprets this and executes the appropriate action.
    """

    # Core action to take
    action: str
    # Supported actions:
    # - "PROCESS_WITH_LLM": Send message to LLM for processing
    # - "EXECUTE_TOOLS": Execute the provided tool calls
    # - "ASK_USER": Ask the user a clarifying question
    # - "PROPOSE_EDIT": Show an edit proposal for approval
    # - "DISCOVER_RESOURCES": Scan for project files
    # - "READ_AND_SUMMARIZE": Read resources and build summary
    # - "EXECUTE_STEP": Execute a plan step
    # - "TRANSITION": Move to a new phase
    # - "WAIT": Wait for user input
    # - "DONE": Processing complete

    # Action-specific payload
    payload: dict[str, Any] = field(default_factory=dict)

    # Task context (if applicable)
    task_id: str | None = None
    phase: str | None = None

    # Optional system prompt addendum for LLM calls
    system_addendum: str | None = None


@dataclass
class ControllerContext:
    """
    Context passed to the controller for decision-making.

    Contains information about the current state that controllers
    can use to make routing and processing decisions.
    """

    # Current working directory
    cwd: str = ""

    # Recent message history (last N messages)
    history: list[Any] = field(default_factory=list)

    # Agent's node ID
    agent_id: str = ""

    # Repository snapshot (file listing)
    repo_snapshot: dict[str, Any] | None = None

    # The incoming message being processed
    message: Any = None

    # Additional context from the agent
    extra: dict[str, Any] = field(default_factory=dict)


class BaseController(ABC):
    """
    Abstract base class for all controller implementations.

    Controllers manage the flow of messages and tasks. They:
    1. Receive incoming messages and decide how to route them
    2. Track tasks and their phases
    3. Manage edit proposals and approvals
    4. Persist state across sessions

    Subclasses must implement:
    - on_message(): Handle incoming user messages
    - on_llm_response(): Handle LLM responses and tool calls
    - load_state(): Load persisted state
    - save_state(): Persist state

    Lifecycle:
    1. Controller is instantiated with config
    2. load_state() is called to restore persisted state
    3. For each message:
       a. on_message() is called to decide routing
       b. If LLM is invoked, on_llm_response() is called with results
       c. save_state() is called after mutations
    4. save_state() is called on shutdown
    """

    @abstractmethod
    async def on_message(
        self, message: Any, context: ControllerContext
    ) -> ControllerDecision:
        """
        Process an incoming user message and decide what to do.

        This is the main entry point for routing. The controller examines
        the message, considers existing tasks, and returns a decision about
        how to handle it.

        Args:
            message: The incoming Message object
            context: ControllerContext with cwd, history, agent_id, etc.

        Returns:
            ControllerDecision indicating next action
        """
        pass

    @abstractmethod
    async def on_llm_response(
        self,
        response: str,
        tool_calls: list[Any],
        context: ControllerContext,
    ) -> ControllerDecision:
        """
        Process an LLM response and decide next action.

        Called after the LLM produces a response. The controller can:
        - Let tool calls proceed normally
        - Intercept and modify the flow
        - Update task state based on the response

        Args:
            response: The LLM's text response
            tool_calls: List of tool calls from the LLM
            context: Current context including task state

        Returns:
            ControllerDecision for next action
        """
        pass

    @abstractmethod
    async def load_state(self) -> None:
        """
        Load persisted state on startup.

        Called once when the controller is initialized. Should load
        any saved tasks, preferences, and other state from disk.
        """
        pass

    @abstractmethod
    async def save_state(self) -> None:
        """
        Persist state after mutations.

        Called after any state change that should be preserved.
        Should save tasks, preferences, and other state to disk.
        """
        pass

    def get_active_task(self) -> Any | None:
        """
        Get the currently active task, if any.

        Subclasses should override this to return their active task.

        Returns:
            The active task object, or None if no task is active
        """
        return None

    def get_tasks(self) -> list[Any]:
        """
        Get all tasks managed by this controller.

        Subclasses should override this to return their task list.

        Returns:
            List of task objects
        """
        return []

    async def handle_command(self, command: str, args: list[str]) -> str | None:
        """
        Handle a user command (e.g., /tasks, /approve).

        Subclasses should override this to handle controller-specific commands.

        Args:
            command: The command name (without leading /)
            args: Command arguments

        Returns:
            Response string to show to the user, or None if command not handled
        """
        return None
