"""
Passthrough controller - preserves existing hello-world behavior.

This is the default controller that passes every message directly to
the LLM without any task tracking or routing logic. It exists to:

1. Preserve backward compatibility
2. Serve as a baseline for comparison
3. Be the safest option when controller logic isn't needed
"""

from typing import Any

from .base import BaseController, ControllerDecision, ControllerContext


class PassthroughController(BaseController):
    """
    Default controller that passes every message directly to the LLM.

    This preserves the existing hello-world behavior where:
    - Every message goes to the LLM
    - Tool calls are executed as the LLM requests
    - No task tracking or workflow management
    - No edit approval flows

    This is a stateless controller - there's nothing to persist.
    """

    async def on_message(
        self, message: Any, context: ControllerContext
    ) -> ControllerDecision:
        """
        Always process with LLM - no routing or task tracking.

        Args:
            message: The incoming Message object
            context: ControllerContext (unused in passthrough mode)

        Returns:
            ControllerDecision to process with LLM
        """
        return ControllerDecision(
            action="PROCESS_WITH_LLM",
            payload={"message": message},
        )

    async def on_llm_response(
        self,
        response: str,
        tool_calls: list[Any],
        context: ControllerContext,
    ) -> ControllerDecision:
        """
        Let existing AgentNode logic handle tool calls.

        Args:
            response: The LLM's text response
            tool_calls: List of tool calls from the LLM
            context: Current context (unused in passthrough mode)

        Returns:
            ControllerDecision to execute tools or signal completion
        """
        if tool_calls:
            return ControllerDecision(
                action="EXECUTE_TOOLS",
                payload={"tool_calls": tool_calls},
            )
        return ControllerDecision(
            action="DONE",
            payload={"response": response},
        )

    async def load_state(self) -> None:
        """No state to load in passthrough mode."""
        pass

    async def save_state(self) -> None:
        """No state to save in passthrough mode."""
        pass
