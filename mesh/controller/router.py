"""
Router LLM - intelligently classifies messages and routes to tasks.

Phase 3 implementation:
- Classifies incoming messages (CREATE_TASK / ROUTE_TO_TASK / DIRECT_ANSWER)
- Extracts task metadata (title, description)
- Uses a small, fast LLM (configured via controller config)
"""

import json
import logging
from typing import Any, Literal

from ..llm import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


class RouterLLM:
    """
    LLM-based message classifier for task routing.

    Uses a fast, cheap model (gpt-4o-mini, gpt-4-turbo, etc.) to decide:
    - CREATE_TASK: User wants help with a new task
    - ROUTE_TO_TASK: Message relates to an existing active task
    - DIRECT_ANSWER: Simple question that doesn't need task tracking
    """

    def __init__(self, model: str = "gpt-4o-mini", backend: str = "openai"):
        """
        Initialize the router LLM.

        Args:
            model: LLM model to use (default: gpt-4o-mini)
            backend: Backend type (default: openai)
        """
        self.model = model
        self.backend = backend

        # Lazy-initialized LLM client
        self._client: LLMClient | None = None

    def _get_client(self) -> LLMClient:
        """Get or create the LLM client."""
        if self._client is None:
            config = LLMConfig.from_env(backend=self.backend)
            config.model = self.model
            self._client = LLMClient(config)  # node_id not needed — only calls complete(), no subprocesses
        return self._client

    async def classify_message(
        self,
        message: str,
        active_task: dict[str, Any] | None = None,
        recent_messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Classify an incoming message and decide routing.

        Args:
            message: The user's message text
            active_task: Currently active task (if any) with id, title, phase
            recent_messages: Last few messages for context (optional)

        Returns:
            Classification dict with:
            - action: "CREATE_TASK" | "ROUTE_TO_TASK" | "DIRECT_ANSWER"
            - task_id: ID of task to route to (if ROUTE_TO_TASK)
            - task_title: Extracted title (if CREATE_TASK)
            - task_description: Extracted description (if CREATE_TASK)
            - confidence: Confidence level (0.0-1.0)
            - reasoning: Brief explanation of the decision
        """
        # Build classification prompt
        prompt_text = self._build_classification_prompt(message, active_task, recent_messages)

        # Call LLM with structured output
        client = self._get_client()
        response = await client.complete(
            prompt=prompt_text,
            temperature=0.0,  # Deterministic classification
            max_tokens=500,
        )

        # Parse JSON response (strip markdown code blocks if present)
        try:
            # Strip markdown code blocks
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            result = json.loads(response)
            return self._validate_classification(result)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse router response: {response}")
            # Default to DIRECT_ANSWER on parse failure
            return {
                "action": "DIRECT_ANSWER",
                "confidence": 0.5,
                "reasoning": "Failed to parse classification, defaulting to direct answer",
            }

    def _build_classification_prompt(
        self,
        message: str,
        active_task: dict[str, Any] | None,
        recent_messages: list[dict[str, Any]] | None,
    ) -> str:
        """Build the classification prompt."""
        context_parts = []

        # Active task context
        if active_task:
            context_parts.append(
                f"**Active Task:**\n"
                f"- ID: {active_task.get('id', 'unknown')}\n"
                f"- Title: {active_task.get('title', 'Untitled')}\n"
                f"- Phase: {active_task.get('phase', 'unknown')}\n"
            )
        else:
            context_parts.append("**Active Task:** None")

        # Recent message context (last 3 messages)
        if recent_messages and len(recent_messages) > 0:
            context_parts.append("\n**Recent Conversation:**")
            for msg in recent_messages[-3:]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                # Truncate long messages
                if len(content) > 200:
                    content = content[:200] + "..."
                context_parts.append(f"- {role}: {content}")

        context = "\n".join(context_parts)

        # Main prompt
        prompt = f"""You are a task routing assistant. Your job is to classify incoming messages and decide how to handle them.

{context}

**Current Message:**
{message}

**Classification Task:**
Decide which action to take:

1. **CREATE_TASK**: The user is requesting help with a new task that requires planning and execution.
   - Examples: "Can you help me refactor this module?", "I need to add authentication to my app"
   - Use this when the request is actionable, multi-step, or involves coding/writing/research

2. **ROUTE_TO_TASK**: The message relates to the currently active task.
   - Examples: "Yes, proceed", "Actually, let's use a different approach", "Can you explain that part?"
   - Use this when there's an active task and the message is clearly continuing that conversation

3. **DIRECT_ANSWER**: The message is a simple question or command that doesn't need task tracking.
   - Examples: "What time is it?", "What's the weather?", "/help", "Thanks!", "ping"
   - Use this for greetings, simple questions, acknowledgments, or one-off commands

**Output Format (JSON):**
```json
{{
  "action": "CREATE_TASK" | "ROUTE_TO_TASK" | "DIRECT_ANSWER",
  "task_id": "<id of task to route to, if ROUTE_TO_TASK>",
  "task_title": "<short title if CREATE_TASK, max 60 chars>",
  "task_description": "<longer description if CREATE_TASK, max 200 chars>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation of why you chose this action>"
}}
```

**Rules:**
- If there's an active task and the message seems related, prefer ROUTE_TO_TASK
- Only CREATE_TASK if the request is clearly a new, distinct task
- Use DIRECT_ANSWER for anything that doesn't need multi-turn task tracking
- task_title and task_description are ONLY needed if action is CREATE_TASK
- Keep task_title concise (under 60 characters)
- confidence should reflect how certain you are (0.7+ is confident, 0.5-0.7 is uncertain)

Respond with ONLY the JSON object, no other text."""

        return prompt

    def _validate_classification(self, result: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize the classification result."""
        action = result.get("action", "DIRECT_ANSWER")

        # Ensure valid action
        if action not in ("CREATE_TASK", "ROUTE_TO_TASK", "DIRECT_ANSWER"):
            logger.warning(f"Invalid action '{action}', defaulting to DIRECT_ANSWER")
            action = "DIRECT_ANSWER"

        # Build normalized result
        normalized = {
            "action": action,
            "confidence": float(result.get("confidence", 0.5)),
            "reasoning": result.get("reasoning", ""),
        }

        # Add action-specific fields
        if action == "ROUTE_TO_TASK":
            normalized["task_id"] = result.get("task_id")

        if action == "CREATE_TASK":
            normalized["task_title"] = result.get("task_title", "Untitled Task")
            normalized["task_description"] = result.get("task_description", "")

            # Truncate if too long
            if len(normalized["task_title"]) > 60:
                normalized["task_title"] = normalized["task_title"][:57] + "..."
            if len(normalized["task_description"]) > 200:
                normalized["task_description"] = normalized["task_description"][:197] + "..."

        return normalized
