"""
Relevance Router - Decides if channel messages should be processed by an agent.

This is a pre-controller router that uses an LLM to score message relevance
on a 0.0-1.0 scale. Used for all controllers (including passthrough) to
filter channel messages.

Flow:
1. Direct messages (to=agent:*) → Always process (bypass router)
2. Channel messages:
   a. Check if nickname mentioned → Fast-path score=1.0
   b. Otherwise call LLM → Score 0.0-1.0
   c. Score >= threshold → Process; else → Log & ignore
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

from .config import RelevanceRouterConfig
from .llm import LLMClient, LLMConfig
from .protocol import Message

logger = logging.getLogger(__name__)


@dataclass
class RelevanceResult:
    """Result of relevance classification."""

    score: float      # 0.0 = definitely ignore, 1.0 = must respond
    reason: str       # Brief explanation
    bypassed: bool = False  # True if fast-pathed (nickname mention or direct message)


class RelevanceRouter:
    """
    LLM-based relevance router for channel messages.

    Decides whether an agent should process a channel message based on:
    - Whether the agent is mentioned
    - Whether the topic is relevant to the agent's role
    - Whether a response is needed

    Returns a confidence score (0.0-1.0) that can be thresholded.
    """

    def __init__(
        self,
        config: RelevanceRouterConfig | None = None,
        agent_nickname: str = "",
        agent_description: str = "",
        nicknames: list[str] | None = None,
    ):
        """
        Initialize the relevance router.

        Args:
            config: Router configuration (uses defaults if None)
            agent_nickname: The agent's primary nickname
            agent_description: Description of the agent's role/capabilities
            nicknames: All nicknames to check for mentions (defaults to [agent_nickname])
        """
        self.config = config or RelevanceRouterConfig()
        self.agent_nickname = agent_nickname
        self.agent_description = agent_description
        self.nicknames = nicknames or ([agent_nickname] if agent_nickname else [])

        # Lazy-initialized LLM client
        self._client: LLMClient | None = None

    def _get_client(self) -> LLMClient:
        """Get or create the LLM client."""
        if self._client is None:
            llm_config = LLMConfig.from_env(backend=self.config.backend)
            llm_config.model = self.config.model
            self._client = LLMClient(llm_config)  # node_id not needed — only calls complete(), no subprocesses
        return self._client

    def _is_nicknamed_mention(self, content: str) -> bool:
        """Check if any nickname is mentioned in the content."""
        content_lower = content.lower()
        for nickname in self.nicknames:
            if not nickname:
                continue
            # Case-insensitive word boundary check
            pattern = rf'\b{re.escape(nickname.lower())}\b'
            if re.search(pattern, content_lower):
                return True
        return False

    async def classify(
        self,
        msg: Message,
        controller_state: dict[str, Any] | None = None,
        recent_messages: list[dict[str, Any]] | None = None,
    ) -> RelevanceResult:
        """
        Classify whether a message should be processed by this agent.

        Args:
            msg: The incoming message
            controller_state: Current controller state (task info, phase, etc.)
            recent_messages: Recent conversation history for context

        Returns:
            RelevanceResult with score (0.0-1.0), reason, and bypass flag
        """
        # Fast-path: Direct messages always processed
        if self.config.bypass_direct and msg.to_node:
            if msg.to_node.startswith("agent:"):
                return RelevanceResult(
                    score=1.0,
                    reason="Direct message to agent",
                    bypassed=True,
                )

        # Fast-path: Nickname mention
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if self.config.bypass_mentions and self._is_nicknamed_mention(content):
            return RelevanceResult(
                score=1.0,
                reason=f"Agent nickname mentioned in message",
                bypassed=True,
            )

        # Call LLM for relevance scoring
        return await self._llm_classify(msg, content, controller_state, recent_messages)

    async def _llm_classify(
        self,
        msg: Message,
        content: str,
        controller_state: dict[str, Any] | None,
        recent_messages: list[dict[str, Any]] | None,
    ) -> RelevanceResult:
        """Use LLM to score message relevance."""
        prompt = self._build_prompt(msg, content, controller_state, recent_messages)

        try:
            client = self._get_client()
            response = await client.complete(
                prompt=prompt,
                temperature=0.0,  # Deterministic scoring
                max_tokens=100,
            )

            return self._parse_response(response)
        except Exception as e:
            logger.warning(f"Relevance router LLM call failed: {e}, defaulting to relevant")
            # Default to relevant on error - better to process than drop
            return RelevanceResult(
                score=1.0,
                reason=f"LLM call failed, passing through: {e}",
                bypassed=True,
            )

    def _build_prompt(
        self,
        msg: Message,
        content: str,
        controller_state: dict[str, Any] | None,
        recent_messages: list[dict[str, Any]] | None,
    ) -> str:
        """Build the relevance classification prompt."""
        # Build context parts
        context_parts = []

        # Agent info
        context_parts.append(
            f"**Agent:** {self.agent_nickname}\n"
            f"**Description:** {self.agent_description or 'General-purpose assistant'}"
        )

        # Controller state
        if controller_state:
            state_summary = []
            if "active_task" in controller_state and controller_state["active_task"]:
                task = controller_state["active_task"]
                state_summary.append(f"- Active task: {task.get('title', 'Untitled')} ({task.get('phase', 'unknown')})")
            if "task_count" in controller_state:
                state_summary.append(f"- Total tasks: {controller_state['task_count']}")

            if state_summary:
                context_parts.append("\n**Current State:**\n" + "\n".join(state_summary))
            else:
                context_parts.append("\n**Current State:** Idle (no active task)")
        else:
            context_parts.append("\n**Current State:** No controller state")

        # Recent messages - highlight which ones are from this agent
        if recent_messages and len(recent_messages) > 0:
            context_parts.append("\n**Recent Conversation:**")
            for m in recent_messages[-5:]:  # Show more context
                sender = m.get("from", m.get("role", "unknown"))
                text = m.get("content", "")
                if len(text) > 200:
                    text = text[:200] + "..."
                # Highlight messages from this agent
                if self.agent_nickname and self.agent_nickname.lower() in sender.lower():
                    context_parts.append(f"- **[THIS AGENT]** {sender}: {text}")
                else:
                    context_parts.append(f"- {sender}: {text}")

        context = "\n".join(context_parts)

        # Truncate message content if very long
        display_content = content
        if len(display_content) > 500:
            display_content = display_content[:500] + "..."

        prompt = f"""You are deciding if agent "{self.agent_nickname}" should RESPOND to this channel message.

{context}

**New Message from {msg.from_node}:**
{display_content}

Rate from 0-10 whether this agent should respond:
- 10: Agent is directly addressed (e.g., @{self.agent_nickname}, hey {self.agent_nickname}, {self.agent_nickname}:)
- 9-10: The new message is a DIRECT FOLLOW-UP to this agent's most recent response (marked [THIS AGENT] above)
- 8-9: Clear continuation of a conversation thread this agent was participating in
- 6-7: Topic matches agent's role AND no one else responded yet
- 3-5: Tangentially related, but another agent or no response may be better
- 1-2: General chatter, observation, or rhetorical - no response needed
- 0: Addressed to someone else, or completely outside agent's domain

IMPORTANT: If the most recent message in the conversation was from [THIS AGENT], and the new message looks like a follow-up question or acknowledgment, score 9-10.

Respond with a single line: SCORE: <0-10> REASON: <brief explanation>"""

        return prompt

    def _parse_response(self, response: str) -> RelevanceResult:
        """Parse the LLM response into a RelevanceResult."""
        response = response.strip()

        # Try to parse "SCORE: X REASON: Y" format
        score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
        reason_match = re.search(r'REASON:\s*(.+)$', response, re.IGNORECASE)

        if score_match:
            raw_score = float(score_match.group(1))
            # Normalize 0-10 to 0.0-1.0
            score = min(1.0, max(0.0, raw_score / 10.0))
        else:
            # Try to find any number in the response
            num_match = re.search(r'(\d+(?:\.\d+)?)', response)
            if num_match:
                raw_score = float(num_match.group(1))
                score = min(1.0, max(0.0, raw_score / 10.0))
            else:
                score = 0.5  # Default to mid-range

        reason = reason_match.group(1).strip() if reason_match else response[:100]

        return RelevanceResult(score=score, reason=reason)

    def should_process(self, result: RelevanceResult) -> bool:
        """Check if the result score meets the threshold for processing."""
        return result.score >= self.config.threshold
