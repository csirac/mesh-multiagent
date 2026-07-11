"""
User preference extraction for mesh agents.

Periodically extracts user patterns, preferences, and workflows from
agent history and prepends them to the agent's context.

Storage: ~/.mesh/history/{agent}.preferences.json
Trigger: Every N messages (configurable) + stale check on startup
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLMClient, LLMConfig, HistoryMessage
    from .node import HistoryEntry

logger = logging.getLogger(__name__)

# Default extraction model
DEFAULT_EXTRACTION_MODEL = "opus"
DEFAULT_EXTRACTION_BACKEND = "zai"

# Default trigger thresholds
DEFAULT_MESSAGE_THRESHOLD = 50  # Extract every N messages
DEFAULT_CONTEXT_LIMIT = 100_000  # Max tokens to consider from history
DEFAULT_STALE_HOURS = 24  # Extract on startup if older than this

# Prompt for extracting user preferences
PREFERENCE_EXTRACTION_PROMPT = r"""You are analyzing a conversation history between a user and an AI agent.

Your task is to extract RECURRING patterns, preferences, and workflows from this conversation.
Focus on things that would help the agent serve this user better in future interactions.

## What to Extract

1. **Communication Style Preferences**
   - How does the user like to receive information? (brief vs detailed, bullet points vs prose)
   - Do they prefer technical depth or high-level summaries?
   - Any formatting preferences observed?

2. **Workflow Patterns**
   - What tasks does this user commonly request?
   - Are there repeated sequences of actions?
   - What tools/features do they use frequently?

3. **Domain Knowledge**
   - What topics or domains is this user working in?
   - What level of expertise do they have in these areas?
   - Any specialized terminology or conventions they use?

4. **Decision Patterns**
   - How do they make decisions when presented with options?
   - Do they prefer recommendations or explorations?
   - Any biases toward certain approaches (e.g., simple vs comprehensive)?

5. **Correction Patterns**
   - What has the agent gotten wrong that the user corrected?
   - Any recurring misunderstandings to avoid?
   - Feedback patterns that indicate preference?

## Output Format

Produce a concise summary (aim for 200-500 words) structured as:

```
<user_preferences>
## Communication
[Observed preferences about how to communicate]

## Common Tasks
[Recurring task patterns]

## Domain Context
[Relevant domain knowledge and expertise level]

## Preferences & Corrections
[Key preferences and things to remember/avoid]
</user_preferences>
```

If you cannot identify clear patterns (e.g., too little history or too varied), output:
```
<user_preferences>
Insufficient history to extract patterns.
</user_preferences>
```

## Conversation History

The following is the conversation history to analyze. Note that tool calls have been
summarized (arguments omitted, just tool names shown) to save space.

{history_text}

Now extract the user's preferences and patterns:
"""


@dataclass
class PreferenceState:
    """Stored preference extraction state."""
    preferences_text: str  # The extracted preferences
    messages_at_extraction: int  # Number of history entries when extracted
    created_at: str  # ISO timestamp
    model_used: str  # Model that performed extraction

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PreferenceState":
        """Deserialize from dictionary."""
        return cls(
            preferences_text=data["preferences_text"],
            messages_at_extraction=data["messages_at_extraction"],
            created_at=data["created_at"],
            model_used=data["model_used"],
        )


class PreferenceExtractor:
    """
    Extracts and manages user preference patterns from agent history.

    Preferences are extracted periodically (based on message count) and
    persisted to disk. They are prepended to the agent's context to
    inform future interactions.
    """

    def __init__(
        self,
        history_file: Path | None,
        message_threshold: int = DEFAULT_MESSAGE_THRESHOLD,
        context_limit: int = DEFAULT_CONTEXT_LIMIT,
        stale_hours: int = DEFAULT_STALE_HOURS,
        extraction_model: str = DEFAULT_EXTRACTION_MODEL,
        extraction_backend: str = DEFAULT_EXTRACTION_BACKEND,
        llm_client: "LLMClient | None" = None,
    ):
        """
        Initialize the preference extractor.

        Args:
            history_file: Path to the agent's history file (used to derive preferences path)
            message_threshold: Extract every N messages
            context_limit: Max tokens to consider from history
            stale_hours: Extract on startup if preferences older than this
            extraction_model: Model to use for extraction
            extraction_backend: Backend for extraction model
            llm_client: Pre-configured LLM client (with cc_fallback_homes etc.).
                        If provided, used instead of creating a fresh client.
        """
        self._history_file = history_file
        self._message_threshold = message_threshold
        self._context_limit = context_limit
        self._stale_hours = stale_hours
        self._extraction_model = extraction_model
        self._extraction_backend = extraction_backend
        self._llm_client = llm_client

        # Current state
        self._preferences: PreferenceState | None = None
        self._extracting = False
        self._extraction_task: asyncio.Task | None = None
        self._messages_since_extraction = 0

    @property
    def preferences_file(self) -> Path | None:
        """Path to preferences file (derived from history file)."""
        if not self._history_file:
            return None
        return self._history_file.with_suffix(".preferences.json")

    def load_preferences(self) -> bool:
        """
        Load preferences from disk.

        Returns:
            True if loaded successfully, False otherwise.
        """
        prefs_file = self.preferences_file
        if not prefs_file or not prefs_file.exists():
            return False

        try:
            with open(prefs_file, "r") as f:
                data = json.load(f)
            self._preferences = PreferenceState.from_dict(data)
            logger.info(
                f"Loaded preferences from {prefs_file}: "
                f"extracted at {self._preferences.messages_at_extraction} messages"
            )
            return True
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to load preferences from {prefs_file}: {e}")
            return False

    def save_preferences(self, state: PreferenceState) -> bool:
        """
        Save preferences to disk.

        Returns:
            True if saved successfully, False otherwise.
        """
        prefs_file = self.preferences_file
        if not prefs_file:
            return False

        try:
            prefs_file.parent.mkdir(parents=True, exist_ok=True)
            with open(prefs_file, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
            logger.debug(f"Saved preferences to {prefs_file}")
            return True
        except (OSError, IOError) as e:
            logger.error(f"Failed to save preferences to {prefs_file}: {e}")
            return False

    def is_stale(self) -> bool:
        """
        Check if preferences are stale (older than stale_hours).

        Returns:
            True if stale or no preferences exist.
        """
        if not self._preferences:
            return True

        try:
            created = datetime.fromisoformat(self._preferences.created_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age_hours = (now - created).total_seconds() / 3600
            return age_hours > self._stale_hours
        except (ValueError, TypeError):
            return True

    def should_extract(self, current_message_count: int) -> bool:
        """
        Check if we should trigger extraction based on message count.

        Args:
            current_message_count: Current number of history entries.

        Returns:
            True if extraction should be triggered.
        """
        if self._extracting:
            return False

        if not self._preferences:
            # No preferences yet - extract if we have enough history
            return current_message_count >= self._message_threshold

        # Check if we've accumulated enough new messages
        new_messages = current_message_count - self._preferences.messages_at_extraction
        return new_messages >= self._message_threshold

    def get_preference_block(self) -> str:
        """
        Get the preferences formatted for inclusion in context.

        Returns:
            Formatted preferences block, or empty string if none.
        """
        if not self._preferences:
            return ""

        text = self._preferences.preferences_text.strip()
        if not text or "Insufficient history" in text:
            return ""

        # If already wrapped in <user_preferences>, return as-is
        if text.startswith("<user_preferences>"):
            return text + "\n\n"

        return f"<user_preferences>\n{text}\n</user_preferences>\n\n"

    def format_history_for_extraction(
        self,
        history: list["HistoryEntry"],
        context_limit: int | None = None,
    ) -> str:
        """
        Format history for the extraction prompt.

        Omits tool results, but includes tool call signatures (just names).
        Limits to context_limit tokens.

        Args:
            history: The agent's full history.
            context_limit: Max tokens to include (defaults to self._context_limit).

        Returns:
            Formatted history text.
        """
        from .llm import estimate_tokens
        from .protocol import MessageType

        context_limit = context_limit or self._context_limit

        lines = []
        total_tokens = 0

        # Process history in reverse (most recent first) to prioritize recent patterns
        for entry in reversed(history):
            msg = entry.message

            # Skip non-message types
            if msg.type != MessageType.MESSAGE:
                continue

            content = msg.content if isinstance(msg.content, str) else str(msg.content)

            # Process content: remove tool results, summarize tool calls
            processed_content = self._process_content_for_extraction(content)

            # Build the entry
            direction = "outgoing" if entry.direction == "outgoing" else "incoming"
            from_node = msg.from_node if direction == "incoming" else "agent"

            entry_text = f"[{from_node}]: {processed_content}\n"
            entry_tokens = estimate_tokens(entry_text)

            if total_tokens + entry_tokens > context_limit:
                break

            lines.insert(0, entry_text)  # Insert at beginning (we're going in reverse)
            total_tokens += entry_tokens

        return "".join(lines)

    def _process_content_for_extraction(self, content: str) -> str:
        """
        Process message content for extraction.

        - Removes tool results (<mesh_result>...</mesh_result>)
        - Summarizes tool calls to just tool names
        - Preserves user messages and agent responses

        Args:
            content: The raw message content.

        Returns:
            Processed content.
        """
        import re

        # Remove tool results entirely (supports both old and new format)
        content = re.sub(
            r'<(?:tool-result|mesh_result)[^>]*>.*?</(?:tool-result|mesh_result)>',
            '',
            content,
            flags=re.DOTALL
        )

        # Also remove "Tool execution results:" sections
        content = re.sub(
            r'Tool execution results:\s*(?:<(?:tool-result|mesh_result)[^>]*>.*?</(?:tool-result|mesh_result)>\s*)*',
            '',
            content,
            flags=re.DOTALL
        )

        # Summarize tool calls: <mesh_call name="X">...</mesh_call> -> [Tool: X]
        # Also supports old format for backward compatibility
        def summarize_tool(match):
            name = match.group(1)
            return f"[Tool: {name}]"

        content = re.sub(
            r'<(?:tool|mesh_call)\s+name="([^"]+)"[^>]*>.*?</(?:tool|mesh_call)>',
            summarize_tool,
            content,
            flags=re.DOTALL
        )

        # Clean up extra whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = content.strip()

        return content

    async def maybe_extract(
        self,
        history: list["HistoryEntry"],
        llm_config: "LLMConfig",
    ) -> None:
        """
        Check if extraction should run, and trigger it if so.

        This is non-blocking - extraction runs in the background.

        Args:
            history: The agent's full history.
            llm_config: LLM config (used to create extraction client).
        """
        if not self.should_extract(len(history)):
            return

        logger.info(
            f"Triggering preference extraction "
            f"({len(history)} messages, threshold={self._message_threshold})"
        )

        self._extracting = True
        self._extraction_task = asyncio.create_task(
            self._run_extraction(history, llm_config)
        )

    async def maybe_extract_on_startup(
        self,
        history: list["HistoryEntry"],
        llm_config: "LLMConfig",
    ) -> None:
        """
        Check if extraction should run on startup (due to staleness).

        Args:
            history: The agent's full history.
            llm_config: LLM config (used to create extraction client).
        """
        if not self.is_stale():
            return

        # Only extract if we have enough history
        if len(history) < self._message_threshold:
            logger.debug(
                f"Preferences stale but insufficient history "
                f"({len(history)} < {self._message_threshold})"
            )
            return

        logger.info(
            f"Preferences stale (>{self._stale_hours}h), triggering extraction"
        )

        self._extracting = True
        self._extraction_task = asyncio.create_task(
            self._run_extraction(history, llm_config)
        )

    async def _run_extraction(
        self,
        history: list["HistoryEntry"],
        base_config: "LLMConfig",
    ) -> None:
        """
        Run preference extraction.

        Creates a new LLM client with the extraction model and runs the
        extraction prompt against the formatted history.

        Args:
            history: The agent's full history.
            base_config: Base LLM config (we'll override backend/model).
        """
        from .llm import LLMClient, LLMConfig

        try:
            # Format history for extraction
            history_text = self.format_history_for_extraction(history)

            if not history_text.strip():
                logger.warning("No valid history content for preference extraction")
                return

            # Build prompt
            prompt = PREFERENCE_EXTRACTION_PROMPT.format(history_text=history_text)

            # Use pre-configured client if available (has cc_fallback_homes),
            # otherwise fall back to creating a fresh client from env.
            if self._llm_client:
                logger.debug(f"Running preference extraction with pre-configured client")
                result = await self._llm_client.complete(
                    prompt, model=self._extraction_model
                )
            else:
                extraction_config = LLMConfig.from_env(backend=self._extraction_backend)
                extraction_config.model = self._extraction_model
                extraction_config.max_tokens = 2000
                extraction_config.temperature = 0.3
                logger.debug(f"Running preference extraction with {self._extraction_backend}:{self._extraction_model}")
                client = LLMClient(extraction_config)  # node_id not needed — only calls complete(), no subprocesses
                async with client:
                    result = await client.complete(prompt, model=self._extraction_model)

            # Parse result
            result = result.strip()

            # Create new preference state
            now = datetime.now(timezone.utc).isoformat()
            new_state = PreferenceState(
                preferences_text=result,
                messages_at_extraction=len(history),
                created_at=now,
                model_used=f"{self._extraction_backend}:{self._extraction_model}",
            )

            # Update and persist
            self._preferences = new_state
            self.save_preferences(new_state)

            logger.info(
                f"Preference extraction complete: "
                f"{len(result)} chars from {len(history)} messages"
            )

        except Exception as e:
            logger.exception(f"Preference extraction failed: {e}")
        finally:
            self._extracting = False
            self._extraction_task = None
