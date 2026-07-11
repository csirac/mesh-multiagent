"""
Tests for user preference extraction.
"""

import pytest
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from mesh.preferences import (
    PreferenceExtractor,
    PreferenceState,
    PREFERENCE_EXTRACTION_PROMPT,
    DEFAULT_MESSAGE_THRESHOLD,
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_STALE_HOURS,
    DEFAULT_EXTRACTION_MODEL,
    DEFAULT_EXTRACTION_BACKEND,
)
from mesh.node import HistoryEntry
from mesh.protocol import Message, MessageType


def make_history_entry(
    from_node: str,
    content: str,
    direction: str = "incoming",
) -> HistoryEntry:
    """Helper to create a history entry."""
    msg = Message(
        from_node=from_node,
        to_node="agent:test:alice",
        type=MessageType.MESSAGE,
        content=content,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    return HistoryEntry(message=msg, direction=direction)


class TestPreferenceState:
    """Tests for PreferenceState dataclass."""

    def test_to_dict(self):
        state = PreferenceState(
            preferences_text="Test preferences",
            messages_at_extraction=50,
            created_at="2024-01-15T10:00:00Z",
            model_used="claude-code:sonnet",
        )
        result = state.to_dict()

        assert result["preferences_text"] == "Test preferences"
        assert result["messages_at_extraction"] == 50
        assert result["created_at"] == "2024-01-15T10:00:00Z"
        assert result["model_used"] == "claude-code:sonnet"

    def test_from_dict(self):
        data = {
            "preferences_text": "Test preferences",
            "messages_at_extraction": 50,
            "created_at": "2024-01-15T10:00:00Z",
            "model_used": "claude-code:sonnet",
        }
        state = PreferenceState.from_dict(data)

        assert state.preferences_text == "Test preferences"
        assert state.messages_at_extraction == 50
        assert state.created_at == "2024-01-15T10:00:00Z"
        assert state.model_used == "claude-code:sonnet"


class TestPreferenceExtractor:
    """Tests for PreferenceExtractor class."""

    def test_init_with_defaults(self, tmp_path):
        history_file = tmp_path / "test.json"
        extractor = PreferenceExtractor(history_file=history_file)

        assert extractor._message_threshold == DEFAULT_MESSAGE_THRESHOLD
        assert extractor._context_limit == DEFAULT_CONTEXT_LIMIT
        assert extractor._stale_hours == DEFAULT_STALE_HOURS
        assert extractor._extraction_model == DEFAULT_EXTRACTION_MODEL
        assert extractor._extraction_backend == DEFAULT_EXTRACTION_BACKEND

    def test_init_with_custom_values(self, tmp_path):
        history_file = tmp_path / "test.json"
        extractor = PreferenceExtractor(
            history_file=history_file,
            message_threshold=100,
            context_limit=50_000,
            stale_hours=48,
            extraction_model="opus",
            extraction_backend="openai",
        )

        assert extractor._message_threshold == 100
        assert extractor._context_limit == 50_000
        assert extractor._stale_hours == 48
        assert extractor._extraction_model == "opus"
        assert extractor._extraction_backend == "openai"

    def test_preferences_file_path(self, tmp_path):
        history_file = tmp_path / "agent-test-alice.json"
        extractor = PreferenceExtractor(history_file=history_file)

        expected = tmp_path / "agent-test-alice.preferences.json"
        assert extractor.preferences_file == expected

    def test_preferences_file_path_none(self):
        extractor = PreferenceExtractor(history_file=None)
        assert extractor.preferences_file is None


class TestPreferencePersistence:
    """Tests for saving and loading preferences."""

    def test_save_and_load_preferences(self, tmp_path):
        history_file = tmp_path / "test.json"
        extractor = PreferenceExtractor(history_file=history_file)

        state = PreferenceState(
            preferences_text="## Communication\nUser prefers concise responses.",
            messages_at_extraction=75,
            created_at=datetime.now(timezone.utc).isoformat(),
            model_used="claude-code:sonnet",
        )

        # Save
        assert extractor.save_preferences(state) is True
        assert extractor.preferences_file.exists()

        # Load
        assert extractor.load_preferences() is True
        assert extractor._preferences.preferences_text == state.preferences_text
        assert extractor._preferences.messages_at_extraction == 75

    def test_load_preferences_nonexistent(self, tmp_path):
        history_file = tmp_path / "test.json"
        extractor = PreferenceExtractor(history_file=history_file)

        assert extractor.load_preferences() is False
        assert extractor._preferences is None

    def test_save_preferences_no_history_file(self):
        extractor = PreferenceExtractor(history_file=None)
        state = PreferenceState(
            preferences_text="Test",
            messages_at_extraction=50,
            created_at=datetime.now(timezone.utc).isoformat(),
            model_used="claude-code:sonnet",
        )
        assert extractor.save_preferences(state) is False


class TestStalenessCheck:
    """Tests for staleness detection."""

    def test_is_stale_no_preferences(self, tmp_path):
        history_file = tmp_path / "test.json"
        extractor = PreferenceExtractor(history_file=history_file)

        assert extractor.is_stale() is True

    def test_is_stale_recent_preferences(self, tmp_path):
        history_file = tmp_path / "test.json"
        extractor = PreferenceExtractor(
            history_file=history_file,
            stale_hours=24,
        )

        # Set preferences created 1 hour ago
        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        extractor._preferences = PreferenceState(
            preferences_text="Test",
            messages_at_extraction=50,
            created_at=one_hour_ago.isoformat(),
            model_used="claude-code:sonnet",
        )

        assert extractor.is_stale() is False

    def test_is_stale_old_preferences(self, tmp_path):
        history_file = tmp_path / "test.json"
        extractor = PreferenceExtractor(
            history_file=history_file,
            stale_hours=24,
        )

        # Set preferences created 48 hours ago
        two_days_ago = datetime.now(timezone.utc) - timedelta(hours=48)
        extractor._preferences = PreferenceState(
            preferences_text="Test",
            messages_at_extraction=50,
            created_at=two_days_ago.isoformat(),
            model_used="claude-code:sonnet",
        )

        assert extractor.is_stale() is True


class TestExtractionTrigger:
    """Tests for should_extract logic."""

    def test_should_extract_no_preferences(self, tmp_path):
        history_file = tmp_path / "test.json"
        extractor = PreferenceExtractor(
            history_file=history_file,
            message_threshold=50,
        )

        # Not enough messages
        assert extractor.should_extract(30) is False
        # Enough messages
        assert extractor.should_extract(50) is True
        assert extractor.should_extract(100) is True

    def test_should_extract_with_preferences(self, tmp_path):
        history_file = tmp_path / "test.json"
        extractor = PreferenceExtractor(
            history_file=history_file,
            message_threshold=50,
        )

        extractor._preferences = PreferenceState(
            preferences_text="Test",
            messages_at_extraction=100,  # Last extracted at 100 messages
            created_at=datetime.now(timezone.utc).isoformat(),
            model_used="claude-code:sonnet",
        )

        # Not enough new messages since last extraction
        assert extractor.should_extract(120) is False  # 20 new messages
        # Enough new messages
        assert extractor.should_extract(150) is True  # 50 new messages
        assert extractor.should_extract(200) is True  # 100 new messages

    def test_should_extract_while_extracting(self, tmp_path):
        history_file = tmp_path / "test.json"
        extractor = PreferenceExtractor(
            history_file=history_file,
            message_threshold=50,
        )
        extractor._extracting = True

        assert extractor.should_extract(100) is False


class TestPreferenceBlock:
    """Tests for get_preference_block."""

    def test_get_preference_block_none(self, tmp_path):
        history_file = tmp_path / "test.json"
        extractor = PreferenceExtractor(history_file=history_file)

        assert extractor.get_preference_block() == ""

    def test_get_preference_block_insufficient(self, tmp_path):
        history_file = tmp_path / "test.json"
        extractor = PreferenceExtractor(history_file=history_file)
        extractor._preferences = PreferenceState(
            preferences_text="Insufficient history to extract patterns.",
            messages_at_extraction=10,
            created_at=datetime.now(timezone.utc).isoformat(),
            model_used="claude-code:sonnet",
        )

        assert extractor.get_preference_block() == ""

    def test_get_preference_block_valid(self, tmp_path):
        history_file = tmp_path / "test.json"
        extractor = PreferenceExtractor(history_file=history_file)
        extractor._preferences = PreferenceState(
            preferences_text="## Communication\nUser prefers concise responses.",
            messages_at_extraction=100,
            created_at=datetime.now(timezone.utc).isoformat(),
            model_used="claude-code:sonnet",
        )

        result = extractor.get_preference_block()
        assert "<user_preferences>" in result
        assert "## Communication" in result
        assert "</user_preferences>" in result

    def test_get_preference_block_already_wrapped(self, tmp_path):
        history_file = tmp_path / "test.json"
        extractor = PreferenceExtractor(history_file=history_file)
        wrapped = "<user_preferences>\n## Communication\nConcise.\n</user_preferences>"
        extractor._preferences = PreferenceState(
            preferences_text=wrapped,
            messages_at_extraction=100,
            created_at=datetime.now(timezone.utc).isoformat(),
            model_used="claude-code:sonnet",
        )

        result = extractor.get_preference_block()
        # Should not double-wrap
        assert result.count("<user_preferences>") == 1


class TestContentProcessing:
    """Tests for history content processing."""

    def test_process_content_removes_tool_results(self, tmp_path):
        history_file = tmp_path / "test.json"
        extractor = PreferenceExtractor(history_file=history_file)

        content = """Here is the response.
<mesh_result name="bash_exec">
stdout: lots of output here
stderr:
returncode: 0
</mesh_result>
And more text."""

        result = extractor._process_content_for_extraction(content)
        assert "<mesh_result" not in result
        assert "lots of output" not in result
        assert "Here is the response" in result
        assert "And more text" in result

    def test_process_content_summarizes_tool_calls(self, tmp_path):
        history_file = tmp_path / "test.json"
        extractor = PreferenceExtractor(history_file=history_file)

        content = """Let me search for that.
<mesh_call name="exa_search">
<query>python async patterns</query>
<num_results>5</num_results>
</mesh_call>
Done."""

        result = extractor._process_content_for_extraction(content)
        assert "<mesh_call name=" not in result
        assert "<query>" not in result
        assert "[Tool: exa_search]" in result
        assert "Let me search for that" in result
        assert "Done" in result

    def test_format_history_for_extraction(self, tmp_path):
        history_file = tmp_path / "test.json"
        extractor = PreferenceExtractor(
            history_file=history_file,
            context_limit=10_000,
        )

        history = [
            make_history_entry("user:testuser", "Help me with Python"),
            make_history_entry(
                "agent:coder:alice",
                "Sure! <mesh_call name=\"bash_exec\"><cmd>python --version</cmd></mesh_call>",
                direction="outgoing",
            ),
            make_history_entry("user:testuser", "Thanks!"),
        ]

        result = extractor.format_history_for_extraction(history)

        assert "[user:testuser]:" in result
        assert "[agent]:" in result
        assert "Help me with Python" in result
        assert "Thanks!" in result
        assert "[Tool: bash_exec]" in result
        assert "<cmd>" not in result


class TestExtractionPrompt:
    """Tests for the extraction prompt."""

    def test_prompt_has_placeholder(self):
        assert "{history_text}" in PREFERENCE_EXTRACTION_PROMPT

    def test_prompt_formatting(self):
        formatted = PREFERENCE_EXTRACTION_PROMPT.format(history_text="test history")
        assert "test history" in formatted
        assert "{history_text}" not in formatted
