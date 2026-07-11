"""
Unit tests for mesh/relevance_router.py
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mesh.config import RelevanceRouterConfig
from mesh.relevance_router import (
    RelevanceRouter,
    RelevanceResult,
)
from mesh.protocol import Message, MessageType


class TestRelevanceResult:
    """Tests for RelevanceResult dataclass."""

    def test_basic_creation(self):
        result = RelevanceResult(score=0.8, reason="Agent mentioned")
        assert result.score == 0.8
        assert result.reason == "Agent mentioned"
        assert result.bypassed is False

    def test_bypassed_flag(self):
        result = RelevanceResult(score=1.0, reason="Direct message", bypassed=True)
        assert result.bypassed is True


class TestRelevanceRouterConfig:
    """Tests for RelevanceRouterConfig defaults."""

    def test_default_values(self):
        config = RelevanceRouterConfig()
        assert config.threshold == 0.7
        assert config.bypass_direct is True
        assert config.bypass_mentions is False
        assert config.model == "gpt-4o-mini"
        assert config.backend == "openai"

    def test_custom_values(self):
        config = RelevanceRouterConfig(
            threshold=0.7,
            bypass_direct=False,
            bypass_mentions=False,
            model="gpt-4",
            backend="anthropic",
        )
        assert config.threshold == 0.7
        assert config.bypass_direct is False


class TestRelevanceRouter:
    """Tests for RelevanceRouter class."""

    @pytest.fixture
    def router(self):
        """Create a router with default config."""
        return RelevanceRouter(
            agent_nickname="alice",
            agent_description="A helpful coding assistant",
            nicknames=["alice", "coder"],
        )

    @pytest.fixture
    def channel_message(self):
        """Create a sample channel message."""
        return Message(
            id="msg-1",
            type=MessageType.MESSAGE,
            from_node="user:testuser",
            to_node="channel:dev",
            content="Can someone help with this bug?",
        )

    @pytest.fixture
    def direct_message(self):
        """Create a sample direct message."""
        return Message(
            id="msg-2",
            type=MessageType.MESSAGE,
            from_node="user:testuser",
            to_node="agent:coder:alice",
            content="Hey alice, can you help?",
        )

    # --- Bypass Tests ---

    @pytest.mark.asyncio
    async def test_bypass_direct_message(self, router, direct_message):
        """Direct messages should be bypassed and score 1.0."""
        result = await router.classify(direct_message)
        assert result.score == 1.0
        assert result.bypassed is True
        assert "Direct message" in result.reason

    @pytest.mark.asyncio
    async def test_bypass_nickname_mention_when_enabled(self, channel_message):
        """Messages mentioning nickname should be bypassed when bypass_mentions=True."""
        config = RelevanceRouterConfig(bypass_mentions=True)
        router = RelevanceRouter(
            config=config,
            agent_nickname="alice",
            agent_description="A helpful coding assistant",
            nicknames=["alice", "coder"],
        )
        channel_message.content = "Hey alice, what do you think?"
        result = await router.classify(channel_message)
        assert result.score == 1.0
        assert result.bypassed is True
        assert "nickname" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_bypass_nickname_case_insensitive_when_enabled(self, channel_message):
        """Nickname matching should be case-insensitive when bypass_mentions=True."""
        config = RelevanceRouterConfig(bypass_mentions=True)
        router = RelevanceRouter(
            config=config,
            agent_nickname="alice",
            agent_description="A helpful coding assistant",
            nicknames=["alice", "coder"],
        )
        channel_message.content = "ALICE can you check this?"
        result = await router.classify(channel_message)
        assert result.score == 1.0
        assert result.bypassed is True

    @pytest.mark.asyncio
    async def test_bypass_alternate_nickname_when_enabled(self, channel_message):
        """Should also match alternate nicknames when bypass_mentions=True."""
        config = RelevanceRouterConfig(bypass_mentions=True)
        router = RelevanceRouter(
            config=config,
            agent_nickname="alice",
            agent_description="A helpful coding assistant",
            nicknames=["alice", "coder"],
        )
        channel_message.content = "hey coder, what's wrong here?"
        result = await router.classify(channel_message)
        assert result.score == 1.0
        assert result.bypassed is True

    @pytest.mark.asyncio
    async def test_nickname_mention_goes_to_llm_by_default(self, router, channel_message):
        """By default (bypass_mentions=False), nickname mentions go to LLM."""
        channel_message.content = "Hey alice, what do you think?"
        result = await router.classify(channel_message)
        # Should NOT be bypassed - should call LLM
        assert result.bypassed is False
        # Score depends on LLM response, but should be non-zero since name is mentioned
        assert result.score > 0

    # --- No Bypass Tests ---

    @pytest.mark.asyncio
    async def test_no_bypass_disabled_direct(self, channel_message):
        """When bypass_direct=False, direct messages should go to LLM."""
        config = RelevanceRouterConfig(bypass_direct=False, bypass_mentions=False)
        router = RelevanceRouter(
            config=config,
            agent_nickname="alice",
            nicknames=["alice"],
        )

        # Mock the LLM call
        with patch.object(router, '_llm_classify', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = RelevanceResult(score=0.8, reason="Test")

            msg = Message(
                id="msg-1",
                type=MessageType.MESSAGE,
                from_node="user:testuser",
                to_node="agent:coder:alice",
                content="Hey alice",
            )
            result = await router.classify(msg)

            # Should have called LLM since bypass is disabled
            mock_llm.assert_called_once()
            assert result.score == 0.8

    @pytest.mark.asyncio
    async def test_no_bypass_disabled_mentions(self, channel_message):
        """When bypass_mentions=False, nickname mentions should go to LLM."""
        config = RelevanceRouterConfig(bypass_mentions=False)
        router = RelevanceRouter(
            config=config,
            agent_nickname="alice",
            nicknames=["alice"],
        )

        with patch.object(router, '_llm_classify', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = RelevanceResult(score=0.9, reason="Test")

            channel_message.content = "alice can you help?"
            result = await router.classify(channel_message)

            mock_llm.assert_called_once()

    # --- Nickname Matching Tests ---

    def test_is_nicknamed_mention_basic(self, router):
        """Test basic nickname detection."""
        assert router._is_nicknamed_mention("hello alice") is True
        assert router._is_nicknamed_mention("hello bob") is False

    def test_is_nicknamed_mention_word_boundary(self, router):
        """Nicknames should match at word boundaries."""
        assert router._is_nicknamed_mention("alice, can you help?") is True
        assert router._is_nicknamed_mention("alice's code") is True
        # Should not match partial words
        assert router._is_nicknamed_mention("malice") is False

    def test_is_nicknamed_mention_empty(self):
        """Empty nicknames should not cause issues."""
        router = RelevanceRouter(nicknames=["", None, "valid"])
        assert router._is_nicknamed_mention("hello valid") is True
        assert router._is_nicknamed_mention("hello") is False

    # --- Threshold Tests ---

    def test_should_process_above_threshold(self, router):
        """Scores at or above threshold should process."""
        result = RelevanceResult(score=0.7, reason="Test")
        assert router.should_process(result) is True

        result = RelevanceResult(score=0.8, reason="Test")
        assert router.should_process(result) is True

    def test_should_process_below_threshold(self, router):
        """Scores below threshold should not process."""
        result = RelevanceResult(score=0.6, reason="Test")
        assert router.should_process(result) is False

        result = RelevanceResult(score=0.0, reason="Test")
        assert router.should_process(result) is False

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        config = RelevanceRouterConfig(threshold=0.8)
        router = RelevanceRouter(config=config)

        assert router.should_process(RelevanceResult(score=0.7, reason="")) is False
        assert router.should_process(RelevanceResult(score=0.8, reason="")) is True

    # --- Response Parsing Tests ---

    def test_parse_response_standard_format(self, router):
        """Test parsing standard SCORE: X REASON: Y format."""
        response = "SCORE: 7 REASON: Topic is relevant to coding tasks"
        result = router._parse_response(response)
        assert result.score == 0.7
        assert "relevant" in result.reason.lower()

    def test_parse_response_decimal_score(self, router):
        """Test parsing decimal scores."""
        response = "SCORE: 8.5 REASON: Very relevant"
        result = router._parse_response(response)
        assert result.score == 0.85

    def test_parse_response_max_score(self, router):
        """Test that scores above 10 are clamped to 1.0."""
        response = "SCORE: 15 REASON: Test"
        result = router._parse_response(response)
        assert result.score == 1.0

    def test_parse_response_no_format(self, router):
        """Test fallback when response doesn't match format."""
        response = "This message scores 6 out of 10"
        result = router._parse_response(response)
        assert result.score == 0.6

    def test_parse_response_no_number(self, router):
        """Test fallback when no number found."""
        response = "I don't know how to score this"
        result = router._parse_response(response)
        assert result.score == 0.5  # Default mid-range

    # --- LLM Integration Tests ---

    @pytest.mark.asyncio
    async def test_llm_classify_called_for_unmentioned(self, router, channel_message):
        """LLM should be called when message doesn't mention nickname."""
        with patch.object(router, '_llm_classify', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = RelevanceResult(score=0.3, reason="Not relevant")

            result = await router.classify(channel_message)

            mock_llm.assert_called_once()
            assert result.score == 0.3
            assert result.bypassed is False

    @pytest.mark.asyncio
    async def test_llm_error_returns_default_score(self, router, channel_message):
        """LLM errors should bypass with score 1.0 (pass through)."""
        with patch.object(router, '_get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.complete = AsyncMock(side_effect=Exception("API error"))
            mock_get_client.return_value = mock_client

            result = await router.classify(channel_message)

            assert result.score == 1.0
            assert result.bypassed is True
            assert "failed" in result.reason.lower()

    # --- Prompt Building Tests ---

    def test_build_prompt_includes_agent_info(self, router, channel_message):
        """Prompt should include agent nickname and description."""
        prompt = router._build_prompt(
            channel_message,
            "test content",
            None,
            None,
        )
        assert "alice" in prompt
        assert "coding assistant" in prompt.lower()

    def test_build_prompt_includes_controller_state(self, router, channel_message):
        """Prompt should include controller state when provided."""
        controller_state = {
            "active_task": {
                "title": "Fix auth bug",
                "phase": "executing",
            }
        }
        prompt = router._build_prompt(
            channel_message,
            "test content",
            controller_state,
            None,
        )
        assert "Fix auth bug" in prompt
        assert "executing" in prompt

    def test_build_prompt_truncates_long_content(self, router, channel_message):
        """Long message content should be truncated."""
        long_content = "x" * 1000
        prompt = router._build_prompt(
            channel_message,
            long_content,
            None,
            None,
        )
        assert len(prompt) < 2000  # Reasonable prompt length
        assert "..." in prompt
