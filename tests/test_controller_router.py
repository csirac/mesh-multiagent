"""
Unit tests for mesh/controller/router.py (Phase 3).

Tests the RouterLLM message classification logic.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from mesh.controller.router import RouterLLM


class TestRouterLLM:
    """Test the RouterLLM classifier."""

    @pytest.fixture
    def router(self):
        """Create a router instance."""
        return RouterLLM(model="gpt-4o-mini", backend="openai")

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client that returns valid JSON."""
        client = AsyncMock()
        client.complete = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_classify_create_task(self, router, mock_llm_client):
        """Test classification of a message that should create a task."""
        # Mock LLM response
        response = json.dumps({
            "action": "CREATE_TASK",
            "task_title": "Add authentication system",
            "task_description": "Implement JWT-based authentication for the API",
            "confidence": 0.9,
            "reasoning": "User is requesting a new multi-step feature"
        })
        mock_llm_client.complete.return_value = response

        with patch.object(router, '_get_client', return_value=mock_llm_client):
            result = await router.classify_message(
                message="Can you help me add authentication to my app?"
            )

        assert result["action"] == "CREATE_TASK"
        assert result["task_title"] == "Add authentication system"
        assert result["task_description"] == "Implement JWT-based authentication for the API"
        assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_classify_route_to_task(self, router, mock_llm_client):
        """Test classification of a message that should route to active task."""
        response = json.dumps({
            "action": "ROUTE_TO_TASK",
            "task_id": "task-20260203-001",
            "confidence": 0.95,
            "reasoning": "Message is clearly continuing the active task conversation"
        })
        mock_llm_client.complete.return_value = response

        active_task = {
            "id": "task-20260203-001",
            "title": "Refactor API endpoints",
            "phase": "executing"
        }

        with patch.object(router, '_get_client', return_value=mock_llm_client):
            result = await router.classify_message(
                message="Yes, proceed with that approach",
                active_task=active_task
            )

        assert result["action"] == "ROUTE_TO_TASK"
        assert result["task_id"] == "task-20260203-001"
        assert result["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_classify_direct_answer(self, router, mock_llm_client):
        """Test classification of a simple question."""
        response = json.dumps({
            "action": "DIRECT_ANSWER",
            "confidence": 0.99,
            "reasoning": "Simple greeting, doesn't need task tracking"
        })
        mock_llm_client.complete.return_value = response

        with patch.object(router, '_get_client', return_value=mock_llm_client):
            result = await router.classify_message(message="Hi there!")

        assert result["action"] == "DIRECT_ANSWER"
        assert result["confidence"] == 0.99

    @pytest.mark.asyncio
    async def test_classify_with_recent_messages(self, router, mock_llm_client):
        """Test classification with recent conversation context."""
        response = json.dumps({
            "action": "ROUTE_TO_TASK",
            "task_id": "task-20260203-001",
            "confidence": 0.85,
            "reasoning": "Following up on previous discussion"
        })
        mock_llm_client.complete.return_value = response

        recent_messages = [
            {"role": "user", "content": "Can you help me fix the bug in login.py?"},
            {"role": "assistant", "content": "I can help. Let me read that file."},
            {"role": "user", "content": "Actually, let's use a different approach"}
        ]

        with patch.object(router, '_get_client', return_value=mock_llm_client):
            result = await router.classify_message(
                message="Actually, let's use a different approach",
                recent_messages=recent_messages
            )

        assert result["action"] == "ROUTE_TO_TASK"

    @pytest.mark.asyncio
    async def test_classify_invalid_json(self, router, mock_llm_client):
        """Test handling of invalid JSON response."""
        # Mock LLM returns malformed JSON
        mock_llm_client.complete.return_value = "This is not valid JSON"

        with patch.object(router, '_get_client', return_value=mock_llm_client):
            result = await router.classify_message(message="Help me")

        # Should default to DIRECT_ANSWER
        assert result["action"] == "DIRECT_ANSWER"
        assert result["confidence"] == 0.5

    @pytest.mark.asyncio
    async def test_validate_classification_invalid_action(self, router):
        """Test validation of classification with invalid action."""
        invalid_result = {
            "action": "INVALID_ACTION",
            "confidence": 0.8,
            "reasoning": "Test"
        }

        normalized = router._validate_classification(invalid_result)

        # Should default to DIRECT_ANSWER
        assert normalized["action"] == "DIRECT_ANSWER"

    def test_validate_classification_truncates_long_title(self, router):
        """Test that long task titles are truncated."""
        result = {
            "action": "CREATE_TASK",
            "task_title": "A" * 100,  # 100 chars, should be truncated
            "task_description": "Short description",
            "confidence": 0.9,
            "reasoning": "Test"
        }

        normalized = router._validate_classification(result)

        assert len(normalized["task_title"]) <= 60
        assert normalized["task_title"].endswith("...")

    def test_validate_classification_truncates_long_description(self, router):
        """Test that long descriptions are truncated."""
        result = {
            "action": "CREATE_TASK",
            "task_title": "Short title",
            "task_description": "D" * 250,  # 250 chars, should be truncated
            "confidence": 0.9,
            "reasoning": "Test"
        }

        normalized = router._validate_classification(result)

        assert len(normalized["task_description"]) <= 200
        assert normalized["task_description"].endswith("...")

    def test_build_classification_prompt_with_active_task(self, router):
        """Test prompt building with active task context."""
        active_task = {
            "id": "task-20260203-001",
            "title": "Refactor code",
            "phase": "executing"
        }

        prompt = router._build_classification_prompt(
            message="Can you help with this?",
            active_task=active_task,
            recent_messages=None
        )

        assert "task-20260203-001" in prompt
        assert "Refactor code" in prompt
        assert "executing" in prompt

    def test_build_classification_prompt_without_active_task(self, router):
        """Test prompt building without active task."""
        prompt = router._build_classification_prompt(
            message="Can you help with this?",
            active_task=None,
            recent_messages=None
        )

        assert "**Active Task:** None" in prompt

    def test_build_classification_prompt_with_recent_messages(self, router):
        """Test prompt building with recent message context."""
        recent_messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"}
        ]

        prompt = router._build_classification_prompt(
            message="Current message",
            active_task=None,
            recent_messages=recent_messages
        )

        assert "**Recent Conversation:**" in prompt
        assert "Message 1" in prompt
        assert "Response 1" in prompt
        assert "Message 2" in prompt

    def test_build_classification_prompt_truncates_long_messages(self, router):
        """Test that long messages in context are truncated."""
        recent_messages = [
            {"role": "user", "content": "X" * 300}  # Very long message
        ]

        prompt = router._build_classification_prompt(
            message="Current",
            active_task=None,
            recent_messages=recent_messages
        )

        # Should truncate to 200 chars + "..."
        assert "..." in prompt
