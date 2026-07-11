"""
Integration tests for Phase 3 router in TaskFSMController.

Tests the on_message routing logic with the router LLM.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from mesh.controller.task_fsm import TaskFSMController
from mesh.controller.base import ControllerContext
from mesh.config import ControllerConfig


class TestTaskFSMControllerPhase3:
    """Test Phase 3 router integration in TaskFSMController."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create a controller config with temp paths."""
        return ControllerConfig(
            mode="task-fsm-v0",
            tasks_path=str(tmp_path / "tasks.json"),
            config_path=str(tmp_path / "config.json"),
            router_model="gpt-4o-mini",
        )

    @pytest.fixture
    def controller(self, config):
        """Create a controller instance."""
        return TaskFSMController(config)

    @pytest.fixture
    def context(self):
        """Create a controller context."""
        return ControllerContext(
            cwd="/home/test",
            agent_id="agent:test:1",
            history=[],
        )

    @pytest.fixture
    def message(self):
        """Create a mock message object."""
        msg = MagicMock()
        msg.content = "Can you help me add authentication?"
        return msg

    @pytest.mark.asyncio
    async def test_on_message_creates_task(self, controller, message, context):
        """Test that CREATE_TASK classification creates a new task."""
        # Mock router classification
        mock_classification = {
            "action": "CREATE_TASK",
            "task_title": "Add authentication",
            "task_description": "Implement auth system",
            "confidence": 0.9,
            "reasoning": "User wants new feature"
        }

        with patch.object(controller._router, 'classify_message', new=AsyncMock(return_value=mock_classification)):
            decision = await controller.on_message(message, context)

        # Should create a task and activate it
        assert decision.action == "PROCESS_WITH_LLM"
        assert decision.task_id is not None
        assert decision.task_id.startswith("task-")
        assert decision.system_addendum is not None
        assert "Add authentication" in decision.system_addendum

        # Check that task was created
        tasks = controller.get_tasks()
        assert len(tasks) == 1
        assert tasks[0].title == "Add authentication"
        assert tasks[0].description == "Implement auth system"
        assert tasks[0].original_request == "Can you help me add authentication?"

    @pytest.mark.asyncio
    async def test_on_message_routes_to_active_task(self, controller, message, context):
        """Test that ROUTE_TO_TASK routes to the active task."""
        # Create an active task first
        task = controller.create_task("Existing task", "Description")
        controller.set_active_task(task.id)

        # Mock router classification
        mock_classification = {
            "action": "ROUTE_TO_TASK",
            "task_id": task.id,
            "confidence": 0.95,
            "reasoning": "Continuing active task"
        }

        with patch.object(controller._router, 'classify_message', new=AsyncMock(return_value=mock_classification)):
            decision = await controller.on_message(message, context)

        # Should route to the active task
        assert decision.action == "PROCESS_WITH_LLM"
        assert decision.task_id == task.id

    @pytest.mark.asyncio
    async def test_on_message_direct_answer(self, controller, message, context):
        """Test that DIRECT_ANSWER processes without task context."""
        # Mock router classification
        mock_classification = {
            "action": "DIRECT_ANSWER",
            "confidence": 0.99,
            "reasoning": "Simple question"
        }

        with patch.object(controller._router, 'classify_message', new=AsyncMock(return_value=mock_classification)):
            decision = await controller.on_message(message, context)

        # Should process without task context
        assert decision.action == "PROCESS_WITH_LLM"
        assert decision.task_id is None
        assert len(controller.get_tasks()) == 0  # No task created

    @pytest.mark.asyncio
    async def test_on_message_router_error_fallback(self, controller, message, context):
        """Test that router errors fall back to DIRECT_ANSWER."""
        # Mock router to raise an exception
        with patch.object(controller._router, 'classify_message', new=AsyncMock(side_effect=Exception("Router failed"))):
            decision = await controller.on_message(message, context)

        # Should fall back to direct answer
        assert decision.action == "PROCESS_WITH_LLM"
        assert decision.task_id is None
        assert len(controller.get_tasks()) == 0

    @pytest.mark.asyncio
    async def test_on_message_empty_content(self, controller, context):
        """Test handling of messages with no text content."""
        empty_msg = MagicMock()
        empty_msg.content = ""

        decision = await controller.on_message(empty_msg, context)

        # Should pass through without calling router
        assert decision.action == "PROCESS_WITH_LLM"

    @pytest.mark.asyncio
    async def test_on_message_passes_context_to_router(self, controller, message, context):
        """Test that active task and history are passed to router."""
        # Create an active task
        task = controller.create_task("Active task", "Description")
        controller.set_active_task(task.id)

        # Add some history
        history_msg1 = MagicMock()
        history_msg1.role = "user"
        history_msg1.content = "Previous message"
        history_msg2 = MagicMock()
        history_msg2.role = "assistant"
        history_msg2.content = "Previous response"
        context.history = [history_msg1, history_msg2]

        mock_classification = {
            "action": "DIRECT_ANSWER",
            "confidence": 0.8,
            "reasoning": "Test"
        }

        with patch.object(controller._router, 'classify_message', new=AsyncMock(return_value=mock_classification)) as mock_classify:
            await controller.on_message(message, context)

            # Check that router was called with correct context
            call_kwargs = mock_classify.call_args[1]
            assert call_kwargs["active_task"]["id"] == task.id
            assert call_kwargs["active_task"]["title"] == "Active task"
            assert len(call_kwargs["recent_messages"]) == 2

    def test_extract_message_text_from_string_content(self, controller):
        """Test extracting text from message with string content."""
        msg = MagicMock()
        msg.content = "Hello world"

        text = controller._extract_message_text(msg)
        assert text == "Hello world"

    def test_extract_message_text_from_dict_content(self, controller):
        """Test extracting text from message with dict content."""
        msg = MagicMock()
        msg.content = {"text": "Hello world"}

        text = controller._extract_message_text(msg)
        assert text == "Hello world"

    def test_extract_message_text_from_dict_message(self, controller):
        """Test extracting text from dict-based message."""
        msg = {"content": "Hello world"}

        text = controller._extract_message_text(msg)
        assert text == "Hello world"

    def test_extract_message_text_fallback(self, controller):
        """Test fallback to str() conversion."""
        msg = "Plain string message"

        text = controller._extract_message_text(msg)
        assert text == "Plain string message"
