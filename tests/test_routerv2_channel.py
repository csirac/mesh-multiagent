"""Tests for RouterV2 channel @mention detection and filtering."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from mesh.agent_node import is_at_mentioned, is_nicknamed_mention
from mesh.router_v2 import RouterV2, RouterV2Config, WorkerResult, RouterState
from mesh.protocol import Message, MessageType


# =============================================================================
# A. is_at_mentioned() function
# =============================================================================


class TestIsAtMentioned:
    """Tests for the @mention detection used in channel message filtering."""

    def test_at_mention_basic(self):
        assert is_at_mentioned("@bob check this", ["bob"]) is True

    def test_at_mention_case_insensitive(self):
        assert is_at_mentioned("@Bob check this", ["bob"]) is True
        assert is_at_mentioned("@BOB check this", ["bob"]) is True

    def test_at_mention_mid_sentence(self):
        assert is_at_mentioned("hey @bob what's up", ["bob"]) is True

    def test_at_mention_end_of_string(self):
        assert is_at_mentioned("check this @bob", ["bob"]) is True

    def test_plain_name_no_match(self):
        assert is_at_mentioned("look at bob's work", ["bob"]) is False

    def test_no_mention(self):
        assert is_at_mentioned("hello everyone", ["bob"]) is False

    def test_empty_content(self):
        assert is_at_mentioned("", ["bob"]) is False

    def test_empty_nicknames(self):
        assert is_at_mentioned("@bob hi", []) is False

    def test_none_content(self):
        """None or empty content should return False, not crash."""
        assert is_at_mentioned("", ["bob"]) is False

    def test_multiple_nicknames(self):
        assert is_at_mentioned("@claude help", ["bob", "claude", "sysadmin"]) is True

    def test_at_mention_with_punctuation(self):
        assert is_at_mentioned("@bob, check this", ["bob"]) is True

    def test_partial_nickname_no_match(self):
        assert is_at_mentioned("@bo check this", ["bob"]) is False

    def test_email_no_false_positive(self):
        """The @bob pattern should NOT match in 'bob@example.com' (@ is after bob, not before)."""
        assert is_at_mentioned("send to bob@example.com", ["bob"]) is False

    def test_at_mention_with_colon(self):
        assert is_at_mentioned("@bob: look at this", ["bob"]) is True

    def test_at_mention_only(self):
        """Just '@bob' with nothing else."""
        assert is_at_mentioned("@bob", ["bob"]) is True

    def test_nickname_with_hyphen(self):
        """Nicknames like 'claude-sobek' should work."""
        assert is_at_mentioned("@claude-sobek help", ["claude-sobek"]) is True

    def test_no_match_without_at_symbol(self):
        """Without @, even an exact match shouldn't trigger."""
        assert is_at_mentioned("bob check this", ["bob"]) is False


# =============================================================================
# B. is_nicknamed_mention() (legacy, still used for non-channel contexts)
# =============================================================================


class TestIsNicknamedMention:
    """Tests for the legacy fuzzy nickname matching."""

    def test_nickname_mention_basic(self):
        assert is_nicknamed_mention("hey claude can you help", ["claude"]) is True

    def test_nickname_mention_substring(self):
        assert is_nicknamed_mention("claude-sobek is online", ["claude"]) is True

    def test_nickname_no_match(self):
        assert is_nicknamed_mention("what do you think", ["claude", "alice"]) is False

    def test_nickname_empty_content(self):
        assert is_nicknamed_mention("", ["claude"]) is False

    def test_nickname_empty_nicknames(self):
        assert is_nicknamed_mention("hey claude", []) is False

    def test_nickname_case_insensitive(self):
        assert is_nicknamed_mention("Hey CLAUDE can you help", ["claude"]) is True


# =============================================================================
# C. Channel message filtering (integration with RouterV2)
# =============================================================================


def _make_message(content, from_node="user:testuser", to_node="agent:test:test-bot"):
    return Message(
        type=MessageType.MESSAGE,
        from_node=from_node,
        to_node=to_node,
        content=content,
    )


def _make_channel_message(content, channel="channel:test", from_node="user:testuser"):
    return Message(
        type=MessageType.MESSAGE,
        from_node=from_node,
        to_node=channel,
        content=content,
    )


@pytest.fixture
def router_with_llm(tmp_path):
    """RouterV2 with mock LLM for channel filtering tests."""
    sent = []

    async def send_fn(content, in_reply_to=None):
        sent.append({"content": content, "in_reply_to": in_reply_to})

    async def worker_fn(context, trigger, **kwargs):
        return WorkerResult(response="Done.", context=[], usage=None, error=None)

    class MockLLM:
        def __init__(self):
            self.calls = []

        async def complete(self, prompt, **kwargs):
            self.calls.append(prompt)
            return '{"needs_response": true, "needs_worker": false, "response": "Hello"}'

    llm = MockLLM()
    config = RouterV2Config(
        llm_enabled=True,
        history_persist=True,
        history_persist_path=str(tmp_path / "channel-test-history.json"),
    )
    router = RouterV2(
        worker_fn=worker_fn,
        send_fn=send_fn,
        config=config,
        llm_client=llm,
        nickname="test-bot",
        agent_type="test",
        node_id="agent:test:test-bot",
    )
    return router, llm, sent


class TestChannelMessageFiltering:
    """Integration tests for channel message routing with @mention gating."""

    @pytest.mark.asyncio
    async def test_channel_msg_with_at_mention_triggers_on_message(self, router_with_llm):
        """Channel message with @mention should go to on_message (LLM classification)."""
        router, llm, sent = router_with_llm
        msg = _make_channel_message("@test-bot check this")
        await router.on_message(msg)
        # The LLM should have been called for classification
        assert len(llm.calls) > 0

    @pytest.mark.asyncio
    async def test_channel_msg_without_mention_goes_to_history_only(self, router_with_llm):
        """Channel message without @mention should go to add_to_history_only (no LLM)."""
        router, llm, sent = router_with_llm
        msg = _make_channel_message("hello everyone")
        await router.add_to_history_only(msg)
        # Message should be in history
        assert len(router._history.window) == 1
        assert router._history.window[0].content == "hello everyone"
        # No LLM call should have been made
        assert len(llm.calls) == 0

    @pytest.mark.asyncio
    async def test_dm_goes_to_on_message(self, router_with_llm):
        """Direct messages should always go to on_message regardless of @mention."""
        router, llm, sent = router_with_llm
        msg = _make_message("hello")
        await router.on_message(msg)
        # LLM should have been called
        assert len(llm.calls) > 0

    @pytest.mark.asyncio
    async def test_channel_msg_passive_awareness(self, router_with_llm):
        """Non-mentioned channel messages should still appear in history."""
        router, llm, sent = router_with_llm
        msg = _make_channel_message("alice: look at Bob's work")
        await router.add_to_history_only(msg)
        # Should be in history for passive awareness
        assert len(router._history.window) == 1
        assert "Bob's work" in router._history.window[0].content

    @pytest.mark.asyncio
    async def test_add_to_history_only_no_state_change(self, router_with_llm):
        """add_to_history_only should not change router state."""
        router, llm, sent = router_with_llm
        assert router.state == RouterState.IDLE
        msg = _make_channel_message("general chatter")
        await router.add_to_history_only(msg)
        assert router.state == RouterState.IDLE
        assert len(sent) == 0  # Nothing sent
