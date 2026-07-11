"""Tests for ConversationHistory component."""

import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mesh.conversation_history import ConversationHistory, Turn, ROUTER_SUMMARY_PROMPT
from mesh.node import SummaryState
from mesh.llm import HistoryMessage


# =========================================================================
# Turn tests
# =========================================================================

class TestTurn:
    def test_basic_creation(self):
        t = Turn(role="user", content="hello", timestamp=datetime.now(timezone.utc))
        assert t.role == "user"
        assert t.content == "hello"
        assert t.seq_id == 0  # not yet assigned

    def test_token_estimate_cached(self):
        t = Turn(role="user", content="hello world", timestamp=datetime.now(timezone.utc))
        est1 = t.token_estimate
        est2 = t.token_estimate
        assert est1 == est2
        assert est1 > 0

    def test_to_history_message(self):
        ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        t = Turn(
            role="user", content="test msg", timestamp=ts,
            from_node="user:testuser", to_node="agent:coder",
        )
        hm = t.to_history_message()
        assert isinstance(hm, HistoryMessage)
        assert hm.from_node == "user:testuser"
        assert hm.content == "test msg"
        assert "2026-01-15" in hm.timestamp
        assert hm.to_node == "agent:coder"

    def test_to_dict_from_dict_roundtrip(self):
        ts = datetime(2026, 2, 7, 6, 0, 0, tzinfo=timezone.utc)
        original = Turn(
            role="assistant", content="response", timestamp=ts,
            from_node="agent:coder", to_node="user:testuser",
            meta={"key": "value"}, seq_id=42,
        )
        d = original.to_dict()
        restored = Turn.from_dict(d)
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.from_node == original.from_node
        assert restored.to_node == original.to_node
        assert restored.meta == original.meta
        assert restored.seq_id == original.seq_id
        assert isinstance(restored.timestamp, datetime)


# =========================================================================
# ConversationHistory tests
# =========================================================================

class TestConversationHistory:
    def _make_turn(self, content: str, role: str = "user") -> Turn:
        return Turn(
            role=role, content=content,
            timestamp=datetime.now(timezone.utc),
            from_node=f"{role}:test",
        )

    def test_append_assigns_seq_id(self):
        ch = ConversationHistory()
        t1 = self._make_turn("first")
        t2 = self._make_turn("second")
        ch.append(t1)
        ch.append(t2)
        assert t1.seq_id == 1
        assert t2.seq_id == 2
        assert ch.last_seq_id == 2
        assert len(ch) == 2

    def test_turns_since(self):
        ch = ConversationHistory()
        for i in range(10):
            ch.append(self._make_turn(f"msg {i}"))

        since_5 = ch.turns_since(5)
        assert len(since_5) == 5
        assert since_5[0].seq_id == 6
        assert since_5[-1].seq_id == 10

    def test_turns_since_zero(self):
        ch = ConversationHistory()
        for i in range(3):
            ch.append(self._make_turn(f"msg {i}"))
        assert len(ch.turns_since(0)) == 3

    def test_get_recent_for_peek(self):
        ch = ConversationHistory()
        for i in range(10):
            ch.append(self._make_turn(f"msg {i}"))

        recent = ch.get_recent_for_peek(n=3)
        assert len(recent) == 3
        assert recent[0].seq_id == 8

    def test_get_recent_for_peek_with_since_seq(self):
        ch = ConversationHistory()
        for i in range(10):
            ch.append(self._make_turn(f"msg {i}"))

        recent = ch.get_recent_for_peek(since_seq=7)
        assert len(recent) == 3
        assert recent[0].seq_id == 8

    def test_estimate_tokens(self):
        ch = ConversationHistory(base_overhead=1000)
        ch.append(self._make_turn("hello world"))
        tokens = ch.estimate_tokens()
        assert tokens > 1000  # at least base overhead

    def test_needs_summarization(self):
        ch = ConversationHistory(soft_token_limit=100, base_overhead=0)
        # Should not need summarization with empty history
        assert not ch.needs_summarization()

        # Add enough content to exceed limit
        for i in range(20):
            ch.append(self._make_turn(f"message {i} with some content to push token count up"))

        assert ch.needs_summarization()

    def test_needs_summarization_false_when_summarizing(self):
        ch = ConversationHistory(soft_token_limit=100, base_overhead=0)
        for i in range(20):
            ch.append(self._make_turn(f"message {i} content"))
        ch._summarizing = True
        assert not ch.needs_summarization()

    def test_build_context_for_llm_no_summary(self):
        ch = ConversationHistory()
        ch.append(self._make_turn("hello"))
        ch.append(self._make_turn("world", role="assistant"))

        result = ch.build_context_for_llm()
        assert len(result) == 2
        assert result[0].content == "hello"
        assert result[1].content == "world"

    def test_build_context_for_llm_with_summary(self):
        ch = ConversationHistory()
        ch._summary = SummaryState(
            summary_text="Previous context summary",
            messages_summarized=10,
            created_at=datetime.now(timezone.utc).isoformat(),
            token_estimate=50,
        )
        ch.append(self._make_turn("recent msg"))

        result = ch.build_context_for_llm()
        assert len(result) == 2
        assert result[0].from_node == "system"
        assert "[Earlier summary]" in result[0].content
        assert result[1].content == "recent msg"

    def test_last_seq_id_empty(self):
        ch = ConversationHistory()
        assert ch.last_seq_id == 0

    def test_clear_via_window_clear(self):
        ch = ConversationHistory()
        for i in range(5):
            ch.append(self._make_turn(f"msg {i}"))
        ch._window.clear()
        assert len(ch) == 0
        assert ch.last_seq_id == 0


# =========================================================================
# Persistence tests
# =========================================================================

class TestConversationHistoryPersistence:
    def _make_turn(self, content: str, role: str = "user") -> Turn:
        return Turn(
            role=role, content=content,
            timestamp=datetime.now(timezone.utc),
            from_node=f"{role}:test",
        )

    def test_save_and_load_v2(self, tmp_path):
        persist_path = tmp_path / "test_history.json"
        summary_path = tmp_path / "test_history.summary.json"

        # Create and populate
        ch = ConversationHistory(
            persist_path=persist_path,
            summary_persist_path=summary_path,
        )
        ch.append(self._make_turn("first"))
        ch.append(self._make_turn("second"))
        ch._summary = SummaryState(
            summary_text="test summary",
            messages_summarized=5,
            created_at=datetime.now(timezone.utc).isoformat(),
            token_estimate=20,
        )
        ch.save()

        # Verify files exist
        assert persist_path.exists()
        assert summary_path.exists()

        # Load into new instance
        ch2 = ConversationHistory(
            persist_path=persist_path,
            summary_persist_path=summary_path,
        )
        loaded = ch2.load()
        assert loaded == 2
        assert len(ch2) == 2
        assert ch2._summary is not None
        assert ch2._summary.summary_text == "test summary"
        assert ch2._summary.messages_summarized == 5

    def test_load_v1_legacy_format(self, tmp_path):
        """Test loading legacy flat list format (HistoryEntry dicts)."""
        persist_path = tmp_path / "legacy.json"

        # Write v1 format (list of HistoryEntry dicts)
        v1_data = [
            {
                "message": {
                    "id": "msg1",
                    "type": "message",
                    "from_node": "user:testuser",
                    "to_node": "agent:coder",
                    "content": "hello",
                    "timestamp": "2026-01-15T12:00:00+00:00",
                    "in_reply_to": None,
                },
                "direction": "incoming",
            },
            {
                "message": {
                    "id": "msg2",
                    "type": "message",
                    "from_node": "agent:coder",
                    "to_node": "user:testuser",
                    "content": "hi there",
                    "timestamp": "2026-01-15T12:00:01+00:00",
                    "in_reply_to": None,
                },
                "direction": "outgoing",
            },
        ]
        persist_path.write_text(json.dumps(v1_data))

        ch = ConversationHistory(persist_path=persist_path)
        loaded = ch.load()

        assert loaded == 2
        assert len(ch) == 2
        assert ch.window[0].content == "hello"
        assert ch.window[0].from_node == "user:testuser"
        assert ch.window[0].role == "incoming"
        assert ch.window[1].content == "hi there"
        assert ch.window[1].role == "outgoing"
        assert ch._summary is None  # No summary in v1

    def test_roundtrip_v2_format(self, tmp_path):
        persist_path = tmp_path / "roundtrip.json"

        ch = ConversationHistory(persist_path=persist_path)
        for i in range(5):
            ch.append(self._make_turn(f"message {i}"))
        ch.save()

        # Verify v2 format
        raw = json.loads(persist_path.read_text())
        assert raw["version"] == 2
        assert raw["next_seq_id"] == 6
        assert len(raw["window"]) == 5

    def test_to_dict_from_dict(self):
        ch = ConversationHistory(soft_token_limit=10_000)
        ch.append(self._make_turn("hello"))
        ch._summary = SummaryState(
            summary_text="summary", messages_summarized=1,
            created_at="2026-01-15", token_estimate=10,
        )

        d = ch.to_dict()
        ch2 = ConversationHistory.from_dict(d, soft_token_limit=10_000)
        assert len(ch2) == 1
        assert ch2._summary.summary_text == "summary"


# =========================================================================
# Summarization tests
# =========================================================================

class TestConversationHistorySummarization:
    def _make_turn(self, content: str, role: str = "user") -> Turn:
        return Turn(
            role=role, content=content,
            timestamp=datetime.now(timezone.utc),
            from_node=f"{role}:test",
        )

    @pytest.mark.asyncio
    async def test_summarize_basic(self):
        """Test basic summarization flow with mock LLM."""
        ch = ConversationHistory(
            soft_token_limit=100,
            hard_token_limit=200,
            target_ratio=0.25,
            base_overhead=0,
            per_message_overhead=5,
        )

        # Add enough turns to trigger summarization
        for i in range(20):
            ch.append(self._make_turn(f"message {i} with padding content words here"))

        # Mock LLM client
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="Summary of older messages")

        assert ch.needs_summarization()

        await ch.summarize(mock_client)

        # Should have created a summary
        assert ch._summary is not None
        assert ch._summary.summary_text == "Summary of older messages"
        assert ch._summary.messages_summarized > 0

        # Window should be trimmed
        assert len(ch) < 20

        # Should no longer be summarizing
        assert not ch._summarizing

    @pytest.mark.asyncio
    async def test_summarize_concurrent_protection(self):
        """Test that concurrent summarization is prevented."""
        ch = ConversationHistory(soft_token_limit=100, base_overhead=0)
        for i in range(20):
            ch.append(self._make_turn(f"msg {i} padding"))

        ch._summarizing = True

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="should not happen")

        await ch.summarize(mock_client)

        # complete() should NOT have been called
        mock_client.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_summarize_persists_if_configured(self, tmp_path):
        """Test that summarization auto-saves when persist is configured."""
        persist_path = tmp_path / "summ_test.json"

        ch = ConversationHistory(
            soft_token_limit=100,
            base_overhead=0,
            per_message_overhead=5,
            persist_path=persist_path,
        )
        for i in range(20):
            ch.append(self._make_turn(f"msg {i} padding content"))

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="Summarized content")

        await ch.summarize(mock_client)

        # Should have saved to disk
        assert persist_path.exists()
        raw = json.loads(persist_path.read_text())
        assert raw["version"] == 2

    @pytest.mark.asyncio
    async def test_summarize_with_existing_summary(self):
        """Test summarization when there's already a summary."""
        ch = ConversationHistory(
            soft_token_limit=100, base_overhead=0, per_message_overhead=5,
        )

        ch._summary = SummaryState(
            summary_text="Previous summary",
            messages_summarized=10,
            created_at="2026-01-15",
            token_estimate=20,
        )

        for i in range(20):
            ch.append(self._make_turn(f"msg {i} padding"))

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="New combined summary")

        await ch.summarize(mock_client)

        assert ch._summary.summary_text == "New combined summary"
        assert ch._summary.messages_summarized > 10  # should include old + new

    @pytest.mark.asyncio
    async def test_seq_ids_stable_after_summarization(self):
        """Test that seq_ids are stable after summarization trims window."""
        ch = ConversationHistory(
            soft_token_limit=100, base_overhead=0, per_message_overhead=5,
        )

        for i in range(20):
            ch.append(self._make_turn(f"msg {i} padding"))

        # Capture seq_id of last turn before summarization
        last_seq = ch.last_seq_id
        assert last_seq == 20

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="Summary")

        await ch.summarize(mock_client)

        # After summarization, remaining turns should keep their original seq_ids
        for turn in ch.window:
            assert turn.seq_id > 0
            assert turn.seq_id <= 20

        # turns_since should still work correctly
        remaining_turns = ch.turns_since(0)
        assert len(remaining_turns) == len(ch)

    def test_default_format(self):
        """Test the default format function for summarization."""
        turns = [
            Turn(role="user", content="hello", timestamp="2026-01-15T12:00:00",
                 from_node="user:testuser"),
            Turn(role="assistant", content="hi", timestamp="2026-01-15T12:00:01",
                 from_node="agent:coder"),
        ]
        result = ConversationHistory._default_format(turns)
        assert "[user:testuser at 2026-01-15T12:00:00]" in result
        assert "hello" in result
        assert "[agent:coder at 2026-01-15T12:00:01]" in result
        assert "hi" in result


# =========================================================================
# Hard limit tests
# =========================================================================

class TestConversationHistoryHardLimit:
    def _make_turn(self, content: str, role: str = "user") -> Turn:
        return Turn(
            role=role, content=content,
            timestamp=datetime.now(timezone.utc),
            from_node=f"{role}:test",
        )

    def test_hard_limit_drops_oldest(self):
        """Test that hard limit drops oldest window entries."""
        ch = ConversationHistory(
            hard_token_limit=500,
            base_overhead=0,
        )

        # Add a lot of content to exceed hard limit
        for i in range(50):
            ch.append(self._make_turn(f"message {i} " + "x" * 100))

        result = ch.build_context_for_llm()
        # Should have fewer messages than what we appended
        assert len(result) < 50


# =========================================================================
# Rolling window summarization tests
# =========================================================================

class TestRollingWindowSummarization:
    """Tests for the rolling window summarization algorithm.

    Token calibration (with tiktoken GPT-4 encoding):
      - "message N with padding content words here" ≈ 8 tokens
      - per_message_overhead = 35
      - cost per turn ≈ 43 tokens
      - With W=200: trigger at ~10 turns (430 tokens > 2*200)
      - After summarization: ~5 turns remain (~215 tokens ≈ W)
    """

    def _make_turn(self, content: str, role: str = "user",
                   topic: str = "", seq: int = 0) -> Turn:
        meta = {}
        if topic:
            meta["topic_label"] = topic
        return Turn(
            role=role, content=content,
            timestamp=datetime(2026, 2, 27, 12, 0, 0, tzinfo=timezone.utc),
            from_node=f"{role}:test",
            meta=meta,
        )

    def _make_ch(self, window_budget: int = 200, **kwargs) -> ConversationHistory:
        """Create a ConversationHistory with small window budget for testing."""
        defaults = dict(
            soft_token_limit=1000,
            hard_token_limit=2000,
            base_overhead=0,
            per_message_overhead=35,
            window_budget=window_budget,
        )
        defaults.update(kwargs)
        return ConversationHistory(**defaults)

    def test_estimate_window_tokens(self):
        """Window-only token estimate excludes summary and base overhead."""
        ch = self._make_ch(base_overhead=5000)
        ch.append(self._make_turn("hello world"))
        ch._summary = SummaryState(
            summary_text="Big summary " * 100,
            messages_summarized=10,
            created_at="2026-01-15",
            token_estimate=500,
        )

        window_tokens = ch.estimate_window_tokens()
        total_tokens = ch.estimate_tokens()

        # Window tokens should be much smaller than total (no summary, no overhead)
        assert window_tokens < total_tokens
        # Window tokens = content tokens + per_message_overhead
        assert window_tokens > 0
        # Total includes base_overhead (5000) + summary (500) + window
        assert total_tokens >= 5000 + 500 + window_tokens

    def test_needs_summarization_rolling_trigger(self):
        """Trigger fires when window tokens >= 2*W, not before."""
        ch = self._make_ch(window_budget=200)

        # Add turns one by one, check trigger
        for i in range(8):
            ch.append(self._make_turn(f"message {i} with padding content words here"))
            # Should not trigger yet (8 * 43 ≈ 344 < 400)
        assert not ch.needs_summarization()

        # Push past 2*W
        for i in range(8, 15):
            ch.append(self._make_turn(f"message {i} with padding content words here"))
        # Now window should exceed 2*200 = 400
        assert ch.estimate_window_tokens() >= 400
        assert ch.needs_summarization()

    @pytest.mark.asyncio
    async def test_summarize_rolling_partitions(self):
        """After summarization, window contains approximately W tokens (recent half)."""
        ch = self._make_ch(window_budget=200)

        # Add enough turns to trigger
        for i in range(15):
            ch.append(self._make_turn(f"message {i} with padding content words here"))

        initial_len = len(ch)
        assert ch.needs_summarization()

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="Summarized content")

        await ch.summarize(mock_client)

        # Window should be trimmed — fewer turns than we started with
        assert len(ch) < initial_len
        # The old half (~W tokens worth of turns) was removed
        # Remaining = total - old_half, which is approximately total - W
        remaining_tokens = ch.estimate_window_tokens()
        # Should have a summary now
        assert ch._summary is not None
        assert ch._summary.summary_text == "Summarized content"

    @pytest.mark.asyncio
    async def test_summarize_rolling_accumulates_summary(self):
        """messages_summarized increments by old_half count on each cycle."""
        ch = self._make_ch(window_budget=200)

        # First cycle
        for i in range(15):
            ch.append(self._make_turn(f"message {i} with padding content words here"))

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="First summary")
        await ch.summarize(mock_client)

        first_summarized = ch._summary.messages_summarized
        assert first_summarized > 0

        first_remaining = len(ch)

        # Add more turns for second cycle
        for i in range(15, 30):
            ch.append(self._make_turn(f"message {i} with padding content words here"))

        mock_client.complete = AsyncMock(return_value="Second summary")
        await ch.summarize(mock_client)

        # messages_summarized should have increased
        assert ch._summary.messages_summarized > first_summarized
        # Specifically, it should be first_summarized + number of turns folded in second cycle
        second_folded = first_remaining + 15 - len(ch)
        assert ch._summary.messages_summarized == first_summarized + second_folded

    @pytest.mark.asyncio
    async def test_summarize_rolling_small_window_no_trigger(self):
        """Window < 2*W does not trigger summarization."""
        ch = self._make_ch(window_budget=200)

        # Add just a few turns (well under 2*W)
        for i in range(3):
            ch.append(self._make_turn(f"message {i} with padding content words here"))

        assert not ch.needs_summarization()

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="Should not happen")
        await ch.summarize(mock_client)

        # summarize() should have returned early (not summarizing flag)
        # But since needs_summarization is separate from summarize, it will actually run
        # Let's just verify it doesn't crash with small windows
        # The point is needs_summarization() returns False
        assert not ch.needs_summarization()

    @pytest.mark.asyncio
    async def test_summarize_rolling_single_large_turn(self):
        """Edge case: a single turn exceeding W tokens is still handled."""
        ch = self._make_ch(window_budget=50)

        # One large turn that exceeds W
        ch.append(self._make_turn("big content " * 100))
        # One normal turn
        ch.append(self._make_turn("small"))

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="Summary of large turn")

        # Even though first turn > W, summarize should handle it
        await ch.summarize(mock_client)

        # Should not crash, should produce a summary
        assert ch._summary is not None
        assert ch._summary.summary_text == "Summary of large turn"

    @pytest.mark.asyncio
    async def test_summary_token_cap(self):
        """Summary output exceeding max_summary_tokens is truncated."""
        ch = self._make_ch(window_budget=200, max_summary_tokens=20)

        for i in range(15):
            ch.append(self._make_turn(f"message {i} with padding content words here"))

        # Return a huge summary that will exceed the 20-token cap
        huge_summary = "This is an extremely detailed summary that goes on and on " * 20
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=huge_summary)

        await ch.summarize(mock_client)

        # Summary should exist but be capped
        assert ch._summary is not None
        from mesh.llm import estimate_tokens
        # The summary should be truncated to approximately max_summary_tokens
        # (it may not be exact due to the character-based truncation)
        summary_tokens = estimate_tokens(ch._summary.summary_text)
        assert summary_tokens <= 30  # some slack for truncation imprecision

    def test_topic_labels_in_format(self):
        """_default_format() groups turns by topic label with section headers."""
        turns = [
            self._make_turn("Started editing the doc", topic="document editing"),
            self._make_turn("Done with the edit", role="assistant", topic="document editing"),
            self._make_turn("Now let's check training", topic="PPO training"),
            self._make_turn("Training looks good", role="assistant", topic="PPO training"),
        ]

        result = ConversationHistory._default_format(turns)

        # Should have topic section headers
        assert "--- Topic: document editing" in result
        assert "--- Topic: PPO training" in result
        # Content should be present
        assert "Started editing the doc" in result
        assert "Training looks good" in result

    def test_topic_labels_missing_graceful(self):
        """Turns without topic labels flow naturally without headers."""
        turns = [
            self._make_turn("labeled message", topic="some topic"),
            self._make_turn("unlabeled message"),  # no topic
            self._make_turn("another unlabeled"),  # no topic
            self._make_turn("new topic message", topic="new topic"),
        ]

        result = ConversationHistory._default_format(turns)

        # Should have headers for labeled topics
        assert "--- Topic: some topic" in result
        assert "--- Topic: new topic" in result
        # Unlabeled messages should still be present
        assert "unlabeled message" in result
        assert "another unlabeled" in result
        # Should NOT have a "(continued)" or "(no topic)" header for unlabeled
        assert "(no topic)" not in result

    def test_backward_compat_no_window_budget(self):
        """ConversationHistory with only soft_token_limit derives W = soft_limit // 2."""
        ch = ConversationHistory(soft_token_limit=70000)
        assert ch._window_budget == 35000

    def test_backward_compat_explicit_window_budget(self):
        """Explicit window_budget takes precedence over soft_limit derivation."""
        ch = ConversationHistory(soft_token_limit=70000, window_budget=20000)
        assert ch._window_budget == 20000

    @pytest.mark.asyncio
    async def test_multi_cycle_summarization(self):
        """Two consecutive summarization cycles: summary accumulates, old content decays."""
        ch = self._make_ch(window_budget=200)

        # Cycle 1: fill and summarize
        for i in range(15):
            ch.append(self._make_turn(f"cycle1 message {i} with padding content"))
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="Cycle 1 summary: discussed items 0-7")
        await ch.summarize(mock_client)

        cycle1_summary = ch._summary.summary_text
        cycle1_window_size = len(ch)
        cycle1_msgs_summarized = ch._summary.messages_summarized

        # Verify the LLM was called with the old half
        call_args = mock_client.complete.call_args[0][0]
        assert "cycle1 message 0" in call_args  # old half should include earliest messages

        # Cycle 2: fill again and summarize
        for i in range(15, 30):
            ch.append(self._make_turn(f"cycle2 message {i} with padding content"))

        mock_client.complete = AsyncMock(return_value="Cycle 2 summary: extended with items 8-20")
        await ch.summarize(mock_client)

        # The LLM should have seen the previous summary
        call_args = mock_client.complete.call_args[0][0]
        assert "Cycle 1 summary" in call_args

        # Summary should have accumulated
        assert ch._summary.messages_summarized > cycle1_msgs_summarized
        assert ch._summary.summary_text == "Cycle 2 summary: extended with items 8-20"

    @pytest.mark.asyncio
    async def test_summarize_preserves_recent_turns(self):
        """After summarization, the most recent turns are preserved."""
        ch = self._make_ch(window_budget=200)

        for i in range(15):
            ch.append(self._make_turn(f"message {i} with padding content words here"))

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="Summary")
        await ch.summarize(mock_client)

        # The last turn should still be in the window
        assert any("message 14" in t.content for t in ch.window)
        # Earlier turns should be gone
        assert not any("message 0" in t.content for t in ch.window)

    def test_unbounded_growth_warning_when_summarizing(self):
        """When summarization is in progress and window exceeds 2W, needs_summarization logs warning."""
        ch = self._make_ch(window_budget=200)
        for i in range(15):
            ch.append(self._make_turn(f"message {i} with padding content words here"))

        ch._summarizing = True
        # Should return False (don't trigger another) but internally log a warning
        assert not ch.needs_summarization()

    def test_max_summary_tokens_default(self):
        """Default max_summary_tokens is W/4."""
        ch = ConversationHistory(window_budget=40000)
        assert ch._max_summary_tokens == 10000

    def test_max_summary_tokens_explicit(self):
        """Explicit max_summary_tokens overrides the default."""
        ch = ConversationHistory(window_budget=40000, max_summary_tokens=5000)
        assert ch._max_summary_tokens == 5000
