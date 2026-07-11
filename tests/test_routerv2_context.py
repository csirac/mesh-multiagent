"""Tests for RouterV2 history operations, snapshot isolation, context merge, and persistence."""

import asyncio
import json
import pytest

from mesh.router_v2 import RouterV2, RouterV2Config, WorkerResult, RouterState
from mesh.conversation_history import ConversationHistory, Turn
from mesh.protocol import Message, MessageType

from datetime import datetime, timezone


# =============================================================================
# Helpers
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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def router_with_history(tmp_path):
    """RouterV2 with history persistence for context tests."""
    sent = []

    async def send_fn(content, in_reply_to=None):
        sent.append({"content": content, "in_reply_to": in_reply_to})

    async def worker_fn(context, trigger, **kwargs):
        return WorkerResult(response="Done.", context=[], usage=None, error=None)

    history_path = tmp_path / "context-test-history.json"
    config = RouterV2Config(
        llm_enabled=False,
        history_persist=True,
        history_persist_path=str(history_path),
    )
    router = RouterV2(
        worker_fn=worker_fn,
        send_fn=send_fn,
        config=config,
        nickname="test-bot",
        agent_type="test",
        node_id="agent:test:test-bot",
    )
    return router, sent, history_path


@pytest.fixture
def slow_worker_router(tmp_path):
    """RouterV2 with a slow worker for snapshot isolation tests."""
    sent = []
    worker_event = asyncio.Event()
    worker_received_snapshot = []

    async def send_fn(content, in_reply_to=None):
        sent.append({"content": content, "in_reply_to": in_reply_to})

    async def worker_fn(context, trigger, **kwargs):
        worker_received_snapshot.extend(context)
        await worker_event.wait()
        return WorkerResult(response="Done.", context=[], usage=None, error=None)

    history_path = tmp_path / "slow-context-history.json"
    config = RouterV2Config(
        llm_enabled=False,
        history_persist=True,
        history_persist_path=str(history_path),
    )
    router = RouterV2(
        worker_fn=worker_fn,
        send_fn=send_fn,
        config=config,
        nickname="test-bot",
        agent_type="test",
        node_id="agent:test:test-bot",
    )
    return router, sent, worker_event, worker_received_snapshot


# =============================================================================
# A. History operations
# =============================================================================


class TestHistoryOperations:
    """Tests for message tracking in history."""

    @pytest.mark.asyncio
    async def test_incoming_message_added_to_history(self, router_with_history):
        """on_message() should add the message to _history.window."""
        router, sent, _ = router_with_history
        msg = _make_message("hello")
        await router.on_message(msg)
        await asyncio.sleep(0.1)

        # The trigger message should be in history
        contents = [t.content for t in router._history.window]
        assert "hello" in contents

    @pytest.mark.asyncio
    async def test_outgoing_message_added_to_history(self, router_with_history):
        """_send_and_store() should add the response to history with router_response meta."""
        router, sent, _ = router_with_history

        # Directly call _send_and_store (used by router acks, busy responses, etc.)
        msg = _make_message("hello")
        await router._send_and_store("I'm here!", msg)

        # Find outgoing turns with router_response meta
        outgoing = [
            t for t in router._history.window
            if t.meta and t.meta.get("router_response")
        ]
        assert len(outgoing) >= 1
        assert outgoing[0].content == "I'm here!"

    @pytest.mark.asyncio
    async def test_history_tracks_user_messages(self, router_with_history):
        """User messages should update _latest_user_message."""
        router, sent, _ = router_with_history
        msg = _make_message("what's the weather?", from_node="user:testuser")
        await router.on_message(msg)
        await asyncio.sleep(0.1)

        assert router._latest_user_message == "what's the weather?"

    @pytest.mark.asyncio
    async def test_add_to_history_only(self, router_with_history):
        """add_to_history_only() adds to history without triggering LLM or worker."""
        router, sent, _ = router_with_history
        msg = _make_channel_message("general chatter")
        await router.add_to_history_only(msg)

        assert len(router._history.window) == 1
        assert router._history.window[0].content == "general chatter"
        assert router.state == RouterState.IDLE
        assert len(sent) == 0

    @pytest.mark.asyncio
    async def test_add_to_history_only_no_worker_dispatch(self, router_with_history):
        """add_to_history_only() should not increment worker counter."""
        router, sent, _ = router_with_history
        msg = _make_channel_message("background message")
        await router.add_to_history_only(msg)

        assert router._worker_id_counter == 0

    @pytest.mark.asyncio
    async def test_context_property_returns_copy(self, router_with_history):
        """The context property should return a copy, not a reference."""
        router, sent, _ = router_with_history
        msg = _make_message("test")
        await router.add_to_history_only(msg)

        ctx1 = router.context
        ctx2 = router.context
        assert ctx1 is not ctx2
        assert len(ctx1) == len(ctx2)

    @pytest.mark.asyncio
    async def test_clear_context(self, router_with_history):
        """clear_context() should empty the history."""
        router, sent, _ = router_with_history
        await router.add_to_history_only(_make_message("msg1"))
        await router.add_to_history_only(_make_message("msg2"))
        assert len(router._history.window) == 2

        router.clear_context()
        assert len(router._history.window) == 0


# =============================================================================
# B. Snapshot isolation
# =============================================================================


class TestSnapshotIsolation:
    """Tests for worker snapshot isolation from router history."""

    @pytest.mark.asyncio
    async def test_worker_active_during_busy(self, slow_worker_router):
        """Worker task should be active during BUSY state."""
        router, sent, worker_event, _ = slow_worker_router
        # Add some history first
        await router.add_to_history_only(_make_message("context 1"))
        await router.add_to_history_only(_make_message("context 2"))

        msg = _make_message("do something")
        await router.on_message(msg)
        await asyncio.sleep(0.05)

        # Should have an active worker task
        assert router._worker_task is not None
        assert not router._worker_task.done()

        worker_event.set()
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_messages_during_busy_in_history(self, slow_worker_router):
        """Messages during BUSY go to router history."""
        router, sent, worker_event, received = slow_worker_router
        msg = _make_message("trigger task")
        await router.on_message(msg)
        await asyncio.sleep(0.05)
        assert router.state == RouterState.BUSY

        # Send a message while busy
        msg2 = _make_message("are you there?")
        await router.on_message(msg2)
        await asyncio.sleep(0.05)

        # The new message should be in router history
        contents = [t.content for t in router._history.window]
        assert "are you there?" in contents

        worker_event.set()
        await asyncio.sleep(0.1)


# =============================================================================
# C. Context merge
# =============================================================================


class TestContextMerge:
    """Tests for merging worker context back into router history."""

    @pytest.mark.asyncio
    async def test_merge_adds_worker_origin_tag(self, tmp_path):
        """Merged entries should have meta['worker_origin'] set."""
        sent = []
        worker_turns = []

        async def send_fn(content, in_reply_to=None):
            sent.append({"content": content, "in_reply_to": in_reply_to})

        async def worker_fn(context, trigger, **kwargs):
            # Simulate worker adding entries to the snapshot
            context.append(Turn(
                role="tool",
                content="ran git status",
                timestamp=datetime.now(timezone.utc),
                from_node="tool:git",
                to_node="agent:test:test-bot",
            ))
            context.append(Turn(
                role="assistant",
                content="Here are the results",
                timestamp=datetime.now(timezone.utc),
                from_node="agent:test:test-bot",
                to_node="user:testuser",
            ))
            return WorkerResult(response="Done.", context=context, usage=None, error=None)

        config = RouterV2Config(
            llm_enabled=False,
            history_persist=True,
            history_persist_path=str(tmp_path / "merge-test.json"),
        )
        router = RouterV2(
            worker_fn=worker_fn,
            send_fn=send_fn,
            config=config,
            nickname="test-bot",
            agent_type="test",
            node_id="agent:test:test-bot",
        )
        msg = _make_message("do something")
        await router.on_message(msg)
        await asyncio.sleep(0.2)

        # Find merged entries with worker_origin tag
        worker_entries = [
            t for t in router._history.window
            if t.meta and t.meta.get("worker_origin")
        ]
        assert len(worker_entries) >= 2
        assert all("test-bot-worker1" in t.meta["worker_origin"] for t in worker_entries)

    @pytest.mark.asyncio
    async def test_merge_preserves_content(self, tmp_path):
        """Merged entries should preserve their original content."""
        sent = []

        async def send_fn(content, in_reply_to=None):
            sent.append({"content": content, "in_reply_to": in_reply_to})

        async def worker_fn(context, trigger, **kwargs):
            context.append(Turn(
                role="tool",
                content="tool output: success",
                timestamp=datetime.now(timezone.utc),
                from_node="tool:test",
            ))
            return WorkerResult(response="Done.", context=context, usage=None, error=None)

        config = RouterV2Config(
            llm_enabled=False,
            history_persist=True,
            history_persist_path=str(tmp_path / "preserve-test.json"),
        )
        router = RouterV2(
            worker_fn=worker_fn,
            send_fn=send_fn,
            config=config,
            nickname="test-bot",
            agent_type="test",
            node_id="agent:test:test-bot",
        )
        msg = _make_message("run tests")
        await router.on_message(msg)
        await asyncio.sleep(0.2)

        contents = [t.content for t in router._history.window]
        assert "tool output: success" in contents

    @pytest.mark.asyncio
    async def test_merge_handles_empty_context(self, router_with_history):
        """Worker returning empty context should not crash merge."""
        router, sent, _ = router_with_history
        msg = _make_message("simple question")
        await router.on_message(msg)
        await asyncio.sleep(0.1)

        # Should complete without error
        assert router.state == RouterState.IDLE


# =============================================================================
# D. Persistence
# =============================================================================


class TestPersistence:
    """Tests for history save/load to disk."""

    @pytest.mark.asyncio
    async def test_save_and_load_roundtrip(self, tmp_path):
        """Save history, create new router, load → history restored."""
        history_path = tmp_path / "roundtrip-test.json"

        sent = []
        async def send_fn(content, in_reply_to=None):
            sent.append(content)

        async def worker_fn(context, trigger, **kwargs):
            return WorkerResult(response="Done.", context=[], usage=None, error=None)

        config = RouterV2Config(
            llm_enabled=False,
            history_persist=True,
            history_persist_path=str(history_path),
        )

        # Create first router and add messages
        router1 = RouterV2(
            worker_fn=worker_fn, send_fn=send_fn, config=config,
            nickname="test-bot", agent_type="test", node_id="agent:test:test-bot",
        )
        await router1.add_to_history_only(_make_message("message 1"))
        await router1.add_to_history_only(_make_message("message 2"))
        router1.save_history()

        # Create second router and load
        router2 = RouterV2(
            worker_fn=worker_fn, send_fn=send_fn, config=config,
            nickname="test-bot", agent_type="test", node_id="agent:test:test-bot",
        )
        loaded = router2.load_history()

        assert loaded == 2
        assert len(router2._history.window) == 2
        assert router2._history.window[0].content == "message 1"
        assert router2._history.window[1].content == "message 2"

    @pytest.mark.asyncio
    async def test_history_persisted_after_worker_complete(self, tmp_path):
        """Regression for e5010879: history should be on disk after worker completion."""
        history_path = tmp_path / "persist-complete-test.json"
        sent = []

        async def send_fn(content, in_reply_to=None):
            sent.append(content)

        async def worker_fn(context, trigger, **kwargs):
            return WorkerResult(response="Done.", context=[], usage=None, error=None)

        config = RouterV2Config(
            llm_enabled=False,
            history_persist=True,
            history_persist_path=str(history_path),
        )
        router = RouterV2(
            worker_fn=worker_fn, send_fn=send_fn, config=config,
            nickname="test-bot", agent_type="test", node_id="agent:test:test-bot",
        )
        msg = _make_message("persist me")
        await router.on_message(msg)
        await asyncio.sleep(0.2)

        assert history_path.exists()
        data = json.loads(history_path.read_text())
        # v2 format has "window" key
        if "window" in data:
            contents = [t["content"] for t in data["window"]]
        else:
            contents = [t["content"] for t in data]
        assert "persist me" in contents

    @pytest.mark.asyncio
    async def test_context_restored_from_disk(self, tmp_path):
        """Regression for 802d9de6: new RouterV2 instance sees prior conversation."""
        history_path = tmp_path / "restore-test.json"
        sent = []

        async def send_fn(content, in_reply_to=None):
            sent.append(content)

        async def worker_fn(context, trigger, **kwargs):
            return WorkerResult(response="Done.", context=[], usage=None, error=None)

        config = RouterV2Config(
            llm_enabled=False,
            history_persist=True,
            history_persist_path=str(history_path),
        )

        # First router: add some conversation and save
        router1 = RouterV2(
            worker_fn=worker_fn, send_fn=send_fn, config=config,
            nickname="test-bot", agent_type="test", node_id="agent:test:test-bot",
        )
        await router1.add_to_history_only(_make_message("old message"))
        router1.save_history()

        # Second router: load from disk
        router2 = RouterV2(
            worker_fn=worker_fn, send_fn=send_fn, config=config,
            nickname="test-bot", agent_type="test", node_id="agent:test:test-bot",
        )
        loaded = router2.load_history()
        assert loaded >= 1
        contents = [t.content for t in router2._history.window]
        assert "old message" in contents

    @pytest.mark.asyncio
    async def test_save_after_worker_error(self, tmp_path):
        """History should be persisted even after worker errors."""
        history_path = tmp_path / "error-persist-test.json"
        sent = []

        async def send_fn(content, in_reply_to=None):
            sent.append(content)

        async def worker_fn(context, trigger, **kwargs):
            raise RuntimeError("kaboom")

        config = RouterV2Config(
            llm_enabled=False,
            history_persist=True,
            history_persist_path=str(history_path),
        )
        router = RouterV2(
            worker_fn=worker_fn, send_fn=send_fn, config=config,
            nickname="test-bot", agent_type="test", node_id="agent:test:test-bot",
        )
        msg = _make_message("trigger error")
        await router.on_message(msg)
        await asyncio.sleep(0.2)

        assert history_path.exists()


# =============================================================================
# E. Summarization triggers
# =============================================================================


class TestSummarizationTriggers:
    """Tests for summarization trigger conditions."""

    @pytest.mark.asyncio
    async def test_summarization_not_triggered_below_limit(self, router_with_history):
        """Below soft limit, no summarization should be triggered."""
        router, sent, _ = router_with_history
        # Add a small message — well below any token limit
        await router.add_to_history_only(_make_message("short"))

        # No summarization task should exist
        # (No summarization_client is set, so _check_and_trigger_summarization is a no-op anyway)
        assert not hasattr(router._history, '_summarization_task') or \
               router._history._summarization_task is None

    @pytest.mark.asyncio
    async def test_summarization_not_triggered_without_client(self, router_with_history):
        """Without a summarization client, summarization should never trigger."""
        router, sent, _ = router_with_history
        # Even with lots of history, no summarization without a client
        for i in range(100):
            await router.add_to_history_only(_make_message(f"message {i} " * 100))

        # Should not crash and no summarization task
        assert router._history._summarization_task is None


# =============================================================================
# F. Diagnostics
# =============================================================================


class TestDiagnostics:
    """Tests for get_diagnostics()."""

    def test_diagnostics_idle(self, router_with_history):
        """Diagnostics in IDLE state."""
        router, _, _ = router_with_history
        diag = router.get_diagnostics()
        assert diag["state"] == "idle"
        assert diag["worker_active"] is False
        assert diag["worker_id"] is None

    @pytest.mark.asyncio
    async def test_diagnostics_busy(self, tmp_path):
        """Diagnostics in BUSY state should show worker info."""
        sent = []
        worker_event = asyncio.Event()

        async def send_fn(content, in_reply_to=None):
            sent.append(content)

        async def worker_fn(context, trigger, **kwargs):
            await worker_event.wait()
            return WorkerResult(response="Done.", context=[], usage=None, error=None)

        config = RouterV2Config(
            llm_enabled=False,
            history_persist=True,
            history_persist_path=str(tmp_path / "diag-test.json"),
        )
        router = RouterV2(
            worker_fn=worker_fn, send_fn=send_fn, config=config,
            nickname="test-bot", agent_type="test", node_id="agent:test:test-bot",
        )
        msg = _make_message("do work")
        await router.on_message(msg)
        await asyncio.sleep(0.05)

        diag = router.get_diagnostics()
        assert diag["state"] == "busy"
        assert diag["worker_active"] is True
        assert diag["worker_id"] == "test-bot-worker1"
        assert diag["pending_trigger_from"] == "user:testuser"

        worker_event.set()
        await asyncio.sleep(0.1)
