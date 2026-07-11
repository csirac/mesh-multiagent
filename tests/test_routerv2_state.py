"""Tests for RouterV2 state machine, worker lifecycle, and concurrency."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from mesh.router_v2 import RouterV2, RouterV2Config, WorkerResult, RouterState
from mesh.protocol import Message, MessageType


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


class MockLLM:
    """Configurable mock LLM that returns preset responses."""

    def __init__(self, responses=None):
        self.responses = responses or [
            '{"needs_response": true, "needs_worker": false, "response": "Hello"}'
        ]
        self._call_count = 0
        self.calls = []

    async def complete(self, prompt, **kwargs):
        self.calls.append(prompt)
        response = self.responses[min(self._call_count, len(self.responses) - 1)]
        self._call_count += 1
        return response


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def state_router(tmp_path):
    """RouterV2 with non-LLM mode for state machine tests."""
    sent = []

    async def send_fn(content, in_reply_to=None):
        sent.append({"content": content, "in_reply_to": in_reply_to})

    async def worker_fn(context, trigger, **kwargs):
        return WorkerResult(response="Done.", context=[], usage=None, error=None)

    config = RouterV2Config(
        llm_enabled=False,
        history_persist=True,
        history_persist_path=str(tmp_path / "state-test-history.json"),
    )
    router = RouterV2(
        worker_fn=worker_fn,
        send_fn=send_fn,
        config=config,
        nickname="test-bot",
        agent_type="test",
        node_id="agent:test:test-bot",
    )
    return router, sent


@pytest.fixture
def llm_router(tmp_path):
    """RouterV2 with mock LLM for LLM-enabled state tests."""
    sent = []

    async def send_fn(content, in_reply_to=None):
        sent.append({"content": content, "in_reply_to": in_reply_to})

    async def worker_fn(context, trigger, **kwargs):
        return WorkerResult(response="Done.", context=[], usage=None, error=None)

    llm = MockLLM([
        '{"needs_response": true, "needs_worker": true, "response": "On it."}',
    ])
    config = RouterV2Config(
        llm_enabled=True,
        history_persist=True,
        history_persist_path=str(tmp_path / "llm-state-test-history.json"),
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


@pytest.fixture
def slow_worker_router(tmp_path):
    """RouterV2 with a slow worker for testing busy state."""
    sent = []
    worker_event = asyncio.Event()

    async def send_fn(content, in_reply_to=None):
        sent.append({"content": content, "in_reply_to": in_reply_to})

    async def worker_fn(context, trigger, **kwargs):
        await worker_event.wait()
        return WorkerResult(response="Done.", context=[], usage=None, error=None)

    config = RouterV2Config(
        llm_enabled=False,
        history_persist=True,
        history_persist_path=str(tmp_path / "slow-worker-history.json"),
    )
    router = RouterV2(
        worker_fn=worker_fn,
        send_fn=send_fn,
        config=config,
        nickname="test-bot",
        agent_type="test",
        node_id="agent:test:test-bot",
    )
    return router, sent, worker_event


@pytest.fixture
def error_worker_router(tmp_path):
    """RouterV2 with a worker that raises an error."""
    sent = []

    async def send_fn(content, in_reply_to=None):
        sent.append({"content": content, "in_reply_to": in_reply_to})

    async def worker_fn(context, trigger, **kwargs):
        raise RuntimeError("Worker exploded")

    config = RouterV2Config(
        llm_enabled=False,
        history_persist=True,
        history_persist_path=str(tmp_path / "error-worker-history.json"),
    )
    router = RouterV2(
        worker_fn=worker_fn,
        send_fn=send_fn,
        config=config,
        nickname="test-bot",
        agent_type="test",
        node_id="agent:test:test-bot",
    )
    return router, sent


# =============================================================================
# A. State transitions
# =============================================================================


class TestStateTransitions:
    """Tests for IDLE/BUSY state transitions."""

    @pytest.mark.asyncio
    async def test_initial_state_is_idle(self, state_router):
        router, sent = state_router
        assert router.state == RouterState.IDLE

    @pytest.mark.asyncio
    async def test_idle_to_busy_on_worker_dispatch(self, state_router):
        """Non-LLM mode always dispatches a worker, transitioning IDLE → BUSY."""
        router, sent = state_router
        msg = _make_message("do something")
        await router.on_message(msg)
        # Worker starts and runs to completion very quickly
        # Give event loop a tick to let the worker task start
        await asyncio.sleep(0.05)
        # Worker already completed since it's instant — should be back to IDLE
        assert router.state == RouterState.IDLE

    @pytest.mark.asyncio
    async def test_busy_to_idle_on_worker_complete(self, slow_worker_router):
        """Worker completion transitions BUSY → IDLE."""
        router, sent, worker_event = slow_worker_router
        msg = _make_message("do something")
        await router.on_message(msg)
        await asyncio.sleep(0.05)
        assert router.state == RouterState.BUSY

        # Let the worker finish
        worker_event.set()
        await asyncio.sleep(0.1)
        assert router.state == RouterState.IDLE

    @pytest.mark.asyncio
    async def test_busy_to_idle_on_worker_error(self, error_worker_router):
        """Worker error transitions BUSY → IDLE."""
        router, sent = error_worker_router
        msg = _make_message("do something")
        await router.on_message(msg)
        await asyncio.sleep(0.1)
        assert router.state == RouterState.IDLE

    @pytest.mark.asyncio
    async def test_busy_to_idle_on_cancel(self, slow_worker_router):
        """cancel_worker() transitions BUSY → IDLE."""
        router, sent, worker_event = slow_worker_router
        msg = _make_message("do something")
        await router.on_message(msg)
        await asyncio.sleep(0.05)
        assert router.state == RouterState.BUSY

        cancelled = await router.cancel_worker()
        assert cancelled is True
        assert router.state == RouterState.IDLE

    @pytest.mark.asyncio
    async def test_stays_idle_on_direct_response(self, llm_router):
        """LLM classifies needs_worker=false → state stays IDLE."""
        router, llm, sent = llm_router
        # Override LLM to return needs_worker=false
        llm.responses = [
            '{"needs_response": true, "needs_worker": false, "response": "Hello there!"}'
        ]
        msg = _make_message("hello")
        await router.on_message(msg)
        await asyncio.sleep(0.05)
        assert router.state == RouterState.IDLE
        assert any("Hello there!" in s["content"] for s in sent)

    @pytest.mark.asyncio
    async def test_stays_idle_on_no_response(self, llm_router):
        """LLM classifies needs_response=false → state stays IDLE, nothing sent."""
        router, llm, sent = llm_router
        llm.responses = ['{"needs_response": false}']
        msg = _make_message("some noise")
        await router.on_message(msg)
        await asyncio.sleep(0.05)
        assert router.state == RouterState.IDLE
        assert len(sent) == 0


# =============================================================================
# B. Worker lifecycle
# =============================================================================


class TestWorkerLifecycle:
    """Tests for worker ID tracking, snapshots, and completion handling."""

    @pytest.mark.asyncio
    async def test_worker_id_increments(self, state_router):
        """Worker IDs should increment: test-bot-worker1, test-bot-worker2."""
        router, sent = state_router
        msg1 = _make_message("task 1")
        await router.on_message(msg1)
        await asyncio.sleep(0.1)
        assert router._worker_id_counter == 1

        msg2 = _make_message("task 2")
        await router.on_message(msg2)
        await asyncio.sleep(0.1)
        assert router._worker_id_counter == 2

    @pytest.mark.asyncio
    async def test_worker_receives_snapshot(self, tmp_path):
        """Worker fn receives a list of Turn objects (the snapshot)."""
        received_context = []

        async def send_fn(content, in_reply_to=None):
            pass

        async def worker_fn(context, trigger, **kwargs):
            received_context.extend(context)
            return WorkerResult(response="Done.", context=[], usage=None, error=None)

        config = RouterV2Config(
            llm_enabled=False,
            history_persist=True,
            history_persist_path=str(tmp_path / "snapshot-test.json"),
        )
        router = RouterV2(
            worker_fn=worker_fn,
            send_fn=send_fn,
            config=config,
            nickname="test-bot",
            agent_type="test",
            node_id="agent:test:test-bot",
        )
        msg = _make_message("hello")
        await router.on_message(msg)
        await asyncio.sleep(0.1)

        # Worker should have received a snapshot with at least the trigger message
        assert len(received_context) >= 1
        assert any("hello" in str(t.content) for t in received_context)

    @pytest.mark.asyncio
    async def test_ack_sent_before_worker(self, llm_router):
        """With LLM enabled, ack is sent before worker starts."""
        router, llm, sent = llm_router
        # LLM returns needs_worker=true with an ack
        llm.responses = [
            '{"needs_response": true, "needs_worker": true, "response": "On it."}',
            # Second call is for completion response
            "Here are the results.",
        ]
        msg = _make_message("do something complex")
        await router.on_message(msg)
        await asyncio.sleep(0.2)

        # First sent message should be the ack
        assert len(sent) >= 1
        assert sent[0]["content"] == "On it."

    @pytest.mark.asyncio
    async def test_completion_transitions_to_idle(self, state_router):
        """Worker completion transitions router back to IDLE (direct-send: worker sends during execution)."""
        router, sent = state_router
        msg = _make_message("do something")
        await router.on_message(msg)
        await asyncio.sleep(0.1)

        # With direct-send, the router doesn't send on completion.
        # The worker is responsible for sending during execution.
        # Test verifies the state transition.
        assert router.state == RouterState.IDLE
        assert router._worker_task is None

    @pytest.mark.asyncio
    async def test_worker_error_sends_error_msg(self, error_worker_router):
        """Worker error sends an error message to the user."""
        router, sent = error_worker_router
        msg = _make_message("do something")
        await router.on_message(msg)
        await asyncio.sleep(0.1)

        assert len(sent) >= 1
        assert any("error" in s["content"].lower() for s in sent)
        assert any("Worker exploded" in s["content"] for s in sent)

    @pytest.mark.asyncio
    async def test_cleanup_after_completion(self, state_router):
        """After worker completes, worker state is cleaned up."""
        router, sent = state_router
        msg = _make_message("do something")
        await router.on_message(msg)
        await asyncio.sleep(0.1)

        assert router._worker_task is None


# =============================================================================
# C. Busy handling
# =============================================================================


class TestBusyHandling:
    """Tests for message handling while worker is busy."""

    @pytest.mark.asyncio
    async def test_busy_non_llm_sends_canned(self, slow_worker_router):
        """Message while BUSY (non-LLM) sends a canned busy response."""
        router, sent, worker_event = slow_worker_router
        # Start a worker
        msg1 = _make_message("first task")
        await router.on_message(msg1)
        await asyncio.sleep(0.05)
        assert router.state == RouterState.BUSY

        # Send another message while busy
        msg2 = _make_message("second task")
        await router.on_message(msg2)
        await asyncio.sleep(0.05)

        # Should get a busy response
        assert any("finish" in s["content"].lower() or "first" in s["content"].lower() for s in sent)

        worker_event.set()
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_cancel_during_busy(self, slow_worker_router):
        """Cancel request while BUSY cancels worker and sends confirmation."""
        router, sent, worker_event = slow_worker_router
        msg1 = _make_message("first task")
        await router.on_message(msg1)
        await asyncio.sleep(0.05)
        assert router.state == RouterState.BUSY

        cancel_msg = _make_message("stop the worker")
        await router.on_message(cancel_msg)
        await asyncio.sleep(0.1)

        assert router.state == RouterState.IDLE
        assert any("cancel" in s["content"].lower() for s in sent)

    @pytest.mark.asyncio
    async def test_status_during_busy(self, slow_worker_router):
        """Status query while BUSY sends elapsed time info."""
        router, sent, worker_event = slow_worker_router
        msg1 = _make_message("first task")
        await router.on_message(msg1)
        await asyncio.sleep(0.05)
        assert router.state == RouterState.BUSY

        status_msg = _make_message("status")
        await router.on_message(status_msg)
        await asyncio.sleep(0.05)

        assert any("working" in s["content"].lower() or "elapsed" in s["content"].lower() for s in sent)

        worker_event.set()
        await asyncio.sleep(0.1)


# =============================================================================
# D. Concurrency & locking
# =============================================================================


class TestConcurrency:
    """Tests for state lock and race condition prevention."""

    @pytest.mark.asyncio
    async def test_no_double_dispatch(self, slow_worker_router):
        """Two rapid messages while IDLE → only one worker started."""
        router, sent, worker_event = slow_worker_router
        msg1 = _make_message("task 1")
        msg2 = _make_message("task 2")

        # Send both quickly
        await asyncio.gather(
            router.on_message(msg1),
            router.on_message(msg2),
        )
        await asyncio.sleep(0.05)

        # Only one worker should have been dispatched
        assert router._worker_id_counter == 1

        worker_event.set()
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_cancel_no_deadlock(self, slow_worker_router):
        """cancel_worker() from outside should complete without deadlock."""
        router, sent, worker_event = slow_worker_router
        msg = _make_message("do something")
        await router.on_message(msg)
        await asyncio.sleep(0.05)

        # cancel_worker acquires the lock itself — should not deadlock
        try:
            cancelled = await asyncio.wait_for(router.cancel_worker(), timeout=2.0)
            assert cancelled is True
        except asyncio.TimeoutError:
            pytest.fail("cancel_worker() deadlocked")

    @pytest.mark.asyncio
    async def test_cancel_from_on_message_no_deadlock(self, slow_worker_router):
        """Regression for c6c10429: cancel via on_message uses _cancel_worker_unlocked."""
        router, sent, worker_event = slow_worker_router
        msg = _make_message("do something")
        await router.on_message(msg)
        await asyncio.sleep(0.05)

        # Cancel from on_message (which already holds the lock)
        cancel_msg = _make_message("cancel the worker")
        try:
            await asyncio.wait_for(router.on_message(cancel_msg), timeout=2.0)
        except asyncio.TimeoutError:
            pytest.fail("Cancel from on_message deadlocked (c6c10429 regression)")

        assert router.state == RouterState.IDLE


# =============================================================================
# E. Error handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in classification and worker."""

    @pytest.mark.asyncio
    async def test_llm_classification_error_on_channel_is_silent(self, tmp_path):
        """Classification error on channel message → silent failure."""
        sent = []

        async def send_fn(content, in_reply_to=None):
            sent.append({"content": content, "in_reply_to": in_reply_to})

        async def worker_fn(context, trigger, **kwargs):
            return WorkerResult(response="Done.", context=[], usage=None, error=None)

        llm = MockLLM()
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
        config = RouterV2Config(
            llm_enabled=True,
            history_persist=True,
            history_persist_path=str(tmp_path / "err-channel-history.json"),
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
        msg = _make_channel_message("@test-bot help")
        await router.on_message(msg)
        await asyncio.sleep(0.1)

        # Should stay silent on channel errors
        assert len(sent) == 0

    @pytest.mark.asyncio
    async def test_llm_classification_error_on_dm_sends_notice(self, tmp_path):
        """Classification error on DM → error notice sent, fallback to worker."""
        sent = []

        async def send_fn(content, in_reply_to=None):
            sent.append({"content": content, "in_reply_to": in_reply_to})

        async def worker_fn(context, trigger, **kwargs):
            return WorkerResult(response="Done.", context=[], usage=None, error=None)

        llm = MockLLM()
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
        config = RouterV2Config(
            llm_enabled=True,
            history_persist=True,
            history_persist_path=str(tmp_path / "err-dm-history.json"),
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
        msg = _make_message("help me")
        await router.on_message(msg)
        await asyncio.sleep(0.2)

        # Error notice should have been sent
        assert any("error" in s["content"].lower() or "LLM" in s["content"] for s in sent)
        # Worker should have been dispatched as fallback
        assert router._worker_id_counter >= 1

    @pytest.mark.asyncio
    async def test_worker_error_persists_history(self, tmp_path):
        """Worker error should persist history to disk."""
        sent = []

        async def send_fn(content, in_reply_to=None):
            sent.append({"content": content, "in_reply_to": in_reply_to})

        async def worker_fn(context, trigger, **kwargs):
            raise RuntimeError("Worker exploded")

        history_path = tmp_path / "persist-error-history.json"
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
        msg = _make_message("do something")
        await router.on_message(msg)
        await asyncio.sleep(0.2)

        # History file should exist on disk (regression for e5010879)
        assert history_path.exists()

    @pytest.mark.asyncio
    async def test_save_failure_doesnt_crash_completion(self, tmp_path):
        """save_history() error should not crash worker completion."""
        sent = []

        async def send_fn(content, in_reply_to=None):
            sent.append({"content": content, "in_reply_to": in_reply_to})

        async def worker_fn(context, trigger, **kwargs):
            return WorkerResult(response="Done.", context=[], usage=None, error=None)

        config = RouterV2Config(
            llm_enabled=False,
            history_persist=True,
            history_persist_path=str(tmp_path / "crash-test-history.json"),
        )
        router = RouterV2(
            worker_fn=worker_fn,
            send_fn=send_fn,
            config=config,
            nickname="test-bot",
            agent_type="test",
            node_id="agent:test:test-bot",
        )

        # Make save_history raise
        original_save = router.save_history
        def broken_save():
            raise IOError("Disk full")
        router.save_history = broken_save

        msg = _make_message("do something")
        await router.on_message(msg)
        await asyncio.sleep(0.2)

        # Worker should still complete (direct-send: router doesn't send on completion)
        assert router.state == RouterState.IDLE
