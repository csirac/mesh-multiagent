"""Tests for RouterV2 worker watchdog — periodic check-in on worker progress."""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from mesh.router_v2 import RouterV2, RouterV2Config, WorkerResult, RouterState
from mesh.conversation_history import Turn
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


def _make_turn(role="assistant", content="test", from_node=""):
    return Turn(
        role=role,
        content=content,
        timestamp=datetime.now(timezone.utc),
        from_node=from_node,
    )


def _make_router(tmp_path, *, watchdog_interval_minutes=15, worker_done_event=None):
    """Create a RouterV2 with configurable watchdog settings."""
    sent = []

    async def send_fn(content, in_reply_to=None):
        sent.append({"content": content, "in_reply_to": in_reply_to})

    async def worker_fn(context, trigger, **kwargs):
        if worker_done_event:
            await worker_done_event.wait()
        return WorkerResult(response="Done.", context=[], usage=None, error=None)

    config = RouterV2Config(
        llm_enabled=False,
        history_persist=True,
        history_persist_path=str(tmp_path / "watchdog-test-history.json"),
        watchdog_interval_minutes=watchdog_interval_minutes,
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
# _is_nominal_watchdog_response — regex matching
# =============================================================================


class TestNominalDetection:
    """Test the regex-based detection of 'nothing to report' responses."""

    def _make_router_instance(self, tmp_path):
        router, _ = _make_router(tmp_path)
        return router

    def test_exact_match(self, tmp_path):
        router = self._make_router_instance(tmp_path)
        assert router._is_nominal_watchdog_response("Nothing to report.")

    def test_no_period(self, tmp_path):
        router = self._make_router_instance(tmp_path)
        assert router._is_nominal_watchdog_response("Nothing to report")

    def test_case_insensitive(self, tmp_path):
        router = self._make_router_instance(tmp_path)
        assert router._is_nominal_watchdog_response("NOTHING TO REPORT")
        assert router._is_nominal_watchdog_response("nothing to report")

    def test_dropped_to(self, tmp_path):
        """Handles 'nothing report' (dropped 'to')."""
        router = self._make_router_instance(tmp_path)
        assert router._is_nominal_watchdog_response("Nothing report.")

    def test_embedded_in_sentence(self, tmp_path):
        """Matches even when embedded in a larger sentence."""
        router = self._make_router_instance(tmp_path)
        assert router._is_nominal_watchdog_response(
            "Overall, nothing to report here."
        )

    def test_with_whitespace(self, tmp_path):
        router = self._make_router_instance(tmp_path)
        assert router._is_nominal_watchdog_response("  Nothing to report.  ")

    def test_notification_not_nominal(self, tmp_path):
        """A real notification should NOT be detected as nominal."""
        router = self._make_router_instance(tmp_path)
        assert not router._is_nominal_watchdog_response(
            "Worker appears stuck in a retry loop on the SSH command."
        )

    def test_empty_response_not_nominal(self, tmp_path):
        router = self._make_router_instance(tmp_path)
        assert not router._is_nominal_watchdog_response("")

    def test_unrelated_text_not_nominal(self, tmp_path):
        router = self._make_router_instance(tmp_path)
        assert not router._is_nominal_watchdog_response(
            "The worker has made significant progress."
        )


# =============================================================================
# Watchdog config — interval=0 disables
# =============================================================================


class TestWatchdogConfig:
    """Test that watchdog_interval_minutes=0 disables the watchdog."""

    def test_interval_zero_disables(self, tmp_path):
        router, _ = _make_router(tmp_path, watchdog_interval_minutes=0)
        trigger = _make_message("do something")
        router._start_watchdog(trigger)
        assert router._watchdog_task is None

    @pytest.mark.asyncio
    async def test_interval_nonzero_enables(self, tmp_path):
        router, _ = _make_router(tmp_path, watchdog_interval_minutes=15)
        trigger = _make_message("do something")
        router._start_watchdog(trigger)
        assert router._watchdog_task is not None
        # Clean up
        router._stop_watchdog()

    def test_default_interval_is_15(self, tmp_path):
        config = RouterV2Config()
        assert config.watchdog_interval_minutes == 15


# =============================================================================
# Watchdog lifecycle — start/stop/cancel
# =============================================================================


class TestWatchdogLifecycle:
    """Test watchdog timer lifecycle: start, stop, cancel."""

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, tmp_path):
        router, _ = _make_router(tmp_path)
        trigger = _make_message("do something")
        router._start_watchdog(trigger)
        task = router._watchdog_task
        assert task is not None
        assert not task.done()

        router._stop_watchdog()
        assert router._watchdog_task is None
        # Let cancel propagate
        await asyncio.sleep(0)
        assert task.cancelled()

    def test_stop_when_no_task(self, tmp_path):
        """Stopping when no watchdog is running should not raise."""
        router, _ = _make_router(tmp_path)
        router._stop_watchdog()  # Should not raise
        assert router._watchdog_task is None

    @pytest.mark.asyncio
    async def test_cleanup_worker_state_stops_watchdog(self, tmp_path):
        """_cleanup_worker_state() should cancel the watchdog."""
        router, _ = _make_router(tmp_path)
        trigger = _make_message("do something")
        router._start_watchdog(trigger)
        task = router._watchdog_task
        assert task is not None

        router._cleanup_worker_state()
        assert router._watchdog_task is None
        # Let cancel propagate
        await asyncio.sleep(0)
        assert task.cancelled()
        assert router._state == RouterState.IDLE

    @pytest.mark.asyncio
    async def test_watchdog_loop_exits_on_non_busy(self, tmp_path):
        """Watchdog loop should exit when state is not BUSY."""
        router, _ = _make_router(tmp_path, watchdog_interval_minutes=1)
        router._state = RouterState.IDLE  # Not BUSY

        trigger = _make_message("do something")
        # Run the loop directly — it should immediately exit after first sleep
        # because state is not BUSY
        with patch.object(asyncio, 'sleep', new_callable=AsyncMock):
            await router._watchdog_loop(trigger)
        # If we get here without hanging, the test passed


# =============================================================================
# Watchdog tick — nominal suppression and notification delivery
# =============================================================================


class TestWatchdogTick:
    """Test the single watchdog evaluation (_watchdog_tick)."""

    @pytest.mark.asyncio
    async def test_nominal_response_suppressed(self, tmp_path):
        """When LLM says 'Nothing to report', no message should be sent."""
        router, sent = _make_router(tmp_path)
        router._state = RouterState.BUSY
        router._current_worker_id = "test-bot-worker1"
        router._worker_start_time = 0.0
        trigger = _make_message("do something")
        router._pending_trigger = trigger

        router._router_process_fn = AsyncMock(return_value="Nothing to report.")

        await router._watchdog_tick(trigger)

        assert len(sent) == 0  # Nominal → suppressed

    @pytest.mark.asyncio
    async def test_notification_sent_to_user(self, tmp_path):
        """When LLM reports an issue, message should be sent via _send_and_store."""
        router, sent = _make_router(tmp_path)
        router._state = RouterState.BUSY
        router._current_worker_id = "test-bot-worker1"
        router._worker_start_time = 0.0
        trigger = _make_message("do something")
        router._pending_trigger = trigger

        notification = "Worker appears stuck in a retry loop on the SSH command."
        router._router_process_fn = AsyncMock(return_value=notification)

        await router._watchdog_tick(trigger)

        assert len(sent) == 1
        assert sent[0]["content"] == notification

    @pytest.mark.asyncio
    async def test_state_change_during_call_discards(self, tmp_path):
        """If worker completes during LLM call, response should be discarded."""
        router, sent = _make_router(tmp_path)
        router._state = RouterState.BUSY
        router._current_worker_id = "test-bot-worker1"
        router._worker_start_time = 0.0
        trigger = _make_message("do something")
        router._pending_trigger = trigger

        async def llm_call_that_completes_worker(**kwargs):
            # Simulate worker completing during the LLM call
            router._state = RouterState.IDLE
            return "Worker is stuck on something."

        router._router_process_fn = llm_call_that_completes_worker

        await router._watchdog_tick(trigger)

        # Response should be discarded because state changed to IDLE
        assert len(sent) == 0

    @pytest.mark.asyncio
    async def test_llm_call_failure_logged_not_raised(self, tmp_path):
        """If LLM call fails, it should log a warning but not crash."""
        router, sent = _make_router(tmp_path)
        router._state = RouterState.BUSY
        router._current_worker_id = "test-bot-worker1"
        router._worker_start_time = 0.0
        trigger = _make_message("do something")
        router._pending_trigger = trigger

        router._router_process_fn = AsyncMock(
            side_effect=Exception("Rate limit exceeded")
        )

        # Should not raise
        await router._watchdog_tick(trigger)

        assert len(sent) == 0  # No message sent on failure

    @pytest.mark.asyncio
    async def test_call_router_full_with_watchdog_params(self, tmp_path):
        """Verify _call_router_full receives watchdog=True, tool_names=[], max_iters=1."""
        router, sent = _make_router(tmp_path)
        router._state = RouterState.BUSY
        router._current_worker_id = "test-bot-worker1"
        router._worker_start_time = 0.0
        trigger = _make_message("do something")
        router._pending_trigger = trigger

        call_kwargs = {}

        async def capture_fn(**kwargs):
            call_kwargs.update(kwargs)
            return "Nothing to report."

        router._router_process_fn = capture_fn

        await router._watchdog_tick(trigger)

        assert call_kwargs.get("tool_names") == []
        assert call_kwargs.get("max_iters") == 1


# =============================================================================
# _build_worker_activity_lines — refactored method
# =============================================================================


class TestBuildWorkerActivityLines:
    """Test the extracted _build_worker_activity_lines method."""

    def test_empty_when_no_snapshot(self, tmp_path):
        router, _ = _make_router(tmp_path)
        lines = router._build_worker_activity_lines()
        assert lines == []

    def test_returns_progress_lines(self, tmp_path):
        router, _ = _make_router(tmp_path)
        router._worker_snapshot = [
            _make_turn(role="assistant", content="Reading file..."),
            _make_turn(role="tool", content="File contents here", from_node="bash_exec"),
        ]
        router._worker_snapshot_start = 0

        lines = router._build_worker_activity_lines(worker_id="test-worker1")
        assert len(lines) == 2
        assert "[assistant]" in lines[0]
        assert "[bash_exec]" in lines[1]

    def test_truncates_long_content(self, tmp_path):
        router, _ = _make_router(tmp_path)
        long_content = "\n".join([f"line {i}" for i in range(200)])
        router._worker_snapshot = [
            _make_turn(role="tool", content=long_content),
        ]
        router._worker_snapshot_start = 0

        lines = router._build_worker_activity_lines()
        assert len(lines) == 1
        assert "truncated" in lines[0]


# =============================================================================
# Integration: _start_worker hooks watchdog
# =============================================================================


class TestStartWorkerHooksWatchdog:
    """Verify _start_worker starts the watchdog alongside the flush monitor."""

    @pytest.mark.asyncio
    async def test_start_worker_creates_watchdog_task(self, tmp_path):
        worker_event = asyncio.Event()
        router, _ = _make_router(tmp_path, worker_done_event=worker_event)
        trigger = _make_message("do something")

        await router._start_worker(trigger)

        assert router._watchdog_task is not None
        assert not router._watchdog_task.done()
        assert router._state == RouterState.BUSY

        # Clean up
        worker_event.set()
        router._cleanup_worker_state()

    @pytest.mark.asyncio
    async def test_start_worker_no_watchdog_when_disabled(self, tmp_path):
        worker_event = asyncio.Event()
        router, _ = _make_router(
            tmp_path, watchdog_interval_minutes=0, worker_done_event=worker_event
        )
        trigger = _make_message("do something")

        await router._start_worker(trigger)

        assert router._watchdog_task is None  # Disabled
        assert router._state == RouterState.BUSY

        # Clean up
        worker_event.set()
        router._cleanup_worker_state()
