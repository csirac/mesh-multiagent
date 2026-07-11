"""Tests for verbatim buffered-message delivery on worker completion
(deliver_buffered_verbatim flag) and the double-dispatch guard.

Root cause context (2026-07-08): in RouterV2 full mode with synthesis
enabled, worker send_message calls to the dispatch origin are buffered by
capturing_send; on synthesis SUCCESS the buffer was discarded and only the
router's synthesized description was delivered (Jul 5 weekly digest: 6,501
buffered chars dropped, user received a meta-ack). The fix delivers the
buffered content verbatim — concatenated into ONE message — and skips
synthesis; synthesis still covers the empty-buffer case.

All LLM interaction is mocked; these tests spend zero API tokens.
"""

import asyncio
import json
import time
from datetime import datetime, timezone

import pytest

from mesh.protocol import Message, MessageType
from mesh.router_v2 import RouterState, RouterV2, RouterV2Config, WorkerResult


def _make_trigger() -> Message:
    return Message(
        type=MessageType.MESSAGE,
        from_node="user:testuser",
        to_node="agent:assistant:alice",
        content="[Scheduled Wake — wake-test] do the briefing",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _make_router(deliver_verbatim: bool, synthesized_text: str | None = "SYNTH"):
    """Build a minimally-wired RouterV2 for _handle_worker_complete tests."""
    r = RouterV2.__new__(RouterV2)
    r._state_lock = asyncio.Lock()
    r._state = RouterState.BUSY
    r._current_worker_id = "alice-worker99"
    r._worker_start_time = None
    r._nickname = "alice"
    r._node_id = "agent:assistant:alice"
    r._config = RouterV2Config(
        synthesize_enabled=True,
        deliver_buffered_verbatim=deliver_verbatim,
        trace_as_history_enabled=False,
    )
    r._llm_client = object()  # truthy → synthesis branch is eligible
    r._worker_snapshot = []
    r._worker_snapshot_start = 0
    r._memory = None
    r._flush_tools_already_flushed = 0

    r.sent = []            # records (content, trigger) from _send_and_store
    r.turns = []           # records appended Turns
    r.synth_calls = []     # records synthesis invocations

    r._stop_flush_monitor = lambda: None
    r._cleanup_worker_state = lambda: None
    r.save_history = lambda: None
    r._build_worker_trace = lambda wid, result: "TRACE"
    r._build_worker_digest = lambda wid: "DIGEST"
    r._append_turn = lambda turn: r.turns.append(turn)

    async def _send_and_store(content, trigger, meta=None):
        r.sent.append((content, trigger, meta))

    async def _synthesize(trace_text, trigger):
        r.synth_calls.append(trace_text)
        return synthesized_text

    r._send_and_store = _send_and_store
    r._synthesize_worker_output = _synthesize
    return r


def _result(buffered):
    return WorkerResult(
        response="worker final text",
        context=[],
        buffered_messages=buffered,
    )


def test_flag_on_buffered_delivered_verbatim_no_synthesis():
    r = _make_router(deliver_verbatim=True)
    trig = _make_trigger()
    body = "Full 6,501-char weekly digest content"
    asyncio.run(r._handle_worker_complete(_result([("user:testuser", body)]), trig))

    assert r.synth_calls == [], "synthesis must NOT run when buffer has origin messages"
    assert len(r.sent) == 1, "exactly one message delivered"
    content, _, meta = r.sent[0]
    assert content == body, "buffered content delivered byte-for-byte"
    assert meta and meta.get("verbatim_buffered_delivery") is True


def test_flag_on_empty_buffer_synthesis_runs():
    r = _make_router(deliver_verbatim=True)
    trig = _make_trigger()
    asyncio.run(r._handle_worker_complete(_result([]), trig))

    assert r.synth_calls == ["TRACE"], "synthesis runs when nothing buffered for origin"
    assert len(r.sent) == 1
    assert r.sent[0][0] == "SYNTH"


def test_flag_on_third_party_buffer_only_synthesis_runs():
    # Buffered messages NOT addressed to the origin don't trigger verbatim path.
    r = _make_router(deliver_verbatim=True)
    trig = _make_trigger()
    asyncio.run(r._handle_worker_complete(
        _result([("agent:assistant:alice", "note to self")]), trig))

    assert r.synth_calls == ["TRACE"]
    assert len(r.sent) == 1
    assert r.sent[0][0] == "SYNTH"


def test_flag_off_behavior_unchanged_synthesis_delivered_buffer_discarded():
    r = _make_router(deliver_verbatim=False)
    trig = _make_trigger()
    asyncio.run(r._handle_worker_complete(
        _result([("user:testuser", "the full digest")]), trig))

    assert r.synth_calls == ["TRACE"], "flag off: synthesis runs as before"
    assert len(r.sent) == 1
    assert r.sent[0][0] == "SYNTH", "flag off: synthesized text delivered, buffer discarded"


def test_flag_off_synthesis_failure_fallback_unchanged():
    r = _make_router(deliver_verbatim=False, synthesized_text=None)
    trig = _make_trigger()
    asyncio.run(r._handle_worker_complete(
        _result([("user:testuser", "the full digest")]), trig))

    assert len(r.sent) == 1
    assert r.sent[0][0] == "the full digest", "existing synthesis-failure fallback intact"


def test_flag_on_multiple_buffered_concatenated_in_order_single_delivery():
    r = _make_router(deliver_verbatim=True)
    trig = _make_trigger()
    asyncio.run(r._handle_worker_complete(
        _result([("user:testuser", "FIRST"),
                 ("agent:assistant:alice", "IGNORED-third-party"),
                 ("user:testuser", "SECOND"),
                 ("user:testuser", "THIRD")]), trig))

    assert r.synth_calls == []
    assert len(r.sent) == 1, "N buffered messages → exactly ONE delivery"
    content = r.sent[0][0]
    assert "— message 1 of 3 —" in content
    assert "— message 3 of 3 —" in content
    assert "IGNORED-third-party" not in content
    order = [content.index("FIRST"), content.index("SECOND"), content.index("THIRD")]
    assert order == sorted(order), "messages concatenated in original order"


def test_double_dispatch_guard_refuses_while_worker_running():
    """Jul 6 defect: one router turn dispatched twice (worker_launch tool +
    dispatch_worker directive) — _start_worker must refuse the second call."""
    r = RouterV2.__new__(RouterV2)
    r._nickname = "alice"
    r._state = RouterState.BUSY
    r._current_worker_id = "alice-worker60"

    class _FakeTask:
        def done(self):
            return False

    r._worker_task = _FakeTask()
    launched = asyncio.run(r._start_worker(_make_trigger()))
    assert launched is False, "second dispatch while a worker runs must be refused"
    assert r._current_worker_id == "alice-worker60", "running worker state untouched"


def test_dispatch_guard_allows_after_worker_done():
    """Completed worker task must not block the next dispatch (guard only
    fires on a live task)."""
    r = RouterV2.__new__(RouterV2)
    r._nickname = "alice"
    r._state = RouterState.BUSY

    class _DoneTask:
        def done(self):
            return True

    r._worker_task = _DoneTask()
    # The guard passing means execution proceeds into the launch body, which
    # needs full wiring — stub the first attribute it touches to observe.
    proceeded = {}

    class _Sentinel:
        def __set_name__(self, owner, name):
            pass

    # _start_worker first sets _pending_trigger then _worker_start_time, then
    # touches _worker_id_counter. Give it a counter and intercept at
    # _select_memory_context via config gate off, then fail fast at
    # _build_worker_context.
    r._worker_id_counter = 0
    r._config = RouterV2Config()

    def _boom():
        proceeded["yes"] = True
        raise RuntimeError("stop-here")

    r._build_worker_context = _boom
    with pytest.raises(RuntimeError, match="stop-here"):
        asyncio.run(r._start_worker(_make_trigger()))
    assert proceeded.get("yes"), "guard must not block once the prior task is done"


def test_tool_worker_launch_clean_launch_reports_dispatched():
    """Jul 8 defect (cacc515e): _start_worker's success path fell off the end
    returning None, so _tool_worker_launch reported "already_running" for
    every SUCCESSFUL tool-path launch — routers believed their own dispatches
    were refused and attempted the work inline. The success path must return
    True and the tool result must say "dispatched"."""
    r = RouterV2.__new__(RouterV2)
    r._nickname = "alice"
    r._node_id = "agent:assistant:alice"
    r._state = RouterState.IDLE
    r._worker_task = None
    r._current_worker_id = None
    r._current_task_description = ""
    r._worker_start_time = None
    r._pending_trigger = None
    r._worker_id_counter = 0
    r._config = RouterV2Config()
    r._trigger_nodes = lambda: ("user:testuser", "agent:assistant:alice")
    r._build_worker_context = lambda: []
    r._start_flush_monitor = lambda trigger: None
    r._start_watchdog = lambda trigger: None

    async def _noop_worker(trigger):
        return None

    r._run_worker = _noop_worker

    async def _go():
        raw = await r._tool_worker_launch("Fix the double-print bug")
        await asyncio.sleep(0)  # let the no-op worker task settle
        return raw

    payload = json.loads(asyncio.run(_go()))
    assert payload["status"] == "dispatched", (
        "clean launch must report dispatched, never already_running")
    assert payload["worker_id"] == "alice-worker1"
    assert r._state == RouterState.BUSY
    assert r._current_task_description == "Fix the double-print bug"


def test_tool_worker_launch_refusal_payload_names_running_worker():
    """A GENUINE refusal must be distinguishable from the Jul 8 false-refusal
    bug: the payload carries the running worker's id, task snippet, and
    elapsed time, and must not clobber the running worker's task
    description."""
    r = RouterV2.__new__(RouterV2)
    r._nickname = "alice"
    r._node_id = "agent:assistant:alice"
    r._state = RouterState.BUSY
    r._current_worker_id = "alice-worker7"
    r._current_task_description = "long-running fold review"
    r._worker_start_time = time.monotonic() - 90

    class _LiveTask:
        def done(self):
            return False

    r._worker_task = _LiveTask()
    r._trigger_nodes = lambda: ("user:testuser", "agent:assistant:alice")

    payload = json.loads(asyncio.run(r._tool_worker_launch("a new task")))
    assert payload["status"] == "already_running"
    assert payload["running_worker_id"] == "alice-worker7"
    assert payload["running_worker_task"] == "long-running fold review"
    assert payload["running_worker_elapsed_seconds"] >= 89
    assert r._current_task_description == "long-running fold review", (
        "refused dispatch must not overwrite the running worker's task")
