"""Tests for the native harness session mode.

Covers:
- protocol session.* event roundtrip
- HarnessSessionManager wiring into RouterV2 (tool exposure, worker-launch
  removal, dispatch_worker hard gate, subprocess argv construction)
- the session subcommand lifecycle end-to-end as a real subprocess, driven over
  stdin with NO LLM dependency (start idle → status → abort → clean exit)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from unittest.mock import AsyncMock

import pytest

from mesh.harness import protocol as hp
from mesh.router_v2 import (
    RouterV2, RouterV2Config, WorkerResult,
    HARNESS_SESSION_INTERACTIVE_TOOLS,
)
from mesh.llm import HistoryMessage, LLMClient, LLMConfig
from mesh.protocol import Message, MessageType


# ---------------------------------------------------------------------------
# protocol roundtrip
# ---------------------------------------------------------------------------

def test_session_event_types_registered():
    for t in ("session.started", "session.awaiting_input", "session.checkpoint",
              "session.context_exhausted", "session.reset_ack", "session.status"):
        assert t in hp.EventType.__args__


def test_session_event_emit_parse_roundtrip(capsys):
    hp.emit_session_awaiting_input("final text", 5, {"input_tokens": 10})
    line = capsys.readouterr().out.strip()
    ev = hp.parse_event(line)
    assert ev is not None
    assert ev.type == "session.awaiting_input"
    assert ev.data["final_text"] == "final text"
    assert ev.data["iteration"] == 5


# ---------------------------------------------------------------------------
# RouterV2 wiring
# ---------------------------------------------------------------------------

def _make_router(tmp_path, *, enabled: bool, cfg=None):
    async def noop_send(*a, **k):
        pass

    async def noop_worker(context, trigger):
        return WorkerResult(response="Done.", context=[], usage=None, error=None)

    config = RouterV2Config(
        llm_enabled=False, history_persist=True,
        history_persist_path=str(tmp_path / "h.json"),
    )
    return RouterV2(
        worker_fn=noop_worker, send_fn=noop_send, config=config,
        nickname="t", agent_type="test", node_id="agent:test:t",
        harness_session_tools=enabled, harness_session_llm_config=cfg,
    )


def _session_cfg():
    cfg = LLMConfig(backend="mesh-harness", model="local-27b")
    cfg.harness_backend = "openai"
    cfg.harness_base_url = "http://localhost:8002/v1"
    cfg.harness_api_key = "dummy"
    cfg.harness_toolset = "harness"
    cfg.harness_soft_limit = 200_000
    return cfg


class _RouterLLM:
    def __init__(self, backend: str):
        self.config = LLMConfig(backend=backend, model="test-model")


def _trigger():
    return Message(
        type=MessageType.MESSAGE,
        from_node="user:testuser",
        to_node="agent:test:t",
        content="launch a worker",
    )


def test_enabled_exposes_tools_and_removes_worker_launch(tmp_path):
    r = _make_router(tmp_path, enabled=True, cfg=_session_cfg())
    names = set(r._router_tool_names)
    assert HARNESS_SESSION_INTERACTIVE_TOOLS <= names
    assert "worker_launch" not in names
    assert "worker_status" not in names
    # stop is registered as a handler but only exposed in monitor mode
    assert "harness_stop_session" not in names
    for t in ("harness_start_session", "harness_send_input",
              "harness_get_status", "harness_stop_session"):
        assert t in r._worker_tool_handlers
    assert r._harness_session_enabled is True


def test_disabled_leaves_worker_launch_and_no_session_tools(tmp_path):
    r = _make_router(tmp_path, enabled=False)
    names = set(r._router_tool_names)
    assert "harness_start_session" not in names
    assert "worker_launch" in names
    assert r._harness_session_enabled is False


@pytest.mark.parametrize("backend", ["claude-code", "codex"])
def test_harness_router_uses_consistent_block_dispatch(tmp_path, backend):
    r = _make_router(tmp_path, enabled=False)
    r._config.router_mode = "full"
    r._llm_client = _RouterLLM(backend)
    captured = {}

    async def capture_fn(**kwargs):
        captured.update(kwargs)
        return (
            "On it.\n"
            "<dispatch_worker>\n"
            "task: inspect the repository read-only\n"
            "</dispatch_worker>"
        )

    r._router_process_fn = capture_fn
    r._start_worker = AsyncMock()

    asyncio.run(r._handle_idle_with_llm(_trigger()))

    names = set(captured["tool_names"])
    assert "worker_launch" not in names
    assert "worker_status" not in names
    assert captured["max_iters"] == 1
    assert "worker_launch" not in captured["instructions"]
    assert "<dispatch_worker>" in captured["instructions"]
    assert r._current_task_description == "inspect the repository read-only"
    r._start_worker.assert_awaited_once()


def test_codex_long_history_ends_with_dispatch_reminder(tmp_path):
    r = _make_router(tmp_path, enabled=False)
    r._llm_client = _RouterLLM("codex")
    captured = {}

    async def capture_fn(**kwargs):
        captured.update(kwargs)
        return "ok"

    r._router_process_fn = capture_fn
    trigger = _trigger()
    assert asyncio.run(r._call_router_full(trigger)) == "ok"

    history = [
        HistoryMessage(
            from_node="agent:test:t" if i % 2 else "user:testuser",
            to_node="user:testuser" if i % 2 else "agent:test:t",
            content=(
                "Prior failed launcher narrative: collab spawn failed; "
                f"historical turn {i}."
            ),
            timestamp=f"2026-07-11 11:{i:02d} CDT",
        )
        for i in range(54)
    ]
    history.append(
        HistoryMessage(
            from_node=trigger.from_node,
            to_node=trigger.to_node,
            content=trigger.content,
            timestamp="2026-07-11 12:14 CDT",
        )
    )
    prompt = LLMClient.format_history_xml(
        None,
        history,
        "agent:test:t",
        system_prompt=captured["system_prompt"],
        instructions=captured["instructions"],
        trigger_msg=trigger,
    )

    reminder = "FINAL ROUTER-DISPATCH RULE"
    assert reminder in captured["instructions"]
    assert captured["instructions"].rstrip().endswith("</dispatch_worker>")
    assert prompt.rindex(reminder) > prompt.rindex("</message_received>")
    assert prompt.rstrip().endswith("</instructions>")


def test_direct_router_advertises_offered_worker_launch(tmp_path):
    r = _make_router(tmp_path, enabled=False)
    r._llm_client = _RouterLLM("openai")
    captured = {}

    async def capture_fn(**kwargs):
        captured.update(kwargs)
        return "ok"

    r._router_process_fn = capture_fn
    assert asyncio.run(r._call_router_full(_trigger())) == "ok"

    assert "worker_launch" in captured["tool_names"]
    assert "worker_launch" in captured["instructions"]
    assert captured["max_iters"] > 1


def test_dispatch_worker_blocked_when_session_enabled(tmp_path):
    r = _make_router(tmp_path, enabled=True, cfg=_session_cfg())
    parsed = r._parse_router_response(
        "I'll handle it.\n<dispatch_worker>\ntask: do thing\n</dispatch_worker>"
    )
    assert parsed["dispatch_worker"] is False
    assert parsed["response"] == "I'll handle it."


def test_session_cmd_builder(tmp_path):
    r = _make_router(tmp_path, enabled=True, cfg=_session_cfg())
    cmd = r._harness_session_mgr._build_session_cmd("/tmp/work", 50, 123_456, 8)
    assert cmd[1:4] == ["-m", "mesh.harness", "session"]
    assert "--backend" in cmd and "openai" in cmd
    assert "--base-url" in cmd and "http://localhost:8002/v1" in cmd
    assert "--soft-limit" in cmd and "123456" in cmd
    assert "--max-iters" in cmd and "50" in cmd
    assert "--checkpoint-interval" in cmd and "8" in cmd
    assert "--cwd" in cmd and "/tmp/work" in cmd
    # api key is masked in logs
    masked = r._harness_session_mgr._mask_cmd(cmd)
    assert "dummy" not in " ".join(masked)


def test_start_session_errors_without_config(tmp_path):
    r = _make_router(tmp_path, enabled=True, cfg=None)
    out = asyncio.run(
        r._harness_session_mgr._tool_harness_start_session(task="x")
    )
    assert json.loads(out)["status"] == "error"


# ---------------------------------------------------------------------------
# Live subprocess lifecycle (no LLM): start idle → status → abort → exit 0
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_subprocess_idle_abort_lifecycle():
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "mesh.harness", "session",
        "--backend", "openai", "--model", "local-27b",
        "--base-url", "http://localhost:9", "--api-key", "dummy",
        "--toolset", "harness", "--soft-limit", "50000",
        "--node-id", "agent:test:smoke",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    events: list[str] = []

    async def pump():
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            t = line.decode().strip()
            if not t:
                continue
            try:
                ev = json.loads(t)
            except json.JSONDecodeError:
                continue
            events.append(ev["type"])
            if ev["type"] == "session.awaiting_input":
                proc.stdin.write((json.dumps({"type": "status"}) + "\n").encode())
                await proc.stdin.drain()
                proc.stdin.write((json.dumps({"type": "abort"}) + "\n").encode())
                await proc.stdin.drain()

    pt = asyncio.create_task(pump())
    try:
        await asyncio.wait_for(proc.wait(), timeout=30)
    except asyncio.TimeoutError:
        proc.kill()
        pytest.fail(f"session did not exit; events={events}")
    await pt

    assert proc.returncode == 0
    assert "session.started" in events
    assert "session.awaiting_input" in events
    assert "session.status" in events
    assert "thread.finished" in events
