"""Hermetic Codex-router regression coverage for the harness text path."""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from mesh.agent_node import AgentNode
from mesh.config import NodeConfig
from mesh.llm import CodexExecutionError, LLMClient, LLMConfig
from mesh.protocol import Message, MessageType
from mesh.router_v2 import DispatchReceipt, RouterV2, RouterV2Config, WorkerResult
from mesh.tools import get_registry


class _Reader:
    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)

    async def read(self, _size: int = -1) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""


class _Stdin:
    def write(self, _data: bytes) -> None:
        pass

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        pass


class _CodexProcess:
    def __init__(self, *, returncode: int, stdout: bytes = b"", stderr: bytes = b""):
        self.returncode = returncode
        self.stdin = _Stdin()
        self.stdout = _Reader([stdout, b""])
        self.stderr = _Reader([stderr])
        self.pid = 12345

    async def wait(self) -> int:
        return self.returncode


def _codex_client() -> LLMClient:
    return LLMClient(
        LLMConfig(
            backend="codex",
            model="test-codex",
            codex_binary="/test/codex",
        )
    )


def _codex_event(text: str) -> bytes:
    return (
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": text},
            }
        )
        + "\n"
    ).encode()


def test_codex_nonzero_exit_without_text_raises_typed_error(monkeypatch):
    client = _codex_client()

    async def create_process(*_args, **_kwargs):
        return _CodexProcess(returncode=17, stderr=b"codex test failure")

    monkeypatch.setattr("mesh.llm.asyncio.create_subprocess_exec", create_process)

    with pytest.raises(CodexExecutionError) as exc_info:
        asyncio.run(client._complete_codex("test prompt", "test-codex"))

    assert exc_info.value.returncode == 17
    assert exc_info.value.stderr == "codex test failure"
    assert client._last_usage["backend"] == "codex"


def test_codex_nonzero_exit_preserves_usable_partial_text(monkeypatch):
    client = _codex_client()

    async def create_process(*_args, **_kwargs):
        return _CodexProcess(
            returncode=17,
            stdout=_codex_event("usable partial response"),
            stderr=b"late cleanup failure",
        )

    monkeypatch.setattr("mesh.llm.asyncio.create_subprocess_exec", create_process)

    assert asyncio.run(client._complete_codex("test prompt", "test-codex")) == (
        "usable partial response"
    )


def test_codex_zero_exit_normal_response_is_unchanged(monkeypatch):
    client = _codex_client()

    async def create_process(*_args, **_kwargs):
        return _CodexProcess(returncode=0, stdout=_codex_event("normal response"))

    monkeypatch.setattr("mesh.llm.asyncio.create_subprocess_exec", create_process)

    assert asyncio.run(client._complete_codex("test prompt", "test-codex")) == (
        "normal response"
    )


def _message(content: str, *, sender: str = "user:testuser", metadata=None) -> Message:
    return Message(
        id=f"codex-router-{sender.replace(':', '-')}",
        type=MessageType.MESSAGE,
        from_node=sender,
        to_node="agent:coder:autopilot",
        content=content,
        metadata=metadata or {},
    )


def _make_codex_router():
    # Importing the implementation module registers dossier_read in the test's
    # global registry. The handler is replaced below; no filesystem state is read.
    from mesh import tool_implementations  # noqa: F401

    client = _codex_client()
    agent = AgentNode(
        NodeConfig(
            id="agent:coder:autopilot",
            agent_type="coder",
            nickname="autopilot",
            tools=["dossier_read"],
        ),
        llm_config=client.config,
        tool_registry=get_registry(),
        persist=False,
    )
    agent._tool_socket_path = None

    sent: list[str] = []

    async def send(content, *_args, **_kwargs):
        sent.append(content)

    async def worker(_context, _trigger):
        return WorkerResult(response="unused", context=[])

    router_ref = {}

    async def router_process(**kwargs):
        # RouterV2 only passes this kwarg when selecting a deep client. The
        # production AgentNode closure supplies its normal light client and
        # durable router history here.
        kwargs.setdefault("llm_client", client)
        kwargs.setdefault("router_history", router_ref["router"]._history)
        return await agent._router_process_with_llm(**kwargs)

    router = RouterV2(
        worker_fn=worker,
        send_fn=send,
        config=RouterV2Config(
            llm_enabled=True,
            router_mode="full",
            router_max_iters=3,
            history_persist=False,
            autonomous_agent_mode_enabled=True,
            autonomous_projects=["project:mesh-autopilot"],
            autonomous_mandate_prompt="PHASE 1 — PLAN",
            autonomous_continuation_mandate_prompt="PHASE 2 — EXECUTE & CLOSE",
        ),
        node_id="agent:coder:autopilot",
        nickname="autopilot",
        agent_type="coder",
        llm_client=client,
        router_process_fn=router_process,
    )
    router_ref["router"] = router
    agent._router_v2 = router
    return agent, client, router, sent


@pytest.mark.filterwarnings(
    "ignore:XML tool path invoked for backend=codex:DeprecationWarning"
)
def test_mocked_codex_full_router_salvages_tools_dispatches_and_continues(monkeypatch):
    """Drive Codex through AgentNode's real text-tool and dispatch seams."""
    agent, client, router, sent = _make_codex_router()
    session_plan = (
        "SESSION PLAN\n"
        "GOAL=G-T021\n"
        "TASKS=T-021\n"
        "EVIDENCE=exercise the Codex harness dispatch path without a subprocess\n"
        "FIRST=read the project dossier before admitting the worker"
    )
    task = (
        "Inspect the Codex harness-router dispatch path, verify that its textual "
        "dispatch reaches the authoritative admission seam, and report the exact "
        "worker trigger metadata with file and line evidence."
    )
    responses = [
        '<mesh_call name="dossier_read"><entity_key>project:mesh-autopilot</entity_key></mesh_call>',
        f"{session_plan}\n\n<dispatch_worker>\ntask: {task}\n</dispatch_worker>",
        "Worker report reviewed; continue the autonomous session without a new dispatch.",
    ]
    codex_prompts: list[str] = []

    async def fake_complete_codex(_self, prompt, _model, callback=None):
        assert callback is not None
        codex_prompts.append(prompt)
        return responses.pop(0)

    monkeypatch.setattr(LLMClient, "_complete_codex", fake_complete_codex)
    executed_tools = []

    async def execute_dossier_read(call, original_sender, trigger_msg_id=None, **_kwargs):
        executed_tools.append((call.name, call.arguments, original_sender, trigger_msg_id))
        return "DOSSIER RESULT: current task is T-021"

    agent._execute_single_tool_with_confirmation = execute_dossier_read
    admitted = []

    async def start_worker(trigger):
        admitted.append(trigger)
        router._last_dispatch_receipt = DispatchReceipt(
            dispatch_key="dk-codex-e2e",
            status="running",
            worker_id="autopilot-worker1",
            slot_index=0,
            origin_message_id=trigger.id,
            router_turn_id="turn-codex-e2e",
            task_description=trigger.metadata["worker_task_description"],
            backend=None,
        )
        return True

    router._start_worker = start_worker
    # The seam under test is textual dispatch, not a deployment's persisted
    # daily ledger. Keep the test hermetic even when a developer has exhausted
    # a real project budget in their own ~/.mesh state.
    import mesh.project_dossier as project_dossier

    monkeypatch.setattr(
        project_dossier,
        "check_budget",
        lambda *_args, **_kwargs: {
            "remaining": 20,
            "limit": 20,
            "used": 0,
            "resets_at": "2099-01-01T00:00:00+00:00",
        },
    )
    wake = _message(
        "Run the autonomous T-021 session.",
        metadata={
            "autonomous_session": True,
            "autonomous_session_id": "as-codex-e2e",
            "autonomous_project_key": "project:mesh-autopilot",
        },
    )
    router._append_to_history(wake)

    asyncio.run(router._handle_idle_with_llm(wake))

    # (a) The Codex text-door XML is parsed and executed; its result re-enters
    # the next Codex prompt instead of terminating the router loop.
    assert executed_tools == [
        (
            "dossier_read",
            {"entity_key": "project:mesh-autopilot"},
            "user:testuser",
            wake.id,
        )
    ]
    assert "DOSSIER RESULT: current task is T-021" in codex_prompts[1]
    # (b) The textual dispatch reaches RouterV2's sole admission seam.
    assert len(admitted) == 1
    assert router._last_dispatch_receipt.source == "xml"
    # (c) T-020's session-plan binding survives onto the admitted trigger.
    assert admitted[0].metadata["autonomous_session_plan"] == session_plan
    assert admitted[0].metadata["worker_task_description"].endswith(task)
    assert admitted[0].metadata["worker_task_description"].startswith(
        "[PROJECT: project:mesh-autopilot]"
    )

    report = _message(
        "Worker completed the requested inspection.",
        sender="worker:autopilot-worker1",
        metadata={
            "autonomous_session": True,
            "autonomous_session_id": "as-codex-e2e",
            "autonomous_project_key": "project:mesh-autopilot",
            "autonomous_session_plan": session_plan,
            "worker_report": True,
        },
    )
    router._append_to_history(report)
    asyncio.run(router._handle_idle_with_llm(report))

    # (d) A worker report remains a continuation: it gets the carried plan and
    # does not admit a second worker.
    assert len(admitted) == 1
    assert "[SESSION PLAN CARRY-FORWARD]" in codex_prompts[2]
    assert session_plan in codex_prompts[2]
    assert any("Worker report reviewed" in content for content in sent)


def test_harness_light_deep_plan_logs_explicit_fallback(caplog):
    with caplog.at_level(logging.WARNING, logger="mesh.agent_node"):
        node = AgentNode(
            config=NodeConfig(
                id="agent:researcher:reme",
                agent_type="researcher",
                nickname="reme",
                router_mode="full",
                router_deep_enabled=True,
                router_deep_backend="claude-code-fable",
                autonomous_plan_backend="deep",
                router_history_persist=False,
                tools=[],
            ),
            llm_config=LLMConfig(backend="openai", model="test"),
            persist=False,
        )
        node._router_v2_llm_config = LLMConfig(
            backend="codex", model="light-codex"
        )
        node._router_deep_llm_config = LLMConfig(
            backend="claude-code", model="deep-fable"
        )
        node._init_router_v2()

    assert node._router_v2._deep_llm_client is None
    assert any(
        "Autonomous PLAN deep fallback for agent:researcher:reme" in record.message
        and "requested deep backend=claude-code-fable" in record.message
        and "deep PLAN route is NOT running" in record.message
        for record in caplog.records
    )


@pytest.mark.filterwarnings(
    "ignore:XML tool path invoked for backend=codex:DeprecationWarning"
)
def test_codex_execution_error_is_contained_by_router_loop(monkeypatch):
    agent, client, router, sent = _make_codex_router()

    async def fail_codex(_self, _prompt, _model, callback=None):
        raise CodexExecutionError(23, "mocked Codex failure")

    monkeypatch.setattr(LLMClient, "_complete_codex", fail_codex)
    wake = _message("Handle a failed Codex router turn.")
    router._append_to_history(wake)

    asyncio.run(router._handle_idle_with_llm(wake))

    assert sent
    assert "Codex subprocess exited 23" in sent[-1]
