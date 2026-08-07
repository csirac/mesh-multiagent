"""Unit and router-integration tests for governed procedural memory."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from mesh.memory.essay_fold import MetaReviewItem, scan_meta_review
from mesh.agent_node import AgentNode
from mesh.procedural_memory import (
    SkillCardError,
    SkillStore,
    detect_formation_signal,
    sha256_file,
)
from mesh.protocol import Message, MessageType
from mesh.router_v2 import RouterV2, RouterV2Config, WorkerResult


def _card(
    source: Path,
    *,
    card_id: str = "qwen-recovery",
    owner: str = "bob",
    status: str = "active",
    approved_fingerprint: str | None = None,
) -> dict[str, Any]:
    approved = status == "active"
    return {
        "schema_version": 1,
        "id": card_id,
        "version": 1,
        "status": status,
        "owner_agent": owner,
        "purpose": "Recover the Qwen vLLM service on ComputeHost.",
        "triggers": ["qwen endpoint down", "restart vllm on computehost"],
        "preconditions": [
            {
                "key": "host", "operator": "equals", "value": "computehost",
                "aliases": ["gpu host"], "required": True,
            },
            {
                "key": "service", "operator": "equals", "value": "vllm",
                "aliases": ["qwen", "qwen-serve"], "required": True,
            },
        ],
        "authority": {
            "reads": "allowed", "diagnostics": "allowed",
            "service_restart": "explicit_user_request",
            "destructive_actions": "forbidden",
        },
        "procedure_source": [{
            "kind": "file", "host": "localhost", "path": str(source),
            "approved_fingerprint": (
                approved_fingerprint
                if approved_fingerprint is not None
                else sha256_file(source)
            ),
        }],
        "required_invariants": [
            {"id": "tp", "statement": "Tensor parallelism is four."}
        ],
        "verification": [
            {"id": "model-list", "probe": "Query /v1/models.", "expected": "HTTP 200"}
        ],
        "rollback": {"description": "Restore approved source.", "source": "backup"},
        "evidence": ["m_000000000001"],
        "proposed_by": {
            "mechanism": "manual_pilot", "agent": owner, "fold_round": None,
            "run_id": "test", "proposed_at": "2026-07-17T00:00:00+00:00",
        },
        "approved_by": "user:approver" if approved else None,
        "approved_at": "2026-07-17T00:00:00+00:00" if approved else None,
        "supersedes_version": None,
        "last_reviewed_at": None,
        "outcomes": [],
    }


def _write_card(store: SkillStore, card: dict[str, Any]) -> Path:
    store.ensure_layout()
    path = store.agent_dir / f"{card['id']}.yaml"
    path.write_text(yaml.safe_dump(card, sort_keys=False), encoding="utf-8")
    return path


def _pilot_qwen_card(source: Path) -> dict[str, Any]:
    card = _card(source, card_id="qwen-computehost-production-recovery")
    card["purpose"] = (
        "Restore the production Qwen3.6-27B vLLM service used by mesh routers."
    )
    card["triggers"] = [
        "qwen endpoint down",
        "vllm not responding on computehost",
        "model serve restart needed on computehost",
        "auto tool choice or qwen tool parser error",
        "context-length regression after a qwen restart",
        "leaked qwen thinking tags",
    ]
    card["preconditions"] = [
        {
            "key": "host", "operator": "equals", "value": "computehost",
            "aliases": ["gpu host", "qwen host"], "required": True,
        },
        {
            "key": "service", "operator": "equals", "value": "vllm",
            "aliases": ["qwen", "qwen-serve", "qwen-serve.service"],
            "required": True,
        },
        {
            "key": "port", "operator": "equals", "value": "8002",
            "aliases": ["localhost 8002", "qwen endpoint"],
            "required": True,
        },
    ]
    return card


@pytest.fixture
def active_store(tmp_path: Path) -> tuple[SkillStore, Path]:
    source = tmp_path / "approved-source.txt"
    source.write_text("approved\n", encoding="utf-8")
    store = SkillStore("bob", root=tmp_path / "skills")
    _write_card(store, _card(source))
    store.rebuild_index()
    return store, source


@pytest.fixture
def pilot_qwen_store(tmp_path: Path) -> SkillStore:
    source = tmp_path / "approved-source.txt"
    source.write_text("approved\n", encoding="utf-8")
    store = SkillStore("bob", root=tmp_path / "skills")
    _write_card(store, _pilot_qwen_card(source))
    store.rebuild_index()
    return store


def test_proposed_card_is_excluded_from_index(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("approved\n", encoding="utf-8")
    store = SkillStore("bob", root=tmp_path / "skills")
    _write_card(store, _card(source, status="proposed"))

    index = store.rebuild_index()

    assert index["cards"] == []
    assert store.render_index_block() == ""


def test_selection_combines_bm25_and_typed_preconditions(active_store):
    store, _ = active_store

    selections = store.select_with_scores("restart qwen service on computehost")

    assert [item.card["id"] for item in selections] == ["qwen-recovery"]
    assert selections[0].matched == ("host", "service")
    assert selections[0].precondition_factor == 1.0
    assert "selection_diagnostics" in store.render_selected_block(selections)


def test_contradicted_required_precondition_disqualifies(active_store):
    store, _ = active_store

    selections = store.select_with_scores("restart qwen service host=coder1")

    assert selections == []


@pytest.mark.parametrize(
    "task",
    [
        "The production model API on ComputeHost stopped answering after a host hiccup. "
        "Give me the safe diagnostic and recovery checklist for the Qwen service, "
        "including how you would prove it is healthy afterward. Do not execute it.",
        "host=computehost service=vllm port=8002 is unavailable. Prepare the exact "
        "production recovery plan and the post-recovery checks; this is planning only.",
        "After yesterday's model-server maintenance, requests that include tools and "
        "tool_choice=auto now return HTTP 400 from the production endpoint. Diagnose "
        "the likely relaunch regression and give a complete read-only remediation plan.",
        "Alice's large router prompt used to fit, but after the Qwen service came back "
        "it fails near 130k tokens. What production serving invariants and probes "
        "should we check before approving another relaunch?",
        "The local production Qwen API is returning internal thinking markup in "
        "ordinary assistant content. Give the durable service-level fix checklist "
        "and verification plan, without changing anything.",
        "The production Qwen router endpoint vanished overnight. I need a first-response "
        "checklist that preserves tool calling, long context, and clean reasoning "
        "output. Do not run commands.",
        "On ComputeHost, the model API at localhost:8002 is unhealthy. Lay out what must "
        "remain true in the approved production launch and how to validate it, but "
        "take no action.",
    ],
)
def test_pilot_positive_prompts_still_select_qwen_card(pilot_qwen_store, task):
    selections = pilot_qwen_store.select_with_scores(task)

    assert [item.card["id"] for item in selections] == [
        "qwen-computehost-production-recovery"
    ]


@pytest.mark.parametrize(
    "task",
    [
        "A model server is sluggish. Explain what information you need before deciding "
        "whether any restart is appropriate; do not assume a host, service, or model.",
        "On ComputeHost, service=computehost-qwen-tunnel.service is down while the model process "
        "itself is healthy. Plan recovery of the reverse tunnel only; do not restart "
        "the model.",
        "host=relayhost service=qwen-test port=18002 is a laptop-scale test server. What "
        "should I inspect before stopping it? Keep the ComputeHost production service out "
        "of scope.",
    ],
)
def test_pilot_false_positive_prompts_do_not_select_qwen_card(
    pilot_qwen_store,
    task,
):
    assert pilot_qwen_store.select_with_scores(task) == []


def test_source_drift_suppresses_retrieval(active_store):
    store, source = active_store
    source.write_text("drifted\n", encoding="utf-8")

    assert store.select_with_scores("restart qwen service on computehost") == []
    findings = store.scan_meta_review()
    assert any(finding.tier == 1 and "source" in finding.reason for finding in findings)


def test_outcome_append_is_immutable_receipt_and_refreshes_index(active_store):
    store, _ = active_store
    before = store.load_index()["cards"][0]["card_fingerprint"]

    receipt = store.append_outcome(
        "qwen-recovery",
        task_summary="restart qwen on computehost",
        task_ref="task-1",
        result="unknown",
        disposition="unknown",
    )

    card = store.load_card("qwen-recovery")
    after = store.load_index()["cards"][0]["card_fingerprint"]
    assert card["outcomes"] == [receipt]
    assert before != after
    assert store.select("restart qwen on computehost")[0]["id"] == "qwen-recovery"


def test_active_card_requires_human_approval(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("approved\n", encoding="utf-8")
    store = SkillStore("bob", root=tmp_path / "skills")
    card = _card(source)
    card["approved_by"] = None
    _write_card(store, card)

    with pytest.raises(SkillCardError, match="approval by user:approver"):
        store.rebuild_index()

    card["approved_by"] = "agent:sysadmin:bob"
    _write_card(store, card)
    with pytest.raises(SkillCardError, match="approval by user:approver"):
        store.rebuild_index()


def test_structural_first_success_formation_signal():
    signal = detect_formation_signal(
        "Qwen recovery completed and verified. Restart the service again with "
        "--tool-call-parser qwen3_coder --max-model-len 262144 and "
        "CUDA_HOME=/opt/cuda."
    )
    assert signal.eligible is True
    assert signal.detail_count >= 2


def test_fold_meta_review_includes_skill_candidate(tmp_path: Path):
    db_path = tmp_path / "memory.db"
    sqlite3.connect(db_path).close()

    items = scan_meta_review(
        str(db_path),
        "",
        skills_owner="bob",
        skills_root=str(tmp_path / "skills"),
        procedural_window_text=(
            "Deployment completed and verified using --backend codex-sol "
            "--max-tokens 32000 at /home/testuser/apps/tool/config.yaml."
        ),
    )

    assert any(item.action == MetaReviewItem.SKILL_CANDIDATE for item in items)


def test_meta_review_uses_two_tiers(active_store):
    store, _ = active_store
    old = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    store.append_outcome(
        "qwen-recovery",
        task_summary="old task",
        task_ref="old",
        selected_at=old,
    )

    findings = store.scan_meta_review(now=datetime.now(timezone.utc))
    report = store.format_meta_review(findings)
    assert any(finding.tier == 2 and "30 days" in finding.reason for finding in findings)
    assert "Tier 2 — For review" in report


@pytest.mark.asyncio
async def test_router_injects_index_selects_card_and_logs_outcome(active_store, tmp_path):
    store, _ = active_store
    captured: dict[str, Any] = {}

    async def send_fn(content, in_reply_to=None):
        return None

    async def worker_fn(context, trigger):
        captured.update(trigger.metadata)
        return WorkerResult(response="Done.", context=context)

    router = RouterV2(
        worker_fn=worker_fn,
        send_fn=send_fn,
        config=RouterV2Config(
            llm_enabled=False,
            synthesize_enabled=False,
            history_persist=False,
            watchdog_interval_minutes=0,
        ),
        nickname="bob",
        agent_type="sysadmin",
        node_id="agent:sysadmin:bob",
    )
    router._skill_store = store

    prompt = await router._build_router_prompt("Answer.", include_tools=False)
    assert "<governed_procedural_memory_index>" in prompt

    router._current_task_description = "restart qwen service on computehost"
    trigger = Message(
        type=MessageType.MESSAGE,
        from_node="user:testuser",
        to_node="agent:sysadmin:bob",
        content="restart qwen service on computehost",
        metadata={"canonical_memory_id": "m_000000000002"},
    )
    assert await router._start_worker(trigger) is True
    await router._worker_task

    assert captured["governed_skill_ids"] == ["qwen-recovery"]
    assert "<governed_procedural_memory>" in captured["governed_skill_context"]
    outcomes = store.load_card("qwen-recovery")["outcomes"]
    assert len(outcomes) == 1
    assert outcomes[0]["result"] == "unknown"
    assert outcomes[0]["memory_id"] == "m_000000000002"


@pytest.mark.asyncio
async def test_agent_worker_prompt_and_cc_briefing_receive_selected_card(
    tmp_path: Path,
):
    skill_context = "<governed_procedural_memory>card</governed_procedural_memory>"
    digest_path = tmp_path / "bob_digest.md"
    digest_path.write_text(
        "## Timeline\n- prior recovery [m_000000000001]\n",
        encoding="utf-8",
    )
    trigger = Message(
        type=MessageType.MESSAGE,
        from_node="user:testuser",
        to_node="agent:sysadmin:bob",
        content="recover qwen",
        metadata={"governed_skill_context": skill_context},
    )
    node = AgentNode.__new__(AgentNode)
    node._preference_extractor = SimpleNamespace(get_preference_block=lambda: "")
    node._memory_system = None
    node.system_prompt = "base prompt"
    node.config = SimpleNamespace(
        trace_as_history_enabled=False,
        worker_digest_injection=True,
        standing_digest_path=str(digest_path),
    )

    system_prompt, _, _ = await node._build_system_prompt_for_llm(trigger, True)
    assert skill_context in system_prompt
    assert "<worker_memory_context>" in system_prompt
    assert "[m_000000000001]" in system_prompt

    node.llm_config = SimpleNamespace(
        cc_worker_briefing=True,
        backend="claude-code",
    )
    node._router_v2 = SimpleNamespace(
        _current_task_description="recover qwen",
        _history=SimpleNamespace(window=[]),
    )

    async def briefing(_trigger):
        return "briefing"

    node._ensure_briefing = briefing
    node._log_worker_dispatch = lambda **kwargs: None
    _, cc_system_prompt, _ = await node._build_worker_instructions(
        trigger,
        True,
        False,
        None,
        "",
        "",
        None,
        0,
        1,
    )
    assert skill_context in cc_system_prompt
    assert "<worker_memory_context>" in cc_system_prompt
    assert "[m_000000000001]" in cc_system_prompt
