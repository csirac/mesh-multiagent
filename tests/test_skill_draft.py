"""On-demand governed skill-card drafting and proposal persistence tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import yaml

import mesh.tool_implementations  # noqa: F401 - register tool schemas
import mesh.router_v2 as router_v2_module
from mesh.procedural_memory import (
    SKILL_DRAFT_TRACE_MAX_CHARS,
    SkillCardError,
    SkillDraftPackage,
    SkillStore,
    bounded_worker_trace,
    build_skill_draft_package,
    persist_completed_worker_trace,
    sha256_file,
)
from mesh.router_v2 import RouterV2, RouterV2Config, WorkerResult
from mesh.tools import get_registry


def _proposal(source: Path, *, owner: str = "hypatia", card_id: str = "proof-audit") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": card_id,
        "version": 1,
        "status": "proposed",
        "owner_agent": owner,
        "purpose": "Repeat a full proof audit with exact verification artifacts.",
        "triggers": ["run the full proof audit again", "verify the proof from scratch"],
        "preconditions": [
            {
                "key": "artifact",
                "operator": "present",
                "value": "proof-note",
                "aliases": ["latex proof"],
                "required": True,
            }
        ],
        "authority": {
            "reads": "allowed",
            "file_write": "explicit_user_request",
            "external_publish": "explicit_user_request",
            "destructive_actions": "forbidden",
        },
        "procedure_source": [
            {
                "kind": "runbook",
                "host": "localhost",
                "path": str(source),
                "approved_fingerprint": sha256_file(source),
            }
        ],
        "required_invariants": [
            {
                "id": "independent-derivation",
                "statement": "The audit re-derives load-bearing identities independently.",
            }
        ],
        "verification": [
            {
                "id": "audit-artifacts",
                "probe": "Run the exact verification scripts and compile the note.",
                "expected": "All scripts exit zero and LaTeX has no errors.",
            }
        ],
        "rollback": {
            "description": "Restore the pre-audit document if edits were requested.",
            "source": "Version control",
        },
        "caveats": {
            "trace_assessment": "canonical",
            "pitfalls": ["do not reuse the failed approximate verifier"],
            "unverified_steps": ["none identified after source cross-check"],
            "unexplored_alternatives": ["an alternate proof assistant was not tested"],
        },
        "evidence": [],
        "proposed_by": {
            "mechanism": "user_requested_tool",
            "agent": owner,
            "fold_round": None,
            "run_id": "skill-draft-test",
            "proposed_at": "2026-07-18T17:48:00-05:00",
        },
        "approved_by": None,
        "approved_at": None,
        "supersedes_version": None,
        "last_reviewed_at": None,
        "outcomes": [],
    }


def _active(card: dict[str, Any]) -> dict[str, Any]:
    active = dict(card)
    active["status"] = "active"
    active["approved_by"] = "user:approver"
    active["approved_at"] = "2026-07-18T17:50:00-05:00"
    active["proposed_by"] = dict(card["proposed_by"])
    active["procedure_source"] = [dict(item) for item in card["procedure_source"]]
    return active


def test_skill_draft_tool_is_registered_with_typed_sources():
    tool = get_registry().get("skill_draft")

    assert tool is not None
    assert [parameter.name for parameter in tool.parameters] == [
        "task_summary",
        "source_files",
        "trace_path",
    ]
    assert tool.parameters[0].required is True
    assert tool.parameters[1].type == "array"
    assert tool.parameters[1].required is False
    assert tool.parameters[2].type == "string"
    assert tool.parameters[2].required is False


def test_context_package_includes_episode_sources_schema_and_example(tmp_path: Path):
    source = tmp_path / "proof-runbook.md"
    source.write_text("Run sympy verifier with --exact.\n", encoding="utf-8")
    spec = tmp_path / "spec.md"
    spec.write_text(
        "## 3. Card Schema\nschema\n\n## 4. Directory Layout\nlayout\n\n"
        "## 5. Lifecycle\nlifecycle\n\n## 6. Formation\nformation\n\n"
        "## 10. Governance and Safety\ngovernance\n\n## 11. Pilot\npilot\n",
        encoding="utf-8",
    )
    example = tmp_path / "example.yaml"
    example.write_text("schema_version: 1\nid: example\n", encoding="utf-8")

    package = build_skill_draft_package(
        "hypatia",
        "Capture the independent proof-verification workflow.",
        recent_context="[worker_report=true]\nAll exact checks passed.",
        source_files=[source],
        trace_root=tmp_path / "worker-traces",
        staging_root=tmp_path / "staging",
        run_id="draft-001",
        spec_path=spec,
        example_path=example,
    )

    assert package.staging_path == tmp_path / "staging/hypatia/draft-001/card.yaml"
    assert "All exact checks passed" in package.task
    assert str(source) in package.task
    assert source.read_text(encoding="utf-8").strip() in package.task
    assert sha256_file(source) in package.task
    assert "<skill_card_spec_excerpt>" in package.task
    assert "<worked_example>" in package.task
    assert "scripts/write_skill_proposal.py" in package.task
    assert "approved_by: null" in package.task
    assert "No completed worker trace was available" in package.task
    assert "DISTILL; DO NOT TRANSCRIBE" in package.task
    assert "caveats.trace_assessment" in package.task


def test_latest_completed_worker_trace_is_located_and_fingerprinted(tmp_path: Path):
    trace_root = tmp_path / "worker-traces"
    older = persist_completed_worker_trace(
        "hypatia",
        "hypatia-worker1",
        "first attempt failed\nthen fixed the proof",
        trace_root=trace_root,
    )
    drafting = persist_completed_worker_trace(
        "hypatia",
        "hypatia-worker2",
        "drafting trace must not become procedure evidence",
        kind="skill_draft",
        trace_root=trace_root,
    )
    newer = persist_completed_worker_trace(
        "hypatia",
        "hypatia-worker3",
        "permission denied\nfinal verified command succeeded",
        trace_root=trace_root,
    )
    # Make ordering deterministic independent of filesystem timestamp precision.
    older.touch()
    newer.touch()
    assert drafting.is_file()

    package = build_skill_draft_package(
        "hypatia",
        "Capture the latest proof procedure.",
        recent_context="worker completed",
        trace_root=trace_root,
        staging_root=tmp_path / "staging",
        run_id="draft-latest-trace",
    )

    assert package.trace_path == newer
    assert package.trace_fingerprint == sha256_file(newer)
    assert sha256_file(newer) in package.task
    assert "final verified command succeeded" in package.task
    assert str(drafting) not in package.task


def test_explicit_trace_path_is_honored_over_latest(tmp_path: Path):
    trace_root = tmp_path / "worker-traces"
    selected = persist_completed_worker_trace(
        "hypatia",
        "hypatia-worker-old",
        "older but explicitly requested procedure",
        trace_root=trace_root,
    )
    persist_completed_worker_trace(
        "hypatia",
        "hypatia-worker-new",
        "newer unrelated procedure",
        trace_root=trace_root,
    )

    package = build_skill_draft_package(
        "hypatia",
        "Capture the older procedure.",
        recent_context="user selected an older task",
        trace_path=selected,
        trace_root=trace_root,
        staging_root=tmp_path / "staging",
        run_id="draft-explicit-trace",
    )

    assert package.trace_path == selected.resolve()
    assert "older but explicitly requested procedure" in package.task
    assert "newer unrelated procedure" not in package.task


def test_no_worker_trace_degrades_gracefully(tmp_path: Path):
    package = build_skill_draft_package(
        "hypatia",
        "Capture a router-executed procedure.",
        recent_context="router ran the diagnostic directly",
        trace_root=tmp_path / "missing-traces",
        staging_root=tmp_path / "staging",
        run_id="draft-no-trace",
    )

    assert package.trace_path is None
    assert package.trace_fingerprint is None
    assert package.trace_truncated is False
    assert "No completed worker trace was available" in package.task
    assert "router-executed procedure" in package.task


def test_worker_trace_handoff_respects_cap_and_preserves_failure_and_tail(tmp_path: Path):
    trace_root = tmp_path / "worker-traces"
    trace = persist_completed_worker_trace(
        "hypatia",
        "hypatia-worker-long",
        "prefix\nERROR: first launch used the wrong flag\n"
        + ("detour output\n" * 300)
        + "FINAL: canonical command verified successfully",
        trace_root=trace_root,
    )
    max_chars = 1_200
    package = build_skill_draft_package(
        "hypatia",
        "Capture the bounded trace procedure.",
        recent_context="worker completed",
        trace_path=trace,
        trace_max_chars=max_chars,
        staging_root=tmp_path / "staging",
        run_id="draft-truncated-trace",
    )

    assert package.trace_truncated is True
    start = package.task.index("<completed_worker_trace_source")
    end = package.task.index("</completed_worker_trace_source>", start)
    trace_block = package.task[start:end]
    assert "ERROR: first launch used the wrong flag" in trace_block
    assert "FINAL: canonical command verified successfully" in trace_block
    assert f"included_chars={max_chars!r}" in trace_block

    bounded, truncated = bounded_worker_trace(
        trace.read_text(encoding="utf-8"),
        max_chars=max_chars,
    )
    assert truncated is True
    assert len(bounded) <= max_chars
    assert SKILL_DRAFT_TRACE_MAX_CHARS > max_chars


def test_validated_proposal_writes_to_owner_proposals_only(tmp_path: Path):
    source = tmp_path / "runbook.md"
    source.write_text("verified procedure\n", encoding="utf-8")
    store = SkillStore("hypatia", root=tmp_path / "skills")

    proposal_path = store.write_proposal(
        _proposal(source),
        source_files=[source],
    )

    assert proposal_path == (
        tmp_path / "skills/hypatia/.proposals/proof-audit.yaml"
    )
    assert proposal_path.is_file()
    assert not (tmp_path / "skills/hypatia/proof-audit.yaml").exists()
    assert not store.index_path.exists()
    persisted = yaml.safe_load(proposal_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "proposed"
    assert persisted["approved_by"] is None


def test_worker_submission_cli_returns_review_summary(tmp_path: Path):
    source = tmp_path / "runbook.md"
    source.write_text("verified procedure\n", encoding="utf-8")
    draft = tmp_path / "staging/card.yaml"
    draft.parent.mkdir(parents=True)
    draft.write_text(
        yaml.safe_dump(_proposal(source), sort_keys=False),
        encoding="utf-8",
    )
    script = Path(__file__).parents[1] / "scripts/write_skill_proposal.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--owner",
            "hypatia",
            "--draft",
            str(draft),
            "--source-file",
            str(source),
            "--skills-root",
            str(tmp_path / "skills"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["status"] == "proposed"
    assert payload["card_id"] == "proof-audit"
    assert payload["human_approval_required"] is True
    assert payload["index_modified"] is False
    assert payload["caveats"]["trace_assessment"] == "canonical"
    assert Path(payload["proposal_path"]).is_file()


def test_schema_rejection_never_lands_in_proposals(tmp_path: Path):
    source = tmp_path / "runbook.md"
    source.write_text("verified procedure\n", encoding="utf-8")
    store = SkillStore("hypatia", root=tmp_path / "skills")
    malformed = _proposal(source)
    malformed["verification"] = []

    with pytest.raises(SkillCardError, match="cannot be empty"):
        store.write_proposal(malformed, source_files=[source])

    proposal_dir = tmp_path / "skills/hypatia/.proposals"
    assert not proposal_dir.exists() or list(proposal_dir.iterdir()) == []


def test_skill_draft_proposal_without_caveats_is_rejected(tmp_path: Path):
    source = tmp_path / "runbook.md"
    source.write_text("verified procedure\n", encoding="utf-8")
    store = SkillStore("hypatia", root=tmp_path / "skills")
    missing_caveats = _proposal(source)
    missing_caveats.pop("caveats")

    with pytest.raises(SkillCardError, match="require a non-empty caveats"):
        store.write_proposal(missing_caveats, source_files=[source])

    proposal_dir = tmp_path / "skills/hypatia/.proposals"
    assert not proposal_dir.exists() or list(proposal_dir.iterdir()) == []


def test_proposal_does_not_mutate_active_card_or_index(tmp_path: Path):
    source = tmp_path / "runbook.md"
    source.write_text("verified procedure\n", encoding="utf-8")
    store = SkillStore("hypatia", root=tmp_path / "skills")
    store.ensure_layout()
    active = _active(_proposal(source, card_id="existing-procedure"))
    active_path = store.agent_dir / "existing-procedure.yaml"
    active_path.write_text(yaml.safe_dump(active, sort_keys=False), encoding="utf-8")
    store.rebuild_index()
    active_before = active_path.read_bytes()
    index_before = store.index_path.read_bytes()

    proposal_path = store.write_proposal(_proposal(source), source_files=[source])

    assert proposal_path.is_file()
    assert active_path.read_bytes() == active_before
    assert store.index_path.read_bytes() == index_before
    assert [entry["id"] for entry in store.load_index()["cards"]] == [
        "existing-procedure"
    ]


@pytest.mark.asyncio
async def test_router_skill_draft_uses_normal_worker_slot_and_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}
    staging_path = tmp_path / "stage/card.yaml"
    staging_path.parent.mkdir(parents=True)
    package = SkillDraftPackage(
        run_id="draft-router-test",
        task="PACKAGED DRAFT TASK",
        staging_path=staging_path,
        source_files=(),
    )
    monkeypatch.setattr(
        router_v2_module,
        "build_skill_draft_package",
        lambda *args, **kwargs: package,
    )

    def archive_trace(owner, worker_id, trace_text, **kwargs):
        captured["archived_trace"] = {
            "owner": owner,
            "worker_id": worker_id,
            "trace_text": trace_text,
            **kwargs,
        }
        return tmp_path / "archived.trace.txt"

    monkeypatch.setattr(
        router_v2_module,
        "persist_completed_worker_trace",
        archive_trace,
    )

    async def worker_fn(context, trigger):
        captured["trigger"] = trigger
        return WorkerResult(response="drafted", context=context)

    async def send_fn(*args, **kwargs):
        return None

    router = RouterV2(
        worker_fn=worker_fn,
        send_fn=send_fn,
        config=RouterV2Config(
            llm_enabled=False,
            synthesize_enabled=False,
            history_persist=False,
            watchdog_interval_minutes=0,
            worker_trace_persist=True,
        ),
        nickname="hypatia",
        agent_type="researcher",
        node_id="agent:researcher:hypatia",
        default_worker_backend="claude-code-fable",
    )
    router._skill_store = SkillStore("hypatia", root=tmp_path / "skills")
    router._current_trigger_from_node = "user:testuser"
    router._current_trigger_to_node = "channel:proof-search-tool"

    result = json.loads(
        await router._tool_skill_draft("Capture the proof audit.")
    )
    await asyncio.shield(router._worker_task)

    assert result["status"] == "dispatched"
    assert result["owner_agent"] == "hypatia"
    assert result["human_approval_required"] is True
    trigger = captured["trigger"]
    assert trigger.content == "PACKAGED DRAFT TASK"
    assert trigger.from_node == "user:testuser"
    assert trigger.to_node == "channel:proof-search-tool"
    assert trigger.metadata["skill_draft"] is True
    assert trigger.metadata["skill_draft_run_id"] == "draft-router-test"
    assert captured["archived_trace"]["owner"] == "hypatia"
    assert captured["archived_trace"]["kind"] == "skill_draft"
    assert "drafted" in captured["archived_trace"]["trace_text"]
    assert "skill_draft" in router._worker_tool_handlers
