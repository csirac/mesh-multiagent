"""Tests for autonomous agent mode: dossiers, session reports, budget ledger.

Covers ``docs/plans/autonomous-agent-mode.md`` §3 (dossier constitution),
§4 (immutable session reports), and §8 (worker-admission budget).

Every test drives the real ``mesh.project_dossier`` functions against a
redirected ``~/.mesh`` tree rather than mocks, because the claims under test
are about atomicity, refusal, and immutability — properties a mock cannot
exhibit.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mesh import project_dossier as pd
from mesh.project_dossier import (
    REQUIRED_SECTIONS,
    BudgetExhausted,
    DossierError,
    budget_path,
    check_budget,
    dossier_path,
    edit_dossier,
    init_dossier,
    read_dossier,
    reset_budget,
    render_skeleton,
    report_path,
    safe_entity_key,
    section_errors,
    spend_budget,
    validate_dossier,
    write_report,
)
from mesh.tools import get_registry


KEY = "project:mesh-infra"


@pytest.fixture
def mesh_home(tmp_path, monkeypatch):
    """Redirect the dossier/report/budget roots into a temp tree."""
    monkeypatch.setattr(pd, "DIGESTS_DIR", tmp_path / "digests")
    monkeypatch.setattr(pd, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(pd, "BUDGET_DIR", tmp_path / "budget")
    return tmp_path


@pytest.fixture
def dossier(mesh_home):
    """An initialized dossier for ``project:mesh-infra``."""
    return init_dossier(KEY, owner_agent="coder1")


# ─────────────────────────────────────────────────────────────────────
# Paths and identity
# ─────────────────────────────────────────────────────────────────────


def test_safe_entity_key_replaces_colon(mesh_home):
    assert safe_entity_key(KEY) == "project-mesh-infra"


@pytest.mark.parametrize(
    "bad",
    ["", "mesh-infra", "person:owner", "project:", "project:Mesh", "project:../etc"],
)
def test_invalid_entity_keys_refused(mesh_home, bad):
    """A malformed key can never steer a derived path out of its directory."""
    with pytest.raises(DossierError):
        safe_entity_key(bad)


def test_derived_paths(mesh_home):
    assert dossier_path(KEY) == mesh_home / "digests" / "project-mesh-infra.md"
    assert report_path(KEY, "2026-07-30", 1) == (
        mesh_home / "reports" / "project-mesh-infra-2026-07-30-1.md"
    )
    assert budget_path(KEY) == mesh_home / "budget" / "project-mesh-infra.json"


@pytest.mark.parametrize("bad_date", ["2026-7-30", "20260730", "tomorrow", ""])
def test_report_path_rejects_bad_date(mesh_home, bad_date):
    with pytest.raises(DossierError):
        report_path(KEY, bad_date, 1)


@pytest.mark.parametrize("bad_seq", [0, -1, "abc", None])
def test_report_path_rejects_bad_seq(mesh_home, bad_seq):
    with pytest.raises(DossierError):
        report_path(KEY, "2026-07-30", bad_seq)


# ─────────────────────────────────────────────────────────────────────
# Skeleton and constitution (§3.2)
# ─────────────────────────────────────────────────────────────────────


def test_skeleton_has_all_seven_sections(mesh_home):
    body = render_skeleton(KEY, owner_agent="coder1")
    assert section_errors(body) == []
    for heading in REQUIRED_SECTIONS:
        assert f"## {heading}\n" in body
    assert len(REQUIRED_SECTIONS) == 7


def test_skeleton_validates_against_its_own_key(mesh_home):
    body = render_skeleton(KEY, owner_agent="coder1")
    assert validate_dossier(KEY, body) == []
    # Same bytes, different project — the frontmatter identity check fires.
    errors = validate_dossier("project:rec-fishing", body)
    assert any("entity_key" in e for e in errors)


def test_init_creates_valid_dossier(mesh_home):
    path = init_dossier(KEY, owner_agent="coder1")
    assert path == dossier_path(KEY)
    assert path.exists()
    assert validate_dossier(KEY, path.read_text()) == []


def test_init_never_overwrites(dossier):
    dossier.write_text(dossier.read_text() + "\nseeded by Project Owner\n")
    before = dossier.read_text()
    init_dossier(KEY, owner_agent="someone-else")
    assert dossier.read_text() == before


def test_read_dossier_fails_closed_when_missing(mesh_home):
    with pytest.raises(DossierError, match="No dossier"):
        read_dossier(KEY)


def test_section_errors_detects_missing_duplicate_and_reordered(mesh_home):
    body = render_skeleton(KEY)

    missing = body.replace("## Narrative", "## Nrrative")
    assert any("missing required section: Narrative" in e for e in section_errors(missing))

    duplicated = body + "\n## Goals\n\nsecond copy\n"
    assert any("duplicate section: Goals" in e for e in section_errors(duplicated))

    empty = body.replace(
        "## Open threads\n\n- Dossier is a fresh skeleton and needs seeding "
        "before autonomous work begins.\n",
        "## Open threads\n\n",
    )
    assert any("empty required section: Open threads" in e for e in section_errors(empty))


def test_section_errors_detects_broken_tasks_block(mesh_home):
    body = render_skeleton(KEY)
    assert any(
        "Tasks block" in e for e in section_errors(body.replace(pd.TASKS_END, ""))
    )


# ─────────────────────────────────────────────────────────────────────
# dossier_edit (§3.9)
# ─────────────────────────────────────────────────────────────────────


def test_edit_applies_exact_replacement(dossier):
    result = edit_dossier(KEY, "[pending] [P2]", "[in_progress] [P0]")
    assert result.replacements == 1
    assert "[in_progress] [P0]" in dossier.read_text()
    assert result.tokens <= result.token_budget


def test_edit_validates_sections_after_edit(dossier):
    """A valid edit still passes the whole constitution, not just its own span."""
    result = edit_dossier(KEY, "## Narrative", "## Narrative")
    assert result.replacements == 1
    assert section_errors(dossier.read_text()) == []


def test_edit_refuses_removing_a_required_section(dossier):
    before = dossier.read_text()
    with pytest.raises(DossierError, match="constitution"):
        edit_dossier(KEY, "## Standing decisions", "## Decisions")
    assert dossier.read_text() == before, "refused edit must leave the file byte-identical"


def test_edit_refuses_emptying_a_required_section(dossier):
    before = dossier.read_text()
    body = (
        "- `D-001` The autonomous controller is the sole routine writer of this\n"
        "  dossier. Workers may read it; they must never edit it."
    )
    assert body in before
    with pytest.raises(DossierError, match="empty required section"):
        edit_dossier(KEY, body, "")
    assert dossier.read_text() == before


def _seed_two_markers(key: str) -> None:
    edit_dossier(key, "## Narrative\n", "## Narrative\n\nmarker-text\n")
    edit_dossier(key, "## Open threads\n", "## Open threads\n\nmarker-text\n")


def test_edit_refuses_ambiguous_match(dossier):
    _seed_two_markers(KEY)
    before = dossier.read_text()
    with pytest.raises(DossierError, match="matches 2 locations"):
        edit_dossier(KEY, "marker-text", "x")
    assert dossier.read_text() == before


def test_edit_replace_all_is_opt_in(dossier):
    _seed_two_markers(KEY)
    result = edit_dossier(KEY, "marker-text", "replaced", replace_all=True)
    assert result.replacements == 2
    assert "marker-text" not in dossier.read_text()


def test_edit_refuses_missing_old_text(dossier):
    with pytest.raises(DossierError, match="not found"):
        edit_dossier(KEY, "text that is not present", "x")


def test_edit_refuses_over_budget_body(dossier):
    """The hard token ceiling is enforced; the old dossier survives intact."""
    before = dossier.read_text()
    bloat = "autonomous session evidence paragraph. " * 6000
    with pytest.raises(DossierError, match="over budget"):
        edit_dossier(KEY, "## Narrative\n", f"## Narrative\n\n{bloat}\n")
    assert dossier.read_text() == before


# ─────────────────────────────────────────────────────────────────────
# Session reports (§4.2)
# ─────────────────────────────────────────────────────────────────────


REPORT = """---
schema_version: 1
record_type: autonomous_session_report
session_id: as-20260730T130000Z-a1b2c3d4
project_entity_key: project:mesh-infra
status: partial
---

# Autonomous session as-20260730T130000Z-a1b2c3d4

## Session summary

Ran one worker against T-001.
"""


def test_write_report_creates_file_and_links_timeline(dossier):
    path = write_report(KEY, "2026-07-30", 1, REPORT)

    assert path == report_path(KEY, "2026-07-30", 1)
    assert path.read_text() == REPORT

    text = dossier.read_text()
    assert path.name in text
    assert "2026-07-30 — autonomous session 1" in text
    # Link is relative, and resolves from the dossier's own directory.
    assert "../reports/project-mesh-infra-2026-07-30-1.md" in text
    assert (dossier.parent / "../reports/project-mesh-infra-2026-07-30-1.md").resolve() == (
        path.resolve()
    )
    assert section_errors(text) == []


def test_write_report_links_newest_first(dossier):
    write_report(KEY, "2026-07-30", 1, REPORT)
    write_report(KEY, "2026-07-31", 1, REPORT.replace("a1b2c3d4", "b2c3d4e5"))
    text = dossier.read_text()
    assert text.index("2026-07-31") < text.index("2026-07-30")


def test_write_report_is_idempotent_for_identical_bytes(dossier):
    first = write_report(KEY, "2026-07-30", 1, REPORT)
    after_first = dossier.read_text()
    second = write_report(KEY, "2026-07-30", 1, REPORT)
    assert first == second
    assert dossier.read_text() == after_first, "no duplicate Timeline entry"


def test_write_report_refuses_to_overwrite(dossier):
    path = write_report(KEY, "2026-07-30", 1, REPORT)
    with pytest.raises(DossierError, match="immutable"):
        write_report(KEY, "2026-07-30", 1, REPORT + "\namended\n")
    assert path.read_text() == REPORT


def test_write_report_requires_a_dossier(mesh_home):
    """A report that cannot be linked is not a usable audit trail."""
    with pytest.raises(DossierError, match="No dossier"):
        write_report(KEY, "2026-07-30", 1, REPORT)
    assert not report_path(KEY, "2026-07-30", 1).exists()


def test_write_report_refuses_empty_content(dossier):
    with pytest.raises(DossierError, match="non-empty"):
        write_report(KEY, "2026-07-30", 1, "   ")


# ─────────────────────────────────────────────────────────────────────
# Budget ledger (§8)
# ─────────────────────────────────────────────────────────────────────


def test_check_budget_starts_full(dossier):
    state = check_budget(KEY)
    assert state["used"] == 0
    assert state["limit"] == pd.DEFAULT_WORKERS_PER_DAY
    assert state["remaining"] == pd.DEFAULT_WORKERS_PER_DAY
    assert state["resets_at"]
    assert not budget_path(KEY).exists(), "a read must not create the ledger"


def test_spend_budget_increments_and_persists(dossier):
    assert spend_budget(KEY)["used"] == 1
    assert spend_budget(KEY, 2)["used"] == 3
    assert check_budget(KEY)["remaining"] == pd.DEFAULT_WORKERS_PER_DAY - 3
    # Survives a fresh read of the on-disk ledger.
    assert json.loads(budget_path(KEY).read_text())["used"] == 3


def test_reset_budget_clears_spent_count_and_preserves_configured_limit(mesh_home):
    init_dossier(KEY, owner_agent="coder1", max_workers_per_day=7)
    spend_budget(KEY, 5)

    state = reset_budget(KEY)

    assert state["used"] == 0
    assert state["limit"] == 7
    assert state["remaining"] == 7
    assert set(state) == set(check_budget(KEY))
    assert json.loads(budget_path(KEY).read_text()) == {
        "date": state["date"],
        "used": 0,
    }
    assert check_budget(KEY)["used"] == 0
    assert reset_budget(KEY)["used"] == 0


def test_reset_budget_initializes_a_missing_ledger(dossier):
    assert not budget_path(KEY).exists()

    state = reset_budget(KEY)

    assert state["used"] == 0
    assert state["remaining"] == state["limit"]
    assert json.loads(budget_path(KEY).read_text())["used"] == 0


def test_spend_budget_refuses_when_exhausted(dossier):
    limit = check_budget(KEY)["limit"]
    spend_budget(KEY, limit)
    with pytest.raises(BudgetExhausted, match="exhausted"):
        spend_budget(KEY)
    assert check_budget(KEY)["used"] == limit, "a refusal must not mutate the ledger"


def test_spend_budget_refuses_oversized_single_spend(dossier):
    with pytest.raises(BudgetExhausted):
        spend_budget(KEY, pd.DEFAULT_WORKERS_PER_DAY + 1)
    assert check_budget(KEY)["used"] == 0


def test_budget_limit_comes_from_dossier_frontmatter(mesh_home):
    init_dossier(KEY, owner_agent="coder1", max_workers_per_day=2)
    assert check_budget(KEY)["limit"] == 2
    spend_budget(KEY, 2)
    with pytest.raises(BudgetExhausted):
        spend_budget(KEY)


def test_budget_rolls_over_on_a_new_day(dossier):
    spend_budget(KEY, 2)
    stale = json.loads(budget_path(KEY).read_text())
    stale["date"] = "2020-01-01"
    budget_path(KEY).write_text(json.dumps(stale))
    assert check_budget(KEY)["used"] == 0


def test_budget_survives_a_corrupt_ledger(dossier):
    budget_path(KEY).parent.mkdir(parents=True, exist_ok=True)
    budget_path(KEY).write_text("{ not json")
    assert check_budget(KEY)["used"] == 0
    assert spend_budget(KEY)["used"] == 1


@pytest.mark.parametrize("bad", [0, -1, "abc"])
def test_spend_budget_rejects_bad_count(dossier, bad):
    with pytest.raises(DossierError):
        spend_budget(KEY, bad)


def test_budgets_are_isolated_per_project(mesh_home):
    init_dossier(KEY, owner_agent="coder1")
    init_dossier("project:rec-fishing", owner_agent="tron")
    spend_budget(KEY, 2)
    assert check_budget(KEY)["used"] == 2
    assert check_budget("project:rec-fishing")["used"] == 0


# ─────────────────────────────────────────────────────────────────────
# Tool registration and prompt wiring
# ─────────────────────────────────────────────────────────────────────


DOSSIER_TOOLS = (
    "dossier_read",
    "dossier_edit",
    "dossier_write_report",
    "dossier_check_budget",
    "dossier_spend_budget",
)


def test_dossier_tools_are_registered():
    import mesh.tool_implementations  # noqa: F401 — registers the tools

    names = set(get_registry().list_names())
    for name in DOSSIER_TOOLS:
        assert name in names


def test_dossier_tools_are_router_tools():
    from mesh.router_v2 import ROUTER_TOOL_NAMES

    for name in DOSSIER_TOOLS:
        assert name in ROUTER_TOOL_NAMES


def test_dossier_tools_return_errors_as_strings(mesh_home):
    """Tool wrappers surface refusals as text; they never raise into the loop."""
    from mesh.tool_implementations import (
        dossier_check_budget,
        dossier_edit,
        dossier_read,
        dossier_spend_budget,
        dossier_write_report,
    )

    assert dossier_read(KEY).startswith("Error:")
    assert dossier_edit(KEY, "a", "b").startswith("Error:")
    assert dossier_write_report(KEY, "2026-07-30", 1, REPORT).startswith("Error:")
    assert dossier_read("not-a-project-key").startswith("Error:")
    assert dossier_check_budget("not-a-project-key").startswith("Error:")
    assert dossier_spend_budget("not-a-project-key").startswith("Error:")


def test_dossier_tool_roundtrip(dossier):
    from mesh.tool_implementations import (
        dossier_check_budget,
        dossier_edit,
        dossier_read,
        dossier_spend_budget,
        dossier_write_report,
    )

    assert "## Tasks" in dossier_read(KEY)
    assert dossier_edit(KEY, "[pending] [P2]", "[in_progress] [P0]").startswith(
        "Dossier updated"
    )
    assert "linked from the dossier Timeline" in dossier_write_report(
        KEY, "2026-07-30", 1, REPORT
    )
    assert json.loads(dossier_check_budget(KEY))["used"] == 0
    assert json.loads(dossier_spend_budget(KEY))["used"] == 1


def test_budget_exhaustion_is_a_structured_tool_refusal(dossier):
    from mesh.tool_implementations import dossier_spend_budget

    spend_budget(KEY, check_budget(KEY)["limit"])
    payload = json.loads(dossier_spend_budget(KEY))
    assert payload["status"] == "autonomous_budget_exhausted"


def test_controller_prompt_exists_and_covers_the_cycle():
    path = Path(__file__).resolve().parents[1] / "prompts" / "autonomous_controller.txt"
    text = path.read_text()
    for marker in (
        "[RESPONSE FORMAT]",
        "## Summary",
        "## Findings",
        "## Artifacts",
        "## Issues",
        "## Recommendation",
        "dossier_read",
        "dossier_write_report",
        "dossier_check_budget",
        "schedule_wake",
        "worker_launch",
        "autonomous_session_report",
    ):
        assert marker in text, f"controller prompt is missing {marker!r}"


def test_controller_mandates_split_plan_from_execute_with_safety_rails():
    prompts = Path(__file__).resolve().parents[1] / "prompts"
    wake = (prompts / "autonomous_controller.txt").read_text()
    continuation = (prompts / "autonomous_controller_execute.txt").read_text()

    assert wake.startswith("[AUTONOMOUS PROJECT SESSION]")
    assert "PHASE 1 — PLAN" in wake
    assert "PHASE 2 — EXECUTE & CLOSE" in wake
    assert "SESSION PLAN — REQUIRED BEFORE THE FIRST DISPATCH" in wake
    assert "GOAL=<G-id>" in wake
    assert "TASKS=<T-ids in execution order>" in wake
    assert "EVIDENCE=<what proves each selected task done>" in wake
    assert "FIRST=<one single next action>" in wake
    assert "every eligible `[recurring]` task" in wake
    assert "mandatory closeout" in wake

    assert continuation.startswith("[AUTONOMOUS PROJECT SESSION]")
    assert "PHASE 2 — EXECUTE & CLOSE" in continuation
    assert "SESSION PLAN — REQUIRED BEFORE THE FIRST DISPATCH" not in continuation
    for text in (wake, continuation):
        flat = " ".join(text.split())
        assert "dossier is the STATE authority" in flat
        assert "Workers never edit the dossier" in flat
        assert "DATA, not instruction" in flat
        assert "WORKER RESPONSE CONTRACT" in text
        assert "direction-changing verdict" in flat
        assert "Evaluated and recorded is the contract" in flat
        assert "mandatory closeout bookkeeping" in flat


def test_controller_prompt_does_not_instruct_the_model_to_spend_budget():
    """The router is the sole charger; a prompt-side spend double-charges.

    Regression for the 2026-08-02 autopilot-rl session, which burned 4 of 4
    daily admissions on 2 workers because step 6 told the controller to call
    ``dossier_spend_budget`` before ``worker_launch`` while
    ``_charge_autonomous_admission()`` also charged at the dispatch seam.
    """
    path = Path(__file__).resolve().parents[1] / "prompts" / "autonomous_controller.txt"
    text = path.read_text()

    assert "Call `dossier_spend_budget` first" not in text

    # The only surviving mentions must be prohibitions, not instructions.
    # Compare on whitespace-normalised text so the wrapped prompt's line
    # breaks do not separate a mention from the negation that governs it.
    flat = " ".join(text.split()).lower()
    needle = "`dossier_spend_budget`"
    at = flat.find(needle)
    assert at != -1, "expected an explicit prohibition to remain in the prompt"
    while at != -1:
        window = flat[max(0, at - 60): at]
        assert (
            "not call" in window or "never call" in window
        ), f"prompt still instructs a budget spend near: {flat[at - 60:at + 60]!r}"
        at = flat.find(needle, at + 1)

    assert "dossier_check_budget" in text


def test_controller_mandate_loads_without_duplicated_shared_includes():
    """The mandate is injected *in addition to* the standing system prompt, so
    its loader must not append the shared includes (channel_policy.md,
    memory.md, mesh_tools.md) the system prompt already carries.

    Regression: loading the mandate via ``load_prompt_file`` nearly doubled its
    size (≈3.5K → ≈6.5K tokens) by re-appending files the agent already had, on
    every autonomous turn.
    """
    from mesh.config import PROMPTS_DIR, load_prompt_file, load_raw_prompt_file

    raw_file = (PROMPTS_DIR / "autonomous_controller.txt").read_text().strip()
    raw = load_raw_prompt_file("autonomous_controller.txt")
    full = load_prompt_file("autonomous_controller.txt")

    # Raw loader returns the file verbatim — no appended includes.
    assert raw == raw_file
    assert raw.startswith("[AUTONOMOUS PROJECT SESSION]")
    # The include-appending loader pulls in the three shared files the system
    # prompt already carries; the raw loader must leave them out.
    assert len(full) > len(raw)
    for shared in ("channel_policy.md", "memory.md", "mesh_tools.md"):
        shared_text = (PROMPTS_DIR / shared).read_text().strip()
        assert shared_text in full
        assert shared_text not in raw

    continuation = load_raw_prompt_file("autonomous_controller_execute.txt")
    assert continuation.startswith("[AUTONOMOUS PROJECT SESSION]")
    assert "PHASE 2 — EXECUTE & CLOSE" in continuation
    assert "PHASE 1 — PLAN" not in continuation


def test_autonomous_config_fields_default_off():
    from mesh.config import NodeConfig

    config = NodeConfig(id="agent:coder:test")
    assert config.autonomous_agent_mode_enabled is False
    assert config.autonomous_projects == []
    assert config.autonomous_controller_prompt_file == "autonomous_controller.txt"
    assert config.autonomous_controller_continuation_prompt_file == ""


def _node(**overrides):
    from mesh.agent_node import AgentNode
    from mesh.config import NodeConfig

    kwargs = dict(id="agent:coder:autopilot-test", tools=["send_message"])
    kwargs.update(overrides)
    return AgentNode(NodeConfig(**kwargs), tool_registry=get_registry())


def test_autonomous_control_response_sends_preconstructed_message():
    """Agent control responses bypass ``Node.send(to_node, content)``."""
    import asyncio
    from types import SimpleNamespace

    from mesh.agent_node import AgentNode
    from mesh.protocol import Message, MessageType

    send_calls: list[tuple[tuple, dict]] = []
    sent_messages: list[Message] = []

    async def send(*args, **kwargs):
        send_calls.append((args, kwargs))

    async def send_message(message: Message):
        sent_messages.append(message)

    node = SimpleNamespace(
        node_id="agent:coder:autopilot-test",
        send=send,
        send_message=send_message,
    )

    asyncio.run(
        AgentNode._send_autonomous_control_response(
            node,
            "user:testuser",
            "status",
            accepted=True,
            result={"wakes": []},
        )
    )

    assert send_calls == []
    assert len(sent_messages) == 1
    response = sent_messages[0]
    assert response.type is MessageType.CONTROL
    assert response.to_node == "user:testuser"
    assert response.content["action"] == "autonomous_control_response"
    assert response.content["agent"] == "agent:coder:autopilot-test"


def test_agent_shutdown_sends_preconstructed_message():
    """Shutdown requests bypass ``Node.send(to_node, content)``."""
    import asyncio
    from types import SimpleNamespace

    from mesh.agent_node import AgentNode
    from mesh.protocol import ControlAction, Message, MessageType

    send_calls: list[tuple[tuple, dict]] = []
    sent_messages: list[Message] = []

    async def send(*args, **kwargs):
        send_calls.append((args, kwargs))

    async def send_message(message: Message):
        sent_messages.append(message)

    node = SimpleNamespace(
        node_id="agent:coder:autopilot-test",
        _auth_token="test-token",
        send=send,
        send_message=send_message,
    )

    result = asyncio.run(
        AgentNode._execute_agent_shutdown(
            node,
            {"target": "agent:coder:target", "reason": "test shutdown"},
        )
    )

    assert result.startswith("Shutdown request sent")
    assert send_calls == []
    assert len(sent_messages) == 1
    request = sent_messages[0]
    assert request.type is MessageType.CONTROL
    assert request.to_node == "agent:coder:target"
    assert request.content["action"] == ControlAction.SHUTDOWN.value


def test_enrollment_adds_dossier_tools_and_loads_the_mandate():
    """Enrollment arms the mandate; it does not put the agent under it.

    §10.1: an enrolled agent still holds ordinary conversations.  The mandate
    text is held for per-turn injection, never concatenated into the standing
    system prompt.
    """
    node = _node(
        autonomous_agent_mode_enabled=True,
        autonomous_projects=[KEY],
        system_prompt="base coder prompt",
    )
    for name in DOSSIER_TOOLS:
        assert name in node.enabled_tools
    assert node.system_prompt == "base coder prompt"
    assert "[AUTONOMOUS PROJECT SESSION]" not in node.system_prompt
    assert "[AUTONOMOUS PROJECT SESSION]" in node._autonomous_mandate_prompt
    # The router receives the text so it can inject it per turn.
    assert (
        node._router_v2_config.autonomous_mandate_prompt
        == node._autonomous_mandate_prompt
    )
    # Empty continuation config preserves the prior full-mandate behavior.
    assert (
        node._router_v2_config.autonomous_continuation_mandate_prompt
        == node._autonomous_mandate_prompt
    )


def test_enrollment_loads_the_configured_execute_only_continuation_mandate():
    node = _node(
        autonomous_agent_mode_enabled=True,
        autonomous_projects=[KEY],
        autonomous_controller_continuation_prompt_file=(
            "autonomous_controller_execute.txt"
        ),
    )

    assert "PHASE 1 — PLAN" in node._autonomous_mandate_prompt
    assert "PHASE 2 — EXECUTE & CLOSE" in (
        node._autonomous_continuation_mandate_prompt
    )
    assert "PHASE 1 — PLAN" not in (
        node._autonomous_continuation_mandate_prompt
    )
    assert node._router_v2_config.autonomous_continuation_mandate_prompt == (
        node._autonomous_continuation_mandate_prompt
    )


def test_unenrolled_agent_gets_neither_tools_nor_mandate():
    """The dossier tools are stripped even if a YAML tool list names them."""
    node = _node(tools=["send_message", "dossier_edit"], system_prompt="base coder prompt")
    for name in DOSSIER_TOOLS:
        assert name not in node.enabled_tools
    assert "[AUTONOMOUS PROJECT SESSION]" not in node.system_prompt
    assert node._autonomous_mandate_prompt == ""


def test_recursive_controller_tool_is_off_by_default():
    """§10.3: no second planner competing with the ReAct loop."""
    node = _node(tools=["send_message", "autonomous_controller_run"])
    assert "autonomous_controller_run" not in node.enabled_tools

    enabled = _node(
        tools=["send_message", "autonomous_controller_run"],
        autonomous_recursive_controller_enabled=True,
    )
    assert "autonomous_controller_run" in enabled.enabled_tools


def test_recursive_controller_run_is_refused_when_disabled():
    """Both doors — router tool loop and agent socket — hit the same gate."""
    import asyncio

    node = _node(autonomous_agent_mode_enabled=True, autonomous_projects=[KEY])
    payload = json.loads(
        asyncio.run(
            node._execute_autonomous_controller_run({"smoke": "mesh_infra_smoke"})
        )
    )
    assert payload["status"] == "disabled"
    assert "autonomous_recursive_controller_enabled" in payload["message"]


def test_node_config_supports_an_enrolled_controller_without_live_fleet_config():
    """Public release tests must not open a private deployment mesh.yaml."""
    from mesh.config import NodeConfig

    node = NodeConfig(
        id="agent:coder:example-controller",
        autonomous_agent_mode_enabled=True,
        autonomous_projects=["project:example-project"],
        autonomous_max_workers_per_session=3,
    )
    assert node.autonomous_agent_mode_enabled is True
    assert isinstance(node.autonomous_projects, list)
    assert len(node.autonomous_projects) == 1
    project_key = node.autonomous_projects[0]
    assert isinstance(project_key, str) and project_key.startswith("project:")
    assert isinstance(node.autonomous_max_workers_per_session, int)
    assert node.autonomous_max_workers_per_session > 0
    # Enrollment is opt-in: the shared tool list must not carry dossier tools.
    assert not [t for t in node.tools if t.startswith("dossier_")]


# ─────────────────────────────────────────────────────────────────────
# Hard admission guard at RouterV2._dispatch_worker() (§8.4, §8.7, §8.9)
# ─────────────────────────────────────────────────────────────────────
#
# These drive the real dispatch seam against the real budget ledger.  A mock
# router could not show what is actually under test: that the guard sits
# *inside* the shared seam both dispatch doors funnel through, and that every
# failure mode of the guard itself lets the worker through rather than
# silently denying it.


TAGGED_TASK = (
    "[PROJECT: project:mesh-infra] Inspect the worker dispatch admission "
    "seam in /home/testuser/log/dev/mesh, confirm the budget guard ordering "
    "against plan section 8.7, and report the receipt fields with evidence."
)
UNTAGGED_TASK = (
    "Inspect the worker dispatch admission seam in /home/testuser/log/dev/mesh, "
    "confirm the budget guard ordering against plan section 8.7, and report "
    "the receipt fields with file and line evidence."
)


def _dispatch_trigger(content: str = "The user message that woke the router."):
    from datetime import datetime, timezone

    from mesh.protocol import Message, MessageType

    return Message(
        id="dispatch-trigger",
        type=MessageType.MESSAGE,
        from_node="user:testuser",
        to_node="agent:coder:autopilot",
        content=content,
        timestamp=datetime.now(timezone.utc).isoformat(),
        metadata={},
    )


def _dispatch_router(*, autonomous: bool):
    """A RouterV2 whose only stubbed part is the worker backend itself."""
    from mesh.router_v2 import RouterV2, RouterV2Config, WorkerResult

    async def worker(_context, _trigger):
        return WorkerResult(response="done", context=[])

    async def send(_content, _trigger):
        return None

    config = RouterV2Config(
        max_concurrent_workers=2,
        history_persist=False,
        router_mode="full",
        autonomous_agent_mode_enabled=autonomous,
        autonomous_projects=[KEY] if autonomous else [],
    )
    config.min_worker_brief_chars = 120
    router = RouterV2(
        worker_fn=worker,
        send_fn=send,
        config=config,
        node_id="agent:coder:autopilot",
        nickname="autopilot",
        agent_type="coder",
    )
    router._trigger_nodes = lambda: ("user:testuser", "agent:coder:autopilot")
    return router


def _dispatch(router, task: str, *, source: str = "tool"):
    """Run one dispatch, recording whether the worker was actually started."""
    import asyncio

    from mesh.router_v2 import DispatchReceipt

    started: list = []

    async def start(trigger):
        started.append(trigger)
        router._last_dispatch_receipt = DispatchReceipt(
            dispatch_key="dk-1",
            status="running",
            worker_id="autopilot-worker1",
            slot_index=0,
            origin_message_id=trigger.id,
            router_turn_id="turn-1",
            task_description=trigger.metadata.get("worker_task_description", ""),
            backend=None,
        )
        return True

    router._start_worker = start
    trigger = _dispatch_trigger()
    outcome = asyncio.run(
        router._dispatch_worker(trigger, {"task": task}, source=source)
    )
    return outcome, started, trigger


def test_dispatch_is_refused_when_the_project_budget_is_exhausted(dossier):
    spend_budget(KEY, check_budget(KEY)["limit"])
    router = _dispatch_router(autonomous=True)

    outcome, started, _ = _dispatch(router, TAGGED_TASK)

    assert outcome.status == "autonomous_budget_exhausted"
    assert started == [], "an exhausted budget must not start a worker"
    assert outcome.autonomous_session is True
    assert outcome.project_key == KEY
    assert outcome.session_id
    payload = json.loads(outcome.message.rsplit("\n", 1)[-1])
    assert payload["status"] == "autonomous_budget_exhausted"
    assert payload["project_entity_key"] == KEY
    assert payload["used"] == payload["allowed"] == check_budget(KEY)["limit"]
    assert payload["next_available_at"]


def test_dispatch_succeeds_when_budget_is_available(dossier):
    router = _dispatch_router(autonomous=True)

    outcome, started, trigger = _dispatch(router, TAGGED_TASK)

    assert outcome.status == "running"
    assert len(started) == 1
    assert outcome.autonomous_session is True
    assert outcome.project_key == KEY
    assert trigger.metadata["autonomous_project_key"] == KEY


def test_successful_dispatch_charges_exactly_one_admission(dossier):
    before = check_budget(KEY)["used"]
    router = _dispatch_router(autonomous=True)

    _dispatch(router, TAGGED_TASK)

    assert check_budget(KEY)["used"] == before + 1
    # The charge is durable, not just in-process.
    assert json.loads(budget_path(KEY).read_text())["used"] == before + 1


def test_a_failed_start_still_consumes_the_admission(dossier):
    """§8.7: the charge is a reservation, taken before ``_start_worker()``.

    Charging only on success lets a controller whose backend is down retry
    forever at zero budget cost — a retry storm against broken infrastructure.
    Reserving instead means those launches burn the budget and the controller
    locks itself out, which is the intended failure posture.
    """
    import asyncio

    before = check_budget(KEY)["used"]
    router = _dispatch_router(autonomous=True)

    async def start(trigger):
        router._last_dispatch_receipt = None
        return False

    router._start_worker = start
    outcome = asyncio.run(
        router._dispatch_worker(
            _dispatch_trigger(), {"task": TAGGED_TASK}, source="tool"
        )
    )

    assert outcome.status == "start_failed"
    assert check_budget(KEY)["used"] == before + 1, (
        "a failed start must not be refunded"
    )
    assert json.loads(budget_path(KEY).read_text())["used"] == before + 1


def test_refused_dispatch_does_not_consume_an_admission(dossier):
    limit = check_budget(KEY)["limit"]
    spend_budget(KEY, limit)
    router = _dispatch_router(autonomous=True)

    _dispatch(router, TAGGED_TASK)

    assert check_budget(KEY)["used"] == limit


def test_guard_is_a_no_op_when_autonomous_mode_is_disabled(dossier):
    spend_budget(KEY, check_budget(KEY)["limit"])
    router = _dispatch_router(autonomous=False)

    outcome, started, trigger = _dispatch(router, TAGGED_TASK)

    assert outcome.status == "running", "an unenrolled agent is never gated"
    assert len(started) == 1
    assert outcome.autonomous_session is False
    assert outcome.project_key == ""
    assert "autonomous_session" not in trigger.metadata


def test_corrupt_budget_ledger_fails_open(dossier, monkeypatch, caplog):
    """Budget *infrastructure* failure must not deny a legitimate worker."""
    import mesh.project_dossier as _pd

    def boom(_key):
        raise OSError("ledger unreadable")

    monkeypatch.setattr(_pd, "check_budget", boom)
    router = _dispatch_router(autonomous=True)

    with caplog.at_level("WARNING"):
        outcome, started, _ = _dispatch(router, TAGGED_TASK)

    assert outcome.status == "running"
    assert len(started) == 1
    assert "failing open" in caplog.text


def test_unparseable_ledger_bytes_still_admit_and_charge(dossier):
    """A garbage ledger file rolls to a fresh day rather than blocking."""
    budget_path(KEY).parent.mkdir(parents=True, exist_ok=True)
    budget_path(KEY).write_text("{ not json")
    router = _dispatch_router(autonomous=True)

    outcome, started, _ = _dispatch(router, TAGGED_TASK)

    assert outcome.status == "running"
    assert len(started) == 1
    assert check_budget(KEY)["used"] == 1


def test_missing_project_tag_fails_open(dossier, caplog):
    spend_budget(KEY, check_budget(KEY)["limit"])
    router = _dispatch_router(autonomous=True)

    with caplog.at_level("WARNING"):
        outcome, started, trigger = _dispatch(router, UNTAGGED_TASK)

    assert outcome.status == "running", "prompt enforcement is the primary gate"
    assert len(started) == 1
    assert outcome.autonomous_session is False
    assert "autonomous_session" not in trigger.metadata
    assert "failing open" in caplog.text
    # Fail-open means fail-open: an untagged launch is not charged either.
    assert check_budget(KEY)["used"] == check_budget(KEY)["limit"]


def test_malformed_project_tag_fails_open(dossier, caplog):
    bad = UNTAGGED_TASK + " [PROJECT: project:../../etc]"
    router = _dispatch_router(autonomous=True)

    with caplog.at_level("WARNING"):
        outcome, started, _ = _dispatch(router, bad)

    assert outcome.status == "running"
    assert len(started) == 1
    assert check_budget(KEY)["used"] == 0
    assert "malformed project tag" in caplog.text


def test_duplicate_refusal_is_decided_before_any_admission_is_charged(dossier):
    """§8.7: refusals that never reach _start_worker() cost nothing."""
    router = _dispatch_router(autonomous=True)
    _dispatch(router, TAGGED_TASK)
    assert check_budget(KEY)["used"] == 1

    # A second launch in the same router turn is refused upstream of the guard.
    outcome, started, _ = _dispatch(router, TAGGED_TASK)

    assert outcome.status == "duplicate_in_turn"
    assert started == []
    assert check_budget(KEY)["used"] == 1


# ─────────────────────────────────────────────────────────────────────
# Trusted session metadata on the worker-completion trigger (§17 item 2)
# ─────────────────────────────────────────────────────────────────────


def _complete_via_report(router, trigger, worker_id="autopilot-worker1"):
    """Drive the real report-as-trigger completion path and capture its wake."""
    import asyncio

    from mesh.router_v2 import WorkerLifecycle, WorkerResult

    slot = router._ensure_slot_table()[0]
    slot.worker_id = worker_id
    slot.lifecycle = WorkerLifecycle.RUNNING
    slot.trigger = trigger
    slot.task_description = trigger.metadata.get("worker_task_description", "")
    router._sync_worker_compat_views()

    captured: list = []
    router._enqueue_report_wake = captured.append

    result = WorkerResult(
        response="done",
        context=[],
        report_sent=True,
        buffered_messages=[("user:testuser", "worker report body")],
    )
    asyncio.run(
        router._handle_worker_complete(result, trigger, worker_id=worker_id)
    )
    return captured


def test_completion_trigger_carries_autonomous_session_metadata(dossier):
    router = _dispatch_router(autonomous=True)
    _outcome, _started, trigger = _dispatch(router, TAGGED_TASK)
    session_id = trigger.metadata["autonomous_session_id"]
    trigger.metadata["autonomous_report_to"] = "channel:mesh-infra"

    captured = _complete_via_report(router, trigger)

    assert captured, "the report-as-trigger path must produce a wake trigger"
    meta = captured[0].metadata
    assert meta["worker_report"] is True
    assert meta["autonomous_session"] is True
    assert meta["autonomous_project_key"] == KEY
    assert meta["autonomous_session_id"] == session_id
    assert meta["autonomous_report_to"] == "channel:mesh-infra"
    assert meta["response_destination"] == "channel:mesh-infra"


def test_completion_trigger_is_unmarked_for_an_interactive_dispatch(dossier):
    router = _dispatch_router(autonomous=False)
    _outcome, _started, trigger = _dispatch(router, TAGGED_TASK)

    captured = _complete_via_report(router, trigger)

    assert captured
    meta = captured[0].metadata
    assert meta["worker_report"] is True
    assert "autonomous_session" not in meta
    assert "autonomous_project_key" not in meta


def test_completion_metadata_is_absent_for_interactive_dispatches(dossier):
    router = _dispatch_router(autonomous=False)
    _outcome, _started, trigger = _dispatch(router, TAGGED_TASK)

    assert router.autonomous_completion_metadata(trigger) == {}


def test_completion_metadata_helper_ignores_untrusted_partial_marks():
    """A project key without the router-set flag confers no autonomous scope."""
    from mesh.router_v2 import RouterV2

    trigger = _dispatch_trigger()
    trigger.metadata["autonomous_project_key"] = KEY
    assert RouterV2.autonomous_completion_metadata(trigger) == {}


# ─────────────────────────────────────────────────────────────────────
# Intermediate autonomous-session response routing
# ─────────────────────────────────────────────────────────────────────


def _autonomous_intermediate_trigger(report_to: str):
    trigger = _dispatch_trigger()
    trigger.metadata.update(
        {
            "autonomous_session": True,
            "autonomous_report_to": report_to,
        }
    )
    return trigger


def test_autonomous_intermediate_reply_routes_to_session_channel():
    node = _node()

    assert node._infer_destination_from_trigger(
        _autonomous_intermediate_trigger("channel:mesh-infra")
    ) == "channel:mesh-infra"


def test_autonomous_intermediate_reply_routes_to_session_user():
    node = _node()

    assert node._infer_destination_from_trigger(
        _autonomous_intermediate_trigger("user:jessica")
    ) == "user:jessica"


def test_worker_report_destination_wins_over_autonomous_session_target():
    node = _node()
    trigger = _autonomous_intermediate_trigger("channel:mesh-infra")
    trigger.metadata.update(
        {"worker_report": True, "response_destination": "user:testuser"}
    )

    assert node._infer_destination_from_trigger(trigger) == "user:testuser"


def test_channel_trigger_wins_over_autonomous_session_target():
    node = _node()
    trigger = _autonomous_intermediate_trigger("channel:mesh-infra")
    trigger.to_node = "channel:other"

    assert node._infer_destination_from_trigger(trigger) == "channel:other"


# ─────────────────────────────────────────────────────────────────────
# Autonomous session trigger detection at wake delivery (§7.1, §10.1)
# ─────────────────────────────────────────────────────────────────────
#
# The stamp is applied by the runtime, not by the model: a wake prompt names
# a project, and the agent decides whether that project is one it controls.


WAKE_PROMPT = f"""[AUTONOMOUS PROJECT SESSION]
schema_version: 1
project_entity_key: {KEY}
project_dossier: /home/testuser/.mesh/digests/project-mesh-infra.md
report_to: channel:mesh-infra
max_workers_this_session: 2

Run one bounded autonomous project session."""


def _wake(
    prompt: str,
    wake_id: str = "wake-abc123",
    requested_by: str = "user:testuser",
):
    from mesh.agent_node import ScheduledWake

    return ScheduledWake(
        id=wake_id,
        wake_time=datetime.now(timezone.utc),
        prompt=prompt,
        requested_by=requested_by,
    )


def test_autonomous_wake_is_stamped_with_trusted_session_scope():
    node = _node(autonomous_agent_mode_enabled=True, autonomous_projects=[KEY])

    meta = node.autonomous_wake_metadata(_wake(WAKE_PROMPT))

    assert meta["autonomous_session"] is True
    assert meta["autonomous_project_key"] == KEY
    assert meta["autonomous_report_to"] == "channel:mesh-infra"
    assert meta["autonomous_worker_limit"] == 2
    assert meta["autonomous_trigger_id"] == "wake-abc123"
    assert meta["autonomous_session_id"].startswith("as-")


def test_ordinary_wake_carries_no_autonomous_scope():
    node = _node(autonomous_agent_mode_enabled=True, autonomous_projects=[KEY])

    assert node.autonomous_wake_metadata(_wake("Remind me to check the fold.")) == {}


def test_wake_naming_an_unconfigured_project_is_not_autonomous():
    """A wake prompt cannot enrol the agent in a project it does not control."""
    node = _node(
        autonomous_agent_mode_enabled=True, autonomous_projects=["project:other"]
    )

    assert node.autonomous_wake_metadata(_wake(WAKE_PROMPT)) == {}


def test_wake_with_a_malformed_project_key_is_not_autonomous():
    node = _node(autonomous_agent_mode_enabled=True, autonomous_projects=[KEY])
    bad = WAKE_PROMPT.replace(f"project_entity_key: {KEY}", "project_entity_key: ../etc")

    assert node.autonomous_wake_metadata(_wake(bad)) == {}


def test_unenrolled_agent_never_opens_an_autonomous_session():
    node = _node()

    assert node.autonomous_wake_metadata(_wake(WAKE_PROMPT)) == {}


def test_wake_cannot_raise_the_configured_worker_limit():
    node = _node(
        autonomous_agent_mode_enabled=True,
        autonomous_projects=[KEY],
        autonomous_max_workers_per_session=2,
    )
    greedy = WAKE_PROMPT.replace(
        "max_workers_this_session: 2", "max_workers_this_session: 99"
    )

    assert node.autonomous_wake_metadata(_wake(greedy))["autonomous_worker_limit"] == 2


# ─────────────────────────────────────────────────────────────────────
# report_to derivation: a project's home channel
# ─────────────────────────────────────────────────────────────────────
#
# ``project:mesh-infra`` reports to ``channel:mesh-infra`` when the controller
# is a member of it, so the whole channel sees the session.  Membership is read
# from the agent's own configured channel list — never from a router
# round-trip, which would block the wake path on the network.


def _controller(**overrides):
    kwargs = dict(autonomous_agent_mode_enabled=True, autonomous_projects=[KEY])
    kwargs.update(overrides)
    return _node(**kwargs)


def _wake_prompt_for(node, **payload) -> str:
    """Run ``_autonomous_wake_result`` and return the prompt it authored."""
    captured: dict[str, str] = {}

    def schedule_wake(wake_time, prompt, requested_by="", recurrence=None):
        captured["prompt"] = prompt
        return {"status": "ok", "wake_id": "wake-test"}

    node.schedule_wake = schedule_wake
    node._autonomous_wake_result({"wake_time": "in 1 hour", **payload})
    return captured["prompt"]


def _report_result_for(node, **payload) -> tuple[dict, dict]:
    """Run ``_autonomous_report_result`` and return result plus wake call."""
    captured: dict = {}

    def schedule_wake(wake_time, prompt, requested_by="", recurrence=None):
        captured.update(
            wake_time=wake_time,
            prompt=prompt,
            requested_by=requested_by,
            recurrence=recurrence,
        )
        return {"status": "ok", "wake_id": "wake-report"}

    node.schedule_wake = schedule_wake
    return node._autonomous_report_result(payload), captured


def test_pi_report_wake_uses_the_explicit_boundary_and_writing_worker(
    tmp_path, monkeypatch
):
    import mesh.agent_node as agent_node

    monkeypatch.setattr(agent_node, "AUTO_REPORTS_DIR", tmp_path / "auto-reports")
    node = _controller(channels=["mesh-infra"])

    result, captured = _report_result_for(
        node, project=KEY, since="2026-08-01", requested_by="user:testuser"
    )

    assert captured["wake_time"] == "now"
    assert captured["requested_by"] == "user:testuser"
    assert "report_to: channel:mesh-infra" in captured["prompt"]
    assert 'task_type="writing"' in captured["prompt"]
    assert "sections Overview, Status, and Open Questions" in captured["prompt"]
    assert "since 2026-08-01" in captured["prompt"]
    assert str(tmp_path / "auto-reports" / "mesh-infra") in captured["prompt"]
    assert result["since"] == {"date": "2026-08-01", "source": "explicit"}
    assert result["wake_id"] == "wake-report"
    assert result["output_path_convention"].endswith(
        "/mesh-infra/pi-report-YYYY-MM-DD.{tex,pdf}"
    )


def test_pi_report_wake_uses_the_latest_prior_report_date(tmp_path, monkeypatch):
    import mesh.agent_node as agent_node

    reports_root = tmp_path / "auto-reports"
    report_dir = reports_root / "mesh-infra"
    report_dir.mkdir(parents=True)
    (report_dir / "pi-report-2026-07-31.tex").write_text("old")
    (report_dir / "pi-report-2026-08-03.pdf").write_text("new")
    monkeypatch.setattr(agent_node, "AUTO_REPORTS_DIR", reports_root)

    result, captured = _report_result_for(_controller(), project=KEY)

    assert result["since"] == {
        "date": "2026-08-03",
        "source": "previous_report",
    }
    assert "since 2026-08-03" in captured["prompt"]


def test_pi_report_wake_uses_the_earliest_timeline_date(
    mesh_home, dossier, tmp_path, monkeypatch
):
    import mesh.agent_node as agent_node

    dossier.write_text(
        dossier.read_text().replace(
            "(No autonomous sessions recorded yet.)",
            "- 2026-08-04: later event.\n- 2026-08-01: first event.",
        )
    )
    monkeypatch.setattr(agent_node, "AUTO_REPORTS_DIR", tmp_path / "auto-reports")

    result, captured = _report_result_for(_controller(), project=KEY)

    assert result["since"] == {
        "date": "2026-08-01",
        "source": "dossier_timeline",
    }
    assert "since 2026-08-01" in captured["prompt"]


def test_wake_reports_to_the_project_channel_when_the_agent_is_a_member():
    node = _controller(channels=["mesh-infra"])

    prompt = _wake_prompt_for(node, project=KEY, requested_by="user:testuser")

    assert "report_to: channel:mesh-infra" in prompt


def test_wake_channel_membership_accepts_the_prefixed_form():
    """``channels: [channel:mesh-infra]`` is the same membership."""
    node = _controller(channels=["channel:mesh-infra"])

    assert node._autonomous_report_destination(KEY) == "channel:mesh-infra"


def test_wake_falls_back_to_the_requester_without_the_project_channel():
    node = _controller(channels=["mesh-dev"])

    prompt = _wake_prompt_for(node, project=KEY, requested_by="user:jessica")

    assert "report_to: user:jessica" in prompt


def test_wake_falls_back_to_the_fleet_default_with_no_requester():
    node = _controller(channels=[])

    prompt = _wake_prompt_for(node, project=KEY)

    assert "report_to: user:operator" in prompt


def test_report_destination_ignores_an_unrelated_channel_of_the_same_name():
    """Only the project's own slug names its channel."""
    node = _controller(channels=["rec-fishing"])

    assert node._autonomous_report_destination(KEY, "user:testuser") == "user:testuser"


def test_report_to_inherits_the_session_destination_for_its_own_project():
    node = _controller(channels=["mesh-infra"])
    node._current_autonomous_metadata = {
        "autonomous_project_key": KEY,
        "autonomous_report_to": "user:jessica",
    }

    assert node._autonomous_report_to(KEY) == "user:jessica"


def test_report_to_ignores_session_metadata_for_a_different_project():
    node = _controller(channels=["mesh-infra"])
    node._current_autonomous_metadata = {
        "autonomous_project_key": "project:elsewhere",
        "autonomous_report_to": "user:jessica",
    }

    assert node._autonomous_report_to(KEY) == "channel:mesh-infra"


def test_report_to_keeps_the_fleet_default_without_a_project_channel():
    node = _controller(channels=[])

    assert node._autonomous_report_to(KEY) == "user:operator"


# ─────────────────────────────────────────────────────────────────────
# Per-turn mandate injection (§10.1)
# ─────────────────────────────────────────────────────────────────────


def _mandate_router(
    *,
    autonomous: bool = True,
    mandate: str = "MANDATE-TEXT",
    continuation: str = "",
):
    router = _dispatch_router(autonomous=autonomous)
    router._config.autonomous_mandate_prompt = mandate
    router._config.autonomous_continuation_mandate_prompt = continuation
    return router


def _trigger_with(metadata: dict):
    trigger = _dispatch_trigger()
    trigger.metadata.update(metadata)
    return trigger


def test_mandate_is_injected_for_an_autonomous_wake_turn():
    router = _mandate_router()
    trigger = _trigger_with(
        {"autonomous_session": True, "autonomous_project_key": KEY}
    )

    assert router._autonomous_mandate_block(trigger) == "MANDATE-TEXT"


def test_mandate_is_withheld_from_ordinary_turns():
    router = _mandate_router()

    assert router._autonomous_mandate_block(_dispatch_trigger()) == ""


def test_mandate_survives_a_worker_report_continuation(dossier):
    """A legacy completion without a carrier gets a visible recovery marker."""
    router = _mandate_router(
        mandate="PHASE 1 — PLAN\nPHASE 2 — EXECUTE & CLOSE",
        continuation="PHASE 2 — EXECUTE & CLOSE\nWORKER RESPONSE CONTRACT",
    )
    _outcome, _started, trigger = _dispatch(router, TAGGED_TASK)

    captured = _complete_via_report(router, trigger)

    mandate = router._autonomous_mandate_block(captured[0])
    assert mandate.startswith("PHASE 2 — EXECUTE & CLOSE\nWORKER RESPONSE CONTRACT")
    assert "PHASE 1 — PLAN" not in mandate
    assert "[SESSION PLAN UNAVAILABLE]" in mandate


def test_worker_report_continuation_falls_back_to_wake_mandate_when_unset(dossier):
    """An empty new config field is safe for existing autonomous deployments."""
    router = _mandate_router(mandate="FULL MANDATE")
    _outcome, _started, trigger = _dispatch(router, TAGGED_TASK)

    captured = _complete_via_report(router, trigger)

    mandate = router._autonomous_mandate_block(captured[0])
    assert mandate.startswith("FULL MANDATE")
    assert "[SESSION PLAN UNAVAILABLE]" in mandate


def test_mandate_is_withheld_from_an_interactive_worker_report(dossier):
    router = _mandate_router(autonomous=False)
    _outcome, _started, trigger = _dispatch(router, TAGGED_TASK)

    captured = _complete_via_report(router, trigger)

    assert router._autonomous_mandate_block(captured[0]) == ""


def test_mandate_is_withheld_for_an_out_of_scope_project_claim():
    router = _mandate_router()
    trigger = _trigger_with(
        {"autonomous_session": True, "autonomous_project_key": "project:elsewhere"}
    )

    assert router._autonomous_mandate_block(trigger) == ""


def test_unenrolled_router_never_injects_the_mandate():
    router = _mandate_router(autonomous=False)
    trigger = _trigger_with(
        {"autonomous_session": True, "autonomous_project_key": KEY}
    )

    assert router._autonomous_mandate_block(trigger) == ""


def test_delivered_wake_message_carries_the_session_scope():
    """The stamp must reach the trigger the router actually sees."""
    import asyncio

    node = _node(autonomous_agent_mode_enabled=True, autonomous_projects=[KEY])
    delivered: list = []

    async def capture(msg):
        delivered.append(msg)

    node.on_message = capture
    asyncio.run(node._deliver_wake(_wake(WAKE_PROMPT)))

    assert delivered, "the wake must be routed through on_message()"
    assert delivered[0].metadata["autonomous_session"] is True
    assert delivered[0].metadata["autonomous_project_key"] == KEY


def test_autonomous_wake_frames_its_session_report_destination():
    import asyncio

    node = _node(autonomous_agent_mode_enabled=True, autonomous_projects=[KEY])
    delivered: list = []

    async def capture(msg):
        delivered.append(msg)

    node.on_message = capture
    asyncio.run(node._deliver_wake(_wake(WAKE_PROMPT)))

    assert "send your response to channel:mesh-infra" in delivered[0].content
    assert "send your response to user:testuser" not in delivered[0].content


def test_ordinary_wake_keeps_the_requester_response_framing():
    import asyncio

    node = _node()
    delivered: list = []

    async def capture(msg):
        delivered.append(msg)

    node.on_message = capture
    asyncio.run(node._deliver_wake(_wake("Remind me to check the fold.")))

    assert "send your response to user:testuser" in delivered[0].content


def test_self_scheduled_wake_is_not_delivered_as_a_self_echo():
    """A continuation wake the agent scheduled for itself must still arrive.

    ``_wake_requester_from_trigger`` deliberately records ``self.node_id`` for
    a wake scheduled during a worker-report turn.  ``on_message`` drops any
    message whose ``from_node`` equals ``self.node_id`` as a channel echo, so
    delivering such a wake verbatim silently swallows it and no autonomous
    session ever opens.  Delivery must reframe it as coming from the user.
    """
    import asyncio

    node = _node(autonomous_agent_mode_enabled=True, autonomous_projects=[KEY])
    delivered: list = []

    async def capture(msg):
        delivered.append(msg)

    node.on_message = capture

    # 1. Wake the agent scheduled for itself.
    asyncio.run(node._deliver_wake(_wake(WAKE_PROMPT, requested_by=node.node_id)))
    # 2. Defensive case: a wake still attributed to a finished worker.
    asyncio.run(
        node._deliver_wake(
            _wake(WAKE_PROMPT, requested_by="worker:autopilot-rl-worker4")
        )
    )
    # 3. Ordinary case must be untouched.
    asyncio.run(node._deliver_wake(_wake(WAKE_PROMPT, requested_by="user:testuser")))

    assert len(delivered) == 3
    for msg, expected_from in zip(
        delivered, ("user:operator", "user:operator", "user:testuser"), strict=True
    ):
        assert msg.from_node == expected_from
        assert msg.from_node != node.node_id
        # The stamp must survive the reframing.
        assert msg.metadata["autonomous_session"] is True
        assert msg.metadata["autonomous_project_key"] == KEY
        # Reframing from_node must not redirect the session report: report_to
        # is read from the wake prompt, not from the delivering from_node.
        assert msg.metadata["autonomous_report_to"] == "channel:mesh-infra"


# ─────────────────────────────────────────────────────────────────────
# The durable guard: an explicit synthetic marker on runtime-generated
# messages, so on_message() can drop true channel echoes without having
# to infer intent from from_node.
# ─────────────────────────────────────────────────────────────────────


class _RouterSpy:
    """Stands in for RouterV2 to observe what survives on_message()'s guards."""

    def __init__(self):
        self.seen: list = []
        self.history_only: list = []

    async def on_message(self, msg):
        self.seen.append(msg)

    async def add_to_history_only(self, msg):
        self.history_only.append(msg)


def _direct_msg(node, from_node: str, msg_id: str, metadata: dict | None = None):
    """A direct (non-channel) message addressed to ``node``."""
    from mesh.protocol import Message, MessageType

    return Message(
        type=MessageType.MESSAGE,
        from_node=from_node,
        to_node=node.node_id,
        content="ping",
        id=msg_id,
        metadata=metadata if metadata is not None else {},
    )


def test_delivered_wake_carries_the_synthetic_marker():
    """Every wake _deliver_wake emits is marked as runtime-generated.

    This is the durable half of the wake-swallowing fix: the marker does not
    depend on who requested the wake, so it holds for autonomous sessions and
    ordinary reminders alike.
    """
    import asyncio

    node = _node(autonomous_agent_mode_enabled=True, autonomous_projects=[KEY])
    delivered: list = []

    async def capture(msg):
        delivered.append(msg)

    node.on_message = capture

    asyncio.run(node._deliver_wake(_wake(WAKE_PROMPT, requested_by=node.node_id)))
    asyncio.run(node._deliver_wake(_wake(WAKE_PROMPT, requested_by="user:testuser")))
    # An ordinary reminder gets no autonomous scope but is still synthetic.
    asyncio.run(node._deliver_wake(_wake("Remind me to check the fold.")))

    assert len(delivered) == 3
    for msg in delivered:
        assert msg.metadata.get("synthetic") is True

    # The marker must not manufacture autonomous scope for a plain reminder.
    assert "autonomous_session" not in delivered[2].metadata


def test_echo_guard_keeps_self_addressed_synthetic_messages():
    """A self-from message carrying the marker must reach the router.

    This is the defence that would have saved the swallowed continuation wake
    even without the from_node reframing.
    """
    import asyncio

    node = _node(autonomous_agent_mode_enabled=True, autonomous_projects=[KEY])
    spy = _RouterSpy()
    node._router_v2 = spy

    asyncio.run(
        node.on_message(
            _direct_msg(node, node.node_id, "m-synthetic", {"synthetic": True})
        )
    )

    assert [m.id for m in spy.seen] == ["m-synthetic"]


def test_echo_guard_still_drops_unmarked_self_echoes():
    """A genuine channel echo — self-from, no marker — is still dropped."""
    import asyncio

    node = _node(autonomous_agent_mode_enabled=True, autonomous_projects=[KEY])
    spy = _RouterSpy()
    node._router_v2 = spy

    asyncio.run(node.on_message(_direct_msg(node, node.node_id, "m-echo")))
    # Explicitly-empty and explicitly-false metadata are echoes too.
    asyncio.run(
        node.on_message(
            _direct_msg(node, node.node_id, "m-echo-false", {"synthetic": False})
        )
    )

    assert spy.seen == []
    assert spy.history_only == []


def test_echo_guard_leaves_ordinary_messages_alone():
    """The ordinary case is unchanged: a message from a user is processed."""
    import asyncio

    node = _node(autonomous_agent_mode_enabled=True, autonomous_projects=[KEY])
    spy = _RouterSpy()
    node._router_v2 = spy

    asyncio.run(node.on_message(_direct_msg(node, "user:testuser", "m-user")))
    # ...and an unmarked message from another agent is not an echo either.
    asyncio.run(node.on_message(_direct_msg(node, "agent:coder:bob", "m-bob")))

    assert [m.id for m in spy.seen] == ["m-user", "m-bob"]


def test_wake_delivery_survives_the_echo_guard_end_to_end():
    """_deliver_wake's marker and on_message's guard compose.

    Delivery is driven through the *real* on_message, and from_node is forced
    back to the agent's own id after reframing so the marker is the only thing
    keeping the wake alive.  Without it the wake is silently swallowed and no
    autonomous session ever opens.
    """
    import asyncio

    node = _node(autonomous_agent_mode_enabled=True, autonomous_projects=[KEY])
    spy = _RouterSpy()
    node._router_v2 = spy

    real_on_message = node.on_message

    async def self_framed(msg):
        # Undo the reframing to isolate the durable guard.
        msg.from_node = node.node_id
        await real_on_message(msg)

    node.on_message = self_framed
    asyncio.run(node._deliver_wake(_wake(WAKE_PROMPT, requested_by=node.node_id)))

    assert len(spy.seen) == 1, "the wake must survive the echo guard"
    assert spy.seen[0].from_node == node.node_id
    assert spy.seen[0].metadata["synthetic"] is True
    assert spy.seen[0].metadata["autonomous_project_key"] == KEY


def test_end_to_end_wake_stamp_drives_mandate_injection():
    """§10.1 end to end: wake prompt → trusted stamp → injected mandate."""
    node = _node(autonomous_agent_mode_enabled=True, autonomous_projects=[KEY])
    router = _mandate_router()
    trigger = _trigger_with(node.autonomous_wake_metadata(_wake(WAKE_PROMPT)))

    assert router._autonomous_mandate_block(trigger) == "MANDATE-TEXT"

    ordinary = _trigger_with(
        node.autonomous_wake_metadata(_wake("Remind me to check the fold."))
    )
    assert router._autonomous_mandate_block(ordinary) == ""


def test_mandate_lands_in_the_dynamic_tail_not_the_stable_prefix():
    """Toggling the mandate must not invalidate the cached history prefix."""
    import asyncio

    router = _mandate_router()
    router._system_prompt = "base coder prompt"
    trigger = _trigger_with(
        {"autonomous_session": True, "autonomous_project_key": KEY}
    )

    stable, dynamic = asyncio.run(
        router._build_router_context_blocks(trigger_msg=trigger)
    )
    assert "MANDATE-TEXT" not in "\n".join(stable)
    assert dynamic[-1] == "MANDATE-TEXT"

    stable_plain, dynamic_plain = asyncio.run(
        router._build_router_context_blocks(trigger_msg=_dispatch_trigger())
    )
    assert stable_plain == stable, "the stable prefix must not depend on the turn kind"
    assert "MANDATE-TEXT" not in "\n".join(dynamic_plain)


def test_execute_only_mandate_also_lands_in_the_dynamic_tail():
    """Continuation trimming must not move the dynamic mandate into cache state."""
    import asyncio

    router = _mandate_router(
        mandate="WAKE PLAN + EXECUTE",
        continuation="CONTINUATION EXECUTE ONLY",
    )
    router._system_prompt = "base coder prompt"
    trigger = _trigger_with(
        {
            "autonomous_session": True,
            "autonomous_project_key": KEY,
            "worker_report": True,
        }
    )

    stable, dynamic = asyncio.run(
        router._build_router_context_blocks(trigger_msg=trigger)
    )

    assert "CONTINUATION EXECUTE ONLY" not in "\n".join(stable)
    assert dynamic[-1].startswith("CONTINUATION EXECUTE ONLY")
    assert "[SESSION PLAN UNAVAILABLE]" in dynamic[-1]


# ─────────────────────────────────────────────────────────────────────
# Mechanical project tagging of an autonomous dispatch brief (§8.7 step 3)
# ─────────────────────────────────────────────────────────────────────
#
# The mandate asks the router to write [PROJECT: …] into every brief it owns.
# Prompt enforcement alone did not hold in the first live session: the tag was
# omitted, the guard failed open, and the completion carried no session scope.
# The scope is known to the runtime, so the tag is now stamped from it.


def _autonomous_wake_trigger():
    """A trigger stamped the way the wake runtime stamps a session wake."""
    trigger = _dispatch_trigger("[AUTONOMOUS PROJECT SESSION] wake body")
    trigger.metadata.update(
        {
            "autonomous_session": True,
            "autonomous_session_id": "as-test-0001",
            "autonomous_project_key": KEY,
        }
    )
    return trigger


def _dispatch_with_trigger(router, trigger, task: str):
    import asyncio

    from mesh.router_v2 import DispatchReceipt

    started: list = []

    async def start(_trigger):
        started.append(_trigger)
        router._last_dispatch_receipt = DispatchReceipt(
            dispatch_key="dk-1",
            status="running",
            worker_id="autopilot-worker1",
            slot_index=0,
            origin_message_id=_trigger.id,
            router_turn_id="turn-1",
            task_description=_trigger.metadata.get("worker_task_description", ""),
            backend=None,
        )
        return True

    router._start_worker = start
    outcome = asyncio.run(
        router._dispatch_worker(trigger, {"task": task}, source="tool")
    )
    return outcome, started


SESSION_PLAN = """SESSION PLAN
GOAL=G-003
TASKS=T-015, T-016
EVIDENCE=T-015 has focused regression coverage; T-016 records the result
FIRST=Capture the plan before dispatching the first worker"""


def _record_session_plan(router, trigger):
    """Write a wake-turn plan through the same router history seam as production."""
    from datetime import datetime, timezone

    from mesh.conversation_history import Turn
    from mesh.router_v2 import _CTX_ROUTER_CALL_STATE

    router._init_call_state(trigger)
    try:
        router._append_turn(
            Turn(
                role="outgoing",
                content=SESSION_PLAN,
                timestamp=datetime.now(timezone.utc),
                from_node="agent:coder:autopilot",
            )
        )
    finally:
        _CTX_ROUTER_CALL_STATE.set(None)


def test_session_plan_is_carried_from_wake_history_to_report_continuation(dossier):
    router = _mandate_router(
        mandate="PHASE 1 — PLAN\nPHASE 2 — EXECUTE & CLOSE",
        continuation="PHASE 2 — EXECUTE & CLOSE\nWORKER RESPONSE CONTRACT",
    )
    trigger = _autonomous_wake_trigger()
    _record_session_plan(router, trigger)

    plan_turn = router._history.window[-1]
    assert plan_turn.meta["autonomous_session_id"] == "as-test-0001"
    assert plan_turn.meta["autonomous_session_plan"] == SESSION_PLAN

    outcome, started = _dispatch_with_trigger(router, trigger, UNTAGGED_TASK)
    assert outcome.status == "running"
    assert len(started) == 1
    assert trigger.metadata["autonomous_session_plan"] == SESSION_PLAN

    captured = _complete_via_report(router, trigger)
    mandate = router._autonomous_mandate_block(captured[0])
    assert "[SESSION PLAN CARRY-FORWARD]" in mandate
    assert SESSION_PLAN in mandate
    assert "[SESSION PLAN UNAVAILABLE]" not in mandate


def test_session_plan_survives_history_compaction_in_worker_report_prompt(dossier):
    from datetime import datetime, timezone

    from mesh.conversation_history import ConversationHistory, Turn

    router = _mandate_router(continuation="EXECUTE ONLY")
    # One-token windows force a deterministic drop of the plan after its
    # dispatch trigger has captured the durable carrier.
    router._history = ConversationHistory(
        soft_token_limit=2,
        hard_token_limit=4,
        window_budget=1,
        summarization_enabled=False,
    )
    trigger = _autonomous_wake_trigger()
    _record_session_plan(router, trigger)
    outcome, _started = _dispatch_with_trigger(router, trigger, UNTAGGED_TASK)
    assert outcome.status == "running"
    assert trigger.metadata["autonomous_session_plan"] == SESSION_PLAN

    router._append_turn(
        Turn(
            role="incoming",
            content="newer completion evidence",
            timestamp=datetime.now(timezone.utc),
        )
    )
    dropped = router._history.partition_and_drop_old()
    assert any(turn.content == SESSION_PLAN for turn in dropped)
    assert all(turn.content != SESSION_PLAN for turn in router._history.window)

    captured = _complete_via_report(router, trigger)
    mandate = router._autonomous_mandate_block(captured[0])
    assert "[SESSION PLAN CARRY-FORWARD]" in mandate
    assert SESSION_PLAN in mandate


def test_missing_session_plan_is_marked_instead_of_silently_omitted():
    router = _mandate_router(continuation="EXECUTE ONLY")
    trigger = _trigger_with(
        {
            "autonomous_session": True,
            "autonomous_project_key": KEY,
            "autonomous_session_id": "as-legacy-no-plan",
            "worker_report": True,
        }
    )

    mandate = router._autonomous_mandate_block(trigger)
    assert mandate.startswith("EXECUTE ONLY")
    assert "[SESSION PLAN UNAVAILABLE]" in mandate
    assert "Do not silently assume or invent" in mandate


def test_agent_node_plan_tool_turn_survives_worker_launch_continuation(dossier):
    """The native AgentNode seam captures narration before launch detaches it."""
    import asyncio

    from mesh.agent_node import AgentNode
    from mesh.config import NodeConfig
    from mesh.conversation_history import Turn
    from mesh.llm import LLMConfig
    from mesh.router_v2 import DispatchReceipt, _CTX_ROUTER_CALL_STATE
    from mesh.tools import ToolCall

    router = _mandate_router(
        mandate="PHASE 1 — PLAN\nPHASE 2 — EXECUTE & CLOSE",
        continuation="PHASE 2 — EXECUTE & CLOSE\nWORKER RESPONSE CONTRACT",
    )
    trigger = _autonomous_wake_trigger()
    # Seed the incoming wake directly.  In particular, do not use
    # RouterV2._append_turn(), whose outgoing-only capture hook is the test-only
    # path that masked the production AgentNode defect.
    router._history.append(
        Turn(
            role="incoming",
            content=trigger.content,
            timestamp=datetime.now(timezone.utc),
            from_node=trigger.from_node,
            to_node=trigger.to_node,
        )
    )

    started: list = []

    async def start(dispatch_trigger):
        started.append(dispatch_trigger)
        router._last_dispatch_receipt = DispatchReceipt(
            dispatch_key="dk-agent-node-plan",
            status="running",
            worker_id="autopilot-worker1",
            slot_index=0,
            origin_message_id=dispatch_trigger.id,
            router_turn_id="turn-agent-node-plan",
            task_description=dispatch_trigger.metadata.get(
                "worker_task_description", ""
            ),
            backend=None,
        )
        return True

    router._start_worker = start

    launch_xml = (
        '<mesh_call name="worker_launch">'
        f"<task>{UNTAGGED_TASK}</task>"
        "</mesh_call>"
    )

    class PlanThenLaunchLLM:
        supports_native_reasoning_multiturn = False

        def __init__(self):
            self.calls = 0

        async def complete_with_tools(self, **_kwargs):
            self.calls += 1
            if self.calls != 1:
                raise AssertionError("successful worker_launch must end the loop")
            return SESSION_PLAN, [
                ToolCall(
                    name="worker_launch",
                    arguments={"task": UNTAGGED_TASK},
                    raw_xml=launch_xml,
                    call_id="launch-agent-node-plan",
                )
            ]

    agent = AgentNode(
        NodeConfig(
            id="agent:coder:autopilot",
            agent_type="coder",
            nickname="autopilot",
            tools=[],
        ),
        persist=False,
    )
    agent._router_v2 = router
    agent._tool_socket_path = None
    agent.llm_config = LLMConfig(backend="openai", model="test")

    async def no_store(*_args, **_kwargs):
        return None

    agent._store_tool_context = no_store
    agent._store_cc_tool_context = no_store
    llm = PlanThenLaunchLLM()

    async def run_plan_turn():
        router._init_call_state(trigger)
        try:
            return await agent._router_process_with_llm(
                trigger_msg=trigger,
                system_prompt="system",
                llm_client=llm,
                tool_names=["worker_launch"],
                max_iters=3,
                router_history=router._history,
                instructions="Write the SESSION PLAN, then launch one worker.",
            )
        finally:
            _CTX_ROUTER_CALL_STATE.set(None)

    assert asyncio.run(run_plan_turn()) == ""
    assert llm.calls == 1
    assert len(started) == 1

    assistant_turn = next(
        turn for turn in reversed(router._history.window)
        if turn.role == "assistant"
    )
    assert SESSION_PLAN not in assistant_turn.content
    assert assistant_turn.content == router._last_dispatch_receipt.request_record
    assert f"[PROJECT: {KEY}]" in assistant_turn.content
    assert assistant_turn.meta["autonomous_session_id"] == "as-test-0001"
    assert assistant_turn.meta["autonomous_session_plan"] == SESSION_PLAN

    dispatch_trigger = started[0]
    assert dispatch_trigger.metadata["autonomous_session_plan"] == SESSION_PLAN
    captured = _complete_via_report(router, dispatch_trigger)
    assert captured[0].metadata["autonomous_session_plan"] == SESSION_PLAN
    continuation = router._autonomous_mandate_block(captured[0])
    assert continuation.startswith("PHASE 2 — EXECUTE & CLOSE")
    assert "[SESSION PLAN CARRY-FORWARD]" in continuation
    assert SESSION_PLAN in continuation


def test_untagged_brief_is_stamped_from_the_session_scope(dossier):
    router = _dispatch_router(autonomous=True)
    trigger = _autonomous_wake_trigger()

    outcome, started = _dispatch_with_trigger(router, trigger, UNTAGGED_TASK)

    assert outcome.status == "running"
    assert len(started) == 1
    brief = trigger.metadata["worker_task_description"]
    assert brief.startswith(f"[PROJECT: {KEY}]")
    assert UNTAGGED_TASK in brief
    # The guard now closes: the dispatch is scoped and the budget is charged.
    assert outcome.autonomous_session is True
    assert outcome.project_key == KEY
    assert check_budget(KEY)["used"] == 1


def test_stamping_never_duplicates_an_existing_tag(dossier):
    router = _dispatch_router(autonomous=True)
    trigger = _autonomous_wake_trigger()

    _outcome, _started = _dispatch_with_trigger(router, trigger, TAGGED_TASK)

    brief = trigger.metadata["worker_task_description"]
    assert brief.count("[PROJECT:") == 1
    assert brief == TAGGED_TASK


def test_stamping_is_inert_without_trusted_session_metadata(dossier):
    """An unmarked trigger is still an ordinary dispatch — no forged scope."""
    router = _dispatch_router(autonomous=True)
    trigger = _dispatch_trigger()

    outcome, _started = _dispatch_with_trigger(router, trigger, UNTAGGED_TASK)

    assert "[PROJECT:" not in trigger.metadata["worker_task_description"]
    assert outcome.autonomous_session is False


# ─────────────────────────────────────────────────────────────────────
# Autonomous closeout reroute when a worker skips send_report
# ─────────────────────────────────────────────────────────────────────
#
# The closeout half of the mandate (session report, task completion, next
# wake) only exists inside a tool-capable ReAct turn, and only the
# report-as-trigger path opens one.  A worker that never calls send_report
# used to land in text-only fallback delivery, ending the session mid-cycle.


def _complete_without_report(
    router,
    trigger,
    worker_id="autopilot-worker1",
    *,
    synthesized="synthesized worker summary",
):
    """Drive a real completion in which the worker never called send_report."""
    import asyncio

    from mesh.router_v2 import WorkerLifecycle, WorkerResult

    slot = router._ensure_slot_table()[0]
    slot.worker_id = worker_id
    slot.lifecycle = WorkerLifecycle.RUNNING
    slot.trigger = trigger
    slot.task_description = trigger.metadata.get("worker_task_description", "")
    router._sync_worker_compat_views()

    captured: list = []
    router._enqueue_report_wake = captured.append
    sent: list = []

    async def _send_and_store(content, _trigger, **_kwargs):
        sent.append(content)

    router._send_and_store = _send_and_store
    router._llm_client = object()

    async def _synthesize(_trace, _trigger):
        return synthesized

    router._synthesize_worker_output = _synthesize

    result = WorkerResult(
        response="73 passed, 0 failed",
        context=[],
        report_sent=False,
        buffered_messages=[],
    )
    asyncio.run(
        router._handle_worker_complete(result, trigger, worker_id=worker_id)
    )
    return captured, sent


def test_autonomous_completion_without_send_report_reenters_with_tools(dossier):
    router = _dispatch_router(autonomous=True)
    trigger = _autonomous_wake_trigger()
    _dispatch_with_trigger(router, trigger, UNTAGGED_TASK)
    session_id = trigger.metadata["autonomous_session_id"]

    captured, sent = _complete_without_report(router, trigger)

    assert captured, (
        "an autonomous completion must open a tool-capable router turn even "
        "when the worker skipped send_report"
    )
    meta = captured[0].metadata
    assert meta["worker_report"] is True
    assert meta["worker_report_synthesized"] is True
    assert meta["autonomous_session"] is True
    assert meta["autonomous_project_key"] == KEY
    assert meta["autonomous_session_id"] == session_id
    assert captured[0].content == "synthesized worker summary"
    assert sent == [], "the reroute replaces text-only fallback delivery"


def test_interactive_completion_without_send_report_still_delivers_text(dossier):
    """The reroute is scoped to autonomous sessions; ordinary work is unchanged."""
    router = _dispatch_router(autonomous=False)
    _outcome, _started, trigger = _dispatch(router, TAGGED_TASK)

    captured, sent = _complete_without_report(router, trigger)

    assert captured == [], "an interactive completion opens no report turn"
    assert sent == ["synthesized worker summary"]


def test_reroute_falls_back_to_worker_output_when_synthesis_is_empty(dossier):
    """An empty synthesis must not close the session with an empty report."""
    router = _dispatch_router(autonomous=True)
    trigger = _autonomous_wake_trigger()
    _dispatch_with_trigger(router, trigger, UNTAGGED_TASK)

    captured, _sent = _complete_without_report(router, trigger, synthesized="")

    assert captured
    assert captured[0].content == "73 passed, 0 failed"


# ─────────────────────────────────────────────────────────────────────
# Session scope survives the native worker_launch tool's synthetic trigger
# ─────────────────────────────────────────────────────────────────────
#
# `_tool_worker_launch` builds its own Message rather than forwarding the
# wake.  Left empty, that trigger drops the session scope before the dispatch
# seam ever sees it — which is why the first live session failed open on the
# admission guard and then closed out as an ordinary interactive completion.


def _clear_installed_call_state():
    """Drop the RouterCallState these tests install on the running context.

    ``_init_call_state`` binds a contextvar that outlives the test function,
    and a leaked state is visible to every later test in the same worker.
    """
    from mesh.router_v2 import _CTX_ROUTER_CALL_STATE

    _CTX_ROUTER_CALL_STATE.set(None)


def test_native_worker_launch_carries_the_session_scope(dossier):
    import asyncio

    from mesh.router_v2 import DispatchReceipt

    router = _dispatch_router(autonomous=True)
    router._init_call_state(_autonomous_wake_trigger())

    started: list = []

    async def start(_trigger):
        started.append(_trigger)
        router._last_dispatch_receipt = DispatchReceipt(
            dispatch_key="dk-1",
            status="running",
            worker_id="autopilot-worker1",
            slot_index=0,
            origin_message_id=_trigger.id,
            router_turn_id="turn-1",
            task_description=_trigger.metadata.get("worker_task_description", ""),
            backend=None,
        )
        return True

    router._start_worker = start
    try:
        asyncio.run(router._tool_worker_launch(UNTAGGED_TASK))
    finally:
        _clear_installed_call_state()

    assert len(started) == 1
    dispatch_trigger = started[0]
    assert dispatch_trigger.metadata["autonomous_session"] is True
    assert dispatch_trigger.metadata["autonomous_project_key"] == KEY
    assert dispatch_trigger.metadata["autonomous_session_id"] == "as-test-0001"
    assert dispatch_trigger.metadata["worker_task_description"].startswith(
        f"[PROJECT: {KEY}]"
    )
    # A dispatch that carries scope is a dispatch the guard actually charged.
    assert check_budget(KEY)["used"] == 1
    # And the completion built from that trigger inherits the same scope.
    assert router.autonomous_completion_metadata(dispatch_trigger) == {
        "autonomous_session": True,
        "autonomous_project_key": KEY,
        "autonomous_session_id": "as-test-0001",
        "autonomous_report_to": "",
    }


def test_native_worker_launch_is_unscoped_for_an_ordinary_turn(dossier):
    import asyncio

    from mesh.router_v2 import DispatchReceipt

    router = _dispatch_router(autonomous=True)
    router._init_call_state(_dispatch_trigger())

    started: list = []

    async def start(_trigger):
        started.append(_trigger)
        router._last_dispatch_receipt = DispatchReceipt(
            dispatch_key="dk-1",
            status="running",
            worker_id="autopilot-worker1",
            slot_index=0,
            origin_message_id=_trigger.id,
            router_turn_id="turn-1",
            task_description="",
            backend=None,
        )
        return True

    router._start_worker = start
    try:
        asyncio.run(router._tool_worker_launch(UNTAGGED_TASK))
    finally:
        _clear_installed_call_state()

    assert len(started) == 1
    assert "autonomous_session" not in started[0].metadata
    assert check_budget(KEY)["used"] == 0
