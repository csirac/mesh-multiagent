"""Active mode — the deterministic next-wake scheduler.

Two layers are covered here:

* ``mesh.autonomous_active.plan_active_wake`` — the pure gate logic, tested
  against a frozen clock so pacing and overlap are exact, not approximate;
* ``AgentNode.schedule_active_wake`` and the ``dossier_write_report`` closeout
  hook — the wiring that makes the decision happen without a human or an LLM
  in the loop.

Everything runs against isolated temp state. Nothing touches ~/.mesh, a
production dossier, a real ledger, or a live wake row.
"""

from datetime import datetime, timedelta, timezone

import pytest

from mesh.agent_node import AgentNode, ToolCall
from mesh.autonomous_active import (
    ActiveWakeDecision,
    PendingWake,
    plan_active_wake,
    report_suppresses_next_wake,
    wake_project_key,
)
from mesh.config import NodeConfig
from mesh.isolation import StatePaths
from mesh.project_dossier import active_flag, init_dossier

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)
GAP = 60


def _decide(**overrides) -> ActiveWakeDecision:
    kwargs = {
        "project_key": "project:alpha",
        "active": True,
        "remaining": 3,
        "pending": [],
        "gap_minutes": GAP,
        "now": NOW,
    }
    kwargs.update(overrides)
    return plan_active_wake(**kwargs)


# =============================================================================
# plan_active_wake — the gates
# =============================================================================


def test_active_with_budget_schedules_one_gap_out():
    decision = _decide()
    assert decision.scheduled is True
    assert decision.wake_time == NOW + timedelta(minutes=GAP)
    assert decision.deferred_behind == ""


def test_inactive_project_is_a_no_op():
    decision = _decide(active=False)
    assert decision.scheduled is False
    assert decision.wake_time is None
    assert "not in active mode" in decision.reason


def test_exhausted_budget_is_a_no_op():
    decision = _decide(remaining=0)
    assert decision.scheduled is False
    assert "budget is exhausted" in decision.reason


def test_negative_or_unparseable_budget_is_a_no_op():
    assert _decide(remaining=-1).scheduled is False
    assert _decide(remaining="lots").scheduled is False


def test_a_pending_wake_for_this_project_is_a_no_op():
    pending = [
        PendingWake("wake-aaa", NOW + timedelta(hours=5), "project:alpha"),
    ]
    decision = _decide(pending=pending)
    assert decision.scheduled is False
    assert "already pending" in decision.reason
    assert decision.detail["existing_wake_id"] == "wake-aaa"


def test_a_pending_wake_for_a_different_project_defers_instead_of_blocking():
    # The other project fires 70 minutes out — inside a gap of the naive
    # now+60 candidate, so the new wake serializes behind it.
    other = NOW + timedelta(minutes=70)
    decision = _decide(
        pending=[PendingWake("wake-bbb", other, "project:beta")]
    )
    assert decision.scheduled is True
    assert decision.wake_time == other + timedelta(minutes=GAP)
    assert decision.deferred_behind == "wake-bbb"


def test_a_distant_wake_on_another_project_does_not_defer_anything():
    decision = _decide(
        pending=[
            PendingWake("wake-ccc", NOW + timedelta(hours=9), "project:beta")
        ]
    )
    assert decision.scheduled is True
    assert decision.wake_time == NOW + timedelta(minutes=GAP)
    assert decision.deferred_behind == ""


def test_overlap_deferral_chains_across_several_projects():
    decision = _decide(
        pending=[
            PendingWake("w1", NOW + timedelta(minutes=70), "project:beta"),
            PendingWake("w2", NOW + timedelta(minutes=150), "project:gamma"),
        ]
    )
    # now+60 collides with w1 (70) -> 130; 130 collides with w2 (150) -> 210.
    assert decision.wake_time == NOW + timedelta(minutes=210)
    assert decision.deferred_behind == "w2"


def test_an_ordinary_reminder_also_counts_for_no_overlap():
    """A plain wake has no project key but still occupies the agent's clock."""
    decision = _decide(
        pending=[PendingWake("w-reminder", NOW + timedelta(minutes=65), "")]
    )
    assert decision.scheduled is True
    assert decision.wake_time == NOW + timedelta(minutes=125)


def test_exactly_one_gap_apart_is_not_an_overlap():
    decision = _decide(
        pending=[
            PendingWake("w1", NOW + timedelta(minutes=120), "project:beta")
        ]
    )
    assert decision.wake_time == NOW + timedelta(minutes=GAP)


def test_a_terminal_report_status_suppresses_the_wake():
    decision = _decide(suppressed_reason="session status=blocked")
    assert decision.scheduled is False
    assert decision.reason == "session status=blocked"


def test_gates_are_ordered_inactive_before_suppression():
    """An inactive project reports as inactive, not as suppressed."""
    decision = _decide(active=False, suppressed_reason="session status=failed")
    assert "not in active mode" in decision.reason


# =============================================================================
# Report parsing — the controller's only vote
# =============================================================================


def _report(status="completed", intent="continue") -> str:
    frontmatter = (
        "---\n"
        "schema_version: 1\n"
        "record_type: autonomous_session_report\n"
        "project_entity_key: project:alpha\n"
        f"status: {status}\n"
    )
    if intent is not None:
        frontmatter += f"next_wake_intent: {intent}\n"
    return frontmatter + "---\n\n# Autonomous session\n"


@pytest.mark.parametrize("status", ["blocked", "failed", "no_op"])
def test_terminal_statuses_suppress(status):
    assert report_suppresses_next_wake(_report(status=status)) == (
        f"session status={status}"
    )


@pytest.mark.parametrize("status", ["completed", "partial"])
@pytest.mark.parametrize("intent", ["continue", "none"])
def test_continuing_statuses_do_not_suppress(status, intent):
    # A non-terminal `none` is a legacy/default report value, not a veto.
    assert report_suppresses_next_wake(_report(status=status, intent=intent)) == ""


def test_no_op_report_with_none_suppresses():
    assert report_suppresses_next_wake(_report(status="no_op", intent="none")) == (
        "session status=no_op"
    )


def test_blocked_report_without_intent_suppresses():
    assert report_suppresses_next_wake(_report(status="blocked", intent=None)) == (
        "session status=blocked"
    )


def test_a_report_without_frontmatter_leaves_the_scheduler_armed():
    assert report_suppresses_next_wake("just some prose") == ""
    assert report_suppresses_next_wake("status: blocked\n") == ""
    assert report_suppresses_next_wake("") == ""
    assert report_suppresses_next_wake("---\nstatus: [not valid yaml\n---") == ""


# =============================================================================
# wake_project_key
# =============================================================================


def test_wake_project_key_reads_a_session_wake():
    from mesh.protocol import build_autonomous_wake_prompt

    prompt = build_autonomous_wake_prompt(
        "project:alpha", "/tmp/alpha.md", 2, report_to="user:testuser"
    )
    assert wake_project_key(prompt) == "project:alpha"


def test_wake_project_key_ignores_an_ordinary_reminder():
    assert wake_project_key("remember project_entity_key: project:alpha") == ""
    assert wake_project_key("") == ""


# =============================================================================
# AgentNode wiring — schedule_active_wake and the closeout hook
# =============================================================================

PROJECT = "project:alpha"
OTHER = "project:beta"
AGENT_ID = "agent:coder:autopilot-active"


class FakeWakeStore:
    def __init__(self):
        self.saved: dict[str, dict] = {}

    def save_wake(self, wake_id, wake_time, prompt, requested_by, created_at,
                  recurrence=None):
        self.saved[wake_id] = {"prompt": prompt, "recurrence": recurrence}

    def delete_wake(self, wake_id):
        self.saved.pop(wake_id, None)


class FakeMemorySystem:
    def __init__(self, store):
        self._store = store


@pytest.fixture
def node(tmp_path):
    """An enrolled two-project controller on isolated temp state."""
    state_paths = StatePaths.for_state_root(tmp_path / "state")
    state_paths.digests_dir.mkdir(parents=True, exist_ok=True)
    state_paths.budget_dir.mkdir(parents=True, exist_ok=True)
    for key in (PROJECT, OTHER):
        init_dossier(key, owner_agent=AGENT_ID, max_workers_per_day=4,
                     state_paths=state_paths)

    config = NodeConfig(id=AGENT_ID, agent_type="coder",
                        nickname="autopilot-active", tools=["file_read"])
    config.autonomous_agent_mode_enabled = True
    config.autonomous_projects = [PROJECT, OTHER]
    config.autonomous_max_workers_per_session = 3
    config.autonomous_active_gap_minutes = GAP

    agent = AgentNode(config=config, persist=False)
    agent._tool_socket_path = None
    agent._scoped_state_paths = lambda: state_paths
    agent.state_paths_for_test = state_paths
    agent._memory_system = FakeMemorySystem(FakeWakeStore())
    return agent


def _arm(agent, project=PROJECT, value=True):
    """Toggle active mode through the real op handler, not a raw file write."""
    return agent._autonomous_active_result(
        {"project": project, "value": value, "requested_by": "user:testuser"}
    )


def test_active_flag_defaults_off_on_a_fresh_dossier(node):
    assert active_flag(PROJECT, node.state_paths_for_test) is False
    assert node._autonomous_active(PROJECT) is False


def test_active_op_round_trips_through_the_dossier(node):
    on = _arm(node, value=True)
    assert on["active"] is True
    assert on["previous_active"] is False
    assert on["changed"] is True
    assert active_flag(PROJECT, node.state_paths_for_test) is True

    off = _arm(node, value=False)
    assert off["active"] is False
    assert off["changed"] is True
    assert active_flag(PROJECT, node.state_paths_for_test) is False


def test_active_op_is_idempotent(node):
    _arm(node, value=True)
    again = _arm(node, value=True)
    assert again["changed"] is False
    assert again["active"] is True


def test_active_op_inserts_the_line_into_a_pre_active_dossier(node, tmp_path):
    """A dossier written before active mode existed gains the line, once."""
    from mesh.project_dossier import dossier_path, read_dossier

    path = dossier_path(PROJECT, node.state_paths_for_test)
    path.write_text(read_dossier(PROJECT, node.state_paths_for_test)
                    .replace("active: false\n", ""))
    assert "active:" not in read_dossier(PROJECT, node.state_paths_for_test)

    result = _arm(node, value=True)
    assert result["active"] is True
    after = read_dossier(PROJECT, node.state_paths_for_test)
    assert "max_workers_per_day: 4\nactive: true\n" in after
    assert after.count("active:") == 1


def test_active_op_refuses_a_project_the_agent_does_not_own(node):
    with pytest.raises(ValueError, match="not in this agent's"):
        _arm(node, project="project:not-mine")


def test_active_op_refuses_a_non_boolean_value(node):
    with pytest.raises(ValueError, match="must be a boolean"):
        node._autonomous_active_result({"project": PROJECT, "value": "sort of"})


def test_schedule_active_wake_arms_a_real_paced_wake(node):
    _arm(node)
    before = datetime.now(timezone.utc)
    outcome = node.schedule_active_wake(PROJECT, _report())

    assert outcome["scheduled"] is True
    wake = node._scheduled_wakes[outcome["wake_id"]]
    delta = wake.wake_time - before
    assert timedelta(minutes=GAP - 1) <= delta <= timedelta(minutes=GAP + 1)
    # One-shot only: a recurrence would keep firing past a budget change.
    assert wake.recurrence is None
    # It is a real session wake, not a reminder.
    assert wake_project_key(wake.prompt) == PROJECT
    assert "max_workers_this_session: 3" in wake.prompt


def test_schedule_active_wake_is_a_no_op_when_not_armed(node):
    outcome = node.schedule_active_wake(PROJECT, _report())
    assert outcome["scheduled"] is False
    assert "not in active mode" in outcome["reason"]
    assert node._scheduled_wakes == {}


def test_schedule_active_wake_is_a_no_op_when_the_budget_is_spent(node):
    from mesh.project_dossier import spend_budget

    _arm(node)
    spend_budget(PROJECT, 4, node.state_paths_for_test)
    outcome = node.schedule_active_wake(PROJECT, _report())
    assert outcome["scheduled"] is False
    assert "budget is exhausted" in outcome["reason"]
    assert node._scheduled_wakes == {}


def test_schedule_active_wake_never_double_schedules(node):
    _arm(node)
    first = node.schedule_active_wake(PROJECT, _report())
    second = node.schedule_active_wake(PROJECT, _report())

    assert first["scheduled"] is True
    assert second["scheduled"] is False
    assert "already pending" in second["reason"]
    assert len(node._scheduled_wakes) == 1


def test_two_projects_serialize_instead_of_overlapping(node):
    _arm(node, PROJECT)
    _arm(node, OTHER)

    first = node.schedule_active_wake(PROJECT, _report())
    second = node.schedule_active_wake(OTHER, _report())

    assert first["scheduled"] is True and second["scheduled"] is True
    assert second["deferred_behind"] == first["wake_id"]

    gap = (
        node._scheduled_wakes[second["wake_id"]].wake_time
        - node._scheduled_wakes[first["wake_id"]].wake_time
    )
    assert gap >= timedelta(minutes=GAP)


def test_a_blocked_report_suppresses_the_wake(node):
    _arm(node)
    outcome = node.schedule_active_wake(PROJECT, _report(status="blocked"))
    assert outcome["scheduled"] is False
    assert outcome["reason"] == "session status=blocked"
    assert node._scheduled_wakes == {}


def test_completed_report_with_none_still_schedules_the_wake(node):
    _arm(node)
    outcome = node.schedule_active_wake(PROJECT, _report(intent="none"))
    assert outcome["scheduled"] is True
    assert len(node._scheduled_wakes) == 1


@pytest.mark.asyncio
async def test_writing_a_session_report_schedules_the_next_one(node):
    """The end-to-end guarantee: no human, no LLM decision, a wake appears."""
    _arm(node)
    node._push_mesh_tool_activity = _noop_activity

    result = await node._execute_single_tool_with_confirmation(
        ToolCall(
            name="dossier_write_report",
            raw_xml="",
            arguments={
                "entity_key": PROJECT,
                "date": "2026-08-03",
                "seq": 1,
                "content": _report(),
            },
        ),
        original_sender="user:testuser",
    )

    assert "Session report written" in result
    assert "Active mode scheduled the next session" in result
    assert len(node._scheduled_wakes) == 1
    wake_id = next(iter(node._scheduled_wakes))
    assert wake_id in result


@pytest.mark.asyncio
async def test_writing_a_report_for_an_inactive_project_schedules_nothing(node):
    node._push_mesh_tool_activity = _noop_activity

    result = await node._execute_single_tool_with_confirmation(
        ToolCall(
            name="dossier_write_report",
            raw_xml="",
            arguments={
                "entity_key": PROJECT,
                "date": "2026-08-03",
                "seq": 1,
                "content": _report(),
            },
        ),
        original_sender="user:testuser",
    )

    assert "Active mode scheduled no next session" in result
    assert "not in active mode" in result
    assert node._scheduled_wakes == {}


@pytest.mark.asyncio
async def test_a_report_for_an_unenrolled_project_is_left_alone(node):
    """The hook must not editorialize on a report outside this agent's fleet."""
    node._push_mesh_tool_activity = _noop_activity
    init_dossier("project:elsewhere", owner_agent="agent:coder:someone",
                 state_paths=node.state_paths_for_test)

    result = await node._execute_single_tool_with_confirmation(
        ToolCall(
            name="dossier_write_report",
            raw_xml="",
            arguments={
                "entity_key": "project:elsewhere",
                "date": "2026-08-03",
                "seq": 1,
                "content": _report(),
            },
        ),
        original_sender="user:testuser",
    )

    assert "Active mode" not in result
    assert node._scheduled_wakes == {}


async def _noop_activity(*args, **kwargs):
    return None
