"""Agent-side autonomous-fleet control (AUTONOMOUS_CONTROL).

Every test here runs against isolated temp state: a StatePaths rooted in
``tmp_path`` and a fake wake store.  Nothing touches ~/.mesh, a production
dossier, or a live wake row.
"""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from mesh.agent_node import AgentNode
from mesh.config import NodeConfig
from mesh.isolation import StatePaths
from mesh.protocol import (
    AUTONOMOUS_WAKE_HEADER,
    ControlAction,
    Message,
    MessageType,
    make_autonomous_control,
)
from mesh.router_v2 import RouterState
from mesh.project_dossier import (
    _parse_frontmatter,
    budget_path,
    check_budget,
    init_dossier,
    read_dossier,
    spend_budget,
)

PROJECT = "project:test-fleet"
OTHER_PROJECT = "project:not-mine"
AGENT_ID = "agent:coder:autopilot-test"


class FakeWakeStore:
    """Records wake persistence without touching SQLite."""

    def __init__(self):
        self.saved: dict[str, dict] = {}
        self.deleted: list[str] = []

    def save_wake(self, wake_id, wake_time, prompt, requested_by, created_at,
                  recurrence=None):
        self.saved[wake_id] = {
            "wake_time": wake_time,
            "prompt": prompt,
            "requested_by": requested_by,
            "recurrence": recurrence,
        }

    def delete_wake(self, wake_id):
        self.deleted.append(wake_id)


class FakeMemorySystem:
    def __init__(self, store):
        self._store = store


@pytest.fixture
def agent(tmp_path):
    """An enrolled controller wired to isolated temp state."""
    state_paths = StatePaths.for_state_root(tmp_path / "state")
    state_paths.digests_dir.mkdir(parents=True, exist_ok=True)
    state_paths.budget_dir.mkdir(parents=True, exist_ok=True)
    init_dossier(
        PROJECT,
        owner_agent=AGENT_ID,
        max_workers_per_day=4,
        state_paths=state_paths,
    )

    config = NodeConfig(
        id=AGENT_ID,
        agent_type="coder",
        nickname="autopilot-test",
        tools=["file_read"],
    )
    config.autonomous_agent_mode_enabled = True
    config.autonomous_projects = [PROJECT]
    config.autonomous_max_workers_per_session = 7

    node = AgentNode(config=config, persist=False)
    node._tool_socket_path = None
    node._scoped_state_paths = lambda: state_paths
    node.state_paths_for_test = state_paths
    node._memory_system = FakeMemorySystem(FakeWakeStore())

    node.sent: list[Message] = []

    async def _capture(msg):
        node.sent.append(msg)

    # Both, deliberately: the control-response path routes through
    # ``send_message`` while other paths still call ``send`` directly, and a
    # fixture that stubs only one silently drops the reply into the retry queue.
    node.send = _capture
    node.send_message = _capture
    return node


def _request(op, **kwargs) -> Message:
    """A router-forwarded control message, as the agent would receive it."""
    msg = make_autonomous_control(
        "router",
        op,
        to_node=AGENT_ID,
        agent=AGENT_ID,
        requested_by="user:testuser",
        message_id="msg-origin",
        **kwargs,
    )
    return msg


async def _run(agent, op, **kwargs) -> dict:
    msg = _request(op, **kwargs)
    await agent.on_message(msg)
    assert agent.sent, f"op={op} produced no response"
    response = agent.sent.pop()
    assert response.to_node == "user:testuser"
    assert response.in_reply_to == "msg-origin"
    assert response.content["action"] == ControlAction.AUTONOMOUS_CONTROL_RESPONSE.value
    return response.content


def _future(minutes: int = 30) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()


# =============================================================================
# op=status
# =============================================================================


@pytest.mark.asyncio
async def test_status_returns_enrollment_wakes_and_budget(agent):
    agent.schedule_wake(_future(), "an existing reminder", requested_by="user:testuser")

    content = await _run(agent, "status")
    assert content["accepted"] is True
    result = content["result"]

    assert result["agent"] == AGENT_ID
    assert result["autonomous_agent_mode_enabled"] is True
    assert result["autonomous_projects"] == [PROJECT]
    assert result["autonomous_max_workers_per_session"] == 7

    assert len(result["wakes"]) == 1
    assert result["wakes"][0]["prompt_preview"].startswith("an existing reminder")

    budget = result["budgets"][PROJECT]
    assert budget["limit"] == 4
    assert budget["used"] == 0
    assert budget["remaining"] == 4
    assert "resets_at" in budget


def test_status_reports_idle_without_an_autonomous_session(agent):
    result = agent._autonomous_status_result()

    assert result["session_in_progress"] is False
    assert result["current_session"] == {}


def test_status_reports_current_autonomous_router_turn(agent):
    agent._router_v2 = SimpleNamespace(
        state=RouterState.AUTO,
        _active_worker_slots=lambda: [],
    )
    agent._current_autonomous_metadata = {
        "autonomous_session_id": "as-test-0001",
        "autonomous_project_key": PROJECT,
        "autonomous_report_to": "channel:test-fleet",
        "autonomous_trigger_id": "wake-test",
    }

    result = agent._autonomous_status_result()

    assert result["session_in_progress"] is True
    assert result["current_session"] == {
        "session_id": "as-test-0001",
        "project_key": PROJECT,
        "report_to": "channel:test-fleet",
        "trigger_id": "wake-test",
    }


def test_status_recovers_autonomous_scope_from_an_active_worker_slot(agent):
    agent._router_v2 = SimpleNamespace(
        state=RouterState.BUSY,
        _active_worker_slots=lambda: [
            SimpleNamespace(
                selection_metadata={
                    "autonomous_session": True,
                    "autonomous_session_id": "as-slot-0001",
                    "autonomous_project_key": PROJECT,
                    "autonomous_report_to": "channel:test-fleet",
                    "autonomous_trigger_id": "wake-slot",
                }
            )
        ],
    )

    result = agent._autonomous_status_result()

    assert result["session_in_progress"] is True
    assert result["current_session"]["session_id"] == "as-slot-0001"
    assert result["current_session"]["project_key"] == PROJECT


@pytest.mark.asyncio
async def test_status_push_sends_heartbeat_lite_ping(agent):
    class CapturingConnection:
        is_closed = False

        def __init__(self):
            self.sent = []

        async def send(self, message):
            self.sent.append(message)

    conn = CapturingConnection()
    agent._conn = conn
    agent._memory_system = None

    await agent._push_router_status()

    assert len(conn.sent) == 1
    ping = conn.sent[0]
    assert ping.type == MessageType.CONTROL
    assert ping.to_node == "router"
    assert ping.content["action"] == ControlAction.PING.value
    assert ping.content["status_summary"]["state"] == "idle"


@pytest.mark.asyncio
async def test_status_refused_when_not_enrolled(agent):
    agent.config.autonomous_agent_mode_enabled = False
    content = await _run(agent, "status")
    assert content["accepted"] is False
    assert "not enrolled" in content["error"]


# =============================================================================
# op=wake
# =============================================================================


@pytest.mark.asyncio
async def test_wake_adds_in_memory_wake_and_persists_it(agent):
    content = await _run(agent, "wake", project=PROJECT, wake_time=_future())
    assert content["accepted"] is True
    result = content["result"]
    wake_id = result["wake_id"]

    # In-memory: the live wake table the scheduler actually reads.
    assert wake_id in agent._scheduled_wakes
    # Persisted: survives a restart.
    assert wake_id in agent._memory_system._store.saved

    assert result["project"] == PROJECT
    assert result["max_workers_this_session"] == 7

    prompt = agent._scheduled_wakes[wake_id].prompt
    assert prompt.startswith(AUTONOMOUS_WAKE_HEADER)
    assert f"project_entity_key: {PROJECT}" in prompt
    assert "report_to: user:testuser" in prompt
    assert "max_workers_this_session: 7" in prompt
    # The dossier path is the agent's own (isolated) one, not the host default.
    assert str(agent.state_paths_for_test.digests_dir) in prompt


@pytest.mark.asyncio
async def test_wake_prompt_opens_a_real_autonomous_session(agent):
    """The generated prompt must satisfy the runtime's own session gate."""
    await _run(agent, "wake", project=PROJECT, wake_time=_future())
    wake = next(iter(agent._scheduled_wakes.values()))
    metadata = agent.autonomous_wake_metadata(wake)
    assert metadata["autonomous_session"] is True
    assert metadata["autonomous_project_key"] == PROJECT
    assert metadata["autonomous_worker_limit"] == 7
    assert metadata["autonomous_report_to"] == "user:testuser"


@pytest.mark.asyncio
async def test_wake_appends_extra_instructions(agent):
    await _run(
        agent, "wake", project=PROJECT, wake_time=_future(),
        prompt="Prioritise the blocked task T-004.",
    )
    prompt = next(iter(agent._scheduled_wakes.values())).prompt
    assert prompt.startswith(AUTONOMOUS_WAKE_HEADER)
    assert prompt.endswith("Prioritise the blocked task T-004.")


@pytest.mark.asyncio
async def test_wake_refuses_project_not_in_autonomous_projects(agent):
    content = await _run(agent, "wake", project=OTHER_PROJECT, wake_time=_future())
    assert content["accepted"] is False
    assert "not in this agent's autonomous_projects" in content["error"]
    assert agent._scheduled_wakes == {}


@pytest.mark.asyncio
async def test_wake_refuses_a_time_in_the_past(agent):
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    content = await _run(agent, "wake", project=PROJECT, wake_time=past)
    assert content["accepted"] is False
    assert "future" in content["error"]
    assert agent._scheduled_wakes == {}


@pytest.mark.asyncio
async def test_wake_resolves_the_project_implicitly_for_single_project_agents(agent):
    content = await _run(agent, "wake", wake_time=_future())
    assert content["accepted"] is True
    assert content["result"]["project"] == PROJECT


@pytest.mark.asyncio
async def test_wake_requires_an_explicit_project_when_the_agent_owns_several(agent):
    agent.config.autonomous_projects = [PROJECT, "project:second"]
    content = await _run(agent, "wake", wake_time=_future())
    assert content["accepted"] is False
    assert "project required" in content["error"]


# =============================================================================
# op=cancel
# =============================================================================


@pytest.mark.asyncio
async def test_cancel_removes_the_wake(agent):
    scheduled = agent.schedule_wake(_future(), "doomed", requested_by="user:testuser")
    wake_id = scheduled["wake_id"]

    content = await _run(agent, "cancel", wake_id=wake_id)
    assert content["accepted"] is True
    assert content["result"]["cancelled_id"] == wake_id
    assert wake_id not in agent._scheduled_wakes
    assert wake_id in agent._memory_system._store.deleted


@pytest.mark.asyncio
async def test_cancel_unknown_wake_is_a_structured_error(agent):
    content = await _run(agent, "cancel", wake_id="wake-nonexistent")
    assert content["accepted"] is False
    assert "No scheduled wake" in content["error"]


# =============================================================================
# op=budget
# =============================================================================


@pytest.mark.asyncio
async def test_budget_edits_only_the_max_workers_per_day_line(agent):
    state_paths = agent.state_paths_for_test
    before = read_dossier(PROJECT, state_paths)

    content = await _run(agent, "budget", project=PROJECT, count=12)
    assert content["accepted"] is True
    result = content["result"]
    assert result["changed"] is True
    assert result["previous_limit"] == "4"
    assert result["budget"]["limit"] == 12
    assert result["budget"]["remaining"] == 12

    after = read_dossier(PROJECT, state_paths)
    assert _parse_frontmatter(after)["max_workers_per_day"] == "12"

    # Exactly one line moved, and it is the one we asked for.
    diff = [
        (b, a)
        for b, a in zip(before.splitlines(), after.splitlines())
        if b != a
    ]
    assert diff == [("max_workers_per_day: 4", "max_workers_per_day: 12")]
    assert len(before.splitlines()) == len(after.splitlines())


@pytest.mark.asyncio
async def test_budget_is_idempotent(agent):
    await _run(agent, "budget", project=PROJECT, count=9)
    unchanged = read_dossier(PROJECT, agent.state_paths_for_test)

    content = await _run(agent, "budget", project=PROJECT, count=9)
    assert content["accepted"] is True
    assert content["result"]["changed"] is False
    assert content["result"]["budget"]["limit"] == 9
    assert read_dossier(PROJECT, agent.state_paths_for_test) == unchanged


@pytest.mark.parametrize("count", [0, -1, 51, 999])
@pytest.mark.asyncio
async def test_budget_enforces_the_1_to_50_bound(agent, count):
    content = await _run(agent, "budget", project=PROJECT, count=count)
    assert content["accepted"] is False
    assert "between 1 and 50" in content["error"]
    assert _parse_frontmatter(
        read_dossier(PROJECT, agent.state_paths_for_test)
    )["max_workers_per_day"] == "4"


@pytest.mark.parametrize("count", [1, 50])
@pytest.mark.asyncio
async def test_budget_accepts_the_bounds_themselves(agent, count):
    content = await _run(agent, "budget", project=PROJECT, count=count)
    assert content["accepted"] is True
    assert content["result"]["budget"]["limit"] == count


@pytest.mark.asyncio
async def test_budget_refuses_a_project_not_in_autonomous_projects(agent):
    content = await _run(agent, "budget", project=OTHER_PROJECT, count=5)
    assert content["accepted"] is False
    assert "not in this agent's autonomous_projects" in content["error"]


# =============================================================================
# op=budget-reset
# =============================================================================


@pytest.mark.asyncio
async def test_budget_reset_clears_used_without_changing_the_limit(agent):
    await _run(agent, "budget", project=PROJECT, count=8)
    dossier_before_reset = read_dossier(PROJECT, agent.state_paths_for_test)
    spend_budget(PROJECT, 5, agent.state_paths_for_test)

    content = await _run(agent, "budget-reset", project=PROJECT)

    assert content["accepted"] is True
    budget = content["result"]["budget"]
    assert content["result"]["project"] == PROJECT
    assert budget["used"] == 0
    assert budget["limit"] == 8
    assert budget["remaining"] == 8
    assert check_budget(PROJECT, agent.state_paths_for_test)["used"] == 0
    assert json.loads(
        budget_path(PROJECT, agent.state_paths_for_test).read_text()
    ) == {"date": budget["date"], "used": 0}
    assert read_dossier(PROJECT, agent.state_paths_for_test) == dossier_before_reset


@pytest.mark.asyncio
async def test_budget_reset_refuses_a_project_not_in_autonomous_projects(agent):
    content = await _run(agent, "budget-reset", project=OTHER_PROJECT)
    assert content["accepted"] is False
    assert "not in this agent's autonomous_projects" in content["error"]


# =============================================================================
# op=active
# =============================================================================


@pytest.mark.asyncio
async def test_active_toggles_only_the_active_frontmatter_line(agent):
    state_paths = agent.state_paths_for_test
    before = read_dossier(PROJECT, state_paths)
    assert _parse_frontmatter(before)["active"] == "false"

    content = await _run(agent, "active", project=PROJECT, value=True)
    assert content["accepted"] is True
    result = content["result"]
    assert result["active"] is True
    assert result["previous_active"] is False
    assert result["changed"] is True
    assert result["gap_minutes"] == 60

    after = read_dossier(PROJECT, state_paths)
    assert _parse_frontmatter(after)["active"] == "true"

    diff = [
        (b, a) for b, a in zip(before.splitlines(), after.splitlines()) if b != a
    ]
    assert diff == [("active: false", "active: true")]
    assert len(before.splitlines()) == len(after.splitlines())


@pytest.mark.asyncio
async def test_active_off_disarms(agent):
    await _run(agent, "active", project=PROJECT, value=True)
    content = await _run(agent, "active", project=PROJECT, value=False)
    assert content["result"]["active"] is False
    assert content["result"]["changed"] is True
    assert _parse_frontmatter(
        read_dossier(PROJECT, agent.state_paths_for_test)
    )["active"] == "false"


@pytest.mark.asyncio
async def test_active_accepts_operator_spellings(agent):
    for spelling, expected in (("on", True), ("off", False), ("enable", True)):
        content = await _run(agent, "active", project=PROJECT, value=spelling)
        assert content["accepted"] is True
        assert content["result"]["active"] is expected


@pytest.mark.asyncio
async def test_active_is_idempotent(agent):
    await _run(agent, "active", project=PROJECT, value=True)
    unchanged = read_dossier(PROJECT, agent.state_paths_for_test)
    content = await _run(agent, "active", project=PROJECT, value=True)
    assert content["result"]["changed"] is False
    assert read_dossier(PROJECT, agent.state_paths_for_test) == unchanged


@pytest.mark.asyncio
async def test_active_refuses_a_project_not_in_autonomous_projects(agent):
    content = await _run(agent, "active", project=OTHER_PROJECT, value=True)
    assert content["accepted"] is False
    assert "not in this agent's autonomous_projects" in content["error"]


@pytest.mark.asyncio
async def test_active_refuses_a_missing_or_nonsense_value(agent):
    """Both are payload-level refusals, so they answer the forwarder directly."""
    for kwargs, expected in (
        ({}, "value required for op=active"),
        ({"value": "perhaps"}, "value must be one of"),
    ):
        await agent.on_message(_request("active", project=PROJECT, **kwargs))
        content = agent.sent.pop().content
        assert content["accepted"] is False
        assert expected in content["error"]


@pytest.mark.asyncio
async def test_status_reports_the_active_flag_and_the_pacing_gap(agent):
    content = await _run(agent, "status")
    assert content["result"]["active"] == {PROJECT: False}
    assert content["result"]["autonomous_active_gap_minutes"] == 60

    await _run(agent, "active", project=PROJECT, value=True)
    content = await _run(agent, "status")
    assert content["result"]["active"] == {PROJECT: True}


# =============================================================================
# Dispatch hygiene
# =============================================================================


@pytest.mark.asyncio
async def test_malformed_payload_is_refused_without_raising(agent):
    msg = Message(
        from_node="router",
        to_node=AGENT_ID,
        type=MessageType.CONTROL,
        content={
            "action": ControlAction.AUTONOMOUS_CONTROL.value,
            "payload": {"op": "detonate", "agent": AGENT_ID},
        },
    )
    await agent.on_message(msg)
    response = agent.sent.pop()
    assert response.content["accepted"] is False
    assert "unknown op" in response.content["error"]


@pytest.mark.asyncio
async def test_other_control_actions_are_ignored(agent):
    """The handler must not answer control traffic that is not its own."""
    msg = Message(
        from_node="router",
        to_node=AGENT_ID,
        type=MessageType.CONTROL,
        content={"action": ControlAction.TODO_RESPONSE.value, "todos": {}},
    )
    await agent.on_message(msg)
    assert agent.sent == []


@pytest.mark.asyncio
async def test_results_are_json_serializable(agent):
    """The TUI renders these over the wire — every result must survive JSON."""
    await _run(agent, "wake", project=PROJECT, wake_time=_future())
    for op, kwargs in (
        ("status", {}),
        ("budget", {"project": PROJECT, "count": 6}),
        ("budget-reset", {"project": PROJECT}),
        ("active", {"project": PROJECT, "value": True}),
    ):
        content = await _run(agent, op, **kwargs)
        json.dumps(content)
