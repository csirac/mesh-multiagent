"""Client-side ``/auto`` logic in run_user_tui.py.

Covers the two follow-on directives:

* the status **collector** — one ordered block for the whole fleet, with a
  timeout and inline rendering of unreachable agents;
* the **project** argument on ``/auto wake``, including the backward-compatible
  inference form;
* the **immediate** form — omitting the time starts a session now.

Everything runs against a hand-built ``MeshTUI`` with a fake connection. No
mesh.yaml read, no router, no agent, no network.
"""

import asyncio

import pytest

import run_user_tui
from run_user_tui import MeshTUI
from mesh.protocol import ControlAction

RL_AGENT = "agent:coder:autopilot-rl"
CANARY_AGENT = "agent:coder:autopilot"
REME_AGENT = "agent:researcher:reme"

RL_PROJECT = "project:bluesky-rl"
CANARY_PROJECT = "project:coco-canary"
GOLDEN_PROJECT = "project:golden-age"

ENROLLED = {
    # Deliberately not in nickname order — the collector must sort.
    REME_AGENT: {
        "node_id": REME_AGENT,
        "nickname": "reme",
        "projects": [GOLDEN_PROJECT],
        "max_workers_per_session": 5,
    },
    RL_AGENT: {
        "node_id": RL_AGENT,
        "nickname": "autopilot-rl",
        "projects": [RL_PROJECT],
        "max_workers_per_session": 20,
    },
    CANARY_AGENT: {
        "node_id": CANARY_AGENT,
        "nickname": "autopilot",
        "projects": [CANARY_PROJECT],
        "max_workers_per_session": 2,
    },
}


class FakeConn:
    def __init__(self):
        self.sent = []

    async def send(self, msg):
        self.sent.append(msg)


class FakeNode:
    def __init__(self):
        self._conn = FakeConn()


def make_tui(enrolled=None) -> MeshTUI:
    """A MeshTUI with only what the /auto surface touches."""
    tui = object.__new__(MeshTUI)
    tui.node_id = "user:testuser"
    tui.node = FakeNode()
    tui._auto_status_collector = None
    tui._enrolled_autonomous_agents = lambda: dict(
        ENROLLED if enrolled is None else enrolled
    )
    tui._print_prompt_hint = lambda: None
    return tui


def payloads(tui, op=None) -> list[dict]:
    out = []
    for msg in tui.node._conn.sent:
        content = msg.content
        assert content["action"] == ControlAction.AUTONOMOUS_CONTROL.value
        payload = content["payload"]
        if op is None or payload.get("op") == op:
            out.append(payload)
    return out


def status_response(node_id, projects, per_session, wakes=(), budgets=None):
    return {
        "action": ControlAction.AUTONOMOUS_CONTROL_RESPONSE.value,
        "op": "status",
        "agent": node_id,
        "accepted": True,
        "result": {
            "agent": node_id,
            "autonomous_agent_mode_enabled": True,
            "autonomous_projects": list(projects),
            "autonomous_max_workers_per_session": per_session,
            "budgets": budgets
            or {
                key: {
                    "entity_key": key,
                    "limit": 4,
                    "used": 1,
                    "remaining": 3,
                    "resets_at": "2026-08-04T00:00:00",
                }
                for key in projects
            },
            "wakes": list(wakes),
        },
    }


def error_response(node_id, error, op="status"):
    return {
        "action": ControlAction.AUTONOMOUS_CONTROL_RESPONSE.value,
        "op": op,
        "agent": node_id,
        "accepted": False,
        "error": error,
    }


# =============================================================================
# Directive 2 — /auto wake <agent> [<project>] <time> [-- extra]
# =============================================================================


@pytest.mark.asyncio
async def test_wake_with_bare_project_carries_it_on_the_wire():
    tui = make_tui()
    await tui._handle_auto_command("wake autopilot-rl bluesky-rl in 30 minutes")

    (payload,) = payloads(tui, "wake")
    assert payload["agent"] == RL_AGENT
    assert payload["project"] == RL_PROJECT
    assert payload["wake_time"] == "in 30 minutes"
    assert "prompt" not in payload


@pytest.mark.asyncio
async def test_wake_accepts_a_project_prefixed_project():
    tui = make_tui()
    await tui._handle_auto_command("wake autopilot-rl project:bluesky-rl at 5pm")

    (payload,) = payloads(tui, "wake")
    assert payload["project"] == RL_PROJECT
    assert payload["wake_time"] == "at 5pm"


@pytest.mark.asyncio
async def test_wake_without_a_project_infers_it_for_a_single_project_agent():
    tui = make_tui()
    await tui._handle_auto_command("wake reme in 15 minutes")

    (payload,) = payloads(tui, "wake")
    assert payload["agent"] == REME_AGENT
    assert payload["project"] == GOLDEN_PROJECT
    assert payload["wake_time"] == "in 15 minutes"


@pytest.mark.asyncio
async def test_wake_without_a_project_is_refused_when_the_agent_owns_several(capsys):
    enrolled = dict(ENROLLED)
    enrolled[RL_AGENT] = dict(
        enrolled[RL_AGENT], projects=[RL_PROJECT, "project:second"]
    )
    tui = make_tui(enrolled)

    await tui._handle_auto_command("wake autopilot-rl in 30 minutes")

    assert payloads(tui, "wake") == []
    out = capsys.readouterr().out
    assert "owns 2 projects" in out
    assert RL_PROJECT in out and "project:second" in out


@pytest.mark.asyncio
async def test_wake_refuses_an_unknown_project(capsys):
    tui = make_tui()
    await tui._handle_auto_command("wake autopilot-rl project:does-not-exist in 1 hour")

    assert payloads(tui, "wake") == []
    assert "No enrolled agent owns project:does-not-exist" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_wake_refuses_a_project_owned_by_a_different_agent(capsys):
    tui = make_tui()
    await tui._handle_auto_command("wake autopilot-rl golden-age in 1 hour")

    assert payloads(tui, "wake") == []
    out = capsys.readouterr().out
    assert "belongs to" in out and REME_AGENT in out


@pytest.mark.asyncio
async def test_wake_refuses_an_unknown_agent(capsys):
    tui = make_tui()
    await tui._handle_auto_command("wake nobody in 30 minutes")

    assert payloads(tui, "wake") == []
    assert "Unknown autonomous agent 'nobody'" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_wake_extra_instructions_survive_the_new_syntax():
    tui = make_tui()
    await tui._handle_auto_command(
        "wake autopilot-rl bluesky-rl in 30 minutes -- Prioritise  T-004, then stop."
    )

    (payload,) = payloads(tui, "wake")
    assert payload["project"] == RL_PROJECT
    assert payload["wake_time"] == "in 30 minutes"
    # Verbatim, including the double space the old token-join used to eat.
    assert payload["prompt"] == "Prioritise  T-004, then stop."


@pytest.mark.asyncio
async def test_wake_extra_instructions_still_work_without_a_project():
    tui = make_tui()
    await tui._handle_auto_command("wake reme in 15 minutes -- read the dossier first")

    (payload,) = payloads(tui, "wake")
    assert payload["project"] == GOLDEN_PROJECT
    assert payload["wake_time"] == "in 15 minutes"
    assert payload["prompt"] == "read the dossier first"


@pytest.mark.asyncio
async def test_wake_time_words_are_never_mistaken_for_a_project():
    """A bare first token that is not an owned project is part of the time."""
    tui = make_tui()
    await tui._handle_auto_command("wake reme tomorrow at 09:00")

    (payload,) = payloads(tui, "wake")
    assert payload["project"] == GOLDEN_PROJECT
    assert payload["wake_time"] == "tomorrow at 09:00"


# =============================================================================
# Directive 3 — omitting the time means "now"
#
# Controllers routinely end up with no wake pending, so the shortest possible
# spelling has to start a session. The absence of a time is expressed on the
# wire as the literal "now" rather than an empty field, because op=wake is
# contractually required to carry one.
# =============================================================================


@pytest.mark.asyncio
async def test_wake_without_a_time_is_immediate_for_a_single_project_agent():
    tui = make_tui()
    await tui._handle_auto_command("wake reme")

    (payload,) = payloads(tui, "wake")
    assert payload["agent"] == REME_AGENT
    assert payload["project"] == GOLDEN_PROJECT
    assert payload["wake_time"] == "now"


@pytest.mark.asyncio
async def test_wake_with_an_explicit_project_and_no_time_is_immediate():
    """The only immediate route for an agent that owns several projects."""
    tui = make_tui()
    await tui._handle_auto_command("wake autopilot-rl bluesky-rl")

    (payload,) = payloads(tui, "wake")
    assert payload["project"] == RL_PROJECT
    assert payload["wake_time"] == "now"


@pytest.mark.asyncio
async def test_immediate_wake_still_carries_extra_instructions():
    tui = make_tui()
    await tui._handle_auto_command("wake reme -- read the dossier first")

    (payload,) = payloads(tui, "wake")
    assert payload["wake_time"] == "now"
    assert payload["prompt"] == "read the dossier first"


@pytest.mark.asyncio
async def test_wake_without_a_time_is_refused_when_the_agent_owns_several(capsys):
    """Ambiguity still beats convenience — the operator must name a project."""
    enrolled = dict(ENROLLED)
    enrolled[RL_AGENT] = {
        **ENROLLED[RL_AGENT],
        "projects": [RL_PROJECT, "project:second-thing"],
    }
    tui = make_tui(enrolled)
    await tui._handle_auto_command("wake autopilot-rl")

    assert payloads(tui, "wake") == []
    assert "owns 2 projects" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_wake_without_a_time_is_refused_when_the_agent_owns_none(capsys):
    enrolled = dict(ENROLLED)
    enrolled[RL_AGENT] = {**ENROLLED[RL_AGENT], "projects": []}
    tui = make_tui(enrolled)
    await tui._handle_auto_command("wake autopilot-rl")

    assert payloads(tui, "wake") == []
    assert "owns no projects" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_wake_with_no_agent_at_all_is_a_usage_error(capsys):
    tui = make_tui()
    await tui._handle_auto_command("wake")

    assert payloads(tui, "wake") == []
    assert "Usage: /auto wake" in capsys.readouterr().out


# =============================================================================
# PI report requests — `/report [project] [since YYYY-MM-DD]`
# =============================================================================


@pytest.mark.asyncio
async def test_report_with_project_and_boundary_carries_the_control_payload():
    tui = make_tui()

    await tui._handle_report_command("bluesky-rl 2026-08-01")

    (payload,) = payloads(tui, "report")
    assert payload == {
        "op": "report",
        "agent": RL_AGENT,
        "project": RL_PROJECT,
        "since": "2026-08-01",
    }


@pytest.mark.asyncio
async def test_report_infers_the_project_from_the_current_channel():
    tui = make_tui()
    tui.current_view = "channel:golden-age"

    await tui._handle_report_command("2026-08-01")

    (payload,) = payloads(tui, "report")
    assert payload["agent"] == REME_AGENT
    assert payload["project"] == GOLDEN_PROJECT
    assert payload["since"] == "2026-08-01"


@pytest.mark.asyncio
async def test_report_refuses_a_non_enrolled_project(capsys):
    tui = make_tui()

    await tui._handle_report_command("does-not-exist")

    assert payloads(tui, "report") == []
    assert "No enrolled agent owns project:does-not-exist" in capsys.readouterr().out


# =============================================================================
# Directive 1 — the status collector
# =============================================================================


def deliver(tui, node_id, content):
    """Hand a response to the TUI the way message_receiver would."""
    collector = tui._auto_status_collector
    assert collector is not None, "no collection in flight"
    message_id = next(
        mid for mid, target in collector["pending"].items() if target == node_id
    )
    tui._handle_autonomous_control_response(content, node_id, message_id)


@pytest.mark.asyncio
async def test_status_fans_out_to_every_enrolled_agent():
    tui = make_tui()
    await tui._start_autonomous_status_collection()

    sent = payloads(tui, "status")
    assert {p["agent"] for p in sent} == set(ENROLLED)

    collector = tui._auto_status_collector
    # Render order is alphabetical by nickname: autopilot, autopilot-rl, reme.
    assert collector["queried"] == [CANARY_AGENT, RL_AGENT, REME_AGENT]

    collector["task"].cancel()


@pytest.mark.asyncio
async def test_collector_renders_one_ordered_block(capsys):
    tui = make_tui()
    await tui._start_autonomous_status_collection()
    collector = tui._auto_status_collector

    # Deliver deliberately out of order.
    deliver(tui, REME_AGENT, status_response(REME_AGENT, [GOLDEN_PROJECT], 5))
    deliver(tui, RL_AGENT, status_response(
        RL_AGENT, [RL_PROJECT], 20,
        wakes=[{
            "id": "wake-abc",
            "wake_time_local": "2026-08-03 09:00 CDT",
            "prompt_preview": "[AUTONOMOUS PROJECT SESSION]",
        }],
    ))
    deliver(tui, CANARY_AGENT, status_response(CANARY_AGENT, [CANARY_PROJECT], 2))

    await collector["task"]
    out = capsys.readouterr().out

    # One consolidated header, and the collector is torn down.
    assert out.count("Autonomous fleet (") == 1
    assert "3 of 3 responded" in out
    assert tui._auto_status_collector is None

    # Ordered by nickname regardless of arrival order.
    lower = out.lower()
    assert lower.index("autopilot\x1b") < lower.index("autopilot-rl") < lower.index("reme")

    # Enrollment, budget and wake detail all present.
    assert "20 worker(s)/session" in out
    assert RL_PROJECT in out and "1/4 used today" in out
    assert "3 left" in out and "resets 2026-08-04T00:00:00" in out
    assert "wake-abc" in out and "2026-08-03 09:00 CDT" in out
    assert "no pending wakes" in out
    assert "timed out" not in out


@pytest.mark.asyncio
async def test_collector_renders_an_unreachable_agent_inline(capsys):
    tui = make_tui()
    await tui._start_autonomous_status_collection()
    collector = tui._auto_status_collector

    deliver(tui, CANARY_AGENT, status_response(CANARY_AGENT, [CANARY_PROJECT], 2))
    deliver(tui, RL_AGENT, error_response(
        RL_AGENT, f"{RL_AGENT} is not connected — start the agent before controlling it"
    ))
    deliver(tui, REME_AGENT, status_response(REME_AGENT, [GOLDEN_PROJECT], 5))

    await collector["task"]
    out = capsys.readouterr().out

    # Inline, inside the single block — not a separate "✗ /auto status" line.
    assert out.count("Autonomous fleet (") == 1
    assert "✗ /auto status" not in out
    assert "is not connected" in out
    assert "3 of 3 responded" in out
    # The router error still counts as an answer, and the config line shows.
    assert RL_PROJECT in out


@pytest.mark.asyncio
async def test_collector_times_out_and_names_the_silent_agents(capsys, monkeypatch):
    monkeypatch.setattr(run_user_tui, "AUTO_STATUS_TIMEOUT_SECS", 0.05)
    tui = make_tui()
    await tui._start_autonomous_status_collection()
    collector = tui._auto_status_collector

    deliver(tui, CANARY_AGENT, status_response(CANARY_AGENT, [CANARY_PROJECT], 2))

    await collector["task"]
    out = capsys.readouterr().out

    assert out.count("Autonomous fleet (") == 1
    assert "1 of 3 responded" in out
    assert "timed out after 0.05s waiting for" in out
    # Both silent agents named, by display name.
    named = out.lower().split("waiting for:")[1]
    assert "autopilot-rl" in named and "reme" in named
    # What did arrive is still rendered.
    assert CANARY_PROJECT in out
    assert tui._auto_status_collector is None


@pytest.mark.asyncio
async def test_a_second_collection_is_refused_while_one_is_in_flight(capsys):
    tui = make_tui()
    await tui._start_autonomous_status_collection()
    collector = tui._auto_status_collector
    capsys.readouterr()

    await tui._start_autonomous_status_collection()
    assert "already in flight" in capsys.readouterr().out
    assert tui._auto_status_collector is collector

    collector["task"].cancel()


@pytest.mark.asyncio
async def test_uncorrelated_status_reply_is_rendered_standalone(capsys):
    """A reply with an unknown message id must not be swallowed."""
    tui = make_tui()
    await tui._start_autonomous_status_collection()
    collector = tui._auto_status_collector
    capsys.readouterr()

    tui._handle_autonomous_control_response(
        status_response(RL_AGENT, [RL_PROJECT], 20), RL_AGENT, "not-a-known-id"
    )
    out = capsys.readouterr().out
    assert RL_PROJECT in out
    assert "Autonomous fleet (" not in out
    # And it did not consume a pending slot.
    assert len(collector["pending"]) == 3

    collector["task"].cancel()


@pytest.mark.asyncio
async def test_non_status_ops_bypass_the_collector(capsys):
    tui = make_tui()
    await tui._start_autonomous_status_collection()
    collector = tui._auto_status_collector
    capsys.readouterr()

    tui._handle_autonomous_control_response(
        {
            "action": ControlAction.AUTONOMOUS_CONTROL_RESPONSE.value,
            "op": "wake",
            "agent": RL_AGENT,
            "accepted": True,
            "result": {
                "wake_id": "wake-xyz",
                "wake_time_local": "2026-08-03 09:00 CDT",
                "project": RL_PROJECT,
                "max_workers_this_session": 20,
            },
        },
        RL_AGENT,
        next(iter(collector["pending"])),
    )
    out = capsys.readouterr().out
    assert "✓ Wake wake-xyz scheduled" in out
    assert len(collector["pending"]) == 3

    collector["task"].cancel()


@pytest.mark.asyncio
async def test_status_with_no_enrolled_agents_says_so(capsys):
    tui = make_tui({})
    await tui._handle_auto_command("")

    assert tui.node._conn.sent == []
    assert "No agents are enrolled" in capsys.readouterr().out
    assert tui._auto_status_collector is None


@pytest.mark.asyncio
async def test_status_without_a_connection_does_not_leave_a_collector(capsys):
    tui = make_tui()
    tui.node = None
    await tui._handle_auto_command("")

    assert "Not connected to router" in capsys.readouterr().out
    assert tui._auto_status_collector is None


# =============================================================================
# Ordering helper
# =============================================================================


def test_display_order_is_alphabetical_by_nickname():
    assert MeshTUI._autonomous_display_order(ENROLLED) == [
        CANARY_AGENT,  # autopilot
        RL_AGENT,      # autopilot-rl
        REME_AGENT,    # reme
    ]


def test_display_order_falls_back_to_node_id_without_a_nickname():
    enrolled = {
        "agent:coder:zed": {"nickname": "", "projects": []},
        "agent:coder:abe": {"nickname": "", "projects": []},
    }
    assert MeshTUI._autonomous_display_order(enrolled) == [
        "agent:coder:abe",
        "agent:coder:zed",
    ]


# =============================================================================
# /auto budget
# =============================================================================


@pytest.mark.asyncio
async def test_budget_reset_sends_a_project_without_a_count():
    tui = make_tui()
    await tui._handle_auto_command("budget bluesky-rl reset")

    (payload,) = payloads(tui, "budget-reset")
    assert payload["agent"] == RL_AGENT
    assert payload["project"] == RL_PROJECT
    assert "count" not in payload


@pytest.mark.asyncio
async def test_numeric_budget_still_sends_the_budget_op():
    tui = make_tui()
    await tui._handle_auto_command("budget bluesky-rl 12")

    (payload,) = payloads(tui, "budget")
    assert payload["agent"] == RL_AGENT
    assert payload["project"] == RL_PROJECT
    assert payload["count"] == 12


def test_budget_reset_response_renders_the_restored_budget(capsys):
    tui = make_tui()
    tui._handle_autonomous_control_response(
        {
            "action": ControlAction.AUTONOMOUS_CONTROL_RESPONSE.value,
            "op": "budget-reset",
            "agent": RL_AGENT,
            "accepted": True,
            "result": {
                "project": RL_PROJECT,
                "budget": {
                    "used": 0,
                    "limit": 4,
                    "remaining": 4,
                    "resets_at": "2026-08-08T00:00:00",
                },
            },
        },
        RL_AGENT,
    )
    out = capsys.readouterr().out
    assert RL_PROJECT in out
    assert "used 0/4" in out
    assert "4 remaining" in out


# =============================================================================
# /auto active
# =============================================================================


@pytest.mark.asyncio
async def test_active_on_carries_a_real_bool_on_the_wire():
    tui = make_tui()
    await tui._handle_auto_command("active bluesky-rl on")
    (payload,) = payloads(tui, "active")
    assert payload["agent"] == RL_AGENT
    assert payload["project"] == RL_PROJECT
    assert payload["value"] is True


@pytest.mark.asyncio
async def test_active_off_is_sent_not_dropped():
    """`off` must reach the agent; a falsy value that vanishes would arm it."""
    tui = make_tui()
    await tui._handle_auto_command("active project:golden-age off")
    (payload,) = payloads(tui, "active")
    assert payload["agent"] == REME_AGENT
    assert payload["project"] == GOLDEN_PROJECT
    assert payload["value"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "word,expected",
    [("on", True), ("enable", True), ("true", True),
     ("off", False), ("disable", False), ("false", False)],
)
async def test_active_accepts_the_operator_spellings(word, expected):
    tui = make_tui()
    await tui._handle_auto_command(f"active bluesky-rl {word}")
    (payload,) = payloads(tui, "active")
    assert payload["value"] is expected


@pytest.mark.asyncio
async def test_active_refuses_a_nonsense_flag(capsys):
    tui = make_tui()
    await tui._handle_auto_command("active bluesky-rl maybe")
    assert payloads(tui) == []
    assert "on or off" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_active_refuses_an_unknown_project(capsys):
    tui = make_tui()
    await tui._handle_auto_command("active does-not-exist on")
    assert payloads(tui) == []
    assert "No enrolled agent owns" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_active_with_the_wrong_argument_count_is_a_usage_error(capsys):
    tui = make_tui()
    await tui._handle_auto_command("active bluesky-rl")
    assert payloads(tui) == []
    assert "Usage: /auto active" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_active_response_renders_the_armed_state(capsys):
    tui = make_tui()
    tui._handle_autonomous_control_response(
        {
            "action": ControlAction.AUTONOMOUS_CONTROL_RESPONSE.value,
            "op": "active",
            "agent": RL_AGENT,
            "accepted": True,
            "result": {
                "project": RL_PROJECT,
                "active": True,
                "changed": True,
                "gap_minutes": 60,
                "budget": {"remaining": 7},
            },
        },
        RL_AGENT,
    )
    out = capsys.readouterr().out
    assert RL_PROJECT in out
    assert "ACTIVE" in out
    assert "60" in out


def test_status_block_marks_active_projects(capsys):
    tui = make_tui()
    content = status_response(RL_AGENT, [RL_PROJECT], 20)
    content["result"]["active"] = {RL_PROJECT: True}
    tui._render_autonomous_status_agent(content, RL_AGENT)
    assert "ACTIVE" in capsys.readouterr().out


def test_status_block_marks_session_in_progress(capsys):
    tui = make_tui()
    content = status_response(RL_AGENT, [RL_PROJECT], 20)
    content["result"].update(
        {
            "session_in_progress": True,
            "current_session": {
                "session_id": "as-test-0001",
                "project_key": RL_PROJECT,
            },
        }
    )

    tui._render_autonomous_status_agent(content, RL_AGENT)
    out = capsys.readouterr().out
    assert "session in progress" in out
    assert "as-test-0001" in out
    assert RL_PROJECT in out


def test_status_block_marks_manual_projects(capsys):
    tui = make_tui()
    content = status_response(REME_AGENT, [GOLDEN_PROJECT], 5)
    content["result"]["active"] = {GOLDEN_PROJECT: False}
    tui._render_autonomous_status_agent(content, REME_AGENT)
    out = capsys.readouterr().out
    assert "manual" in out
    assert "ACTIVE" not in out


def test_status_block_from_a_pre_active_agent_claims_nothing(capsys):
    """An agent on older code omits the field — do not render it as disarmed."""
    tui = make_tui()
    content = status_response(REME_AGENT, [GOLDEN_PROJECT], 5)
    content["result"].pop("active", None)
    tui._render_autonomous_status_agent(content, REME_AGENT)
    out = capsys.readouterr().out
    assert "manual" not in out
    assert "ACTIVE" not in out
