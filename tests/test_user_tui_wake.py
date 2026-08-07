"""Client-side ``/wake`` and ``/step`` logic in run_user_tui.py.

Both spellings are the same command: start an autonomous session on a project
*now*. The project comes from the argument when there is one and otherwise from
the channel being viewed, because channels and project slugs share a namespace
(``#rec-fishing`` ↔ ``project:rec-fishing``). The owning controller is derived
from the project, so no agent name is ever typed.

Everything runs against a hand-built ``MeshTUI`` with a fake connection. No
mesh.yaml read, no router, no agent, no network.
"""

import pytest

from run_user_tui import MeshTUI
from mesh.protocol import ControlAction

TRON_AGENT = "agent:coder:tron"
REME_AGENT = "agent:researcher:reme"

FISHING_PROJECT = "project:rec-fishing"
GOLDEN_PROJECT = "project:golden-age"

FISHING_SLUG = "rec-fishing"

ENROLLED = {
    TRON_AGENT: {
        "node_id": TRON_AGENT,
        "nickname": "tron",
        "projects": [FISHING_PROJECT],
        "max_workers_per_session": 3,
    },
    REME_AGENT: {
        "node_id": REME_AGENT,
        "nickname": "reme",
        "projects": [GOLDEN_PROJECT],
        "max_workers_per_session": 5,
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


def make_tui(current_view=None, enrolled=None) -> MeshTUI:
    """A MeshTUI with only what the /wake surface touches."""
    tui = object.__new__(MeshTUI)
    tui.node_id = "user:testuser"
    tui.node = FakeNode()
    tui.current_view = current_view
    tui._enrolled_autonomous_agents = lambda: dict(
        ENROLLED if enrolled is None else enrolled
    )
    return tui


def payloads(tui) -> list[dict]:
    out = []
    for msg in tui.node._conn.sent:
        content = msg.content
        assert content["action"] == ControlAction.AUTONOMOUS_CONTROL.value
        out.append(content["payload"])
    return out


# =============================================================================
# The channel names the project
# =============================================================================


@pytest.mark.parametrize("arg", [None, ""])
@pytest.mark.asyncio
async def test_wake_in_a_project_channel_kicks_its_owner_now(arg):
    tui = make_tui(current_view=f"channel:{FISHING_SLUG}")
    await tui._handle_wake_command(arg)

    (payload,) = payloads(tui)
    assert payload["op"] == "wake"
    assert payload["agent"] == TRON_AGENT
    assert payload["project"] == FISHING_PROJECT
    assert payload["wake_time"] == "now"


@pytest.mark.parametrize("line", ["/wake", "/step", "/WAKE", "/Step"])
@pytest.mark.asyncio
async def test_both_spellings_dispatch_through_handle_command(line):
    """Driven through the real dispatcher, so the registration is under test."""
    tui = make_tui(current_view=f"channel:{FISHING_SLUG}")
    assert await tui.handle_command(line) is not False

    (payload,) = payloads(tui)
    assert payload["op"] == "wake"
    assert payload["agent"] == TRON_AGENT
    assert payload["project"] == FISHING_PROJECT
    assert payload["wake_time"] == "now"


@pytest.mark.parametrize("line", ["/wake rec-fishing", "/step project:rec-fishing"])
@pytest.mark.asyncio
async def test_both_spellings_take_a_project_argument(line):
    tui = make_tui(current_view=None)
    await tui.handle_command(line)

    (payload,) = payloads(tui)
    assert payload["project"] == FISHING_PROJECT
    assert payload["wake_time"] == "now"


@pytest.mark.asyncio
async def test_wake_announces_who_it_is_kicking(capsys):
    tui = make_tui(current_view=f"channel:{FISHING_SLUG}")
    await tui._handle_wake_command(None)

    out = capsys.readouterr().out
    assert FISHING_PROJECT in out
    assert "now" in out.lower()


# =============================================================================
# An explicit argument wins
# =============================================================================


@pytest.mark.parametrize("arg", [FISHING_SLUG, FISHING_PROJECT])
@pytest.mark.asyncio
async def test_an_explicit_project_beats_the_channel(arg):
    """Bare or prefixed, and it overrides the channel being viewed."""
    tui = make_tui(current_view="channel:golden-age")
    await tui._handle_wake_command(arg)

    (payload,) = payloads(tui)
    assert payload["agent"] == TRON_AGENT
    assert payload["project"] == FISHING_PROJECT
    assert payload["wake_time"] == "now"


@pytest.mark.asyncio
async def test_an_explicit_project_works_from_a_direct_message_view():
    tui = make_tui(current_view="agent:coder:tron")
    await tui._handle_wake_command(FISHING_SLUG)

    (payload,) = payloads(tui)
    assert payload["project"] == FISHING_PROJECT


@pytest.mark.asyncio
async def test_extra_instructions_ride_along():
    tui = make_tui(current_view=f"channel:{FISHING_SLUG}")
    await tui._handle_wake_command("-- Finish  the VMS clean, then stop.")

    (payload,) = payloads(tui)
    assert payload["project"] == FISHING_PROJECT
    assert payload["wake_time"] == "now"
    assert payload["prompt"] == "Finish  the VMS clean, then stop."


@pytest.mark.asyncio
async def test_extra_instructions_ride_along_with_an_explicit_project():
    tui = make_tui(current_view=None)
    await tui._handle_wake_command(f"{FISHING_SLUG} -- read the dossier first")

    (payload,) = payloads(tui)
    assert payload["project"] == FISHING_PROJECT
    assert payload["prompt"] == "read the dossier first"


# =============================================================================
# Refusals
# =============================================================================


@pytest.mark.asyncio
async def test_no_argument_outside_a_channel_is_a_usage_error(capsys):
    tui = make_tui(current_view=None)
    await tui._handle_wake_command(None)

    assert payloads(tui) == []
    out = capsys.readouterr().out
    assert "Usage: /wake" in out
    assert "not viewing a channel" in out


@pytest.mark.asyncio
async def test_a_direct_message_view_is_not_a_project(capsys):
    tui = make_tui(current_view="agent:coder:tron")
    await tui._handle_wake_command(None)

    assert payloads(tui) == []
    assert "Usage: /wake" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_a_channel_with_no_enrolled_project_is_refused(capsys):
    tui = make_tui(current_view="channel:mesh-infra")
    await tui._handle_wake_command(None)

    assert payloads(tui) == []
    out = capsys.readouterr().out
    assert "No enrolled agent owns project:mesh-infra" in out


@pytest.mark.asyncio
async def test_an_unknown_explicit_project_is_refused(capsys):
    tui = make_tui(current_view=f"channel:{FISHING_SLUG}")
    await tui._handle_wake_command("does-not-exist")

    assert payloads(tui) == []
    assert "No enrolled agent owns project:does-not-exist" in capsys.readouterr().out
