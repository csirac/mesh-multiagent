"""Client-side ``/dossier`` logic in run_user_tui.py.

``/dossier [project]`` opens a project's dossier in emacs. The project comes
from the argument when there is one and otherwise from the channel being
viewed, because channels and project slugs share a namespace
(``#rec-fishing`` ↔ ``project:rec-fishing``).

Everything runs against a hand-built ``MeshTUI``. No router, no agent, no
network, and no editor is ever launched — ``subprocess.Popen`` is captured.
"""

import subprocess

import pytest

import run_user_tui
from run_user_tui import MeshTUI

SLUG = "rec-fishing"


def make_tui(current_view=None) -> MeshTUI:
    """A MeshTUI with only what the /dossier surface touches."""
    tui = object.__new__(MeshTUI)
    tui.current_view = current_view
    return tui


@pytest.fixture
def digests(tmp_path, monkeypatch):
    """Redirect the dossier root and return it."""
    from mesh import project_dossier as pd

    root = tmp_path / "digests"
    root.mkdir()
    monkeypatch.setattr(pd, "DIGESTS_DIR", root)
    return root


@pytest.fixture(autouse=True)
def headless(monkeypatch):
    """No tmux and no display unless a test asks for one."""
    for var in ("TMUX", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def launched(monkeypatch):
    """Capture detached editor launches instead of running one."""
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return object()

    monkeypatch.setattr(run_user_tui.subprocess, "Popen", fake_popen)
    return calls


class Ran(list):
    """Captured waited-on launches, plus the exit code they should report."""

    returncode = 0
    stderr = ""


@pytest.fixture
def ran(monkeypatch):
    """Capture waited-on launches (tmux) instead of running them."""
    calls = Ran()

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(
            cmd, calls.returncode, stdout="", stderr=calls.stderr
        )

    monkeypatch.setattr(run_user_tui.subprocess, "run", fake_run)
    return calls


@pytest.fixture
def under_tmux(monkeypatch):
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,544691,4")


@pytest.fixture
def with_display(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")


# ── slug resolution ──────────────────────────────────────────────────


def test_slug_is_inferred_from_the_channel_being_viewed():
    tui = make_tui(current_view=f"channel:{SLUG}")

    assert tui._dossier_slug("") == SLUG


def test_an_explicit_bare_slug_wins_over_the_current_channel():
    tui = make_tui(current_view="channel:mesh-infra")

    assert tui._dossier_slug(SLUG) == SLUG


def test_an_explicit_prefixed_key_is_accepted():
    tui = make_tui()

    assert tui._dossier_slug(f"project:{SLUG}") == SLUG


def test_a_direct_message_view_infers_no_project():
    """Only a channel names a project — a DM partner does not."""
    tui = make_tui(current_view="agent:coder:tron")

    assert tui._dossier_slug("") is None


def test_no_argument_and_no_view_infers_no_project():
    assert make_tui()._dossier_slug("") is None


# ── opening the dossier ──────────────────────────────────────────────


def existing_dossier(digests):
    path = digests / f"project-{SLUG}.md"
    path.write_text("# dossier\n")
    return path


def test_under_tmux_the_dossier_opens_in_a_new_tmux_window(
    digests, ran, launched, under_tmux
):
    """The tmux server owns the window, so it needs no DISPLAY and survives us."""
    path = existing_dossier(digests)
    tui = make_tui(current_view=f"channel:{SLUG}")

    tui._handle_dossier_command("")

    assert launched == []
    assert len(ran) == 1
    cmd, _ = ran[0]
    assert cmd == [
        "tmux",
        "new-window",
        "-c",
        str(path.parent),
        "emacs",
        "-nw",
        str(path),
    ]


def test_a_failing_tmux_launch_is_reported_rather_than_claimed_as_success(
    digests, ran, under_tmux, capsys
):
    """The original bug: a dead child behind a cheerful 'Opening ...' line."""
    path = existing_dossier(digests)
    ran.returncode = 1
    ran.stderr = "no server running"
    tui = make_tui(current_view=f"channel:{SLUG}")

    tui._handle_dossier_command("")

    out = capsys.readouterr().out
    assert "no server running" in out
    assert str(path) in out
    assert "Opening" not in out


def test_with_a_display_and_no_tmux_the_dossier_opens_via_emacsclient(
    digests, launched, with_display
):
    path = existing_dossier(digests)
    tui = make_tui(current_view=f"channel:{SLUG}")

    tui._handle_dossier_command("")

    assert len(launched) == 1
    cmd, kwargs = launched[0]
    assert cmd == ["emacsclient", "-n", "-a", "emacs", str(path)]
    # Detached, or quitting the TUI would take the editor with it.
    assert kwargs["start_new_session"] is True


def test_with_neither_tmux_nor_a_display_nothing_is_launched(
    digests, launched, ran, capsys
):
    """Emacs has nowhere to draw — say so instead of launching a doomed child."""
    path = existing_dossier(digests)
    tui = make_tui(current_view=f"channel:{SLUG}")

    tui._handle_dossier_command("")

    assert launched == []
    assert ran == []
    out = capsys.readouterr().out
    assert f"emacs {path}" in out
    assert "Opening" not in out


def test_a_missing_dossier_reports_the_expected_path(digests, launched, capsys):
    tui = make_tui(current_view=f"channel:{SLUG}")

    tui._handle_dossier_command("")

    assert launched == []
    out = capsys.readouterr().out
    assert str(digests / f"project-{SLUG}.md") in out


def test_an_unresolvable_project_prints_usage(digests, launched, capsys):
    tui = make_tui(current_view="agent:coder:tron")

    tui._handle_dossier_command("")

    assert launched == []
    assert "/dossier [project]" in capsys.readouterr().out


def test_a_malformed_slug_is_refused_without_touching_the_filesystem(
    digests, launched, capsys
):
    tui = make_tui()

    tui._handle_dossier_command("../../etc/passwd")

    assert launched == []
    assert "Invalid project entity key" in capsys.readouterr().out
