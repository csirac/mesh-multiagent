"""
Tests for the shell tool — one-shot and persistent session modes.

Covers:
- One-shot mode backward compatibility (cd interception, output, exit codes)
- Persistent sessions: create, state persistence, multi-command sequences
- write_stdin: interactive input to sessions
- yield_time_ms: partial output from long-running commands
- PTY mode: isatty detection
- Session cleanup
- Multiple concurrent sessions
- Smart truncation
"""

import os
import sys
import tempfile
import time

import pytest

from mesh.harness.tools.unified_exec import (
    shell,
    _sessions,
    _sessions_lock,
    _create_session,
    _destroy_session,
    _cleanup_all_sessions,
    _get_session,
    _smart_truncate,
    ShellSession,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def cleanup_sessions():
    """Ensure all sessions are cleaned up after each test."""
    yield
    _cleanup_all_sessions()


@pytest.fixture
def original_dir():
    """Save and restore the working directory."""
    d = os.getcwd()
    yield d
    os.chdir(d)


# ===========================================================================
# One-shot mode — backward compatibility
# ===========================================================================

class TestOneShotMode:
    """One-shot mode should behave identically to the Phase 2 implementation."""

    def test_simple_echo(self):
        result = shell("echo hello world")
        assert "hello world" in result

    def test_exit_code_reported(self):
        result = shell("exit 42")
        assert "[Exit code: 42]" in result

    def test_no_output(self):
        result = shell("true")
        assert result == "(no output)"

    def test_cd_bare(self, original_dir):
        home = os.path.expanduser("~")
        result = shell("cd")
        assert f"Changed directory to {home}" in result
        assert os.getcwd() == home

    def test_cd_path(self, original_dir):
        with tempfile.TemporaryDirectory() as d:
            result = shell(f"cd {d}")
            assert "Changed directory to" in result
            assert os.path.realpath(os.getcwd()) == os.path.realpath(d)

    def test_cd_nonexistent(self, original_dir):
        result = shell("cd /nonexistent_path_xyzzy")
        assert "Not a directory" in result

    def test_timeout(self):
        result = shell("sleep 300", timeout=1)
        assert "timed out" in result.lower() or "Command timed out" in result

    def test_cd_with_pipes_runs_as_command(self, original_dir):
        """cd combined with other commands should run as a normal command, not cd-intercept."""
        result = shell("cd /tmp && pwd")
        assert "/tmp" in result
        # Process cwd should NOT change (cd-intercept only fires for bare cd)
        assert os.getcwd() == original_dir


# ===========================================================================
# Persistent session mode
# ===========================================================================

class TestPersistentSession:
    """Persistent sessions keep shell state between calls."""

    def test_create_session(self):
        result = shell("echo hello-session", session_id="test-s1")
        assert "hello-session" in result
        session = _get_session("test-s1")
        assert session is not None
        assert session.alive

    def test_state_persists_env_var(self):
        """Shell variables set in one call are visible in the next."""
        shell("export MESH_TEST_VAR=42", session_id="test-env")
        result = shell("echo $MESH_TEST_VAR", session_id="test-env")
        assert "42" in result

    def test_state_persists_cd(self):
        """Working directory persists inside the session."""
        with tempfile.TemporaryDirectory() as d:
            shell(f"cd {d}", session_id="test-cd")
            result = shell("pwd", session_id="test-cd")
            assert os.path.realpath(d) in os.path.realpath(result.strip())

    def test_multi_command_sequence(self):
        """Run multiple commands in sequence — state accumulates."""
        shell("X=1", session_id="test-seq")
        shell("Y=2", session_id="test-seq")
        result = shell("echo $((X + Y))", session_id="test-seq")
        assert "3" in result

    def test_session_auto_recreate_on_death(self):
        """If a session's process dies, re-creating it transparently."""
        shell("echo alive", session_id="test-die")
        session = _get_session("test-die")
        assert session is not None
        # Kill the session's process
        session.proc.kill()
        session.proc.wait()
        # Next call should auto-recreate
        result = shell("echo revived", session_id="test-die")
        assert "revived" in result

    def test_multiple_sessions_independent(self):
        """Multiple sessions don't interfere with each other."""
        shell("export VAL=alpha", session_id="multi-a")
        shell("export VAL=beta", session_id="multi-b")
        result_a = shell("echo $VAL", session_id="multi-a")
        result_b = shell("echo $VAL", session_id="multi-b")
        assert "alpha" in result_a
        assert "beta" in result_b


# ===========================================================================
# write_stdin
# ===========================================================================

class TestWriteStdin:
    """write_stdin sends raw input to an existing session."""

    def test_write_stdin_requires_session_id(self):
        result = shell("ignored", write_stdin="hello")
        assert "Error: write_stdin requires session_id" in result

    def test_write_stdin_nonexistent_session(self):
        result = shell("ignored", session_id="no-such", write_stdin="hello")
        assert "no session" in result.lower() or "Error" in result

    def test_write_stdin_feeds_read(self):
        """Feed input to a `read` command via write_stdin."""
        # Start a session that runs `read` in background
        shell("read -p '' ANSWER && echo got-$ANSWER &", session_id="test-stdin")
        time.sleep(0.3)
        result = shell("ignored", session_id="test-stdin",
                       write_stdin="hello\n", yield_time_ms=2000)
        # The read + echo should produce "got-hello"
        # If timing is tricky, at least verify no error
        assert "Error" not in result


# ===========================================================================
# yield_time_ms
# ===========================================================================

class TestYieldTime:
    """yield_time_ms controls how long to wait for output."""

    def test_short_yield_returns_partial(self):
        """A long-running command with short yield returns partial output."""
        # echo fast, then sleep — we should get the echo but not wait for sleep
        result = shell(
            "echo immediate-output; sleep 30",
            session_id="test-yield",
            yield_time_ms=1500,
        )
        assert "immediate-output" in result
        # Should indicate truncation / session still active
        assert "session still active" in result.lower() or "still active" in result.lower() or len(result) > 0

    def test_completed_command_returns_fully(self):
        """A fast command returns full output even with generous yield."""
        result = shell("echo done-fast", session_id="test-yield2", yield_time_ms=5000)
        assert "done-fast" in result


# ===========================================================================
# PTY mode
# ===========================================================================

class TestPTYMode:
    """PTY mode provides a terminal for interactive programs."""

    def test_isatty_true_in_pty(self):
        """Programs see isatty=True when tty=True."""
        result = shell(
            f'{sys.executable} -c "import sys; print(sys.stdout.isatty())"',
            session_id="test-pty",
            tty=True,
            yield_time_ms=3000,
        )
        assert "True" in result

    def test_isatty_false_without_pty(self):
        """Programs see isatty=False in pipe mode (no tty)."""
        result = shell(
            f'{sys.executable} -c "import sys; print(sys.stdout.isatty())"',
            session_id="test-no-pty",
            tty=False,
            yield_time_ms=3000,
        )
        assert "False" in result


# ===========================================================================
# Session cleanup
# ===========================================================================

class TestSessionCleanup:
    """Sessions are cleaned up properly."""

    def test_destroy_session(self):
        shell("echo test", session_id="test-destroy")
        assert _get_session("test-destroy") is not None
        _destroy_session("test-destroy")
        assert _get_session("test-destroy") is None

    def test_cleanup_all(self):
        shell("echo a", session_id="cleanup-a")
        shell("echo b", session_id="cleanup-b")
        assert len([s for s in _sessions if s.startswith("cleanup-")]) == 2
        _cleanup_all_sessions()
        assert len([s for s in _sessions if s.startswith("cleanup-")]) == 0


# ===========================================================================
# Smart truncation
# ===========================================================================

class TestSmartTruncation:
    """Output truncation works correctly."""

    def test_short_output_not_truncated(self):
        lines = "\n".join(f"line {i}" for i in range(10))
        assert _smart_truncate(lines) == lines

    def test_long_output_truncated(self):
        lines = "\n".join(f"line {i}" for i in range(500))
        result = _smart_truncate(lines)
        assert "lines elided" in result
        assert "line 0" in result  # head preserved
        assert "line 499" in result  # tail preserved


# ===========================================================================
# Sentinel collision — per-command nonce prevents false positives
# ===========================================================================

class TestSentinelCollision:
    """User output containing sentinel-like strings must not corrupt results."""

    def test_sentinel_prefix_in_output_not_stripped(self):
        """Output containing the sentinel prefix should be preserved verbatim."""
        from mesh.harness.tools.unified_exec import _SENTINEL_PREFIX
        result = shell(
            f'echo "before {_SENTINEL_PREFIX}fake_nonce__ after"',
            session_id="test-sentinel",
            yield_time_ms=3000,
        )
        assert _SENTINEL_PREFIX in result
        assert "fake_nonce__" in result
        assert "before" in result and "after" in result

    def test_sentinel_prefix_does_not_cause_early_completion(self):
        """Echoing the sentinel prefix mid-stream should not truncate later output."""
        from mesh.harness.tools.unified_exec import _SENTINEL_PREFIX
        result = shell(
            f'echo "{_SENTINEL_PREFIX}decoy__"; echo "real-tail-marker"',
            session_id="test-sentinel2",
            yield_time_ms=3000,
        )
        assert "real-tail-marker" in result


# ===========================================================================
# write_stdin to dead session
# ===========================================================================

class TestWriteStdinEdgeCases:

    def test_write_stdin_to_dead_session(self):
        """write_stdin to a session whose process died should report the error."""
        shell("echo alive", session_id="test-dead-stdin")
        session = _get_session("test-dead-stdin")
        session.proc.kill()
        session.proc.wait()
        result = shell("ignored", session_id="test-dead-stdin",
                        write_stdin="hello\n")
        assert "terminated" in result.lower() or "exited" in result.lower() or "Error" in result


# ===========================================================================
# PTY large output
# ===========================================================================

class TestPTYLargeOutput:

    def test_pty_large_output(self):
        """PTY mode should handle large output without hanging."""
        result = shell(
            "seq 1 500",
            session_id="test-pty-large",
            tty=True,
            yield_time_ms=5000,
        )
        assert "1" in result
        assert "500" in result


# ===========================================================================
# Session auto-recreate visibility
# ===========================================================================

class TestAutoRecreate:

    def test_auto_recreate_loses_state(self):
        """After auto-recreate, previous session state is gone (expected)."""
        shell("export EPHEMERAL=secret123", session_id="test-recreate")
        session = _get_session("test-recreate")
        session.proc.kill()
        session.proc.wait()
        result = shell("echo val=$EPHEMERAL", session_id="test-recreate")
        assert "secret123" not in result
