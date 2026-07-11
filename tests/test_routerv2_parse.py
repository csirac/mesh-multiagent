"""Tests for RouterV2 classification JSON parsing and pattern matching."""

import json
from datetime import datetime

import pytest

from mesh.conversation_history import Turn
from mesh.router_v2 import RouterV2, RouterV2Config, WorkerResult, RouterState


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def router(tmp_path):
    """Create a RouterV2 with mocked dependencies for parse testing."""
    async def noop_send(content, in_reply_to=None):
        pass

    async def noop_worker(context, trigger):
        return WorkerResult(response="Done.", context=[], usage=None, error=None)

    config = RouterV2Config(
        llm_enabled=False,
        history_persist=True,
        history_persist_path=str(tmp_path / "test-parse-history.json"),
    )
    return RouterV2(
        worker_fn=noop_worker,
        send_fn=noop_send,
        config=config,
        nickname="test-bot",
        agent_type="test",
        node_id="agent:test:test-bot",
    )


# =============================================================================
# A. Direct JSON parsing (_parse_classification_response)
# =============================================================================


class TestParseClassificationResponse:
    """Tests for _parse_classification_response."""

    def test_parse_simple_json(self, router):
        raw = '{"needs_response": true, "needs_worker": false, "response": "Hello"}'
        result = router._parse_classification_response(raw)
        assert result["needs_response"] is True
        assert result["needs_worker"] is False
        assert result["response"] == "Hello"

    def test_parse_with_markdown_fences(self, router):
        raw = '```json\n{"needs_response": true, "needs_worker": false, "response": "Hello"}\n```'
        result = router._parse_classification_response(raw)
        assert result["needs_response"] is True
        assert result["needs_worker"] is False
        assert result["response"] == "Hello"

    def test_parse_with_bare_markdown_fences(self, router):
        raw = '```\n{"needs_response": true, "needs_worker": true, "response": "On it."}\n```'
        result = router._parse_classification_response(raw)
        assert result["needs_response"] is True
        assert result["needs_worker"] is True
        assert result["response"] == "On it."

    def test_parse_with_nested_braces(self, router):
        raw = '{"needs_response": true, "needs_worker": false, "response": "Use {MLP, GRU} for this."}'
        result = router._parse_classification_response(raw)
        assert result["needs_response"] is True
        assert result["needs_worker"] is False
        assert "MLP" in result["response"]

    def test_parse_with_python_dicts(self, router):
        raw = '{"needs_response": true, "needs_worker": false, "response": "The config is {\\\"key\\\": \\\"value\\\"}."}'
        result = router._parse_classification_response(raw)
        assert result["needs_response"] is True

    def test_parse_with_code_blocks(self, router):
        response_text = "Here's the code:\\n```python\\ndef foo():\\n    d = {}\\n```"
        raw = json.dumps({
            "needs_response": True,
            "needs_worker": False,
            "response": response_text,
        })
        result = router._parse_classification_response(raw)
        assert result["needs_response"] is True
        assert result["needs_worker"] is False

    def test_parse_missing_needs_response(self, router):
        raw = '{"needs_worker": true, "response": "OK"}'
        result = router._parse_classification_response(raw)
        assert result["needs_response"] is True  # defaults to True
        assert result["needs_worker"] is True

    def test_parse_garbage_input(self, router):
        raw = "I don't understand the format"
        result = router._parse_classification_response(raw)
        assert result["needs_response"] is True
        assert result["needs_worker"] is True
        assert result["response"] == "Let me look into that..."

    def test_parse_empty_string(self, router):
        result = router._parse_classification_response("")
        assert result["needs_response"] is True
        assert result["needs_worker"] is True
        assert result["response"] == "Let me look into that..."

    def test_parse_partial_json(self, router):
        raw = '{"needs_response": true, "needs_wor'
        result = router._parse_classification_response(raw)
        # Should fall back gracefully
        assert result["needs_response"] is True
        assert result["needs_worker"] is True

    def test_parse_multiple_json_objects(self, router):
        raw = 'Here is my analysis:\n{"needs_response": true, "needs_worker": false, "response": "Hello"}\nEnd of response.'
        result = router._parse_classification_response(raw)
        assert result["needs_response"] is True
        assert result["needs_worker"] is False
        assert result["response"] == "Hello"

    def test_parse_json_with_newlines_in_response(self, router):
        raw = json.dumps({
            "needs_response": True,
            "needs_worker": False,
            "response": "Line 1\nLine 2\nLine 3",
        })
        result = router._parse_classification_response(raw)
        assert result["needs_response"] is True
        assert "Line 1" in result["response"]
        assert "Line 3" in result["response"]

    def test_parse_needs_response_false(self, router):
        raw = '{"needs_response": false}'
        result = router._parse_classification_response(raw)
        assert result["needs_response"] is False

    def test_parse_bool_as_string(self, router):
        """LLMs sometimes return "true" as a string instead of boolean."""
        raw = '{"needs_response": true, "needs_worker": false, "response": "Done"}'
        result = router._parse_classification_response(raw)
        assert result["needs_worker"] is False


# =============================================================================
# Regression tests for b6732987 (Ada RL spec nested braces bug)
# =============================================================================


class TestParseRegressionNestedBraces:
    """Regression tests for the Ada RL spec bug where nested braces broke parsing."""

    def test_parse_ada_rl_spec(self, router):
        """Exact scenario from the Ada bug: long response with set notation {MLP, transformer, GRU}."""
        response_text = (
            "I've been sketching the upgraded RL spec with your latest points in mind. "
            "Let me summarize the upgraded picture.\n\n"
            "## 1. Policy network: real deep RL, not just a tiny head\n\n"
            "We can afford a non-trivial policy net:\n\n"
            "- Backbone (examples):\n"
            "  - 2-4 layer MLP with residuals, or\n"
            "  - a small transformer over a handful of state tokens {task, tests, history}, or\n"
            "  - GRU/LSTM if we want an explicit recurrent controller.\n"
            "- Outputs:\n"
            "  - Policy head: categorical over intents.\n"
            "  - Value head: scalar value for actor-critic.\n\n"
            "We can even consider recurrent policies:\n"
            "- Maintain a hidden state across time, so state = (current observation embedding, hidden).\n"
        )
        raw = json.dumps({
            "needs_response": True,
            "needs_worker": False,
            "response": response_text,
        })
        result = router._parse_classification_response(raw)
        assert result["needs_response"] is True
        assert result["needs_worker"] is False
        assert "RL spec" in result["response"]
        assert "{task, tests, history}" in result["response"]

    def test_parse_response_with_dict_literal(self, router):
        """Response containing Python dict literals."""
        raw = json.dumps({
            "needs_response": True,
            "needs_worker": False,
            "response": 'The config looks like: {"host": "localhost", "port": 8080}',
        })
        result = router._parse_classification_response(raw)
        assert result["needs_response"] is True
        assert "localhost" in result["response"]

    def test_parse_never_returns_raw_json_on_garbage(self, router):
        """Fallback response should never start with {"needs_ — that would be a JSON leak."""
        garbage_inputs = [
            "Random text with no JSON",
            "{ invalid json }",
            '{"other_key": "value"}',  # valid JSON but missing needs_response/needs_worker
        ]
        for inp in garbage_inputs:
            result = router._parse_classification_response(inp)
            response = result.get("response", "")
            assert not response.startswith('{"needs_'), f"Raw JSON leaked for input: {inp!r}"

    def test_parse_response_with_deeply_nested_braces(self, router):
        """Response with multiple levels of nesting."""
        raw = json.dumps({
            "needs_response": True,
            "needs_worker": False,
            "response": "Config: {a: {b: {c: 1}}}",
        })
        result = router._parse_classification_response(raw)
        assert result["needs_response"] is True
        assert "Config" in result["response"]


# =============================================================================
# B. Pattern matching
# =============================================================================


class TestIsStatusQuery:
    """Tests for _is_status_query."""

    def test_status_matches(self, router):
        assert router._is_status_query("status") is True
        assert router._is_status_query("what's happening") is True
        assert router._is_status_query("you there") is True
        assert router._is_status_query("still there") is True
        assert router._is_status_query("update?") is True

    def test_status_case_insensitive(self, router):
        assert router._is_status_query("Status") is True
        assert router._is_status_query("STATUS") is True

    def test_status_no_match(self, router):
        assert router._is_status_query("hello") is False
        assert router._is_status_query("fix the bug") is False

    def test_status_substring_match(self, router):
        """Status patterns use substring matching ('in' operator)."""
        assert router._is_status_query("tell me about status codes") is True  # "status" is substring
        assert router._is_status_query("are you still there?") is True  # "still there" is substring


class TestIsCancelRequest:
    """Tests for _is_cancel_request."""

    def test_cancel_exact_match(self, router):
        assert router._is_cancel_request("stop the worker") is True
        assert router._is_cancel_request("cancel the worker") is True

    def test_cancel_case_insensitive(self, router):
        assert router._is_cancel_request("Stop The Worker") is True
        assert router._is_cancel_request("CANCEL THE WORKER") is True

    def test_cancel_exact_phrase_only(self, router):
        """Cancel uses exact phrase matching — extra words make it not match."""
        assert router._is_cancel_request("can you stop the worker please") is False
        assert router._is_cancel_request("cancel") is False
        assert router._is_cancel_request("stop it") is False

    def test_cancel_strips_whitespace(self, router):
        assert router._is_cancel_request("  stop the worker  ") is True


# =============================================================================
# C. Dispatch worker block parsing (multi-line task support)
# =============================================================================


class TestDispatchWorkerParsing:
    """Tests for <dispatch_worker> block parsing in _parse_router_response."""

    def test_single_line_task(self, router):
        raw = (
            "On it.\n"
            "<dispatch_worker>\n"
            "task: fix the authentication bug\n"
            "complexity: simple\n"
            "</dispatch_worker>"
        )
        result = router._parse_router_response(raw)
        assert result["dispatch_worker"] is True
        assert result["task"] == "fix the authentication bug"
        assert "task_complexity" not in result
        assert result["response"] == "On it."

    def test_multi_line_numbered_steps(self, router):
        """Multi-line task with numbered steps — the Ada bug."""
        raw = (
            "I'll handle all of that.\n"
            "<dispatch_worker>\n"
            "task: 1. Stop training process PID 500329 (SIGINT, confirm exit)\n"
            "2. Archive iters 16-30 to checkpoints/archive/\n"
            "3. Truncate log.jsonl and episodes.jsonl to iter ≤15\n"
            "4. Clear __pycache__\n"
            "5. Change configs/train.yaml: leaky_alpha from 0.0 to 0.1\n"
            "6. Reduce accessible batch 5 → 3, restart from iter 15\n"
            "complexity: complex\n"
            "</dispatch_worker>"
        )
        result = router._parse_router_response(raw)
        assert result["dispatch_worker"] is True
        assert "task_complexity" not in result
        # All 6 steps should be present
        assert "1. Stop training" in result["task"]
        assert "2. Archive iters" in result["task"]
        assert "3. Truncate log" in result["task"]
        assert "4. Clear __pycache__" in result["task"]
        assert "5. Change configs" in result["task"]
        assert "6. Reduce accessible" in result["task"]

    def test_pipe_separated_single_line(self, router):
        raw = (
            "<dispatch_worker>\n"
            "task: fix the bug | complexity: simple\n"
            "</dispatch_worker>"
        )
        result = router._parse_router_response(raw)
        assert result["dispatch_worker"] is True
        assert result["task"] == "fix the bug"
        assert "task_complexity" not in result

    def test_multi_line_with_blank_lines(self, router):
        """Multi-line task with blank lines between steps."""
        raw = (
            "<dispatch_worker>\n"
            "task: Execute the full rollback sequence:\n"
            "\n"
            "1. Archive checkpoints to backup dir\n"
            "\n"
            "2. Truncate logs to iter 15\n"
            "\n"
            "3. Edit config and restart\n"
            "complexity: complex\n"
            "</dispatch_worker>"
        )
        result = router._parse_router_response(raw)
        assert result["dispatch_worker"] is True
        assert "Execute the full rollback" in result["task"]
        assert "1. Archive checkpoints" in result["task"]
        assert "2. Truncate logs" in result["task"]
        assert "3. Edit config" in result["task"]

    def test_multi_line_with_bullet_points(self, router):
        """Multi-line task with dash bullet points."""
        raw = (
            "<dispatch_worker>\n"
            "task: Review and fix the deployment:\n"
            "- Check nginx config for errors\n"
            "- Restart the service\n"
            "- Verify logs show clean startup\n"
            "complexity: simple\n"
            "</dispatch_worker>"
        )
        result = router._parse_router_response(raw)
        assert result["dispatch_worker"] is True
        assert "Review and fix" in result["task"]
        assert "Check nginx" in result["task"]
        assert "Restart the service" in result["task"]
        assert "Verify logs" in result["task"]

    def test_multi_line_preserves_newlines(self, router):
        """Verify newlines are preserved in multi-line tasks."""
        raw = (
            "<dispatch_worker>\n"
            "task: Step 1: do A\n"
            "Step 2: do B\n"
            "Step 3: do C\n"
            "complexity: simple\n"
            "</dispatch_worker>"
        )
        result = router._parse_router_response(raw)
        assert result["task"] == "Step 1: do A\nStep 2: do B\nStep 3: do C"

    def test_no_complexity_in_result(self, router):
        """Complexity is no longer returned — all workers are equal."""
        raw = (
            "<dispatch_worker>\n"
            "task: do something important\n"
            "</dispatch_worker>"
        )
        result = router._parse_router_response(raw)
        assert result["dispatch_worker"] is True
        assert result["task"] == "do something important"
        assert "task_complexity" not in result


# =============================================================================
# Worker Context — Unified (same as router)
# =============================================================================

def _make_turn(content, role="assistant", meta=None, from_node="agent:test:bot", to_node=None):
    """Helper to create a Turn."""
    return Turn(
        role=role,
        content=content,
        timestamp=datetime.now(),
        from_node=from_node,
        to_node=to_node,
        meta=meta or {},
    )


def _make_router_for_worker_ctx(tmp_path, window_turns, context_window_tokens=80_000):
    """Create a RouterV2 with pre-populated history window."""
    async def noop_send(content, in_reply_to=None):
        pass

    async def noop_worker(context, trigger):
        return WorkerResult(response="Done.", context=[], usage=None, error=None)

    config = RouterV2Config(
        llm_enabled=False,
        history_persist=True,
        history_persist_path=str(tmp_path / "test-worker-ctx.json"),
        worker_context_window_tokens=context_window_tokens,
    )
    router = RouterV2(
        worker_fn=noop_worker,
        send_fn=noop_send,
        config=config,
        nickname="test-bot",
        agent_type="test",
        node_id="agent:test:test-bot",
    )
    router._history._window = list(window_turns)
    return router


class TestWorkerContextUnified:
    """Tests that worker context is a straight copy of router history (with budget trim)."""

    def test_worker_sees_full_history(self, tmp_path):
        """Worker context includes all turns from router history — no filtering."""
        window = [
            _make_turn("User question", role="incoming"),
            _make_turn("Worker digest", meta={"worker_digest": True}),
            _make_turn("Cancelled the current task.", role="outgoing",
                       meta={"router_response": True, "worker_cancelled": True}),
            _make_turn("cancel the worker", role="incoming"),
            _make_turn("New question", role="incoming"),
        ]
        router = _make_router_for_worker_ctx(tmp_path, window)
        ctx = router._build_worker_context()
        contents = [t.content for t in ctx]
        # Worker sees everything the router sees — including cancel artifacts
        assert "User question" in contents
        assert "Worker digest" in contents
        assert "Cancelled the current task." in contents
        assert "cancel the worker" in contents
        assert "New question" in contents
        assert len(ctx) == 5

    def test_budget_trimming(self, tmp_path):
        """Oldest turns trimmed when over budget, most recent preserved."""
        big_content = "x " * 5000  # ~5000 tokens per turn
        window = [_make_turn(f"turn {i}: {big_content}", role="incoming") for i in range(10)]
        router = _make_router_for_worker_ctx(tmp_path, window, context_window_tokens=20_000)
        ctx = router._build_worker_context()
        contents = [t.content for t in ctx]
        assert any("turn 9" in c for c in contents)
        assert not any("turn 0:" in c for c in contents)


class TestMessageReceivedRendering:
    """Tests for <message_received> structural separation in _build_history_xml."""

    def test_trigger_extracted_from_history(self, tmp_path):
        """Trigger message appears in <message_received>, not in <history>."""
        trigger = _make_turn("Please fix the bug", role="incoming", from_node="user:testuser")
        window = [
            _make_turn("Earlier context", role="incoming", from_node="user:testuser"),
            _make_turn("Got it", role="outgoing"),
            trigger,
        ]
        router = _make_router_for_worker_ctx(tmp_path, window)
        xml = router._build_history_xml(trigger_msg=trigger)
        assert "<message_received" in xml
        assert "Please fix the bug" in xml
        # Trigger should NOT appear inside <history>
        history_block = xml.split("</history>")[0]
        assert "Please fix the bug" not in history_block
        # But earlier messages should still be in history
        assert "Earlier context" in history_block
        assert "Got it" in history_block

    def test_no_trigger_msg_no_message_received(self, tmp_path):
        """Without trigger_msg, no <message_received> block is rendered."""
        window = [
            _make_turn("Hello", role="incoming", from_node="user:testuser"),
            _make_turn("Hi there", role="outgoing"),
        ]
        router = _make_router_for_worker_ctx(tmp_path, window)
        xml = router._build_history_xml(trigger_msg=None)
        assert "<message_received" not in xml
        assert "Hello" in xml
        assert "Hi there" in xml

    def test_trigger_with_to_node(self, tmp_path):
        """Trigger with to_node includes it in <message_received> attrs."""
        trigger = _make_turn("Channel msg", role="incoming",
                             from_node="user:testuser", to_node="channel:general")
        window = [trigger]
        router = _make_router_for_worker_ctx(tmp_path, window)
        xml = router._build_history_xml(trigger_msg=trigger)
        assert 'to="channel:general"' in xml
        assert "<message_received" in xml

    def test_trigger_matches_last_occurrence(self, tmp_path):
        """When content appears twice, trigger matches the last occurrence."""
        window = [
            _make_turn("Do the thing", role="incoming", from_node="user:testuser"),
            _make_turn("OK", role="outgoing"),
            _make_turn("Do the thing", role="incoming", from_node="user:testuser"),
        ]
        trigger = window[2]  # the second "Do the thing"
        router = _make_router_for_worker_ctx(tmp_path, window)
        xml = router._build_history_xml(trigger_msg=trigger)
        history_block = xml.split("</history>")[0]
        # First occurrence should still be in history, second extracted
        assert history_block.count("Do the thing") == 1
        assert "<message_received" in xml

    def test_unmatched_trigger_no_extraction(self, tmp_path):
        """If trigger doesn't match any history entry, all stays in <history>."""
        window = [
            _make_turn("Hello", role="incoming", from_node="user:testuser"),
        ]
        # Create a trigger that won't match
        fake_trigger = _make_turn("Nonexistent message", from_node="user:bob")
        router = _make_router_for_worker_ctx(tmp_path, window)
        xml = router._build_history_xml(trigger_msg=fake_trigger)
        assert "<message_received" not in xml
        assert "Hello" in xml
