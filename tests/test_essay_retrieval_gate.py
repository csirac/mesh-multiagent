"""Unit tests for the essays_retrieval_enabled config gate (Phase 4).

When essays_retrieval_enabled is True, essay_get and essay_list are
dynamically added to the agent's enabled_tools list at startup.
essay_edit is NEVER added — it remains fold-engine-only.
Default is False (no essay tools exposed to live agents).
"""
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesh.config import NodeConfig  # noqa: E402


def _make_config(**overrides) -> NodeConfig:
    """Build a NodeConfig with sensible defaults and optional overrides."""
    defaults = {"id": "agent:test:test-agent"}
    defaults.update(overrides)
    return NodeConfig(**defaults)


def _simulate_wiring(enabled_tools: list[str], essays_retrieval_enabled: bool) -> list[str]:
    """Mirror the wiring logic from agent_node.py."""
    result = list(enabled_tools)
    if essays_retrieval_enabled:
        for _essay_tool in ("essay_get", "essay_list"):
            if result and _essay_tool not in result:
                result.append(_essay_tool)
    return result


# ── Config flag default ──────────────────────────────────────────────

def test_config_default_is_false():
    fields = {f.name: f.default for f in dataclasses.fields(NodeConfig)}
    assert fields.get("essays_retrieval_enabled") is False


# ── Dynamic tool injection ───────────────────────────────────────────

def test_enabled_adds_essay_get_and_list():
    """With flag True, essay_get and essay_list are appended to enabled_tools."""
    result = _simulate_wiring(
        ["send_message", "memory_get"], essays_retrieval_enabled=True)
    assert "essay_get" in result
    assert "essay_list" in result


def test_enabled_never_adds_essay_edit():
    """essay_edit must never be added to enabled_tools, even when retrieval is on."""
    result = _simulate_wiring(
        ["send_message", "memory_get"], essays_retrieval_enabled=True)
    assert "essay_edit" not in result


def test_disabled_does_not_add_essay_tools():
    """With flag False (default), no essay tools are added."""
    result = _simulate_wiring(
        ["send_message", "memory_get"], essays_retrieval_enabled=False)
    assert "essay_get" not in result
    assert "essay_list" not in result


def test_no_duplicates_if_already_present():
    """If essay tools are already in the list, don't add duplicates."""
    result = _simulate_wiring(
        ["send_message", "essay_get", "essay_list"],
        essays_retrieval_enabled=True)
    assert result.count("essay_get") == 1
    assert result.count("essay_list") == 1


def test_disabled_with_empty_tools_list():
    """With an empty tools list and flag False, nothing is added."""
    result = _simulate_wiring([], essays_retrieval_enabled=False)
    assert result == []


def test_wiring_matches_agent_node_pattern():
    """The wiring logic produces the expected appended list."""
    result = _simulate_wiring(
        ["send_message", "memory_get", "memory_search"],
        essays_retrieval_enabled=True)
    assert result == [
        "send_message", "memory_get", "memory_search",
        "essay_get", "essay_list",
    ]


def test_config_flag_coexists_with_fold_flag():
    """essays_retrieval_enabled is independent of essay_fold_enabled."""
    config = _make_config()
    assert config.essays_retrieval_enabled is False
    assert config.essay_fold_enabled is False
    config.essays_retrieval_enabled = True
    assert config.essay_fold_enabled is False
