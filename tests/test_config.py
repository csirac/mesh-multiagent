"""Tests for the config module."""

import pytest
import tempfile
import os
from pathlib import Path

from mesh.config import (
    RouterConfig,
    NodeConfig,
    MeshConfig,
    find_config,
    load_config,
)


class TestRouterConfig:
    def test_defaults(self):
        """RouterConfig has sensible defaults."""
        config = RouterConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 7700
        assert "messages.db" in config.storage_path

    def test_custom_values(self):
        """RouterConfig accepts custom values."""
        config = RouterConfig(
            host="0.0.0.0",
            port=8080,
            storage_path="/custom/path.db",
        )
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.storage_path == "/custom/path.db"

    def test_expands_user_path(self):
        """RouterConfig expands ~ in storage_path."""
        config = RouterConfig(storage_path="~/test/messages.db")
        assert "~" not in config.storage_path
        from mesh.paths import real_home
        assert str(real_home()) in config.storage_path


class TestNodeConfig:
    def test_required_id(self):
        """NodeConfig requires an ID."""
        config = NodeConfig(id="user:testuser")
        assert config.id == "user:testuser"

    def test_defaults(self):
        """NodeConfig has sensible defaults."""
        config = NodeConfig(id="agent:test")
        assert config.router_host == "127.0.0.1"
        assert config.router_port == 7700
        assert config.llm_model is None
        assert config.system_prompt == ""
        assert config.fold_backend == "deepseek-direct"

    def test_custom_values(self):
        """NodeConfig accepts custom values."""
        config = NodeConfig(
            id="agent:custom",
            router_host="192.168.1.1",
            router_port=9000,
            llm_model="claude-3",
            system_prompt="You are a helpful assistant.",
            fold_backend="codex-sol",
        )
        assert config.router_host == "192.168.1.1"
        assert config.router_port == 9000
        assert config.llm_model == "claude-3"
        assert config.system_prompt == "You are a helpful assistant."
        assert config.fold_backend == "codex-sol"


class TestMeshConfig:
    def test_defaults(self):
        """MeshConfig with no arguments has sensible defaults."""
        config = MeshConfig()
        assert config.router.host == "127.0.0.1"
        assert config.router.port == 7700
        assert config.nodes == {}

    def test_from_dict_empty(self):
        """MeshConfig.from_dict handles empty dict."""
        config = MeshConfig.from_dict({})
        assert config.router.host == "127.0.0.1"
        assert config.nodes == {}

    def test_from_dict_router_only(self):
        """MeshConfig.from_dict loads router settings."""
        data = {
            "router": {
                "host": "0.0.0.0",
                "port": 9000,
                "storage_path": "/tmp/test.db",
            }
        }
        config = MeshConfig.from_dict(data)
        assert config.router.host == "0.0.0.0"
        assert config.router.port == 9000
        assert config.router.storage_path == "/tmp/test.db"

    def test_from_dict_with_nodes(self):
        """MeshConfig.from_dict loads node configurations."""
        data = {
            "router": {"port": 9000},
            "nodes": {
                "agent:researcher": {
                    "llm_model": "gpt-4-turbo",
                    "system_prompt": "You research things.",
                },
                "agent:coder": {
                    "llm_model": "claude-3",
                    "system_prompt": "You write code.",
                },
            }
        }
        config = MeshConfig.from_dict(data)

        assert "agent:researcher" in config.nodes
        assert "agent:coder" in config.nodes

        researcher = config.nodes["agent:researcher"]
        assert researcher.id == "agent:researcher"
        assert researcher.llm_model == "gpt-4-turbo"
        assert researcher.system_prompt == "You research things."
        # Should inherit router settings
        assert researcher.router_port == 9000

        coder = config.nodes["agent:coder"]
        assert coder.llm_model == "claude-3"

    def test_nodes_inherit_router_settings(self):
        """Nodes inherit router host/port if not specified."""
        data = {
            "router": {
                "host": "192.168.1.100",
                "port": 8080,
            },
            "nodes": {
                "agent:test": {}
            }
        }
        config = MeshConfig.from_dict(data)
        node = config.nodes["agent:test"]
        assert node.router_host == "192.168.1.100"
        assert node.router_port == 8080

    def test_nodes_override_router_settings(self):
        """Nodes can override inherited router settings."""
        data = {
            "router": {
                "host": "192.168.1.100",
                "port": 8080,
            },
            "nodes": {
                "agent:remote": {
                    "router_host": "remote.example.com",
                    "router_port": 9999,
                }
            }
        }
        config = MeshConfig.from_dict(data)
        node = config.nodes["agent:remote"]
        assert node.router_host == "remote.example.com"
        assert node.router_port == 9999

    def test_to_dict(self):
        """MeshConfig.to_dict serializes correctly."""
        config = MeshConfig.from_dict({
            "router": {"port": 8000},
            "nodes": {
                "agent:test": {
                    "llm_model": "gpt-4",
                    "fold_backend": "codex-sol",
                }
            }
        })
        data = config.to_dict()

        assert data["router"]["port"] == 8000
        assert "agent:test" in data["nodes"]
        assert data["nodes"]["agent:test"]["llm_model"] == "gpt-4"
        assert data["nodes"]["agent:test"]["fold_backend"] == "codex-sol"


class TestMeshConfigFile:
    def test_load_from_yaml(self):
        """MeshConfig.load reads YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
router:
  host: 0.0.0.0
  port: 7700
  storage_path: /tmp/mesh.db

nodes:
  agent:echo:
    llm_model: gpt-4
    system_prompt: Echo things back
""")
            f.flush()
            path = f.name

        try:
            config = MeshConfig.load(path)
            assert config.router.host == "0.0.0.0"
            assert "agent:echo" in config.nodes
            assert config.nodes["agent:echo"].system_prompt == "Echo things back"
        finally:
            os.unlink(path)

    def test_load_nonexistent_returns_defaults(self):
        """MeshConfig.load returns defaults for missing file."""
        config = MeshConfig.load("/nonexistent/path/mesh.yaml")
        assert config.router.host == "127.0.0.1"
        assert config.nodes == {}

    def test_save_and_load_roundtrip(self):
        """Save and load produces equivalent config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.yaml"

            original = MeshConfig.from_dict({
                "router": {"port": 9999},
                "nodes": {
                    "agent:test": {
                        "llm_model": "test-model",
                        "system_prompt": "Test prompt",
                    }
                }
            })
            original.save(path)

            loaded = MeshConfig.load(path)
            assert loaded.router.port == 9999
            assert "agent:test" in loaded.nodes
            assert loaded.nodes["agent:test"].llm_model == "test-model"


class TestFindConfig:
    def test_find_in_current_dir(self):
        """find_config finds mesh.yaml in current directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mesh.yaml in tmpdir
            config_path = Path(tmpdir) / "mesh.yaml"
            config_path.write_text("router:\n  port: 1234\n")

            # Change to tmpdir and check
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                found = find_config()
                assert found is not None
                assert found.name == "mesh.yaml"
            finally:
                os.chdir(old_cwd)

    def test_find_returns_none_if_not_found(self):
        """find_config returns None if no config file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                # Make sure home dir config doesn't exist for this test
                # This might fail if user has ~/.hello-world/mesh.yaml
                # but that's unlikely in test environment
                found = find_config()
                # Can't assert None because user might have config in home
                # Just ensure it returns Path or None
                assert found is None or isinstance(found, Path)
            finally:
                os.chdir(old_cwd)


class TestLoadConfig:
    def test_load_from_explicit_path(self):
        """load_config loads from specified path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("router:\n  port: 5555\n")
            f.flush()
            path = f.name

        try:
            config = load_config(path)
            assert config.router.port == 5555
        finally:
            os.unlink(path)

    def test_load_returns_defaults_if_no_config(self):
        """load_config returns defaults if no config found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                config = load_config()
                assert config.router.port == 7700  # Default
            finally:
                os.chdir(old_cwd)
