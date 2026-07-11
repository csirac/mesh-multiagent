import importlib.util
from pathlib import Path

import pytest
import mesh.paths

flask = pytest.importorskip("flask", reason="Flask required for web-client tests")


def load_serve_module():
    serve_path = Path(__file__).resolve().parents[1] / "web-client" / "serve.py"
    spec = importlib.util.spec_from_file_location("mesh_web_client_serve_for_test", serve_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_notes_dir_defaults_to_shared_mesh_notes_dir():
    serve = load_serve_module()

    assert serve.NOTES_DIR == mesh.paths.NOTES_DIR


def test_notes_endpoints_require_auth_and_support_crud(tmp_path, monkeypatch):
    serve = load_serve_module()
    monkeypatch.setattr(serve, "NOTES_DIR", tmp_path)
    monkeypatch.setattr(serve, "MESH_AUTH_TOKEN", "test-token")
    client = serve.app.test_client()
    headers = {"Authorization": "Bearer test-token", "X-Node-ID": "user:web"}

    assert client.get("/api/notes/agent--coder--alice").status_code == 401
    assert client.get(
        "/api/notes/agent--coder--alice",
        headers={"Authorization": "Bearer wrong-token"},
    ).status_code == 401

    invalid = client.get("/api/notes/bad.id", headers=headers)
    assert invalid.status_code == 400

    missing = client.get("/api/notes/agent--coder--alice", headers=headers)
    assert missing.status_code == 200
    assert missing.get_json() == {"content": ""}

    put_resp = client.put(
        "/api/notes/agent--coder--alice",
        json={"content": "#+TITLE: alice\n\nnote"},
        headers=headers,
    )
    assert put_resp.status_code == 200
    assert (tmp_path / "agent--coder--alice.org").read_text() == "#+TITLE: alice\n\nnote"

    get_resp = client.get("/api/notes/agent--coder--alice", headers=headers)
    assert get_resp.status_code == 200
    assert get_resp.get_json() == {"content": "#+TITLE: alice\n\nnote"}

    delete_resp = client.delete("/api/notes/agent--coder--alice", headers=headers)
    assert delete_resp.status_code == 200
    assert not (tmp_path / "agent--coder--alice.org").exists()
