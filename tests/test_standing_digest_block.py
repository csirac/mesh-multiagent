"""Unit tests for the rev-10 standing-digest read pathway.

_standing_digest_block() is the flag-gated swap point: when
standing_digest_enabled with a readable digest file, the published
standing digest replaces the <memory_toc> block in prompt composition;
in every failure mode it returns "" so callers fall back to the TOC
branch (degrade-to-old-pathway, never memoryless).
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesh.router_v2 import RouterV2  # noqa: E402


def block(tmp_path=None, **cfg):
    self = types.SimpleNamespace(_config=types.SimpleNamespace(**cfg))
    return RouterV2._standing_digest_block(self)


def test_disabled_returns_empty(tmp_path):
    p = tmp_path / "d.md"
    p.write_text("## Timeline\ncontent")
    assert block(standing_digest_enabled=False,
                 standing_digest_path=str(p)) == ""


def test_enabled_wraps_digest(tmp_path):
    p = tmp_path / "d.md"
    p.write_text("## Timeline\n- 2026-07-08: fold landed [m_6c431f179510]\n")
    out = block(standing_digest_enabled=True, standing_digest_path=str(p))
    assert out.startswith("<standing_digest>\n")
    assert out.endswith("\n</standing_digest>")
    assert "[m_6c431f179510]" in out


def test_missing_file_falls_back_empty(tmp_path):
    out = block(standing_digest_enabled=True,
                standing_digest_path=str(tmp_path / "absent.md"))
    assert out == ""


def test_empty_path_falls_back_empty():
    assert block(standing_digest_enabled=True, standing_digest_path="") == ""


def test_empty_file_falls_back_empty(tmp_path):
    p = tmp_path / "d.md"
    p.write_text("   \n")
    assert block(standing_digest_enabled=True,
                 standing_digest_path=str(p)) == ""


def test_config_default_is_off():
    from mesh.config import NodeConfig
    import dataclasses
    fields = {f.name: f.default for f in dataclasses.fields(NodeConfig)}
    assert fields.get("standing_digest_enabled") is False
    assert fields.get("standing_digest_path") == ""
