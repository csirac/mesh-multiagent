"""Focused subprocess tests for the opt-in PEV fold file profile."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def test_fold_profile_enforces_paths_handles_and_measurement(tmp_path: Path) -> None:
    digest = tmp_path / "digest.md"
    heuristics = tmp_path / "heuristics.md"
    state = tmp_path / "handle_state.json"
    source = tmp_path / "rounds" / "round_0000_pev" / "conversation_chunk.md"
    outside = tmp_path.parent / "outside-fold-profile.txt"
    escaped_link = tmp_path / "escaped-source.md"
    digest.write_text(
        "\n\n".join(
            [
                "## Timeline\nExisting [m_111111111111].",
                "## Narrative\nN",
                "## Projects\nP",
                "## People\nP",
                "## Standing decisions & conventions\nS",
                "## Open threads / where-we-are\nO",
                "## Agent narrative\nA",
            ]
        )
    )
    heuristics.write_text("- keep evidence\n")
    source.parent.mkdir(parents=True)
    source.write_text("read-only round evidence\n")
    state.write_text(
        json.dumps(
            {
                "m_handles": {"1": "222222222222"},
                "h_handles": {},
            }
        )
    )
    outside.write_text("do not change")
    escaped_link.symlink_to(outside)
    script = r"""
import json
import mesh.tool_implementations
import mesh.harness.tools
from mesh.tools import get_registry

registry = get_registry()
read = registry.get("file_read").handler
edit = registry.get("file_edit").handler
write = registry.get("file_write").handler
first = read("digest.md", 1, 100)
source = read(__import__("os").environ["FOLD_TEST_SOURCE"], 1, 10)
edited = edit(
    "digest.md",
    "Existing [[H1]].",
    "Existing [[H1]] plus new [[M1]].",
)
denied_source_edit = edit(
    __import__("os").environ["FOLD_TEST_SOURCE"],
    "read-only",
    "changed",
)
denied_outside = write(__import__("os").environ["FOLD_TEST_OUTSIDE"], "changed")
denied_symlink = read(__import__("os").environ["FOLD_TEST_SYMLINK"], 1, 10)
print(json.dumps({
    "read": first,
    "source": source,
    "edited": edited,
    "denied_source_edit": denied_source_edit,
    "denied_outside": denied_outside,
    "denied_symlink": denied_symlink,
}))
"""
    env = os.environ.copy()
    env.update(
        {
            "PEV_FOLD_TOOL_MODE": "1",
            "PEV_FOLD_EDITDIR": str(tmp_path),
            "PEV_FOLD_HANDLE_STATE": str(state),
            "PEV_FOLD_CEILING": "32000",
            "PEV_FOLD_READ_FILES": json.dumps([str(source)]),
            "FOLD_TEST_SOURCE": str(source),
            "FOLD_TEST_OUTSIDE": str(outside),
            "FOLD_TEST_SYMLINK": str(escaped_link),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert "[[H1]]" in result["read"]
    assert "read-only round evidence" in result["source"]
    assert "MEASURED digest size:" in result["edited"]
    assert "[m_111111111111] plus new [m_222222222222]" in digest.read_text()
    assert result["denied_source_edit"].startswith("Error: path not mutable")
    assert result["denied_outside"].startswith("Error: path not mutable")
    assert result["denied_symlink"].startswith("Error: path not readable")
    assert source.read_text() == "read-only round evidence\n"
    assert outside.read_text() == "do not change"
