"""Tests for the bounded get_context reader."""

from __future__ import annotations

from pathlib import Path

from mesh.harness.tools.get_context import MAX_CONTEXT_RADIUS, get_context


def _write_lines(path: Path, count: int = 8) -> None:
    path.write_text(
        "".join(f"line {number}\n" for number in range(1, count + 1)),
        encoding="utf-8",
    )


def test_get_context_returns_centered_numbered_window(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    _write_lines(target)

    result = get_context(str(target), line=4, radius=2)

    assert result.splitlines()[:5] == [
        "   2|line 2",
        "   3|line 3",
        "   4|line 4",
        "   5|line 5",
        "   6|line 6",
    ]
    assert result.endswith("(8 lines total)")


def test_get_context_clips_at_file_boundaries(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    _write_lines(target, count=4)

    assert get_context(str(target), line=1, radius=2).splitlines()[:3] == [
        "   1|line 1",
        "   2|line 2",
        "   3|line 3",
    ]
    assert get_context(str(target), line=4, radius=2).splitlines()[:3] == [
        "   2|line 2",
        "   3|line 3",
        "   4|line 4",
    ]


def test_get_context_rejects_invalid_line_and_radius(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    _write_lines(target, count=2)

    assert get_context(str(target), line=0) == "Error: line must be a positive integer"
    assert "out of range" in get_context(str(target), line=3)
    assert "non-negative integer" in get_context(str(target), line=1, radius=-1)
    assert "non-negative integer" in get_context(
        str(target),
        line=1,
        radius=MAX_CONTEXT_RADIUS + 1,
    )


def test_get_context_resolves_relative_paths_from_harness_cwd(
    monkeypatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "relative.txt"
    _write_lines(target, count=3)
    monkeypatch.chdir(tmp_path)

    result = get_context("relative.txt", line=2, radius=0)

    assert result.startswith("   2|line 2")


def test_get_context_rejects_directories_and_missing_files(tmp_path: Path) -> None:
    assert "is a directory" in get_context(str(tmp_path), line=1)
    assert "File not found" in get_context(str(tmp_path / "missing.txt"), line=1)
