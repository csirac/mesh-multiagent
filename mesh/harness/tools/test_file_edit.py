"""
Tests for the file_edit tool — CC-style exact string replacement.

Covers: basic replacement, uniqueness safety, replace_all, edge cases.
"""

import os
import pytest

from mesh.harness.tools.file_edit import file_edit


@pytest.fixture
def tmp_dir(tmp_path, monkeypatch):
    """Change to a temp directory for file operations."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_file(tmp_dir, name: str, content: str) -> str:
    p = tmp_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return str(p)


class TestFileEdit:
    def test_basic_replacement(self, tmp_dir):
        write_file(tmp_dir, "app.py", "def greet():\n    print('Hi')\n    return 0\n")
        result = file_edit(path="app.py", old_string="print('Hi')", new_string="print('Hello')")
        assert "Updated" in result
        content = (tmp_dir / "app.py").read_text()
        assert "print('Hello')" in content
        assert "print('Hi')" not in content

    def test_file_not_found(self, tmp_dir):
        result = file_edit(path="nonexistent.py", old_string="x", new_string="y")
        assert "Error" in result
        assert "not found" in result

    def test_directory_rejected(self, tmp_dir):
        os.makedirs(tmp_dir / "subdir", exist_ok=True)
        result = file_edit(path="subdir", old_string="x", new_string="y")
        assert "Error" in result
        assert "directory" in result

    def test_old_string_not_found(self, tmp_dir):
        write_file(tmp_dir, "app.py", "hello world\n")
        result = file_edit(path="app.py", old_string="foobar", new_string="baz")
        assert "Error" in result
        assert "not found" in result

    def test_multiple_matches_without_replace_all(self, tmp_dir):
        write_file(tmp_dir, "app.py", "x = 1\nx = 1\nx = 1\n")
        result = file_edit(path="app.py", old_string="x = 1", new_string="x = 2")
        assert "Error" in result
        assert "3 times" in result
        assert "replace_all" in result
        # File must not be modified
        content = (tmp_dir / "app.py").read_text()
        assert content.count("x = 1") == 3

    def test_multiple_matches_with_replace_all(self, tmp_dir):
        write_file(tmp_dir, "app.py", "x = 1\nx = 1\nx = 1\n")
        result = file_edit(
            path="app.py", old_string="x = 1", new_string="x = 2", replace_all=True
        )
        assert "Replaced 3 occurrences" in result
        content = (tmp_dir / "app.py").read_text()
        assert content.count("x = 2") == 3
        assert content.count("x = 1") == 0

    def test_unicode_content_preserved(self, tmp_dir):
        write_file(tmp_dir, "uni.py", "# 日本語コメント\nprint('→ output')\n")
        result = file_edit(
            path="uni.py",
            old_string="print('→ output')",
            new_string="print('← input')",
        )
        assert "Updated" in result
        content = (tmp_dir / "uni.py").read_text()
        assert "# 日本語コメント" in content
        assert "print('← input')" in content

    def test_multiline_old_string(self, tmp_dir):
        write_file(tmp_dir, "app.py", "def f():\n    x = 1\n    return x\n")
        result = file_edit(
            path="app.py",
            old_string="    x = 1\n    return x",
            new_string="    x = 42\n    return x * 2",
        )
        assert "Updated" in result
        content = (tmp_dir / "app.py").read_text()
        assert "x = 42" in content
        assert "return x * 2" in content

    def test_empty_file(self, tmp_dir):
        write_file(tmp_dir, "empty.py", "")
        result = file_edit(path="empty.py", old_string="anything", new_string="new")
        assert "Error" in result
        assert "not found" in result

    def test_replace_with_empty_string(self, tmp_dir):
        write_file(tmp_dir, "app.py", "line1\nline2\nline3\n")
        result = file_edit(path="app.py", old_string="line2\n", new_string="")
        assert "Updated" in result
        content = (tmp_dir / "app.py").read_text()
        assert "line2" not in content
        assert "line1\nline3\n" == content

    def test_exact_match_required_no_fuzzy(self, tmp_dir):
        """file_edit does NOT do fuzzy matching — exact bytes only."""
        write_file(tmp_dir, "app.py", "hello → world\n")
        result = file_edit(path="app.py", old_string="hello -> world", new_string="hi")
        assert "Error" in result
        assert "not found" in result
        # File must be unchanged
        content = (tmp_dir / "app.py").read_text()
        assert "hello → world" in content
