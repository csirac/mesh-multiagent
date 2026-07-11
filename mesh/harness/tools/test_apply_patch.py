"""
Tests for the V4A apply_patch parser and applier.

~40 test cases covering: parsing, fuzzy matching, file operations, edge cases.
"""

import os
import tempfile
import textwrap

import pytest

from mesh.harness.tools.apply_patch import (
    parse_patch,
    apply_patch_text,
    seek_sequence,
    AddFile,
    DeleteFile,
    UpdateFile,
    UpdateChunk,
    _normalise,
    _nfc,
    _ws_collapse,
    _strip_heredoc_wrapper,
    _auto_wrap_envelope,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def tmp_dir(tmp_path, monkeypatch):
    """Change to a temp directory for file operations."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_file(tmp_dir, name: str, content: str) -> str:
    """Helper to create a file in tmp_dir."""
    p = tmp_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return str(p)


# ===========================================================================
# Parser tests
# ===========================================================================

class TestParser:
    def test_empty_patch_raises(self):
        with pytest.raises(ValueError, match="Missing"):
            parse_patch("no patch here")

    def test_add_file(self):
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Add File: hello.txt
            +Hello, world!
            +Line two
            *** End Patch
        """)
        ops = parse_patch(patch)
        assert len(ops) == 1
        assert isinstance(ops[0], AddFile)
        assert ops[0].path == "hello.txt"
        assert ops[0].contents == ["Hello, world!", "Line two"]

    def test_delete_file(self):
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Delete File: obsolete.txt
            *** End Patch
        """)
        ops = parse_patch(patch)
        assert len(ops) == 1
        assert isinstance(ops[0], DeleteFile)
        assert ops[0].path == "obsolete.txt"

    def test_update_file_simple(self):
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Update File: src/app.py
            @@ def greet():
             def greet():
            -    print("Hi")
            +    print("Hello, world!")
             return 0
            *** End Patch
        """)
        ops = parse_patch(patch)
        assert len(ops) == 1
        assert isinstance(ops[0], UpdateFile)
        assert ops[0].path == "src/app.py"
        assert len(ops[0].chunks) == 1
        chunk = ops[0].chunks[0]
        assert chunk.context_lines == ["def greet():"]
        assert chunk.old_lines == ['    print("Hi")']
        assert chunk.new_lines == ['    print("Hello, world!")']
        assert chunk.post_context_lines == ["return 0"]

    def test_update_file_with_move(self):
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Update File: old.py
            *** Move to: new.py
            @@
            -old
            +new
            *** End Patch
        """)
        ops = parse_patch(patch)
        assert len(ops) == 1
        op = ops[0]
        assert isinstance(op, UpdateFile)
        assert op.path == "old.py"
        assert op.move_path == "new.py"

    def test_multiple_ops(self):
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Add File: new.txt
            +content
            *** Delete File: old.txt
            *** Update File: keep.txt
            @@
            -bad
            +good
            *** End Patch
        """)
        ops = parse_patch(patch)
        assert len(ops) == 3
        assert isinstance(ops[0], AddFile)
        assert isinstance(ops[1], DeleteFile)
        assert isinstance(ops[2], UpdateFile)

    def test_heredoc_wrapper_stripped(self):
        patch = textwrap.dedent("""\
            apply_patch <<'EOF'
            *** Begin Patch
            *** Add File: test.txt
            +hello
            *** End Patch
            EOF
        """)
        ops = parse_patch(patch)
        assert len(ops) == 1
        assert isinstance(ops[0], AddFile)

    def test_multiple_chunks(self):
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Update File: multi.py
            @@ first
             line1
            -old1
            +new1
            @@ second
             line5
            -old2
            +new2
            *** End Patch
        """)
        ops = parse_patch(patch)
        assert len(ops) == 1
        op = ops[0]
        assert isinstance(op, UpdateFile)
        assert len(op.chunks) == 2

    def test_eof_marker(self):
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Update File: eof.py
            @@
            -last_line
            +new_last_line
            *** End of File
            *** End Patch
        """)
        ops = parse_patch(patch)
        assert len(ops) == 1
        assert ops[0].chunks[0].is_eof is True

    def test_auto_wrap_no_begin_patch(self):
        """Parser auto-wraps when *** Begin Patch is missing but file ops exist."""
        patch = textwrap.dedent("""\
            *** Update File: test.py
            @@
            -old
            +new
            *** End Patch
        """)
        ops = parse_patch(patch)
        assert len(ops) == 1
        assert isinstance(ops[0], UpdateFile)
        assert ops[0].path == "test.py"

    def test_auto_wrap_preserves_valid_patch(self):
        """Auto-wrap is a no-op when *** Begin Patch is present."""
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Add File: ok.txt
            +content
            *** End Patch
        """)
        assert _auto_wrap_envelope(patch.strip()) == patch.strip()


# ===========================================================================
# seek_sequence tests
# ===========================================================================

class TestSeekSequence:
    def test_exact_match(self):
        lines = ["foo", "bar", "baz"]
        pattern = ["bar", "baz"]
        assert seek_sequence(lines, pattern, 0) == 1

    def test_exact_match_at_start(self):
        lines = ["foo", "bar", "baz"]
        assert seek_sequence(lines, ["foo", "bar"], 0) == 0

    def test_no_match(self):
        lines = ["foo", "bar", "baz"]
        assert seek_sequence(lines, ["qux"], 0) is None

    def test_rstrip_match(self):
        lines = ["foo   ", "bar\t\t"]
        pattern = ["foo", "bar"]
        assert seek_sequence(lines, pattern, 0) == 0

    def test_trim_match(self):
        lines = ["    foo   ", "   bar\t"]
        pattern = ["foo", "bar"]
        assert seek_sequence(lines, pattern, 0) == 0

    def test_unicode_normalise(self):
        lines = ["hello — world"]  # em dash
        pattern = ["hello - world"]  # ASCII dash
        assert seek_sequence(lines, pattern, 0) == 0

    def test_empty_pattern(self):
        assert seek_sequence(["a", "b"], [], 0) == 0
        assert seek_sequence(["a", "b"], [], 5) == 5

    def test_pattern_longer_than_lines(self):
        assert seek_sequence(["one"], ["a", "b", "c"], 0) is None

    def test_eof_search(self):
        lines = ["a", "b", "c", "target"]
        pattern = ["target"]
        assert seek_sequence(lines, pattern, 0, eof=True) == 3

    def test_start_offset(self):
        lines = ["foo", "bar", "foo", "bar"]
        pattern = ["foo", "bar"]
        assert seek_sequence(lines, pattern, 2) == 2

    def test_fancy_quotes_normalised(self):
        lines = ["it’s a “test”"]  # curly quotes
        pattern = ["it's a \"test\""]
        assert seek_sequence(lines, pattern, 0) == 0

    def test_nbsp_normalised(self):
        lines = ["hello world"]  # nbsp
        pattern = ["hello world"]
        assert seek_sequence(lines, pattern, 0) == 0


# ===========================================================================
# Normalise tests
# ===========================================================================

class TestNormalise:
    def test_em_dash(self):
        assert _normalise("hello — world") == "hello - world"

    def test_curly_quotes(self):
        assert _normalise("“quoted”") == '"quoted"'

    def test_nbsp(self):
        assert _normalise("a b") == "a b"

    def test_plain_ascii_unchanged(self):
        assert _normalise("  plain text  ") == "plain text"


# ===========================================================================
# Heredoc wrapper tests
# ===========================================================================

class TestHeredocWrapper:
    def test_strip_apply_patch_prefix(self):
        text = "apply_patch *** Begin Patch\n*** End Patch"
        result = _strip_heredoc_wrapper(text)
        assert result.startswith("*** Begin Patch")

    def test_strip_eof_wrapper(self):
        text = "<<'EOF'\n*** Begin Patch\n*** End Patch\nEOF"
        result = _strip_heredoc_wrapper(text)
        assert "*** Begin Patch" in result
        assert "EOF" not in result.split("*** End Patch")[-1]


# ===========================================================================
# End-to-end apply tests
# ===========================================================================

class TestApplyPatch:
    def test_add_file(self, tmp_dir):
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Add File: newfile.txt
            +Hello, world!
            +Line 2
            *** End Patch
        """)
        result = apply_patch_text(patch)
        assert "Created" in result
        content = (tmp_dir / "newfile.txt").read_text()
        assert content == "Hello, world!\nLine 2\n"

    def test_add_file_nested(self, tmp_dir):
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Add File: deep/nested/file.py
            +print("hello")
            *** End Patch
        """)
        apply_patch_text(patch)
        assert (tmp_dir / "deep" / "nested" / "file.py").exists()

    def test_delete_file(self, tmp_dir):
        write_file(tmp_dir, "to_delete.txt", "content")
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Delete File: to_delete.txt
            *** End Patch
        """)
        result = apply_patch_text(patch)
        assert "Deleted" in result
        assert not (tmp_dir / "to_delete.txt").exists()

    def test_delete_missing_file_warns(self, tmp_dir):
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Delete File: nonexistent.txt
            *** End Patch
        """)
        result = apply_patch_text(patch)
        assert "Warning" in result or "not found" in result

    def test_update_simple(self, tmp_dir):
        write_file(tmp_dir, "app.py", "def greet():\n    print('Hi')\n    return 0\n")
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Update File: app.py
            @@
             def greet():
            -    print('Hi')
            +    print('Hello, world!')
             return 0
            *** End Patch
        """)
        apply_patch_text(patch)
        content = (tmp_dir / "app.py").read_text()
        assert "Hello, world!" in content
        assert "Hi" not in content

    def test_update_with_move(self, tmp_dir):
        write_file(tmp_dir, "old.py", "old_content\n")
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Update File: old.py
            *** Move to: new.py
            @@
            -old_content
            +new_content
            *** End Patch
        """)
        apply_patch_text(patch)
        assert not (tmp_dir / "old.py").exists()
        assert (tmp_dir / "new.py").read_text().strip() == "new_content"

    def test_update_fuzzy_whitespace(self, tmp_dir):
        write_file(tmp_dir, "ws.py", "  line1  \n  line2  \n  line3  \n")
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Update File: ws.py
            @@
             line1
            -line2
            +replaced
             line3
            *** End Patch
        """)
        apply_patch_text(patch)
        content = (tmp_dir / "ws.py").read_text()
        assert "replaced" in content

    def test_update_missing_file_raises(self, tmp_dir):
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Update File: missing.py
            @@
            -old
            +new
            *** End Patch
        """)
        with pytest.raises(FileNotFoundError):
            apply_patch_text(patch)

    def test_update_multiple_chunks(self, tmp_dir):
        write_file(tmp_dir, "multi.py", "a\nb\nc\nd\ne\nf\n")
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Update File: multi.py
            @@
             a
            -b
            +B
             c
            @@
             d
            -e
            +E
             f
            *** End Patch
        """)
        apply_patch_text(patch)
        content = (tmp_dir / "multi.py").read_text()
        assert "B" in content and "E" in content
        assert "b\n" not in content and "e\n" not in content

    def test_combined_operations(self, tmp_dir):
        write_file(tmp_dir, "keep.txt", "old line\n")
        write_file(tmp_dir, "remove.txt", "bye\n")
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Add File: brand_new.txt
            +shiny
            *** Delete File: remove.txt
            *** Update File: keep.txt
            @@
            -old line
            +new line
            *** End Patch
        """)
        apply_patch_text(patch)
        assert (tmp_dir / "brand_new.txt").read_text().strip() == "shiny"
        assert not (tmp_dir / "remove.txt").exists()
        assert (tmp_dir / "keep.txt").read_text().strip() == "new line"

    def test_absolute_path_rejected(self, tmp_dir):
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Add File: /etc/evil.txt
            +bad
            *** End Patch
        """)
        with pytest.raises(ValueError, match="Absolute paths not allowed"):
            apply_patch_text(patch)

    def test_no_ops_raises(self, tmp_dir):
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** End Patch
        """)
        with pytest.raises(ValueError, match="No operations"):
            apply_patch_text(patch)

    def test_add_empty_file(self, tmp_dir):
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Add File: empty.txt
            *** End Patch
        """)
        apply_patch_text(patch)
        assert (tmp_dir / "empty.txt").exists()

    def test_update_unicode_fuzzy(self, tmp_dir):
        write_file(tmp_dir, "unicode.py", "# hello — world\nprint('hi')\n")
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Update File: unicode.py
            @@
             # hello - world
            -print('hi')
            +print('hello')
            *** End Patch
        """)
        apply_patch_text(patch)
        content = (tmp_dir / "unicode.py").read_text()
        assert "print('hello')" in content

    def test_post_context_disambiguates_duplicate_old_lines(self, tmp_dir):
        # old_lines "x = 1" appears twice; post-context anchors to the right one
        write_file(tmp_dir, "dup.py",
            "def first():\n    x = 1\n    return x\n\n"
            "def second():\n    x = 1\n    return x * 2\n"
        )
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Update File: dup.py
            @@
             def second():
            -    x = 1
            +    x = 99
                 return x * 2
            *** End Patch
        """)
        apply_patch_text(patch)
        content = (tmp_dir / "dup.py").read_text()
        lines = content.split("\n")
        # first() should still have x = 1
        first_idx = lines.index("def first():")
        assert lines[first_idx + 1].strip() == "x = 1"
        # second() should have x = 99
        second_idx = lines.index("def second():")
        assert lines[second_idx + 1].strip() == "x = 99"

    def test_post_context_parsed_correctly(self):
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Update File: test.py
            @@
             pre_context
            -old_line
            +new_line
             post_context
            *** End Patch
        """)
        ops = parse_patch(patch)
        chunk = ops[0].chunks[0]
        assert chunk.context_lines == ["pre_context"]
        assert chunk.old_lines == ["old_line"]
        assert chunk.new_lines == ["new_line"]
        assert chunk.post_context_lines == ["post_context"]

    def test_path_traversal_rejected(self, tmp_dir):
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Add File: ../../../etc/passwd
            +evil
            *** End Patch
        """)
        with pytest.raises(ValueError, match="traversal"):
            apply_patch_text(patch)

    def test_auto_wrap_missing_begin_patch(self, tmp_dir):
        """Patch without *** Begin Patch is auto-wrapped and applied."""
        write_file(tmp_dir, "target.py", "old_line\n")
        patch = textwrap.dedent("""\
            *** Update File: target.py
            @@
            -old_line
            +new_line
            *** End Patch
        """)
        result = apply_patch_text(patch)
        assert "Updated" in result
        content = (tmp_dir / "target.py").read_text()
        assert "new_line" in content
        assert "old_line" not in content

    def test_auto_wrap_missing_both_markers(self, tmp_dir):
        """Patch missing both Begin and End markers is auto-wrapped."""
        write_file(tmp_dir, "bare.py", "x = 1\n")
        patch = textwrap.dedent("""\
            *** Update File: bare.py
            @@
            -x = 1
            +x = 2
        """)
        result = apply_patch_text(patch)
        assert "Updated" in result
        content = (tmp_dir / "bare.py").read_text()
        assert "x = 2" in content

    def test_auto_wrap_add_file(self, tmp_dir):
        """Add File without Begin Patch is auto-wrapped."""
        patch = textwrap.dedent("""\
            *** Add File: wrapped.txt
            +content here
        """)
        result = apply_patch_text(patch)
        assert "Created" in result
        assert (tmp_dir / "wrapped.txt").exists()


# ===========================================================================
# Arrow / extended normalise tests
# ===========================================================================

class TestNormaliseExtended:
    def test_right_arrow(self):
        assert _normalise("state → idle") == "state -> idle"

    def test_left_arrow(self):
        assert _normalise("← back") == "<- back"

    def test_double_arrow(self):
        assert _normalise("a ⇒ b") == "a => b"

    def test_ellipsis(self):
        assert _normalise("wait…") == "wait..."

    def test_bullet(self):
        assert _normalise("• item") == "* item"

    def test_long_right_arrow(self):
        assert _normalise("a ⟶ b") == "a -> b"


# ===========================================================================
# NFC normalization tests
# ===========================================================================

class TestNFC:
    def test_nfc_strips(self):
        assert _nfc("  hello  ") == "hello"

    def test_nfc_composed_vs_decomposed(self):
        import unicodedata
        composed = "é"  # é as single codepoint
        decomposed = "é"  # e + combining accent
        assert composed != decomposed
        assert _nfc(composed) == _nfc(decomposed)


# ===========================================================================
# Whitespace collapse tests
# ===========================================================================

class TestWSCollapse:
    def test_collapse_spaces(self):
        assert _ws_collapse("hello    world") == "hello world"

    def test_collapse_mixed(self):
        assert _ws_collapse("a\t\t  b   c") == "a b c"

    def test_collapse_tabs_to_space(self):
        assert _ws_collapse("\thello\tworld\t") == "hello world"


# ===========================================================================
# seek_sequence new fuzzy pass tests
# ===========================================================================

class TestSeekSequenceFuzzy:
    def test_nfc_normalization_match(self):
        """NFC pass matches composed vs decomposed Unicode."""
        import unicodedata
        composed = "équation"    # é as one codepoint
        decomposed = "équation"  # e + combining accent
        lines = ["before", composed, "after"]
        pattern = [decomposed]
        assert seek_sequence(lines, pattern, 0) is not None

    def test_nfc_multiple_candidates_rejects(self):
        """NFC pass rejects when normalization makes multiple locations match."""
        import unicodedata
        composed = "équation"
        decomposed = "équation"
        # Both lines NFC-normalize to the same thing
        lines = ["header", composed, "middle", composed, "footer"]
        pattern = [decomposed]
        # Exact, rstrip, trim, and _normalise passes won't match (different bytes).
        # NFC pass finds 2 candidates — must reject.
        result = seek_sequence(lines, pattern, 0)
        assert result is None

    def test_ws_collapse_match(self):
        """Whitespace-collapse pass matches different whitespace patterns."""
        lines = ["def    greet(  self  ):"]
        pattern = ["def greet( self ):"]
        # Exact/rstrip/trim/norm won't match (different internal spacing)
        # ws_collapse reduces both to "def greet( self ):" and matches
        assert seek_sequence(lines, pattern, 0) == 0

    def test_ws_collapse_multiple_candidates_rejects(self):
        """Whitespace-collapse pass rejects ambiguous matches."""
        lines = [
            "x  =  1",
            "something else",
            "x    =    1",
        ]
        pattern = ["x = 1"]
        # Both line 0 and line 2 collapse to "x = 1" — must reject
        result = seek_sequence(lines, pattern, 0)
        assert result is None

    def test_ws_collapse_unique_match_succeeds(self):
        """Whitespace-collapse pass succeeds with a unique match."""
        lines = [
            "x  =  1",
            "y = 2",
            "z = 3",
        ]
        pattern = ["x = 1"]
        result = seek_sequence(lines, pattern, 0)
        assert result == 0

    def test_arrow_match_via_normalise(self):
        """Arrow characters match via the existing unicode-normalise pass (pass 4)."""
        lines = ["    state → idle", "    next_state = None"]
        pattern = ["    state -> idle", "    next_state = None"]
        assert seek_sequence(lines, pattern, 0) == 0


# ===========================================================================
# End-to-end: fuzzy matching in apply_patch
# ===========================================================================

class TestApplyPatchFuzzy:
    def test_arrow_context_match(self, tmp_dir):
        """apply_patch succeeds when context lines contain → but patch uses ->."""
        write_file(tmp_dir, "fsm.py",
            "class FSM:\n"
            "    def transition(self):\n"
            "        # state → idle\n"
            "        self.state = 'idle'\n"
            "        return True\n"
        )
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Update File: fsm.py
            @@
             # state -> idle
            -        self.state = 'idle'
            +        self.state = 'active'
                     return True
            *** End Patch
        """)
        result = apply_patch_text(patch)
        assert "Updated" in result
        content = (tmp_dir / "fsm.py").read_text()
        assert "'active'" in content

    def test_ws_collapse_context_match(self, tmp_dir):
        """apply_patch succeeds when file has extra whitespace in context lines."""
        write_file(tmp_dir, "spaced.py",
            "def  greet(  name  ):\n"
            "    print(name)\n"
            "    return True\n"
        )
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Update File: spaced.py
            @@
             def greet( name ):
            -    print(name)
            +    print(f"Hello {name}")
                 return True
            *** End Patch
        """)
        result = apply_patch_text(patch)
        assert "Updated" in result
        content = (tmp_dir / "spaced.py").read_text()
        assert "Hello {name}" in content

    def test_ws_collapse_ambiguous_fails(self, tmp_dir):
        """apply_patch fails when whitespace-collapse produces ambiguous context."""
        # Both context and old lines have extra whitespace — no exact match possible,
        # and ws-collapse finds 2 candidates for every fallback attempt.
        write_file(tmp_dir, "ambig.py",
            "x  =  1\n"
            "y  =  2\n"
            "a = 10\n"
            "x    =    1\n"
            "y    =    2\n"
            "b = 20\n"
        )
        patch = textwrap.dedent("""\
            *** Begin Patch
            *** Update File: ambig.py
            @@
             x = 1
            -y = 2
            +y = 99
            *** End Patch
        """)
        with pytest.raises(ValueError, match="Could not find matching context"):
            apply_patch_text(patch)
