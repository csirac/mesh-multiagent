"""
apply_patch tool — V4A envelope format parser and applier.

Ported from Codex's codex-rs/apply-patch Rust implementation.
Supports:
- *** Add File: <path>
- *** Delete File: <path>
- *** Update File: <path> with optional *** Move to: <new_path>
- @@ context hunks with fuzzy matching (4-pass seek_sequence)
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...tools import tool, ToolParameter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool description — embeds the V4A spec verbatim so the LLM knows the format
# ---------------------------------------------------------------------------

TOOL_DESCRIPTION = r"""Apply a patch to the filesystem using the V4A envelope format.

Your patch language is a stripped-down, file-oriented diff format designed to be easy to parse and safe to apply. You can think of it as a high-level envelope:

*** Begin Patch
[ one or more file sections ]
*** End Patch

Within that envelope, you get a sequence of file operations.
You MUST include a header to specify the action you are taking.
Each operation starts with one of three headers:

*** Add File: <path> - create a new file. Every following line is a + line (the initial contents).
*** Delete File: <path> - remove an existing file. Nothing follows.
*** Update File: <path> - patch an existing file in place (optionally with a rename).

May be immediately followed by *** Move to: <new path> if you want to rename the file.
Then one or more "hunks", each introduced by @@ (optionally followed by a hunk header).
Within a hunk each line starts with:

For instructions on [context_before] and [context_after]:
- By default, show 3 lines of code immediately above and 3 lines immediately below each change. If a change is within 3 lines of a previous change, do NOT duplicate the first change's [context_after] lines in the second change's [context_before] lines.
- If 3 lines of context is insufficient to uniquely identify the snippet of code within the file, use the @@ operator to indicate the class or function to which the snippet belongs. For instance, we might have:
@@ class BaseClass
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

- If a code block is repeated so many times in a class or function such that even a single @@ statement and 3 lines of context cannot uniquely identify the snippet of code, you can use multiple @@ statements to jump to the right context. For instance:

@@ class BaseClass
@@   def method():
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

The full grammar definition is below:
Patch := Begin { FileOp } End
Begin := "*** Begin Patch" NEWLINE
End := "*** End Patch" NEWLINE
FileOp := AddFile | DeleteFile | UpdateFile
AddFile := "*** Add File: " path NEWLINE { "+" line NEWLINE }
DeleteFile := "*** Delete File: " path NEWLINE
UpdateFile := "*** Update File: " path NEWLINE [ MoveTo ] { Hunk }
MoveTo := "*** Move to: " newPath NEWLINE
Hunk := "@@" [ header ] NEWLINE { HunkLine } [ "*** End of File" NEWLINE ]
HunkLine := (" " | "-" | "+") text NEWLINE

It is important to remember:
- You must include a header with your intended action (Add/Delete/Update)
- You must prefix new lines with + even when creating a new file
- File references can only be relative, NEVER ABSOLUTE."""

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class UpdateChunk:
    context_lines: list[str] = field(default_factory=list)
    old_lines: list[str] = field(default_factory=list)
    new_lines: list[str] = field(default_factory=list)
    post_context_lines: list[str] = field(default_factory=list)
    is_eof: bool = False


@dataclass
class PatchOp:
    pass


@dataclass
class AddFile(PatchOp):
    path: str = ""
    contents: list[str] = field(default_factory=list)


@dataclass
class DeleteFile(PatchOp):
    path: str = ""


@dataclass
class UpdateFile(PatchOp):
    path: str = ""
    move_path: str | None = None
    chunks: list[UpdateChunk] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_BEGIN = "*** Begin Patch"
_END = "*** End Patch"
_ADD = "*** Add File: "
_DELETE = "*** Delete File: "
_UPDATE = "*** Update File: "
_MOVE = "*** Move to: "
_EOF_MARKER = "*** End of File"
_HUNK = "@@"


def _strip_heredoc_wrapper(text: str) -> str:
    """Strip common heredoc / shell wrappers around the patch text."""
    text = text.strip()
    for prefix in ("apply_patch ", "apply_patch\n"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    # Handle <<'EOF' ... EOF wrapping
    if text.startswith("<<"):
        first_nl = text.index("\n") if "\n" in text else len(text)
        delimiter = text[2:first_nl].strip().strip("'\"")
        text = text[first_nl + 1:]
        if text.rstrip().endswith(delimiter):
            text = text.rstrip()
            text = text[: -len(delimiter)].rstrip("\n")
    return text.strip()


def _auto_wrap_envelope(text: str) -> str:
    """If text has file-op headers but no *** Begin Patch, auto-wrap it."""
    _FILE_OPS = (_ADD, _DELETE, _UPDATE)
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(_BEGIN):
            return text  # already wrapped
        if any(stripped.startswith(op) for op in _FILE_OPS):
            logger.warning("Auto-wrapping patch: missing '*** Begin Patch' envelope marker")
            wrapped = _BEGIN + "\n" + text.rstrip()
            if not wrapped.rstrip().endswith(_END):
                wrapped += "\n" + _END
            return wrapped
        break  # first non-blank line is neither Begin nor a file-op
    return text


def parse_patch(text: str) -> list[PatchOp]:
    """Parse a V4A patch envelope into a list of operations."""
    text = _strip_heredoc_wrapper(text)
    text = _auto_wrap_envelope(text)
    lines = text.split("\n")
    ops: list[PatchOp] = []
    i = 0

    # Find *** Begin Patch
    while i < len(lines) and not lines[i].strip().startswith(_BEGIN):
        i += 1
    if i >= len(lines):
        raise ValueError("Missing '*** Begin Patch' marker")
    i += 1  # skip begin line

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith(_END):
            break

        if stripped.startswith(_ADD):
            path = stripped[len(_ADD):]
            i += 1
            contents: list[str] = []
            while i < len(lines):
                cl = lines[i]
                if cl.startswith("+"):
                    contents.append(cl[1:])
                    i += 1
                elif cl.strip().startswith(("*** ", _END)):
                    break
                elif cl.strip() == "":
                    # Could be empty line in add block — check next
                    if i + 1 < len(lines) and (lines[i + 1].startswith("+") or lines[i + 1].strip().startswith("*** ")):
                        break
                    contents.append("")
                    i += 1
                else:
                    break
            ops.append(AddFile(path=path, contents=contents))
            continue

        if stripped.startswith(_DELETE):
            path = stripped[len(_DELETE):]
            ops.append(DeleteFile(path=path))
            i += 1
            continue

        if stripped.startswith(_UPDATE):
            path = stripped[len(_UPDATE):]
            i += 1
            move_path = None
            if i < len(lines) and lines[i].strip().startswith(_MOVE):
                move_path = lines[i].strip()[len(_MOVE):]
                i += 1

            chunks: list[UpdateChunk] = []
            while i < len(lines):
                cl = lines[i].strip()
                if cl.startswith(_END) or cl.startswith(("*** Add File:", "*** Delete File:", "*** Update File:")):
                    break
                if cl.startswith(_HUNK):
                    i += 1
                    chunk, i = _parse_chunk(lines, i)
                    chunks.append(chunk)
                else:
                    i += 1

            ops.append(UpdateFile(path=path, move_path=move_path, chunks=chunks))
            continue

        i += 1

    return ops


def _parse_chunk(lines: list[str], i: int) -> tuple[UpdateChunk, int]:
    """Parse a single hunk chunk starting after the @@ line."""
    chunk = UpdateChunk()
    in_changes = False

    while i < len(lines):
        if i >= len(lines):
            break
        line = lines[i]
        stripped = line.strip()

        # Check for markers that end this chunk
        if stripped.startswith(_HUNK):
            break
        if stripped.startswith(("*** Add File:", "*** Delete File:", "*** Update File:", _END)):
            break
        if stripped == _EOF_MARKER:
            chunk.is_eof = True
            i += 1
            break

        if line.startswith("-"):
            in_changes = True
            chunk.old_lines.append(line[1:])
            i += 1
        elif line.startswith("+"):
            in_changes = True
            chunk.new_lines.append(line[1:])
            i += 1
        elif line.startswith(" "):
            if in_changes:
                chunk.post_context_lines.append(line[1:])
                i += 1
            else:
                chunk.context_lines.append(line[1:])
                i += 1
        else:
            # Unrecognized line — skip
            i += 1

    return chunk, i


# ---------------------------------------------------------------------------
# Fuzzy line matching — seek_sequence (ported from Codex Rust)
# ---------------------------------------------------------------------------

def _normalise(s: str) -> str:
    """Normalize unicode punctuation to ASCII equivalents, trim whitespace."""
    result = []
    for c in s.strip():
        if c in "‐‑‒–—―−":
            result.append("-")
        elif c in "‘’‚‛":
            result.append("'")
        elif c in "“”„‟":
            result.append('"')
        elif c in "            　":
            result.append(" ")
        elif c in "→⟶➔":
            result.append("->")
        elif c in "←⟵":
            result.append("<-")
        elif c == "⇒":
            result.append("=>")
        elif c == "⇐":
            result.append("<=")
        elif c == "⇔":
            result.append("<=>")
        elif c == "…":
            result.append("...")
        elif c in "·•‣◦":
            result.append("*")
        else:
            result.append(c)
    return "".join(result)


def _nfc(s: str) -> str:
    """NFC-normalize and strip a string."""
    return unicodedata.normalize("NFC", s).strip()


def _ws_collapse(s: str) -> str:
    """Collapse all whitespace runs to single space and strip."""
    return re.sub(r"\s+", " ", s).strip()


def _find_unique_match(
    lines: list[str],
    pattern: list[str],
    start: int,
    end: int,
    match_fn,
    pass_name: str,
) -> int | None:
    """Scan [start, end] for ALL positions matching under match_fn.

    Returns the index only if exactly one candidate is found.
    If multiple candidates match, logs a warning and returns None —
    this is the safety property that prevents ambiguous edits.
    """
    candidates = []
    for i in range(start, end + 1):
        if match_fn(i):
            candidates.append(i)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        logger.warning(
            "Fuzzy pass '%s' found %d candidate matches — rejecting ambiguous "
            "match. The model should add more context lines to disambiguate.",
            pass_name, len(candidates),
        )
    return None


def seek_sequence(
    lines: list[str],
    pattern: list[str],
    start: int = 0,
    eof: bool = False,
) -> int | None:
    """Find the starting index where pattern matches within lines.

    6-pass fuzzy matching (most specific first):
    1. Exact match
    2. Rstrip match (trailing whitespace)
    3. Trim match (leading + trailing whitespace)
    4. Unicode-normalized match (punctuation → ASCII)
    5. NFC normalization (composed/decomposed Unicode) — uniqueness-checked
    6. Whitespace-collapse (runs of whitespace → single space) — uniqueness-checked

    Passes 1–4 return the first match (backward-compatible).
    Passes 5–6 scan the full range and accept only if exactly one
    candidate matches — ambiguous matches are rejected to prevent
    silent wrong edits.
    """
    if not pattern:
        return start
    if len(pattern) > len(lines):
        return None

    search_start = (len(lines) - len(pattern)) if eof and len(lines) >= len(pattern) else start
    end = len(lines) - len(pattern)

    # Pass 1: exact match
    for i in range(search_start, end + 1):
        if lines[i:i + len(pattern)] == pattern:
            return i
    if eof and search_start > start:
        for i in range(start, search_start):
            if lines[i:i + len(pattern)] == pattern:
                return i

    # Pass 2: rstrip match
    def _rstrip_match(idx: int) -> bool:
        return all(
            lines[idx + j].rstrip() == pattern[j].rstrip()
            for j in range(len(pattern))
        )

    for i in range(search_start, end + 1):
        if _rstrip_match(i):
            return i
    if eof and search_start > start:
        for i in range(start, search_start):
            if _rstrip_match(i):
                return i

    # Pass 3: trim match
    def _trim_match(idx: int) -> bool:
        return all(
            lines[idx + j].strip() == pattern[j].strip()
            for j in range(len(pattern))
        )

    for i in range(search_start, end + 1):
        if _trim_match(i):
            return i
    if eof and search_start > start:
        for i in range(start, search_start):
            if _trim_match(i):
                return i

    # Pass 4: unicode-normalized match (punctuation → ASCII)
    def _norm_match(idx: int) -> bool:
        return all(
            _normalise(lines[idx + j]) == _normalise(pattern[j])
            for j in range(len(pattern))
        )

    for i in range(search_start, end + 1):
        if _norm_match(i):
            return i
    if eof and search_start > start:
        for i in range(start, search_start):
            if _norm_match(i):
                return i

    # --- New lenient passes with uniqueness guarantee ---

    # Pass 5: NFC normalization — catches composed vs. decomposed Unicode forms
    def _nfc_match(idx: int) -> bool:
        return all(
            _nfc(lines[idx + j]) == _nfc(pattern[j])
            for j in range(len(pattern))
        )

    result = _find_unique_match(lines, pattern, start, end, _nfc_match, "NFC")
    if result is not None:
        return result

    # Pass 6: whitespace collapse — collapses runs of any whitespace to single space
    def _ws_match(idx: int) -> bool:
        return all(
            _ws_collapse(lines[idx + j]) == _ws_collapse(pattern[j])
            for j in range(len(pattern))
        )

    result = _find_unique_match(lines, pattern, start, end, _ws_match, "whitespace-collapse")
    if result is not None:
        return result

    return None


# ---------------------------------------------------------------------------
# Applier
# ---------------------------------------------------------------------------

def _resolve_path(path: str) -> str:
    """Resolve a relative patch path to an absolute path, rejecting traversal."""
    if os.path.isabs(path):
        raise ValueError(f"Absolute paths not allowed in patches: {path}")
    resolved = os.path.abspath(path)
    cwd = os.getcwd()
    if not (resolved == cwd or resolved.startswith(cwd + os.sep)):
        raise ValueError(f"Path traversal not allowed: {path} resolves outside working directory")
    return resolved


def _apply_add(op: AddFile) -> str:
    fpath = _resolve_path(op.path)
    parent = os.path.dirname(fpath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as f:
        for line in op.contents:
            f.write(line + "\n")
    return f"Created {op.path} ({len(op.contents)} lines)"


def _apply_delete(op: DeleteFile) -> str:
    fpath = _resolve_path(op.path)
    if os.path.isdir(fpath):
        shutil.rmtree(fpath)
        return f"Deleted directory {op.path}"
    elif os.path.exists(fpath):
        os.remove(fpath)
        return f"Deleted {op.path}"
    else:
        return f"Warning: {op.path} not found (already deleted?)"


def _apply_update(op: UpdateFile) -> str:
    fpath = _resolve_path(op.path)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Cannot update {op.path}: file not found")

    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    file_lines = content.split("\n")
    # Remove trailing empty string from split if file ends with newline
    if file_lines and file_lines[-1] == "":
        file_lines.pop()

    cursor = 0
    for chunk_idx, chunk in enumerate(op.chunks):
        is_last = chunk_idx == len(op.chunks) - 1
        is_eof = chunk.is_eof or (is_last and not chunk.context_lines and not chunk.old_lines)

        # Build the full pattern: pre-context + old_lines + post-context
        pattern = chunk.context_lines + chunk.old_lines + chunk.post_context_lines

        if not pattern:
            # No context and no old lines — append new lines at cursor
            for nl in reversed(chunk.new_lines):
                file_lines.insert(cursor, nl)
            continue

        match_idx = seek_sequence(file_lines, pattern, cursor, eof=is_eof)
        if match_idx is None:
            # Try pre-context + old_lines (without post-context)
            if chunk.post_context_lines:
                shorter = chunk.context_lines + chunk.old_lines
                match_idx = seek_sequence(file_lines, shorter, cursor, eof=is_eof)
            # Try just old_lines (no context at all)
            if match_idx is None and chunk.old_lines:
                match_idx = seek_sequence(file_lines, chunk.old_lines, cursor, eof=is_eof)
                if match_idx is not None:
                    del file_lines[match_idx:match_idx + len(chunk.old_lines)]
                    for j, nl in enumerate(chunk.new_lines):
                        file_lines.insert(match_idx + j, nl)
                    cursor = match_idx + len(chunk.new_lines)
                    continue
            # Try from start (out-of-order chunks)
            if match_idx is None and cursor > 0:
                match_idx = seek_sequence(file_lines, pattern, 0, eof=is_eof)
                if match_idx is None and chunk.post_context_lines:
                    match_idx = seek_sequence(file_lines, chunk.context_lines + chunk.old_lines, 0, eof=is_eof)
                if match_idx is None and chunk.old_lines:
                    match_idx = seek_sequence(file_lines, chunk.old_lines, 0, eof=is_eof)
                    if match_idx is not None:
                        del file_lines[match_idx:match_idx + len(chunk.old_lines)]
                        for j, nl in enumerate(chunk.new_lines):
                            file_lines.insert(match_idx + j, nl)
                        cursor = match_idx + len(chunk.new_lines)
                        continue
            if match_idx is None:
                raise ValueError(
                    f"Could not find matching context in {op.path} for chunk {chunk_idx + 1}. "
                    f"Pattern ({len(pattern)} lines): {pattern[:3]}..."
                )

        # Found the pattern. Replace old_lines portion with new_lines.
        # The pattern is [context_lines..., old_lines..., post_context_lines...]
        # We keep context and post-context in place, replace only old_lines.
        replace_start = match_idx + len(chunk.context_lines)
        replace_end = replace_start + len(chunk.old_lines)
        file_lines[replace_start:replace_end] = chunk.new_lines
        cursor = replace_start + len(chunk.new_lines)

    # Write back
    target_path = _resolve_path(op.move_path) if op.move_path else fpath
    if op.move_path:
        parent = os.path.dirname(target_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write("\n".join(file_lines) + "\n")
    if op.move_path and fpath != target_path:
        os.remove(fpath)

    parts = [f"Updated {op.path}"]
    if op.move_path:
        parts.append(f" -> {op.move_path}")
    parts.append(f" ({len(op.chunks)} chunk(s))")
    return "".join(parts)


def apply_patch_text(patch_text: str) -> str:
    """Parse and apply a V4A patch. Returns a summary of operations performed."""
    ops = parse_patch(patch_text)
    if not ops:
        raise ValueError("No operations found in patch")

    results: list[str] = []
    for op in ops:
        if isinstance(op, AddFile):
            results.append(_apply_add(op))
        elif isinstance(op, DeleteFile):
            results.append(_apply_delete(op))
        elif isinstance(op, UpdateFile):
            results.append(_apply_update(op))
    return "\n".join(results)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

@tool(
    name="apply_patch",
    description=TOOL_DESCRIPTION,
    parameters=[
        ToolParameter(
            name="patch",
            type="string",
            description="The patch text in V4A envelope format",
            required=True,
        ),
    ],
)
def apply_patch(patch: str) -> str:
    """Apply a V4A format patch to the filesystem."""
    try:
        return apply_patch_text(patch)
    except Exception as e:
        return f"Error applying patch: {e}"
