"""Opt-in standing-digest file tools for PEV fold subprocesses.

The generic harness file tools intentionally serve normal coding tasks.  A
standing-digest fold has a narrower contract: only digest.md and heuristics.md
are mutable, explicitly listed round artifacts are read-only, citations are
represented to the model by short handles, and every digest mutation reports a
real post-expansion token measurement.

The wrapper enables this profile through ``PEV_FOLD_TOOL_MODE=1`` before
launching PEV.  With that variable absent this module registers nothing, so the
normal harness toolset is unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

from ...tools import ToolParameter, tool


_ENABLED = os.environ.get("PEV_FOLD_TOOL_MODE") == "1"
_HANDLE_RE = re.compile(r"\[\[M(\d{1,4})\]\]")
_H_HANDLE_RE = re.compile(r"\[\[H(\d{1,5})\]\]")
_SINGLE_TAG_RE = re.compile(r"\[m_([0-9a-f]{12})\]")
_CITATION_GROUP_RE = re.compile(
    r"\[\s*m_[0-9a-f]{12}(?:\s*,\s*m_[0-9a-f]{12})+\s*\]"
)
_CANONICAL_ID_RE = re.compile(
    r"(?<![0-9A-Za-z_])m_([0-9a-f]{12})(?![0-9a-f])"
)
_SECTIONS = (
    "## Timeline",
    "## Narrative",
    "## Projects",
    "## People",
    "## Standing decisions & conventions",
    "## Open threads / where-we-are",
    "## Agent narrative",
)


def _clean_memory_id(value: Any) -> str:
    value = str(value)
    return value[2:] if value.startswith("m_") else value


def _normalise_handles(value: Any) -> dict[int, str]:
    if not isinstance(value, dict):
        return {}
    return {int(key): _clean_memory_id(mid) for key, mid in value.items()}


def _configuration() -> tuple[Path, Path, int]:
    raw_editdir = os.environ.get("PEV_FOLD_EDITDIR", "")
    raw_state = os.environ.get("PEV_FOLD_HANDLE_STATE", "")
    if not raw_editdir or not raw_state:
        raise ValueError(
            "PEV fold tools require PEV_FOLD_EDITDIR and "
            "PEV_FOLD_HANDLE_STATE"
        )
    editdir = Path(raw_editdir).expanduser().resolve(strict=True)
    handle_state = Path(raw_state).expanduser().resolve()
    try:
        handle_state.relative_to(editdir)
    except ValueError as exc:
        raise ValueError("fold handle state must be inside the fold edit directory") from exc
    ceiling = int(os.environ.get("PEV_FOLD_CEILING", "32000"))
    if ceiling < 1:
        raise ValueError("PEV_FOLD_CEILING must be positive")
    return editdir, handle_state, ceiling


def _resolve_path(raw: str, editdir: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = editdir / path
    return path.resolve()


def _configured_read_sources(editdir: Path) -> set[Path]:
    raw_sources = os.environ.get("PEV_FOLD_READ_FILES", "").strip()
    if not raw_sources:
        return set()
    try:
        values = json.loads(raw_sources)
    except json.JSONDecodeError as exc:
        raise ValueError("PEV_FOLD_READ_FILES must be a JSON list") from exc
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value.strip() for value in values
    ):
        raise ValueError("PEV_FOLD_READ_FILES must be a JSON list of paths")
    sources: set[Path] = set()
    for value in values:
        source = _resolve_path(value, editdir)
        try:
            source.relative_to(editdir)
        except ValueError as exc:
            raise ValueError(
                f"fold read source must be inside the fold edit directory: {source}"
            ) from exc
        if not source.is_file():
            raise ValueError(f"fold read source does not exist: {source}")
        sources.add(source)
    return sources


def _resolve_mutable(raw: str) -> tuple[Path, bool]:
    editdir, _handle_state, _ceiling = _configuration()
    path = _resolve_path(raw, editdir)
    digest = (editdir / "digest.md").resolve()
    heuristics = (editdir / "heuristics.md").resolve()
    if path not in {digest, heuristics}:
        raise ValueError(
            "path not mutable; only digest.md and heuristics.md in the fold "
            f"edit directory may be changed (got {path})"
        )
    return path, path == digest


def _resolve_readable(raw: str) -> tuple[Path, bool]:
    editdir, _handle_state, _ceiling = _configuration()
    path = _resolve_path(raw, editdir)
    digest = (editdir / "digest.md").resolve()
    heuristics = (editdir / "heuristics.md").resolve()
    if path not in {digest, heuristics} | _configured_read_sources(editdir):
        raise ValueError(
            "path not readable; use digest.md, heuristics.md, or an explicitly "
            f"listed fold source artifact (got {path})"
        )
    return path, path == digest


def _load_handles() -> tuple[dict[int, str], dict[int, str]]:
    _editdir, state_path, _ceiling = _configuration()
    if not state_path.exists():
        return {}, {}
    data = json.loads(state_path.read_text(encoding="utf-8"))
    return (
        _normalise_handles(data.get("m_handles")),
        _normalise_handles(data.get("h_handles")),
    )


def _save_handles(
    m_handles: dict[int, str],
    h_handles: dict[int, str],
) -> None:
    _editdir, state_path, _ceiling = _configuration()
    temporary = state_path.with_name(f".{state_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            {"m_handles": m_handles, "h_handles": h_handles},
            indent=1,
        ),
        encoding="utf-8",
    )
    temporary.replace(state_path)


def _handle_for_id(mid: str, mapping: dict[int, str]) -> int:
    mid = _clean_memory_id(mid)
    for number, existing in mapping.items():
        if existing == mid:
            return number
    number = max(mapping, default=0) + 1
    mapping[number] = mid
    return number


def _encode_existing(text: str, mapping: dict[int, str]) -> str:
    def handle(mid: str) -> str:
        return f"[[H{_handle_for_id(mid, mapping)}]]"

    def replace_group(match: re.Match[str]) -> str:
        return ", ".join(
            handle(mid) for mid in _CANONICAL_ID_RE.findall(match.group(0))
        )

    text = _CITATION_GROUP_RE.sub(replace_group, text)
    text = _SINGLE_TAG_RE.sub(lambda match: handle(match.group(1)), text)
    return _CANONICAL_ID_RE.sub(lambda match: handle(match.group(1)), text)


def _decode_all(
    text: str,
    m_handles: dict[int, str],
    h_handles: dict[int, str],
) -> str:
    def decode_m(match: re.Match[str]) -> str:
        number = int(match.group(1))
        return (
            f"[m_{m_handles[number]}]"
            if number in m_handles
            else match.group(0)
        )

    def decode_h(match: re.Match[str]) -> str:
        number = int(match.group(1))
        return (
            f"[m_{h_handles[number]}]"
            if number in h_handles
            else match.group(0)
        )

    return _H_HANDLE_RE.sub(decode_h, _HANDLE_RE.sub(decode_m, text))


def _measurement() -> str:
    editdir, _state_path, ceiling = _configuration()
    digest = editdir / "digest.md"
    text = digest.read_text(encoding="utf-8") if digest.exists() else ""
    m_handles, h_handles = _load_handles()
    text = _decode_all(text, m_handles, h_handles)
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")

    def count(value: str) -> int:
        return len(encoding.encode(value, disallowed_special=()))

    total = count(text)
    available = (
        f"OVER by {total - ceiling}"
        if total > ceiling
        else f"{ceiling - total} available"
    )
    lines = [
        f"MEASURED digest size: {total} tokens "
        f"(ceiling {ceiling}, {available})"
    ]
    positions = sorted(
        (text.find(section), section)
        for section in _SECTIONS
        if text.find(section) >= 0
    )
    for index, (start, section) in enumerate(positions):
        end = positions[index + 1][0] if index + 1 < len(positions) else len(text)
        lines.append(f"  {section.lstrip('# ')}: {count(text[start:end])} tokens")
    return "\n".join(lines)


if _ENABLED:

    @tool(
        name="file_read",
        description=(
            "Read digest.md, heuristics.md, or an explicitly listed read-only "
            "round artifact with line numbers."
        ),
        parameters=[
            ToolParameter(
                "path",
                "string",
                "digest.md, heuristics.md, or a listed round artifact",
            ),
            ToolParameter("start_line", "integer", "1-indexed first line"),
            ToolParameter("num_lines", "integer", "number of lines to read"),
        ],
    )
    def file_read(path: str, start_line: int = 1, num_lines: int = 200) -> str:
        try:
            resolved, is_digest = _resolve_readable(path)
            start = int(start_line)
            count = int(num_lines)
            if start < 1 or count < 1:
                raise ValueError("start_line and num_lines must be positive")
            if not resolved.exists():
                return f"Error: File not found: {resolved}"
            lines = resolved.read_text(encoding="utf-8").splitlines()
            selected = lines[start - 1:start - 1 + count]
            result = "\n".join(
                f"{line_no:4d}|{line}"
                for line_no, line in enumerate(selected, start=start)
            )
            if is_digest:
                m_handles, h_handles = _load_handles()
                result = _encode_existing(result, h_handles)
                _save_handles(m_handles, h_handles)
            return result + f"\n\n({len(lines)} lines total)"
        except Exception as exc:
            return f"Error: {exc}"


    @tool(
        name="file_edit",
        description=(
            "Perform one exact-string replacement in digest.md or "
            "heuristics.md. Digest mutations return a real token measurement."
        ),
        parameters=[
            ToolParameter("path", "string", "digest.md or heuristics.md"),
            ToolParameter("old_string", "string", "exact text to replace"),
            ToolParameter("new_string", "string", "replacement text"),
            ToolParameter(
                "replace_all",
                "boolean",
                "replace all matches",
                required=False,
                default=False,
            ),
        ],
    )
    def file_edit(
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        try:
            resolved, is_digest = _resolve_mutable(path)
            old = str(old_string)
            new = str(new_string)
            if is_digest:
                m_handles, h_handles = _load_handles()
                old = _decode_all(old, m_handles, h_handles)
                new = _decode_all(new, m_handles, h_handles)
            text = resolved.read_text(encoding="utf-8")
            matches = text.count(old)
            if matches == 0:
                raise ValueError("old_string not found")
            if matches > 1 and not replace_all:
                raise ValueError(
                    f"old_string matched {matches} times; make it unique or "
                    "set replace_all"
                )
            replacements = matches if replace_all else 1
            resolved.write_text(
                text.replace(old, new, -1 if replace_all else 1),
                encoding="utf-8",
            )
            result = f"edited {resolved.name}: {replacements} replacement(s)"
            return result + (f"\n\n{_measurement()}" if is_digest else "")
        except Exception as exc:
            return f"Error: {exc}"


    @tool(
        name="file_write",
        description=(
            "Replace all content of digest.md or heuristics.md. Reserve this "
            "for an empty first-fold digest. Digest writes return a real token "
            "measurement."
        ),
        parameters=[
            ToolParameter("path", "string", "digest.md or heuristics.md"),
            ToolParameter("content", "string", "complete replacement content"),
        ],
    )
    def file_write(path: str, content: str) -> str:
        try:
            resolved, is_digest = _resolve_mutable(path)
            value = str(content)
            if is_digest:
                m_handles, h_handles = _load_handles()
                value = _decode_all(value, m_handles, h_handles)
            resolved.write_text(value, encoding="utf-8")
            result = f"wrote {resolved.name}"
            return result + (f"\n\n{_measurement()}" if is_digest else "")
        except Exception as exc:
            return f"Error: {exc}"
