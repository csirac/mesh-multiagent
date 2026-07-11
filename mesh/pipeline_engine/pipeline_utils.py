"""
Shared utilities for the pipeline compiler and handler modules.

Extracted to break circular imports between pipeline_compiler.py
and handler modules (e.g., literature_handlers.py).
"""

import functools
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any


MATH_INSTRUCTION = (
    "Use LaTeX math notation: `$...$` for inline math, `$$...$$` for display math. "
    "Never use Unicode math symbols."
)


@dataclass
class JsonParseResult:
    data: Any
    source: str       # "direct" | "escape_repair" | "fenced_block" | "bracket_scan" | "truncated_recovery" | "raw_text"
    is_json: bool
    warnings: list[str] = field(default_factory=list)


def _strip_json_control_chars(text: str) -> str:
    """Strip control characters that are invalid in JSON strings.

    PDF extractors (pdftotext) embed form-feed (\\x0c) page breaks and other
    control chars that make otherwise-valid JSON unparseable.
    """
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


# Matches either a valid JSON escape (kept as-is) or a lone backslash (doubled).
# The alternation consumes valid escapes whole, so "\\l" in "$\\lambda$" is seen
# as valid-escape + plain "l" rather than backslash + invalid "\l".
_JSON_ESCAPE_RE = re.compile(r'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})|\\')


def _repair_invalid_json_escapes(text: str) -> str:
    """Double backslashes that do not begin a valid JSON escape.

    LLMs writing LaTeX inside JSON strings emit single backslashes
    ("$\\lambda_n$"), producing invalid escapes like \\l that make
    json.loads fail. Valid escapes (\\n, \\", \\uXXXX, ...) are untouched.
    """
    return _JSON_ESCAPE_RE.sub(
        lambda m: m.group(0) if m.group(1) is not None else r'\\', text)


def extract_json_result(text: str) -> JsonParseResult:
    """Parse JSON from LLM output with explicit strategy tracking."""
    warnings: list[str] = []
    text = text.strip()
    text = _strip_json_control_chars(text)

    try:
        return JsonParseResult(json.loads(text), "direct", True)
    except json.JSONDecodeError as e:
        warnings.append(f"direct parse failed: {e.msg} at pos {e.pos}")

    repaired = _repair_invalid_json_escapes(text)
    if repaired != text:
        try:
            return JsonParseResult(json.loads(repaired), "escape_repair", True, warnings)
        except json.JSONDecodeError as e:
            warnings.append(f"escape repair failed: {e.msg} at pos {e.pos}")

    m = re.search(r'```(?:json)?\s*\n(.*?)```', text, re.DOTALL)
    if m:
        try:
            return JsonParseResult(json.loads(m.group(1).strip()), "fenced_block", True, warnings)
        except json.JSONDecodeError as e:
            warnings.append(f"fenced block parse failed: {e.msg}")

    for start, end in [('[', ']'), ('{', '}')]:
        idx_s = text.find(start)
        idx_e = text.rfind(end)
        if idx_s != -1 and idx_e > idx_s:
            candidate = text[idx_s:idx_e + 1]
            try:
                return JsonParseResult(json.loads(candidate), "bracket_scan", True, warnings)
            except json.JSONDecodeError as e:
                warnings.append(f"bracket scan ({start}...{end}) failed: {e.msg}")
            repaired = _repair_invalid_json_escapes(candidate)
            if repaired != candidate:
                try:
                    return JsonParseResult(json.loads(repaired), "escape_repair", True, warnings)
                except json.JSONDecodeError as e:
                    warnings.append(f"bracket scan escape repair ({start}...{end}) failed: {e.msg}")

    idx_s = text.find('[')
    if idx_s != -1:
        candidate = text[idx_s:]
        last_brace = candidate.rfind('}')
        if last_brace != -1:
            truncated = candidate[:last_brace + 1].rstrip().rstrip(',') + ']'
            try:
                result = json.loads(truncated)
                warnings.append(f"recovered {len(result)} items from truncated array")
                print(f"    [info] Recovered {len(result)} items from truncated JSON",
                      file=sys.stderr)
                return JsonParseResult(result, "truncated_recovery", True, warnings)
            except json.JSONDecodeError as e:
                warnings.append(f"truncated recovery failed: {e.msg}")

    warnings.append("all JSON strategies exhausted; returning raw text")
    return JsonParseResult(text, "raw_text", False, warnings)


def extract_json(text: str) -> Any:
    """Parse JSON from LLM output. Returns parsed data or raw text as fallback."""
    return extract_json_result(text).data


# ---------------------------------------------------------------------------
# Handler contract decorator
# ---------------------------------------------------------------------------

class ContractError(Exception):
    """Raised when a handler's input or output violates its declared contract."""


def contract(*, accepts=(dict,), returns=(dict, list, str), required_keys=None):
    """Decorator for deterministic handlers — validates input/output at the boundary.

    Args:
        accepts: Tuple of types the primary input must be (after JSON parsing).
        returns: Tuple of types the output must be.
        required_keys: If primary is a dict, these keys must be present.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(inputs: dict, compiler):
            primary = inputs.get("primary")

            # Parse string inputs to JSON before type-checking
            if isinstance(primary, str):
                parsed = extract_json_result(primary)
                if parsed.is_json:
                    primary = parsed.data
                    inputs = {**inputs, "primary": primary}

            if not isinstance(primary, accepts):
                raise ContractError(
                    f"{fn.__name__}: expected primary input type "
                    f"{' | '.join(t.__name__ for t in accepts)}, "
                    f"got {type(primary).__name__}"
                )

            if required_keys and isinstance(primary, dict):
                missing = [k for k in required_keys if k not in primary]
                if missing:
                    raise ContractError(
                        f"{fn.__name__}: missing required input keys {missing}, "
                        f"got keys {sorted(primary.keys())}"
                    )

            result = fn(inputs, compiler)

            if not isinstance(result, returns):
                raise ContractError(
                    f"{fn.__name__}: expected output type "
                    f"{' | '.join(t.__name__ for t in returns)}, "
                    f"got {type(result).__name__}"
                )
            return result
        return wrapper
    return decorator


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + f"\n\n[... truncated at {max_words} words ...]"
