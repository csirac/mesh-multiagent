"""Fallback parsing for tool calls emitted as assistant text.

Some OpenAI-compatible backends occasionally return no native ``tool_calls``
while still writing an intended tool call in content.  This module converts the
observed fallback formats into normal ``ToolCall`` objects and strips the raw
markup from assistant text before it is stored.
"""

from __future__ import annotations

import html
import json
import logging
import re
import uuid
from typing import Any

from .tools import ToolCall, ToolRegistry, parse_tool_calls

logger = logging.getLogger(__name__)


_DSML_PIPE = "｜"
_DSML_DELIM = _DSML_PIPE * 2
_DSML_NS = r"(?:(?:\w+)" + re.escape(_DSML_DELIM) + r")?"
_DSML_BLOCK_RE = re.compile(
    r"<" + re.escape(_DSML_DELIM) + _DSML_NS
    + r"tool_calls>\s*(.*?)\s*</" + re.escape(_DSML_DELIM)
    + _DSML_NS + r"tool_calls>",
    re.DOTALL,
)
_DSML_INVOKE_RE = re.compile(
    r"<" + re.escape(_DSML_DELIM) + _DSML_NS
    + r'invoke\s+name="([^"]+)"[^>]*>(.*?)</'
    + re.escape(_DSML_DELIM) + _DSML_NS + r"invoke>",
    re.DOTALL,
)
_DSML_PARAM_RE = re.compile(
    r"<" + re.escape(_DSML_DELIM) + _DSML_NS
    + r'parameter\s+name="([^"]+)"[^>]*>(.*?)</'
    + re.escape(_DSML_DELIM) + _DSML_NS + r"parameter>",
    re.DOTALL,
)
_DSML_PIPE_BLOCK_RE = re.compile(
    re.escape(_DSML_DELIM) + r"tool_calls" + re.escape(_DSML_DELIM)
    + r"\s*(.*?)\s*"
    + re.escape(_DSML_DELIM) + r"/tool_calls" + re.escape(_DSML_DELIM),
    re.DOTALL,
)
_DSML_PIPE_TOOL_RE = re.compile(
    re.escape(_DSML_DELIM)
    + r"tool[▁_]([A-Za-z_]\w*)[▁_](.*?)"
    + re.escape(_DSML_DELIM),
    re.DOTALL,
)

_XML_TOOL_CALL_RE = re.compile(
    r'<tool_call\s+name="([^"]+)">(.*?)</tool_call>',
    re.DOTALL,
)
_XML_TOOL_CALLS_BLOCK_RE = re.compile(
    r"<tool_calls>\s*(.*?)\s*</tool_calls>",
    re.DOTALL,
)
_XML_PARAM_RE = re.compile(
    r"<(\w+)>(.*?)</\1>",
    re.DOTALL,
)
_XML_INVOKE_RE = re.compile(
    r'<(?:antml:)?invoke\s+name="([^"]+)"[^>]*>(.*?)</(?:antml:)?invoke>',
    re.DOTALL,
)
_XML_INVOKE_PARAM_RE = re.compile(
    r'<(?:antml:)?parameter\s+name="([^"]+)"[^>]*>(.*?)</(?:antml:)?parameter>',
    re.DOTALL,
)
_XML_INVOKE_BLOCK_RE = re.compile(
    r"<(?:antml:)?function_calls>\s*(.*?)\s*</(?:antml:)?function_calls>",
    re.DOTALL,
)

_CALLING_FENCED_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:\*\*)?Calling:\s*(?:\*\*)?\s*`?"
    r"([A-Za-z_]\w*)`?[ \t]*\n+[ \t]*```(?:json)?[ \t]*\n"
    r"(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)
_CALLING_LINE_RE = re.compile(
    r"(?im)^[ \t]*(?:\*\*)?Calling:\s*(?:\*\*)?\s*`?([A-Za-z_]\w*)`?[ \t]*$"
)

_NARRATION_RE = re.compile(
    r"(?is)^\s*(?:got it[,\s:;.-]*)?"
    r"(?:(?:let me)\s+"
    r"(?:dispatch|kill|re-?read|check|look|send|restart|launch|write|read|fix|do|run|open|call|use)\b"
    r"|(?:i(?:'|’)ll|i will)\s+(?:dispatch|need to|check|look|send|restart|launch|write|read|fix|run|open|call|use)\b"
    r"|i should\s+(?:dispatch|check|look|send|restart|launch|write|read|fix|run|open|call|use)\b)"
)
_CALLING_ANY_RE = re.compile(r"(?i)(?:\*\*)?Calling:\s*(?:\*\*)?")


def _coerce_scalar(value: str) -> Any:
    value = html.unescape(value.strip())
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return value


def _parse_xml_params(body: str) -> dict[str, Any]:
    return {
        m.group(1): _coerce_scalar(m.group(2))
        for m in _XML_PARAM_RE.finditer(body)
    }


def _parse_invoke_params(body: str) -> dict[str, Any]:
    return {
        m.group(1): _coerce_scalar(m.group(2))
        for m in _XML_INVOKE_PARAM_RE.finditer(body)
    }


def _parse_dsml_params(body: str) -> dict[str, Any]:
    return {
        m.group(1): _coerce_scalar(m.group(2))
        for m in _DSML_PARAM_RE.finditer(body)
    }


def _json_object_at(text: str, start: int) -> tuple[dict[str, Any], int, int] | None:
    pos = text.find("{", start)
    if pos < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, rel_end = decoder.raw_decode(text[pos:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj, pos, pos + rel_end


def _arguments_match_schema(arguments: dict[str, Any], tool_def: Any) -> bool:
    params = getattr(tool_def, "parameters", []) or []
    if not params:
        return not arguments

    known = {p.name: p for p in params}
    unknown = set(arguments) - set(known)
    if unknown:
        logger.info(
            "salvage parser rejected %s: unknown argument(s) %s",
            getattr(tool_def, "name", "<tool>"), sorted(unknown),
        )
        return False

    for param in params:
        if param.required and param.name not in arguments:
            logger.info(
                "salvage parser rejected %s: missing required argument %s",
                getattr(tool_def, "name", "<tool>"), param.name,
            )
            return False
        if param.name not in arguments or arguments[param.name] is None:
            continue
        value = arguments[param.name]
        if param.type == "string" and not isinstance(value, str):
            return False
        if param.type == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            return False
        if param.type == "number" and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            return False
        if param.type == "boolean" and not isinstance(value, bool):
            return False
        if param.type == "object" and not isinstance(value, dict):
            return False
        if param.type == "array" and not isinstance(value, list):
            return False
    return True


def _coerce_arguments_for_schema(
    arguments: dict[str, Any],
    tool_def: Any,
) -> dict[str, Any] | None:
    params = getattr(tool_def, "parameters", []) or []
    if not params:
        return dict(arguments)
    known = {p.name: p for p in params}
    out = dict(arguments)
    for name, value in list(out.items()):
        param = known.get(name)
        if param is None or value is None:
            continue
        try:
            if param.type == "string" and not isinstance(value, str):
                out[name] = str(value)
            elif param.type == "integer" and isinstance(value, str):
                out[name] = int(value)
            elif param.type == "number" and isinstance(value, str):
                out[name] = float(value)
            elif param.type == "boolean" and isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes"}:
                    out[name] = True
                elif lowered in {"false", "0", "no"}:
                    out[name] = False
                else:
                    return None
        except (TypeError, ValueError):
            return None
    return out


def _validate_tool_call(
    name: str,
    arguments: dict[str, Any],
    *,
    offered_tool_names: set[str] | list[str] | tuple[str, ...] | None = None,
    tool_registry: ToolRegistry | None = None,
) -> bool:
    if offered_tool_names is not None and name not in set(offered_tool_names):
        logger.info("salvage parser rejected unknown/offered-out tool %s", name)
        return False
    if tool_registry is not None:
        tool_def = tool_registry.get(name)
        if tool_def is None:
            logger.info("salvage parser rejected unregistered tool %s", name)
            return False
        if not _arguments_match_schema(arguments, tool_def):
            logger.info("salvage parser rejected %s: schema mismatch", name)
            return False
    return True


def _make_tool_call(
    name: str,
    arguments: dict[str, Any],
    raw: str,
    *,
    offered_tool_names: set[str] | list[str] | tuple[str, ...] | None = None,
    tool_registry: ToolRegistry | None = None,
) -> ToolCall | None:
    if tool_registry is not None:
        tool_def = tool_registry.get(name)
        if tool_def is None:
            logger.info("salvage parser rejected unregistered tool %s", name)
            return None
        coerced = _coerce_arguments_for_schema(arguments, tool_def)
        if coerced is None:
            logger.info("salvage parser rejected %s: coercion failed", name)
            return None
        arguments = coerced
    if not _validate_tool_call(
        name,
        arguments,
        offered_tool_names=offered_tool_names,
        tool_registry=tool_registry,
    ):
        return None
    return ToolCall(
        name=name,
        arguments=arguments,
        raw_xml=raw,
        call_id=f"salvage-{uuid.uuid4().hex[:8]}",
    )


def extract_salvaged_tool_calls(
    text: str,
    *,
    offered_tool_names: set[str] | list[str] | tuple[str, ...] | None = None,
    tool_registry: ToolRegistry | None = None,
) -> list[ToolCall]:
    """Parse non-native tool calls embedded in assistant content."""
    if not text:
        return []

    calls: list[ToolCall] = []

    dsml_block = _DSML_BLOCK_RE.search(text)
    if dsml_block:
        for m in _DSML_INVOKE_RE.finditer(dsml_block.group(1)):
            call = _make_tool_call(
                m.group(1),
                _parse_dsml_params(m.group(2)),
                m.group(0),
                offered_tool_names=offered_tool_names,
                tool_registry=tool_registry,
            )
            if call:
                calls.append(call)
        if calls:
            return calls

    pipe_block = _DSML_PIPE_BLOCK_RE.search(text)
    if pipe_block:
        for m in _DSML_PIPE_TOOL_RE.finditer(pipe_block.group(1)):
            parsed = _json_object_at(m.group(2), 0)
            if not parsed:
                continue
            call = _make_tool_call(
                m.group(1),
                parsed[0],
                m.group(0),
                offered_tool_names=offered_tool_names,
                tool_registry=tool_registry,
            )
            if call:
                calls.append(call)
        if calls:
            return calls

    block_match = _XML_TOOL_CALLS_BLOCK_RE.search(text)
    search_text = block_match.group(1) if block_match else text
    for m in _XML_TOOL_CALL_RE.finditer(search_text):
        call = _make_tool_call(
            m.group(1),
            _parse_xml_params(m.group(2)),
            m.group(0),
            offered_tool_names=offered_tool_names,
            tool_registry=tool_registry,
        )
        if call:
            calls.append(call)
    if calls:
        return calls

    invoke_block = _XML_INVOKE_BLOCK_RE.search(text)
    invoke_search = invoke_block.group(1) if invoke_block else text
    for m in _XML_INVOKE_RE.finditer(invoke_search):
        call = _make_tool_call(
            m.group(1),
            _parse_invoke_params(m.group(2)),
            m.group(0),
            offered_tool_names=offered_tool_names,
            tool_registry=tool_registry,
        )
        if call:
            calls.append(call)
    if calls:
        return calls

    for call in parse_tool_calls(text):
        if not call.call_id:
            call.call_id = f"salvage-{uuid.uuid4().hex[:8]}"
        if _validate_tool_call(
            call.name,
            call.arguments,
            offered_tool_names=offered_tool_names,
            tool_registry=tool_registry,
        ):
            calls.append(call)
    if calls:
        return calls

    for m in _CALLING_FENCED_RE.finditer(text):
        try:
            arguments = json.loads(m.group(2))
        except json.JSONDecodeError:
            continue
        if not isinstance(arguments, dict):
            continue
        call = _make_tool_call(
            m.group(1),
            arguments,
            m.group(0),
            offered_tool_names=offered_tool_names,
            tool_registry=tool_registry,
        )
        if call:
            calls.append(call)
    if calls:
        return calls

    for m in _CALLING_LINE_RE.finditer(text):
        parsed = _json_object_at(text, m.end())
        if not parsed:
            continue
        arguments, _start, end = parsed
        call = _make_tool_call(
            m.group(1),
            arguments,
            text[m.start():end],
            offered_tool_names=offered_tool_names,
            tool_registry=tool_registry,
        )
        if call:
            calls.append(call)
    return calls


def _remove_raw_calling_json(text: str) -> str:
    spans: list[tuple[int, int]] = []
    for m in _CALLING_LINE_RE.finditer(text):
        parsed = _json_object_at(text, m.end())
        if parsed:
            spans.append((m.start(), parsed[2]))
    if not spans:
        return text
    out: list[str] = []
    pos = 0
    for start, end in spans:
        out.append(text[pos:start])
        pos = end
    out.append(text[pos:])
    return "".join(out)


def strip_salvaged_tool_calls(text: str) -> str:
    """Remove all supported fallback tool-call syntaxes from assistant text."""
    if not text:
        return text
    result = _DSML_BLOCK_RE.sub("", text)
    result = _DSML_PIPE_BLOCK_RE.sub("", result)
    result = _XML_TOOL_CALLS_BLOCK_RE.sub("", result)
    result = _XML_TOOL_CALL_RE.sub("", result)
    result = _XML_INVOKE_BLOCK_RE.sub("", result)
    result = _XML_INVOKE_RE.sub("", result)
    result = re.sub(r"<mesh_call\s+name=\"[^\"]+\">.*?</mesh_call>", "", result, flags=re.DOTALL)
    result = _CALLING_FENCED_RE.sub("", result)
    result = _remove_raw_calling_json(result)
    return result.strip()


def synthesize_send_message(text: str, to_node: str | None = None) -> ToolCall:
    """Convert fallback assistant text into a router ``send_message`` call."""
    content = strip_salvaged_tool_calls(text or "").strip()
    if not content:
        # Suppress generic fallback for channels — caller should handle
        if to_node and to_node.startswith("channel:"):
            content = ""
        else:
            content = "[router] No deliverable response content was produced."
    arguments: dict[str, Any] = {"content": content}
    if to_node:
        arguments["to"] = to_node
    return ToolCall(
        name="send_message",
        arguments=arguments,
        raw_xml=content,
        call_id=f"salvage-{uuid.uuid4().hex[:8]}",
    )


def tool_call_batch_terminates(tool_calls: list[ToolCall]) -> bool:
    """Router loop terminates after any batch that includes ``send_message``."""
    return any(call.name == "send_message" for call in tool_calls)


def looks_like_tool_narration(text: str) -> bool:
    """Return true when text describes a pending tool action instead of answering."""
    if not text:
        return False
    return bool(_CALLING_ANY_RE.search(text) or _NARRATION_RE.search(text))
