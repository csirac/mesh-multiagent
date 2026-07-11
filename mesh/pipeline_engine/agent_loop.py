"""
Standalone agentic tool loop using the OpenAI Chat Completions API.

Works with any OpenAI-compatible endpoint (OpenAI, DeepSeek, OpenRouter,
vLLM, etc.). Provides native research tools (arXiv, PubMed, Exa, etc.)
plus a shell fallback.

This replaces the mesh-harness dependency for the grant review pipeline.
"""

import json
import os
import re
import subprocess
import sys
import time
import asyncio
import inspect
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

from pipeline_tools import TOOL_DEFINITIONS, dispatch_tool

last_usage: dict = {"input_tokens": 0, "output_tokens": 0}


SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": (
            "Execute a shell command and return its output. "
            "Use this for general unix commands, running scripts, "
            "or anything not covered by the dedicated research tools."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 120).",
                },
            },
            "required": ["command"],
        },
    },
}

TOOLS = TOOL_DEFINITIONS + [SHELL_TOOL]

MAX_RESULT_CHARS = 200_000
SHELL_HEAD_LINES = 200
SHELL_TAIL_LINES = 50
VALID_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh"}
REASONING_THINKING_BUDGETS = {
    "none": 0,
    "low": 512,
    "medium": 2048,
    "high": 8192,
    "xhigh": 16384,
}


def _exec_shell(command: str, timeout: int = 120, cwd: str | None = None,
                env: dict | None = None) -> str:
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True, text=True,
            timeout=timeout, cwd=cwd, env=env,
        )
        out = result.stdout
        if result.stderr:
            out += "\n[stderr]\n" + result.stderr
        if result.returncode != 0:
            out += f"\n[exit code: {result.returncode}]"
    except subprocess.TimeoutExpired:
        out = f"[command timed out after {timeout}s]"
    except Exception as e:
        out = f"[error: {e}]"

    lines = out.split("\n")
    if len(lines) > SHELL_HEAD_LINES + SHELL_TAIL_LINES + 5:
        head = "\n".join(lines[:SHELL_HEAD_LINES])
        tail = "\n".join(lines[-SHELL_TAIL_LINES:])
        omitted = len(lines) - SHELL_HEAD_LINES - SHELL_TAIL_LINES
        out = f"{head}\n\n[... {omitted} lines omitted ...]\n\n{tail}"

    if len(out) > MAX_RESULT_CHARS:
        out = out[:MAX_RESULT_CHARS] + f"\n[truncated at {MAX_RESULT_CHARS} chars]"

    return out


def _execute_tool_call(
    name: str, arguments: dict, cwd: str | None = None, env: dict | None = None,
    tool_dispatcher: Callable[[str, dict], Any] | None = None,
) -> str:
    if tool_dispatcher is not None:
        result = tool_dispatcher(name, arguments)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        if result is not None:
            result_str = str(result)
            if len(result_str) > MAX_RESULT_CHARS:
                result_str = result_str[:MAX_RESULT_CHARS] + f"\n[truncated at {MAX_RESULT_CHARS} chars]"
            return result_str
        return f"[unknown tool: {name}]"

    if name == "shell":
        return _exec_shell(
            arguments["command"],
            timeout=arguments.get("timeout", 120),
            cwd=cwd, env=env,
        )
    result = dispatch_tool(name, arguments)
    if result is not None:
        if len(result) > MAX_RESULT_CHARS:
            result = result[:MAX_RESULT_CHARS] + f"\n[truncated at {MAX_RESULT_CHARS} chars]"
        return result
    return f"[unknown tool: {name}]"


def _emit(event_type: str, data: dict, stream: bool = True):
    event = {"type": event_type, "ts": time.time(), "data": data}
    if stream:
        print(json.dumps(event), flush=True)


def _estimate_tokens(messages: list) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content) // 4
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(part.get("text", "")) // 4
    return total


def _normalize_reasoning_effort(value: str | None) -> str | None:
    if value is None:
        return None
    effort = str(value).strip().lower()
    if not effort:
        return None
    if effort not in VALID_REASONING_EFFORTS:
        raise ValueError(
            f"Invalid reasoning_effort '{value}'. "
            f"Expected one of: {', '.join(sorted(VALID_REASONING_EFFORTS))}"
        )
    return effort


def _is_local_vllm_backend(base_url: str, model: str) -> bool:
    base_url_l = base_url.lower()
    model_l = model.lower()
    return (
        "localhost" in base_url_l
        or "127.0.0.1" in base_url_l
        or model_l.startswith("local-")
    )


def _apply_reasoning_controls(
    kwargs: Dict[str, Any],
    *,
    base_url: str,
    model: str,
    reasoning_effort: str | None,
    thinking_budget: int,
) -> None:
    reasoning_effort = _normalize_reasoning_effort(reasoning_effort)

    def set_chat_template_thinking(enabled: bool, budget: int = 0) -> None:
        extra_body = dict(kwargs.get("extra_body") or {})
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_template_kwargs["enable_thinking"] = enabled
        if enabled and budget > 0:
            chat_template_kwargs["thinking_budget"] = budget
        else:
            chat_template_kwargs.pop("thinking_budget", None)
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        kwargs["extra_body"] = extra_body

    if reasoning_effort:
        if _is_local_vllm_backend(base_url, model):
            if reasoning_effort in ("none", "low"):
                set_chat_template_thinking(False)
            else:
                budget = thinking_budget or REASONING_THINKING_BUDGETS[reasoning_effort]
                set_chat_template_thinking(True, budget)
        elif reasoning_effort != "none":
            kwargs["reasoning_effort"] = reasoning_effort
        return

    if thinking_budget > 0:
        if _is_local_vllm_backend(base_url, model):
            set_chat_template_thinking(True, thinking_budget)
        else:
            kwargs["extra_body"] = {
                "enable_thinking": True,
                "thinking_budget": thinking_budget,
            }


# =============================================================================
# XML Tool Call Salvage Path
# =============================================================================
# When the API returns message.tool_calls = [] but the model emitted tool calls
# inline as XML in message.content (a known DeepSeek behavior), parse the XML
# into synthetic tool_call dicts and strip it from the content. This keeps the
# agent loop running instead of silently dropping the calls.
#
# Three formats are supported:
#   1. DSML with fullwidth-pipe delimiters (U+FF5C, not ASCII '|'):
#        <｜｜DSML｜｜tool_calls>
#          <｜｜DSML｜｜invoke name="bash_exec">
#            <｜｜DSML｜｜parameter name="command">ls</｜｜DSML｜｜parameter>
#          </｜｜DSML｜｜invoke>
#        </｜｜DSML｜｜tool_calls>
#   2. DeepSeek <tool_call name="X">...</tool_call>
#   3. Anthropic <invoke name="X"><parameter name="Y">v</parameter></invoke>

_DSML_PIPE = "｜"  # fullwidth vertical bar
_DSML_DELIM = _DSML_PIPE * 2  # '｜｜' (two fullwidth pipes)

_DSML_BLOCK_RE = re.compile(
    r"<" + _DSML_DELIM + r"tool_calls>\s*(.*?)\s*</" + _DSML_DELIM + r"tool_calls>",
    re.DOTALL,
)
_DSML_INVOKE_RE = re.compile(
    r"<" + _DSML_DELIM + r'invoke\s+name="([^"]+)"[^>]*>(.*?)</' + _DSML_DELIM + r"invoke>",
    re.DOTALL,
)
_DSML_PARAM_RE = re.compile(
    r"<" + _DSML_DELIM + r'parameter\s+name="([^"]+)"[^>]*>(.*?)</' + _DSML_DELIM + r"parameter>",
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


def _make_synthetic_tool_call(name: str, arguments: dict) -> dict:
    return {
        "id": f"xml-salvage-{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def _salvage_xml_tool_calls(content: str) -> Tuple[List[dict], str]:
    """Parse tool calls embedded as XML in LLM response text.

    Used as a fallback when ``message.tool_calls`` is empty but the model
    emitted tool calls inline as XML (a known DeepSeek behavior).

    Returns ``(synthetic_tool_calls, cleaned_content)`` where:
      - ``synthetic_tool_calls`` is a list of dicts in OpenAI tool_call shape
        (``{"id", "type", "function": {"name", "arguments"}}``), empty if no
        XML tool calls were found.
      - ``cleaned_content`` is the original content with matched XML blocks
        stripped, so the leaked XML does not pollute conversation history.
    """
    if not content:
        return [], content

    calls: List[dict] = []
    cleaned = content

    # --- Format 1: DSML with fullwidth-pipe delimiters ---
    dsml_block = _DSML_BLOCK_RE.search(content)
    if dsml_block:
        for m in _DSML_INVOKE_RE.finditer(dsml_block.group(1)):
            name = m.group(1)
            body = m.group(2)
            arguments: dict[str, str] = {}
            for pm in _DSML_PARAM_RE.finditer(body):
                arguments[pm.group(1)] = pm.group(2).strip()
            calls.append(_make_synthetic_tool_call(name, arguments))
        if calls:
            cleaned = content.replace(dsml_block.group(0), "")

    # --- Format 2: DeepSeek <tool_call name="X">...</tool_call> ---
    if not calls:
        block_match = _XML_TOOL_CALLS_BLOCK_RE.search(content)
        search_text = block_match.group(1) if block_match else content
        for m in _XML_TOOL_CALL_RE.finditer(search_text):
            name = m.group(1)
            body = m.group(2).strip()
            arguments = {}
            for pm in _XML_PARAM_RE.finditer(body):
                arguments[pm.group(1)] = pm.group(2).strip()
            calls.append(_make_synthetic_tool_call(name, arguments))
        if calls:
            cleaned = content
            if block_match:
                cleaned = cleaned.replace(block_match.group(0), "")
            for m in _XML_TOOL_CALL_RE.finditer(content):
                cleaned = cleaned.replace(m.group(0), "")

    # --- Format 3: Anthropic <invoke>/<parameter> ---
    if not calls:
        invoke_block = _XML_INVOKE_BLOCK_RE.search(content)
        invoke_search = invoke_block.group(1) if invoke_block else content
        for m in _XML_INVOKE_RE.finditer(invoke_search):
            name = m.group(1)
            body = m.group(2).strip()
            arguments = {}
            for pm in _XML_INVOKE_PARAM_RE.finditer(body):
                arguments[pm.group(1)] = pm.group(2).strip()
            calls.append(_make_synthetic_tool_call(name, arguments))
        if calls:
            cleaned = content
            if invoke_block:
                cleaned = cleaned.replace(invoke_block.group(0), "")
            for m in _XML_INVOKE_RE.finditer(content):
                cleaned = cleaned.replace(m.group(0), "")

    return calls, cleaned.strip()


def run_agent(
    prompt: str,
    system_prompt: str,
    *,
    model: str = "deepseek-v4-pro",
    base_url: str = "https://api.deepseek.com/v1",
    api_key: str = "",
    max_output_tokens: int = 131072,
    max_iters: int = 30,
    soft_limit: int = 500_000,
    thinking_budget: int = 0,
    reasoning_effort: str | None = None,
    http_timeout: int = 600,
    cwd: str | None = None,
    env: dict | None = None,
    step_name: str = "step",
    stream_events: bool = True,
    output_dir: str | None = None,
    tool_definitions: list[dict] | None = None,
    tool_dispatcher: Callable[[str, dict], Any] | None = None,
) -> str:
    """Run an agentic tool loop and return the final assistant text.

    Args:
        prompt: The user task prompt.
        system_prompt: System instructions for the agent.
        model: Model name (e.g. deepseek-v4-pro, gpt-4o, gpt-5.1).
        base_url: OpenAI-compatible API base URL.
        api_key: API key for the endpoint.
        max_output_tokens: Max tokens per LLM response.
        max_iters: Safety cap on tool-loop iterations.
        soft_limit: Approximate token budget before forcing synthesis.
        thinking_budget: Reasoning token budget (0 = disable).
        reasoning_effort: Optional reasoning effort override
            ("none", "low", "medium", "high", "xhigh").
        cwd: Working directory for shell commands.
        env: Environment variables for shell commands.
        step_name: Label for logging.
        stream_events: Whether to emit JSONL events to stdout.
        output_dir: Directory for tool_usage.jsonl log file.
        tool_definitions: Optional OpenAI-format tool definitions. Defaults
            to the standalone research tools plus shell.
        tool_dispatcher: Optional dispatcher for tool calls. Defaults to the
            legacy in-process dispatcher.

    Returns:
        The final assistant message text.
    """
    client = OpenAI(
        base_url=base_url, api_key=api_key,
        timeout=httpx.Timeout(http_timeout, connect=30.0),
    )

    tool_log_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        tool_log_path = os.path.join(output_dir, "tool_usage.jsonl")

    tool_stats: Dict[str, Dict[str, Any]] = {}

    messages: list = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    _emit("thread.started", {
        "model": model, "step": step_name,
    }, stream=stream_events)

    total_input_tokens = 0
    total_output_tokens = 0
    final_text = ""
    active_tools = tool_definitions if tool_definitions is not None else TOOLS
    last_iteration_had_tool_calls = False

    for iteration in range(1, max_iters + 1):
        _emit("turn.started", {"iteration": iteration}, stream=stream_events)

        est_tokens = _estimate_tokens(messages)
        if est_tokens > soft_limit * 0.97:
            _emit("context.budget_exceeded", {
                "estimated_tokens": est_tokens,
                "soft_limit": soft_limit,
                "action": "forcing_synthesis",
            }, stream=stream_events)
            messages.append({
                "role": "user",
                "content": (
                    "You are approaching context limits. Produce your FINAL "
                    "output now with everything you have gathered so far. "
                    "Do not make any more tool calls."
                ),
            })
            tools_param: list | None = None
        else:
            tools_param = active_tools

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "temperature": 0.0,
        }
        if tools_param:
            kwargs["tools"] = tools_param
        _apply_reasoning_controls(
            kwargs,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            thinking_budget=thinking_budget,
        )

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            error_text = str(e)
            retried = False
            if "reasoning_effort" in kwargs and "reasoning_effort" in error_text:
                fallback_kwargs = dict(kwargs)
                effort = fallback_kwargs["reasoning_effort"]
                if effort == "xhigh":
                    fallback_kwargs["reasoning_effort"] = "high"
                    _emit("warning", {
                        "message": "reasoning_effort=xhigh rejected; retrying with high",
                        "iteration": iteration,
                    }, stream=stream_events)
                else:
                    fallback_kwargs.pop("reasoning_effort", None)
                    _emit("warning", {
                        "message": "reasoning_effort rejected; retrying without it",
                        "iteration": iteration,
                    }, stream=stream_events)
                try:
                    response = client.chat.completions.create(**fallback_kwargs)
                    retried = True
                except Exception:
                    retried = False
            elif "extra_body" in kwargs and (
                "enable_thinking" in error_text
                or "thinking_budget" in error_text
                or "extra_body" in error_text
            ):
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("extra_body", None)
                _emit("warning", {
                    "message": "thinking controls rejected; retrying without them",
                    "iteration": iteration,
                }, stream=stream_events)
                try:
                    response = client.chat.completions.create(**fallback_kwargs)
                    retried = True
                except Exception:
                    retried = False
            if retried:
                pass
            else:
                _emit("error", {
                    "message": str(e), "iteration": iteration, "fatal": True,
                }, stream=stream_events)
                raise

        choice = response.choices[0]
        message = choice.message

        if response.usage:
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens

        # Normalize native tool_calls to dict form so downstream code has a
        # uniform shape whether calls came from the API or from salvage.
        effective_tool_calls: List[dict] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in (message.tool_calls or [])
        ]

        # Salvage path: if the API returned no tool_calls but the content has
        # XML-formatted tool calls (DeepSeek behavior), parse them out so the
        # agent loop continues instead of silently dropping the calls.
        salvaged_content: str | None = None
        if not effective_tool_calls and message.content:
            salvaged, salvaged_content = _salvage_xml_tool_calls(message.content)
            if salvaged:
                effective_tool_calls = salvaged
                _emit("xml_salvage", {
                    "iteration": iteration,
                    "salvaged_count": len(salvaged),
                    "tool_names": [tc["function"]["name"] for tc in salvaged],
                }, stream=stream_events)

        assistant_msg: Dict[str, Any] = {"role": "assistant"}
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            assistant_msg["reasoning_content"] = reasoning
        if message.content:
            # If we salvaged, store the cleaned text (without leaked XML) in
            # history so future prompts don't see the raw tool-call markup.
            assistant_msg["content"] = (
                salvaged_content if salvaged_content is not None else message.content
            )
            final_text = assistant_msg["content"]
            _emit("assistant.message", {
                "text": assistant_msg["content"][:2000], "iteration": iteration,
            }, stream=stream_events)

        if effective_tool_calls:
            assistant_msg["tool_calls"] = effective_tool_calls

        messages.append(assistant_msg)

        if not effective_tool_calls:
            if not message.content and reasoning:
                final_text = reasoning
            break

        last_iteration_had_tool_calls = True

        for tc in effective_tool_calls:
            tc_id = tc["id"]
            tc_name = tc["function"]["name"]
            tc_arguments = tc["function"]["arguments"]

            try:
                args = json.loads(tc_arguments)
            except json.JSONDecodeError:
                args = {"command": tc_arguments}

            _emit("tool_call", {
                "name": tc_name,
                "arguments": args,
                "call_id": tc_id,
                "salvaged": tc_id.startswith("xml-salvage-"),
            }, stream=stream_events)

            t_start = time.time()
            result_str = _execute_tool_call(
                tc_name, args, cwd=cwd, env=env,
                tool_dispatcher=tool_dispatcher,
            )
            duration_ms = (time.time() - t_start) * 1000

            success = not result_str.startswith("[error")
            tool_name = tc_name

            if tool_name not in tool_stats:
                tool_stats[tool_name] = {
                    "calls": 0, "total_result_bytes": 0,
                    "total_ms": 0.0, "failures": 0,
                }
            tool_stats[tool_name]["calls"] += 1
            tool_stats[tool_name]["total_result_bytes"] += len(result_str)
            tool_stats[tool_name]["total_ms"] += duration_ms
            if not success:
                tool_stats[tool_name]["failures"] += 1

            if tool_log_path:
                args_preview = json.dumps(args)
                if len(args_preview) > 500:
                    args_preview = args_preview[:500] + "..."
                log_entry = {
                    "ts": time.time(),
                    "step": step_name,
                    "iteration": iteration,
                    "tool": tool_name,
                    "call_id": tc_id,
                    "arguments": args_preview,
                    "result_length": len(result_str),
                    "duration_ms": round(duration_ms, 1),
                    "success": success,
                    "result_preview": result_str[:300],
                }
                with open(tool_log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            _emit("tool_result", {
                "name": tool_name,
                "result": result_str[:2000],
                "call_id": tc_id,
                "success": success,
                "duration_ms": round(duration_ms, 1),
            }, stream=stream_events)

            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result_str,
            })

    # If the loop exhausted max_iters while the model was still calling tools,
    # force one final synthesis call with no tools so the model can produce output.
    if last_iteration_had_tool_calls:
        _emit("max_iters_exceeded", {
            "iteration": iteration,
            "action": "forcing_synthesis",
        }, stream=stream_events)
        messages.append({
            "role": "user",
            "content": (
                "You have used all available iterations. Produce your FINAL "
                "output now with everything you have gathered so far. "
                "Do not make any more tool calls."
            ),
        })
        synth_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "temperature": 0.0,
        }
        _apply_reasoning_controls(
            synth_kwargs,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            thinking_budget=thinking_budget,
        )
        try:
            synth_response = client.chat.completions.create(**synth_kwargs)
            synth_msg = synth_response.choices[0].message
            if synth_response.usage:
                total_input_tokens += synth_response.usage.prompt_tokens
                total_output_tokens += synth_response.usage.completion_tokens
            if synth_msg.content:
                final_text = synth_msg.content
            elif getattr(synth_msg, "reasoning_content", None):
                final_text = synth_msg.reasoning_content
        except Exception as e:
            _emit("error", {
                "message": f"synthesis call failed: {e}",
                "fatal": False,
            }, stream=stream_events)

    _emit("thread.finished", {
        "step": step_name,
        "iterations": iteration,
        "final_text": final_text,
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        },
        "tool_stats": tool_stats,
    }, stream=stream_events)

    if tool_stats:
        print(f"\n    Tool usage summary [{step_name}]:", file=sys.stderr)
        for name, stats in sorted(tool_stats.items()):
            avg_ms = stats["total_ms"] / stats["calls"] if stats["calls"] else 0
            print(
                f"      {name}: {stats['calls']} calls, "
                f"{stats['total_result_bytes']:,} bytes, "
                f"avg {avg_ms:.0f}ms"
                + (f", {stats['failures']} failures" if stats["failures"] else ""),
                file=sys.stderr,
            )

    global last_usage
    last_usage = {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens}
    return final_text
