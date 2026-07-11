"""
Pipeline Compiler — compiles and executes YAML pipeline plans.

Takes a plan file expressed in the pipeline language (see pipeline-language-spec.tex)
and executes it step by step, handling model selection, tool routing, context management,
I/O wiring, and budget allocation.

Usage:
    python pipeline_compiler.py literature_survey_plan.yaml \\
        --input proposal.pdf --output-dir survey_output/
    python pipeline_compiler.py literature_survey_plan.yaml \\
        --input proposal.pdf --output-dir survey_output/ --resume 5
    python pipeline_compiler.py literature_survey_plan.yaml \\
        --input proposal.pdf --output-dir survey_output/ --step 5
"""

import argparse
import ast
import importlib
import json
import os
import pwd
import shlex
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import traceback
import uuid
import dataclasses
from dataclasses import dataclass, field
from typing import Any

import httpx
import yaml
from openai import OpenAI

import agent_loop
from agent_loop import run_agent
from pipeline_utils import MATH_INSTRUCTION, extract_json, extract_json_result, _truncate_words, ContractError
from prompt_library import PromptLibrary
from tool_provider import MeshToolProvider, StandaloneToolProvider, ToolProvider

try:
    import jsonschema as _jsonschema
except ImportError:
    _jsonschema = None

VALID_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh"}
REASONING_THINKING_BUDGETS = {
    "none": 0,
    "low": 512,
    "medium": 2048,
    "high": 8192,
    "xhigh": 16384,
}


# ---------------------------------------------------------------------------
# Plan data structures
# ---------------------------------------------------------------------------

@dataclass
class StepSpec:
    name: str
    op: str                             # Search | Extract | Classify | Translate | Synthesize | Gate | While | Pipeline
    type_sig: str                       # e.g. "TaskDescription -> List[Theme]"
    input: list[str]                    # input binding expressions
    prompt: str
    prompt_vars: dict[str, Any] = field(default_factory=dict)
    input_schema: Any = None
    output_schema: Any = None
    tools: list[str] = field(default_factory=list)
    agentic: bool = False
    deterministic: bool = False
    handler: str = ""
    foreach: bool = False
    batch_size: int = 0
    batch_parallel: int = 1
    adaptive_batch: bool = False
    budget: dict = field(default_factory=dict)
    backend_override: str = ""          # "light" | "deep" | ""
    reasoning_effort: str | None = None  # "none" | "low" | "medium" | "high" | "xhigh"
    condition: str = ""                  # Gate condition expression
    validate: dict = field(default_factory=lambda: {"enabled": True, "max_retries": 2})
    while_spec: dict = field(default_factory=dict)  # While op: {classify, translate, execute, max_rounds, toolset}
    foreach_over: str = ""              # Foreach: dotted reference to array (e.g. "plan_changes.changes")
    foreach_as: str = ""                # Foreach: loop variable name
    body: list | None = None            # Foreach: nested body steps (list[StepSpec])
    pipeline_ref: str = ""              # Pipeline op: path to sub-pipeline YAML
    chunk: dict | bool = field(default_factory=dict)  # Auto-chunk config: True/False or {max_chunk_tokens: N}
    cache_from: str = ""                # Stub steps: seed output from this JSON file instead of executing
    allow_empty_input: bool = False     # Permit empty upstream outputs (default: hard error)


@dataclass
class PipelinePlan:
    name: str
    description: str
    config: dict
    steps: list[StepSpec]


def _step_mode_label(step: StepSpec) -> str:
    """Compact execution-mode label for step listings and debug output."""
    if step.op == "Pipeline":
        return f"pipeline:{step.pipeline_ref}"
    if step.op == "Foreach" and step.body:
        return f"foreach:{len(step.body)} body steps"
    if step.op == "While":
        handler = step.while_spec.get("execute", {}).get("handler", "")
        return f"while:{handler}" if handler else "while"
    if step.op == "Foreach" or step.foreach:
        return "foreach"
    if step.deterministic:
        return f"handler:{step.handler}"
    if step.batch_size:
        tag = f"batched:{step.batch_size}"
        if step.batch_parallel > 1:
            tag += f",par:{step.batch_parallel}"
        if step.adaptive_batch:
            tag += ",adaptive"
        return tag
    if step.agentic:
        return "agentic"
    return "single"


def print_step_list(plan: PipelinePlan) -> None:
    """Print a stable, human-readable list of plan steps."""
    print(f"Pipeline: {plan.name}")
    print(f"Steps: {len(plan.steps)}")
    for i, step in enumerate(plan.steps, 1):
        zero = i - 1
        print(
            f"{i:>2} ({zero:>2})  {step.name:<28} "
            f"{step.op:<10} {_step_mode_label(step)}"
        )


# ---------------------------------------------------------------------------
# Plan loading
# ---------------------------------------------------------------------------

def _env_substitute(text: str) -> str:
    return re.sub(r'\$\{(\w+)\}', lambda m: os.environ.get(m.group(1), ""), text)


def _normalize_reasoning_effort(value: Any) -> str | None:
    if value is None:
        return None
    effort = str(value).strip().lower()
    if not effort:
        return None
    if effort not in VALID_REASONING_EFFORTS:
        valid = ", ".join(sorted(VALID_REASONING_EFFORTS))
        raise ValueError(f"Invalid reasoning_effort '{value}'. Expected one of: {valid}")
    return effort


def _is_local_vllm_config(config: dict) -> bool:
    base_url = str(config.get("base_url", "")).lower()
    model = str(config.get("model", "")).lower()
    return (
        "localhost" in base_url
        or "127.0.0.1" in base_url
        or model.startswith("local-")
    )


_vllm_model_len_cache: dict[str, int] = {}

def _get_vllm_max_model_len(config: dict) -> int | None:
    base_url = config.get("base_url", "")
    if base_url in _vllm_model_len_cache:
        return _vllm_model_len_cache[base_url]
    try:
        import httpx as _httpx
        models_url = base_url.rstrip("/").removesuffix("/v1") + "/v1/models"
        resp = _httpx.get(models_url, timeout=5.0)
        data = resp.json().get("data", [])
        if data:
            limit = data[0].get("max_model_len", 0)
            if limit:
                _vllm_model_len_cache[base_url] = limit
                return limit
    except Exception:
        pass
    return None


def _apply_reasoning_controls(kwargs: dict[str, Any], config: dict) -> None:
    """Attach reasoning controls to an OpenAI-compatible request.

    OpenAI/DeepSeek-style endpoints generally accept `reasoning_effort`.
    Local vLLM-served Qwen uses chat-template thinking controls under
    `extra_body.chat_template_kwargs` instead.
    """
    reasoning_effort = _normalize_reasoning_effort(config.get("reasoning_effort"))
    thinking_budget = int(config.get("thinking_budget") or 0)

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
        if _is_local_vllm_config(config):
            if reasoning_effort in ("none", "low"):
                set_chat_template_thinking(False)
            else:
                budget = thinking_budget or REASONING_THINKING_BUDGETS[reasoning_effort]
                set_chat_template_thinking(True, budget)
        elif reasoning_effort != "none":
            kwargs["reasoning_effort"] = reasoning_effort
        return

    if thinking_budget > 0:
        if _is_local_vllm_config(config):
            set_chat_template_thinking(True, min(thinking_budget, 4096))
        else:
            kwargs["extra_body"] = {
                "enable_thinking": True,
                "thinking_budget": min(thinking_budget, 4096),
            }


def _parse_step(step_raw: dict) -> StepSpec:
    input_raw = step_raw.get("input", "task")
    if isinstance(input_raw, str):
        inputs = [s.strip() for s in input_raw.split(",")]
    elif isinstance(input_raw, list):
        inputs = [s.strip() if isinstance(s, str) else str(s) for s in input_raw]
    else:
        inputs = [str(input_raw)]

    while_spec = {}
    if step_raw.get("op") == "While":
        while_spec = {
            "classify": step_raw.get("classify", {}),
            "translate": step_raw.get("translate", {}),
            "execute": step_raw.get("execute", {}),
            "max_rounds": step_raw.get("max_rounds", 3),
            "min_rounds": step_raw.get("min_rounds", 2),
            "toolset": step_raw.get("toolset", []),
            "translate_system": step_raw.get("translate_system", ""),
            "max_evidence_chars": step_raw.get("max_evidence_chars", 50000),
            "refresh_handler": step_raw.get("refresh_handler", ""),
            "escalation": step_raw.get("escalation", []),
            # Extract overrides (depth 1+ in recursive pipelines)
            "extract_toolset": step_raw.get("extract_toolset", []),
            "extract_classify": step_raw.get("extract_classify", {}),
            "extract_translate": step_raw.get("extract_translate", {}),
            "extract_max_rounds": step_raw.get("extract_max_rounds", None),
        }

    body = None
    foreach_over = ""
    foreach_as = ""
    if step_raw.get("op") == "Foreach" and "body" in step_raw:
        foreach_over = step_raw.get("over", "")
        foreach_as = step_raw.get("as", "item")
        if not foreach_over:
            raise ValueError(
                f"Foreach step '{step_raw.get('name', '?')}' has body but no 'over' "
                f"field specifying the iteration array"
            )
        body_raw = step_raw.get("body", [])
        if body_raw:
            body = [_parse_step(bs) for bs in body_raw]
    elif step_raw.get("op") == "While" and "body" in step_raw:
        body_raw = step_raw.get("body", [])
        if body_raw:
            body = [_parse_step(bs) for bs in body_raw]

    return StepSpec(
        name=step_raw["name"],
        op=step_raw["op"],
        type_sig=step_raw.get("type", ""),
        input=inputs,
        prompt=step_raw.get("prompt", ""),
        prompt_vars=step_raw.get("prompt_vars", {}) or {},
        input_schema=step_raw.get("input_schema"),
        output_schema=step_raw.get("output_schema"),
        tools=step_raw.get("tools", []),
        agentic=step_raw.get("agentic", False),
        deterministic=step_raw.get("deterministic", False),
        handler=step_raw.get("handler", ""),
        foreach=step_raw.get("foreach", False),
        batch_size=step_raw.get("batch_size", 0),
        batch_parallel=step_raw.get("batch_parallel", 1),
        adaptive_batch=bool(step_raw.get("adaptive_batch", False)),
        budget=step_raw.get("budget", {}),
        backend_override=step_raw.get("backend", ""),
        reasoning_effort=_normalize_reasoning_effort(step_raw.get("reasoning_effort")),
        condition=step_raw.get("condition", ""),
        validate=step_raw.get("validate", {"enabled": True, "max_retries": 2}),
        while_spec=while_spec,
        foreach_over=foreach_over,
        foreach_as=foreach_as,
        body=body,
        pipeline_ref=step_raw.get("plan", ""),
        chunk=step_raw.get("chunk", {}),
        cache_from=step_raw.get("cache_from", ""),
        allow_empty_input=bool(step_raw.get("allow_empty_input", False)),
    )


def load_plan(path: str) -> PipelinePlan:
    with open(path) as f:
        raw = yaml.safe_load(_env_substitute(f.read()))

    steps = [_parse_step(step_raw) for step_raw in raw.get("steps", [])]

    plan = PipelinePlan(
        name=raw.get("name", "pipeline"),
        description=raw.get("description", ""),
        config=raw.get("config", {}),
        steps=steps,
    )
    _validate_plan_edges(plan)
    return plan


def _validate_plan_edges(plan: PipelinePlan) -> None:
    """Compile-time validation: check that step input bindings reference existing steps
    and that output_schema → input_schema edges are type-compatible."""
    step_names = {s.name for s in plan.steps}
    step_by_name = {s.name: s for s in plan.steps}
    errors = []

    for step in plan.steps:
        for binding in step.input:
            base_name = binding.split("[")[0].split(".")[0].strip()
            if base_name == "task":
                continue
            if base_name not in step_names:
                if step.op == "Pipeline":
                    print(f"    [plan-validate] [warn] Pipeline step '{step.name}' "
                          f"references unresolved input '{base_name}' "
                          f"— will be resolved at runtime from parent context",
                          file=sys.stderr)
                else:
                    errors.append(
                        f"Step '{step.name}' references unknown input '{base_name}'"
                    )
                continue

            source = step_by_name[base_name]
            if step.input_schema and source.output_schema:
                src_type = _schema_output_type(source.output_schema)
                dst_type = _schema_input_type(step.input_schema)
                if src_type and dst_type and not _types_compatible(src_type, dst_type):
                    errors.append(
                        f"Type mismatch: '{source.name}' outputs {src_type} "
                        f"but '{step.name}' expects {dst_type}"
                    )

        if step.body:
            body_names = {bs.name for bs in step.body}
            parent_type = "While" if step.op == "While" else "Foreach"
            collisions = body_names & step_names
            if collisions:
                errors.append(
                    f"{parent_type} '{step.name}': body step name(s) {collisions} "
                    f"collide with outer pipeline step names — rename them"
                )
            allowed = step_names | body_names | {"task"}
            if step.foreach_as:
                allowed.add(step.foreach_as)
            for bs in step.body:
                for binding in bs.input:
                    base_name = binding.split("[")[0].split(".")[0].strip()
                    if base_name in allowed:
                        continue
                    if bs.op == "Pipeline":
                        print(f"    [plan-validate] [warn] Pipeline body step '{bs.name}' "
                              f"in {parent_type} '{step.name}' references unresolved "
                              f"input '{base_name}' — will be resolved at runtime",
                              file=sys.stderr)
                        continue
                    errors.append(
                        f"Body step '{bs.name}' in {parent_type} '{step.name}' "
                        f"references unknown input '{base_name}'"
                    )

    if errors:
        msg = "Plan validation errors:\n  " + "\n  ".join(errors)
        print(f"    [plan-validate] {msg}", file=sys.stderr)
        raise ValueError(msg)


def _schema_output_type(schema: Any) -> str | None:
    """Infer the top-level type from an output_schema declaration."""
    if isinstance(schema, dict):
        t = schema.get("type")
        if t:
            return t
        return "object"
    if isinstance(schema, list):
        return "array"
    if isinstance(schema, str):
        return schema
    return None


def _schema_input_type(schema: Any) -> str | None:
    """Infer expected type from an input_schema declaration."""
    if isinstance(schema, dict):
        t = schema.get("type")
        if t:
            return t
    if isinstance(schema, str):
        return schema
    return None


def _types_compatible(src: str, dst: str) -> bool:
    """Check if a source output type is compatible with a destination input type."""
    if dst == "any" or src == "any":
        return True
    if src == dst:
        return True
    if dst == "string":
        return True  # anything can be serialized to string
    return False



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONTEXT_UPGRADE_THRESHOLD = 100_000  # tokens; upgrade light → deep above this

MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "local-27b": 262_144,
    "qwen-27b": 262_144,
    "Qwen/Qwen3.6-27B": 131_072,
    "Qwen/Qwen3-235B-A22B": 262_144,
    "deepseek-v4-pro": 1_048_576,
    "deepseek-chat": 131_072,
    "gpt-4o": 128_000,
    "gpt-5.5": 1_048_576,
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
}

CONTEXT_SAFETY_MARGIN = 20_000  # tokens reserved for output + overhead

# Minimum output budget for an LLM call to be worth making. Below this the
# model truncates mid-generation (thinking models burn it all on reasoning
# prose and never reach the JSON), so the call is refused instead.
MIN_USABLE_OUTPUT_TOKENS = 2_048

# Conservative chars-per-token used wherever input size is weighed against a
# model's context window (llm_call's output cap, the auto-chunk trigger, and
# chunk sizing). These three MUST agree: when the chunker sized chunks at
# 4 chars/token while llm_call estimated at 2.5, a chunk could pass the
# chunker yet still starve the output budget at call time.
CONTEXT_EST_CHARS_PER_TOKEN = 2.5

def _is_codex_backend(config: dict) -> bool:
    return bool(config.get("codex_backend"))

def _is_cc_backend(config: dict) -> bool:
    return bool(config.get("cc_backend"))

def _is_ci_backend(config: dict) -> bool:
    return bool(config.get("ci_backend"))


class EmptyStepOutputError(RuntimeError):
    """An upstream step produced empty output that a downstream step consumes."""


def _is_empty_output(data: Any) -> bool:
    """True if data carries no usable content (recursively empty).

    Catches the cascade where a step emits e.g. {"sections": [], "citations": []}
    and every downstream step silently runs on nothing.
    """
    if data is None:
        return True
    if isinstance(data, str):
        return not data.strip()
    if isinstance(data, (list, tuple)):
        return all(_is_empty_output(v) for v in data)
    if isinstance(data, dict):
        return all(_is_empty_output(v) for v in data.values())
    return False

def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _read_input(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--input file not found: {path}")
    if path.lower().endswith('.pdf'):
        try:
            result = subprocess.run(
                ['pdftotext', '-layout', path, '-'],
                capture_output=True, text=True, timeout=60,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"pdftotext is required to extract text from PDF input "
                f"'{path}' but is not installed (apt install poppler-utils). "
                f"Refusing to fall back to raw PDF bytes."
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"pdftotext failed on '{path}' (exit {result.returncode}): "
                f"{result.stderr.strip()[:500]}"
            )
        if not result.stdout.strip():
            raise RuntimeError(
                f"pdftotext produced no text for '{path}' — the PDF may be "
                f"scanned images without a text layer"
            )
        return result.stdout
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ---------------------------------------------------------------------------
# Filter predicate parsing and application
# ---------------------------------------------------------------------------

def parse_binding(binding: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Parse 'step_name[field=value]' → (step_name, [(field, op, value)])."""
    m = re.match(r'^(\w+)\[(.+)\]$', binding.strip())
    if not m:
        return binding.strip(), []
    step_name = m.group(1)
    predicates = []
    for part in m.group(2).split(','):
        part = part.strip()
        for op in ('<=', '>=', '!=', '='):
            if op in part:
                fld, val = part.split(op, 1)
                predicates.append((fld.strip(), op, val.strip()))
                break
    return step_name, predicates


def apply_filter(data: Any, predicates: list[tuple[str, str, str]]) -> Any:
    if not predicates or not isinstance(data, list):
        return data
    result = []
    for item in data:
        if not isinstance(item, dict):
            continue
        match = True
        for fld, op, val in predicates:
            item_val = item.get(fld)
            if item_val is None:
                match = False
                break
            try:
                cmp_val = type(item_val)(val) if isinstance(item_val, (int, float)) else val
            except (ValueError, TypeError):
                cmp_val = val
            if op == '=' and item_val != cmp_val:
                match = False
            elif op == '!=' and item_val == cmp_val:
                match = False
            elif op == '<=' and item_val > cmp_val:
                match = False
            elif op == '>=' and item_val < cmp_val:
                match = False
        if match:
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Deterministic handlers
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Pipeline Compiler
# ---------------------------------------------------------------------------

class PipelineCompiler:

    TOOL_ALIASES = {
        "memory": ["memory_search", "memory_get", "memory_list"],
        "notes": ["notes_search", "notes_list", "notes_get", "notes_read"],
        "email": ["gmail_search_emails", "gmail_get_email", "gmail_list_from_date"],
        "calendar": ["calendar_list_on_date", "current_time"],
        "files": ["file_read", "list_dir", "bash_exec", "get_working_directory"],
        "web": ["exa_search", "exa_fetch_full", "extract_url"],
        "literature": [
            "arxiv_search",
            "arxiv_get",
            "arxiv_fulltext",
            "literature_search",
            "literature_fulltext",
            "pubmed_search",
            "pubmed_get",
            "pubmed_fulltext",
        ],
    }

    def __init__(
        self,
        plan: PipelinePlan,
        task_text: str,
        output_dir: str,
        light_config: dict,
        deep_config: dict,
        tool_providers: list[ToolProvider] | None = None,
        prompt_library: PromptLibrary | None = None,
        plan_dir: str | None = None,
    ):
        self.plan = plan
        self.task_text = task_text
        self.output_dir = output_dir
        self.plan_dir = plan_dir
        self.light_config = light_config
        self.deep_config = deep_config
        self.tool_providers = tool_providers or [StandaloneToolProvider()]
        self.prompt_library = prompt_library
        self.outputs: dict[str, Any] = {}
        self._precached_steps: dict[str, Any] = {}
        self.step_timings: list[dict] = []
        self._token_records: list[dict] = []
        self._current_step_name: str = ""
        self._gate_skipped: set[str] = set()
        self._handlers: dict[str, Any] = {}
        self._batch_post_processors: dict[str, Any] = {}
        self._step_post_processors: dict[str, Any] = {}
        self._auto_prompt_vars_ext: Any = None
        max_total = self.plan.config.get("max_total_seconds")
        if max_total and "_deadline" not in self.plan.config:
            self._deadline: float | None = time.time() + max_total
        else:
            self._deadline = self.plan.config.get("_deadline")
        self._load_handler_module()

    def _load_handler_module(self):
        """Load handlers from the plan's config.handler_module if specified."""
        module_name = self.plan.config.get("handler_module")
        if not module_name:
            return
        if self.plan_dir and self.plan_dir not in sys.path:
            sys.path.insert(0, self.plan_dir)
        engine_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        handlers_dir = os.path.join(engine_dir, "handlers")
        for d in (engine_dir, handlers_dir):
            if d not in sys.path:
                sys.path.insert(0, d)
        try:
            mod = importlib.import_module(module_name)
        except ImportError as e:
            print(f"    [warning] Could not import handler module "
                  f"'{module_name}': {e}", file=sys.stderr)
            return
        if hasattr(mod, "HANDLERS"):
            self._handlers.update(mod.HANDLERS)
        if hasattr(mod, "BATCH_POST_PROCESSORS"):
            self._batch_post_processors.update(mod.BATCH_POST_PROCESSORS)
        if hasattr(mod, "STEP_POST_PROCESSORS"):
            self._step_post_processors.update(mod.STEP_POST_PROCESSORS)
        if hasattr(mod, "AUTO_PROMPT_VARS"):
            self._auto_prompt_vars_ext = mod.AUTO_PROMPT_VARS
        loaded = list(self._handlers.keys())
        print(f"    [info] Loaded handler module '{module_name}': "
              f"{len(loaded)} handlers ({', '.join(loaded) or 'none'}), "
              f"{len(self._batch_post_processors)} batch processors, "
              f"{len(self._step_post_processors)} step processors",
              file=sys.stderr)

    # -- Tool resolution ---------------------------------------------------

    def _expand_tool_names(self, names: list[str]) -> list[str]:
        if not names:
            expanded: list[str] = []
            for provider in self.tool_providers:
                expanded.extend(sorted(provider.tool_names()))
            return list(dict.fromkeys(expanded))

        expanded = []
        for name in names:
            alias = self.TOOL_ALIASES.get(name)
            if alias:
                expanded.extend(alias)
            else:
                expanded.append(name)
        return list(dict.fromkeys(expanded))

    def _resolve_tools_for_step(self, step: StepSpec) -> tuple[list[dict], Any]:
        requested = self._expand_tool_names(step.tools)
        definitions: list[dict] = []
        dispatch_map: dict[str, ToolProvider] = {}
        provider_names = {provider.name: provider for provider in self.tool_providers}

        for tool_name in requested:
            for provider in self.tool_providers:
                if tool_name in provider.tool_names():
                    definitions.extend(provider.definitions_for([tool_name]))
                    dispatch_map[tool_name] = provider
                    break
            else:
                print(
                    f"    [warning] Tool '{tool_name}' requested by step "
                    f"'{step.name}' is not available from providers: "
                    f"{', '.join(provider_names)}",
                    file=sys.stderr,
                )

        async def dispatch(tool_name: str, args: dict) -> str:
            provider = dispatch_map.get(tool_name)
            if not provider:
                return f"Unknown tool: {tool_name}"
            return await provider.dispatch(tool_name, args)

        return definitions, dispatch

    # -- Model selection ---------------------------------------------------

    def select_config(self, step: StepSpec) -> dict:
        if step.backend_override == "deep":
            return self.deep_config
        if step.backend_override == "light":
            return self.light_config
        if step.op == "Synthesize":
            return self.deep_config
        return self.light_config

    def config_for_step(self, step: StepSpec, *, escalation: dict | None = None,
                        input_chars: int = 0) -> dict:
        config = self.select_config(step)

        # Context-based escalation: if input exceeds threshold, use deep
        threshold = self.plan.config.get("context_escalation_threshold", 100_000)
        if input_chars > threshold and config is self.light_config:
            print(f"    [escalation] {step.name} input is {input_chars} chars, "
                  f"exceeding threshold {threshold} — using deep backend",
                  file=sys.stderr)
            config = self.deep_config

        # Retry-based escalation override (takes precedence over step-level)
        if escalation:
            if escalation.get("use_deep"):
                config = self.deep_config
            effort = escalation.get("reasoning_effort")
            if effort:
                config = config.copy()
                config["reasoning_effort"] = effort
                return config

        if not step.reasoning_effort:
            return config
        step_config = config.copy()
        step_config["reasoning_effort"] = step.reasoning_effort
        return step_config

    # -- Input resolution --------------------------------------------------

    def resolve_inputs(self, step: StepSpec) -> dict:
        """Resolve input bindings → {name: data, ..., "primary": first_data}.

        Returns empty dict if all non-task inputs resolved to error/empty,
        which signals the caller to skip the step.
        """
        result: dict[str, Any] = {}
        has_real_data = False
        for i, binding in enumerate(step.input):
            binding = binding.strip()

            # Handle dotted-path bindings like "parse_sections.citations"
            if "." in binding and "[" not in binding:
                parts = binding.split(".")
                base = parts[0]
                if base == "task":
                    # task doesn't support sub-paths
                    result["task"] = self.task_text
                    if i == 0:
                        result["primary"] = self.task_text
                    continue
                if base not in self.outputs and not self._ensure_output_loaded(base):
                    print(f"    [warning] Input '{base}' not available",
                          file=sys.stderr)
                    continue
                data = self.outputs[base]
                for attr in parts[1:]:
                    if isinstance(data, dict) and attr in data:
                        data = data[attr]
                    else:
                        print(f"    [warning] Cannot resolve '{binding}': "
                              f"'{attr}' not found in {base}",
                              file=sys.stderr)
                        data = None
                        break
                if data is None:
                    continue
                if _is_empty_output(data):
                    if step.allow_empty_input:
                        print(f"    [warning] Input '{binding}' is empty — "
                              f"allowed by allow_empty_input", file=sys.stderr)
                        continue
                    raise EmptyStepOutputError(
                        f"Step '{step.name}' received empty output from "
                        f"upstream step '{base}' (binding '{binding}'). "
                        f"Fix or rerun '{base}' (--step), or set "
                        f"allow_empty_input: true on '{step.name}' if empty "
                        f"is expected."
                    )
                has_real_data = True
                # Use the full dotted path as the key so multiple refs
                # to same step don't collide
                result[binding] = data
                if i == 0:
                    result["primary"] = data
                continue

            if binding == "task":
                result["task"] = self.task_text
                if i == 0:
                    result["primary"] = self.task_text
                continue

            step_name, predicates = parse_binding(binding)
            if step_name not in self.outputs and not self._ensure_output_loaded(step_name):
                print(f"    [warning] Input '{step_name}' not available",
                      file=sys.stderr)
                continue

            data = self.outputs[step_name]

            if isinstance(data, dict) and "error" in data:
                print(f"    [warning] Input '{step_name}' is an error: "
                      f"{str(data['error'])[:120]}", file=sys.stderr)
                continue

            # Empty output from the producing step is a hard error: running
            # downstream steps on nothing cascades garbage through the whole
            # pipeline (e.g. reviewing an empty paper for 800s).
            if _is_empty_output(data):
                if step.allow_empty_input:
                    print(f"    [warning] Input '{step_name}' is empty — "
                          f"allowed by allow_empty_input", file=sys.stderr)
                    continue
                raise EmptyStepOutputError(
                    f"Step '{step.name}' received empty output from upstream "
                    f"step '{step_name}'. Fix or rerun '{step_name}' (--step), "
                    f"or set allow_empty_input: true on '{step.name}' if empty "
                    f"is expected."
                )

            if predicates:
                data = apply_filter(data, predicates)

            # Filtered-to-empty is distinct from produced-empty: a predicate
            # legitimately matching nothing keeps the existing skip semantics.
            if isinstance(data, list) and len(data) == 0:
                print(f"    [warning] Input '{step_name}' resolved to empty list",
                      file=sys.stderr)
                continue

            has_real_data = True
            result[step_name] = data
            if i == 0:
                result["primary"] = data

        if not has_real_data and "task" not in result:
            return {}

        if "primary" not in result and result:
            result["primary"] = next(iter(result.values()))
        return result

    # -- Prompt building ---------------------------------------------------

    def _auto_prompt_vars(self, step: StepSpec, inputs: dict) -> dict[str, Any]:
        values: dict[str, Any] = {
            "pipeline_name": self.plan.name,
            "step_name": step.name,
            "op": step.op,
            "_depth": self.plan.config.get("_depth", 0),
            "max_recursion_depth": self.plan.config.get("max_recursion_depth", 3),
        }
        primary = inputs.get("primary")
        if isinstance(primary, list):
            values["num_items"] = len(primary)

        codebase_root = self.plan.config.get("codebase_root")
        if codebase_root:
            values["codebase_root"] = codebase_root

        # Make named inputs available to prompt templates. This allows scoped
        # prompts to reference fields directly, e.g. {extract_context.sender},
        # while build_prompt still appends the full input blocks for auditability.
        for key, value in inputs.items():
            if key in ("primary", "_validation_feedback", "task"):
                continue
            values.setdefault(key, value)

        if self._auto_prompt_vars_ext:
            values.update(self._auto_prompt_vars_ext(step.name, inputs))

        values.setdefault("_validation_feedback", "")
        values.update(step.prompt_vars)
        return values

    def _resolve_prompt(self, step: StepSpec, inputs: dict) -> str:
        reference = PromptLibrary.reference_name(step.prompt)
        if not reference:
            return step.prompt
        if not self.prompt_library:
            raise ValueError(
                f"Step '{step.name}' references prompt '{reference}', "
                "but no PromptLibrary is configured"
            )
        return self.prompt_library.resolve(reference, self._auto_prompt_vars(step, inputs))

    def build_prompt(self, step: StepSpec, inputs: dict) -> tuple[str, str]:
        """Build (system_prompt, user_prompt) for an LLM step."""
        op_system = {
            "Search": (
                "You are a research search agent. Use the available tools to find relevant, "
                "high-quality results. Try multiple query formulations — broaden when results "
                "are sparse, narrow when flooded. Cross-reference across sources to verify "
                "claims. Stop searching when additional queries return diminishing new information."
            ),
            "Extract": (
                "You are a data extraction tool. Extract structured data faithfully from the "
                "input, preserving the source's structure and terminology. Use null for missing "
                "fields — never invent or hallucinate values. Output valid JSON only — no "
                "preamble, no commentary."
            ),
            "Classify": (
                "You are a classification tool. Apply the given criteria consistently across "
                "all items, using the same standard of evidence for each. When a case is "
                "genuinely ambiguous, choose the closest match and note the ambiguity if the "
                "schema allows. Output valid JSON only — no preamble."
            ),
            "Translate": (
                "You are a transformation tool. Your job is faithful conversion of input from "
                "one form to another — preserving semantic content while changing representation. "
                "The transformation may be stylistic (rewriting in a different voice or tone), "
                "structural (turning a specification into an implementation, pseudocode into "
                "code, or a plan into concrete actions), or format-level (restructuring data "
                "layout). Do not add, remove, or editorialize beyond what the transformation requires."
            ),
            "Synthesize": (
                "You are a synthesis writer producing well-structured, evidence-based text. "
                "Ground claims in the provided sources — cite or attribute when possible. "
                "Distinguish established facts from inferences or interpretations. Maintain "
                "analytical structure: organize by theme or argument, not by source order."
            ),
        }
        system = op_system.get(step.op, "You are a helpful assistant.") + " " + MATH_INSTRUCTION

        resolved_prompt = self._resolve_prompt(step, inputs).strip()

        primary = inputs.get("primary")
        primary_inlined = False
        if primary is not None and "{input}" in resolved_prompt:
            if isinstance(primary, list) and len(primary) > 10 and isinstance(primary[0], dict):
                serialized = json.dumps(primary, indent=2, ensure_ascii=False)
                if len(serialized) > 2_000_000:
                    compact = self._compact_paper_list(primary)
                    serialized = json.dumps(compact, indent=2, ensure_ascii=False)
                    print(f"    [compact] primary: {len(primary)} items compacted "
                          f"({len(json.dumps(primary)):,} → {len(serialized):,} chars)",
                          file=sys.stderr)
            elif isinstance(primary, (list, dict)):
                serialized = json.dumps(primary, indent=2, ensure_ascii=False)
            elif isinstance(primary, str):
                serialized = _truncate_words(
                    primary, self.plan.config.get('max_input_words', 5000))
            else:
                serialized = str(primary)[:50_000]
            resolved_prompt = resolved_prompt.replace("{input}", serialized, 1)
            primary_inlined = True

        user_parts = [resolved_prompt]

        if step.output_schema is not None:
            schema_text = yaml.safe_dump(
                step.output_schema,
                sort_keys=False,
                allow_unicode=True,
            ).strip()
            user_parts.append(
                "\nOUTPUT SCHEMA:\n"
                "Return valid JSON matching this schema.\n"
                f"```yaml\n{schema_text}\n```"
            )

        for key, data in inputs.items():
            if key in ("primary", "_validation_feedback"):
                continue
            if primary_inlined and data is primary:
                continue
            if key == "task":
                user_parts.append(f"\nINPUT DOCUMENT:\n{_truncate_words(data, self.plan.config.get('max_input_words', 5000))}")
            elif isinstance(data, list) and len(data) > 10 and isinstance(data[0], dict):
                formatted = json.dumps(data, indent=2, ensure_ascii=False)
                if len(formatted) > 2_000_000:
                    compact = self._compact_paper_list(data)
                    formatted = json.dumps(compact, indent=2, ensure_ascii=False)
                    print(f"    [compact] {key}: {len(data)} items compacted "
                          f"({len(json.dumps(data)):,} → {len(formatted):,} chars)",
                          file=sys.stderr)
                user_parts.append(f"\nDATA FROM {key}:\n{formatted}")
            elif isinstance(data, (list, dict)):
                formatted = json.dumps(data, indent=2, ensure_ascii=False)
                user_parts.append(f"\nDATA FROM {key}:\n{formatted}")
            else:
                user_parts.append(f"\nDATA FROM {key}:\n{str(data)[:50_000]}")

        if not primary_inlined and primary is not None:
            already = any(data is primary for k, data in inputs.items()
                          if k not in ("primary",))
            if not already:
                if isinstance(primary, list) and len(primary) > 10 and isinstance(primary[0], dict):
                    formatted = json.dumps(primary, indent=2, ensure_ascii=False)
                    if len(formatted) > 2_000_000:
                        compact = self._compact_paper_list(primary)
                        formatted = json.dumps(compact, indent=2, ensure_ascii=False)
                        print(f"    [compact] primary: {len(primary)} items compacted "
                              f"({len(json.dumps(primary)):,} → {len(formatted):,} chars)",
                              file=sys.stderr)
                    user_parts.append(f"\nINPUT DATA:\n{formatted}")
                elif isinstance(primary, (list, dict)):
                    formatted = json.dumps(primary, indent=2, ensure_ascii=False)
                    user_parts.append(f"\nINPUT DATA:\n{formatted}")
                elif isinstance(primary, str):
                    user_parts.append(f"\nINPUT:\n{_truncate_words(primary, self.plan.config.get('max_input_words', 5000))}")

        vf = inputs.get("_validation_feedback") or ""
        if vf:
            user_parts.append(vf)

        return system, "\n".join(user_parts)

    # -- Auto-compress -----------------------------------------------------

    def should_compress(self, step: StepSpec, inputs: dict) -> bool:
        threshold = self.plan.config.get("defaults", {}).get(
            "compress_threshold", 500_000,
        )
        total = 0
        for key, data in inputs.items():
            if key in ("primary", "task"):
                continue
            if isinstance(data, str):
                total += _estimate_tokens(data)
            elif isinstance(data, (list, dict)):
                total += _estimate_tokens(json.dumps(data, ensure_ascii=False))
        return total > threshold

    def compress(self, inputs: dict) -> dict:
        print(f"    [auto-compress] Input exceeds threshold", file=sys.stderr)
        compressed = {}
        for key, data in inputs.items():
            if key in ("primary", "task"):
                compressed[key] = data
                continue
            if isinstance(data, str):
                tokens = _estimate_tokens(data)
            elif isinstance(data, (list, dict)):
                tokens = _estimate_tokens(json.dumps(data, ensure_ascii=False))
            else:
                compressed[key] = data
                continue

            if tokens > 30_000:
                content = json.dumps(data, indent=2, ensure_ascii=False) if isinstance(
                    data, (list, dict)) else data
                summary = self.llm_call(
                    f"Summarize concisely. Preserve all key facts, names, numbers.\n\n{content[:120_000]}",
                    "You are a summarization tool. Compress while preserving key information.",
                    self.light_config,
                    max_tokens=self.light_config.get("max_output_tokens", 131072),
                )
                compressed[key] = summary
                print(f"    [auto-compress] {key}: {tokens} → ~{_estimate_tokens(summary)} tokens",
                      file=sys.stderr)
            else:
                compressed[key] = data

        if "primary" in inputs:
            compressed["primary"] = inputs["primary"]
        return compressed

    # -- LLM call ----------------------------------------------------------

    def llm_call(
        self, prompt: str, system_prompt: str, config: dict,
        max_tokens: int = 131072, retries: int = 1,
    ) -> str:
        client = OpenAI(
            base_url=config.get("base_url", "https://api.deepseek.com/v1"),
            api_key=config.get("api_key", ""),
            timeout=httpx.Timeout(1200.0, connect=30.0),
        )
        kwargs: dict[str, Any] = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if _is_local_vllm_config(config):
            model_limit = _get_vllm_max_model_len(config)
            if model_limit:
                input_est = int(
                    (len(system_prompt) + len(prompt))
                    / CONTEXT_EST_CHARS_PER_TOKEN
                ) + 500
                thinking_budget = REASONING_THINKING_BUDGETS.get(
                    config.get("reasoning_effort", ""), 0)
                available = model_limit - input_est - thinking_budget
                if available < MIN_USABLE_OUTPUT_TOKENS:
                    # A call whose output budget is squeezed below this floor
                    # cannot produce usable structured output — the model
                    # truncates mid-generation (often mid-reasoning, before
                    # any JSON appears) and every retry fails identically.
                    # Fail loudly instead of cascading garbage downstream.
                    raise RuntimeError(
                        f"Input (~{input_est:,} tokens est.) leaves only "
                        f"{max(available, 0):,} of {model_limit:,} context "
                        f"tokens for output (< {MIN_USABLE_OUTPUT_TOKENS}). "
                        f"Refusing a starved LLM call that would truncate "
                        f"mid-generation. Reduce or chunk the step input."
                    )
                capped = max(1024, available)
                if capped < max_tokens:
                    kwargs["max_tokens"] = capped
        _apply_reasoning_controls(kwargs, config)

        for attempt in range(retries + 1):
            try:
                response = client.chat.completions.create(**kwargs)
                if response.usage:
                    backend = "deep" if config.get("model") == self.deep_config.get("model") and config.get("base_url") == self.deep_config.get("base_url") else "light"
                    cache_tokens = getattr(response.usage, "prompt_cache_hit_tokens", 0) or getattr(response.usage, "cache_read_input_tokens", 0) or 0
                    self._token_records.append({
                        "step": self._current_step_name,
                        "backend": backend,
                        "input_tokens": response.usage.prompt_tokens or 0,
                        "output_tokens": response.usage.completion_tokens or 0,
                        "cache_tokens": cache_tokens,
                    })
                if response.choices[0].finish_reason == "length":
                    print(f"    [warning] LLM output truncated at "
                          f"max_tokens={kwargs['max_tokens']} — downstream "
                          f"JSON parse may fail", file=sys.stderr)
                text = response.choices[0].message.content or ""
                text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
                text = re.sub(r'^.*?</think>\s*', '', text, flags=re.DOTALL)
                return text.strip()
            except Exception as e:
                error_text = str(e)
                if "reasoning_effort" in kwargs and "reasoning_effort" in error_text:
                    effort = kwargs["reasoning_effort"]
                    if effort == "xhigh":
                        kwargs["reasoning_effort"] = "high"
                        print("    [warning] reasoning_effort=xhigh rejected; retrying with high",
                              file=sys.stderr)
                        continue
                    kwargs.pop("reasoning_effort", None)
                    print("    [warning] reasoning_effort rejected; retrying without it",
                          file=sys.stderr)
                    continue
                if "extra_body" in kwargs and (
                    "enable_thinking" in error_text
                    or "thinking_budget" in error_text
                    or "extra_body" in error_text
                ):
                    kwargs.pop("extra_body", None)
                    print("    [warning] thinking controls rejected; retrying without them",
                          file=sys.stderr)
                    continue
                print(f"    [warning] LLM call failed (attempt {attempt+1}): {e}",
                      file=sys.stderr)
                if attempt == retries:
                    return ""
        return ""

    # -- Input validation ----------------------------------------------------

    def _validate_step_input(self, step: StepSpec, inputs: dict) -> str | None:
        """Validate step inputs against input_schema. Returns error string or None."""
        if not step.input_schema:
            return None

        schema = step.input_schema
        primary = inputs.get("primary")

        if isinstance(schema, dict):
            if "type" in schema:
                expected_type = schema["type"]
                if expected_type == "object" and not isinstance(primary, dict):
                    return (
                        f"Step '{step.name}' expects object input, "
                        f"got {type(primary).__name__}"
                    )
                if expected_type == "array" and not isinstance(primary, list):
                    return (
                        f"Step '{step.name}' expects array input, "
                        f"got {type(primary).__name__}"
                    )
                if expected_type == "string" and not isinstance(primary, str):
                    return (
                        f"Step '{step.name}' expects string input, "
                        f"got {type(primary).__name__}"
                    )

                required_keys = schema.get("required_keys", [])
                if required_keys and isinstance(primary, dict):
                    missing = [k for k in required_keys if k not in primary]
                    if missing:
                        return (
                            f"Step '{step.name}' input missing required keys: {missing}"
                        )
            else:
                type_map = {"string": str, "object": dict, "array": list}
                errors = []
                for binding_name, expected_type_str in schema.items():
                    if expected_type_str == "any":
                        continue
                    val = inputs.get(binding_name)
                    if val is None:
                        continue
                    expected_cls = type_map.get(expected_type_str)
                    if expected_cls and not isinstance(val, expected_cls):
                        errors.append(
                            f"binding '{binding_name}': expected {expected_type_str}, "
                            f"got {type(val).__name__}"
                        )
                if errors:
                    return f"Step '{step.name}' input type mismatches: {'; '.join(errors)}"

        elif isinstance(schema, str):
            type_map = {"string": str, "object": dict, "array": list}
            expected = type_map.get(schema)
            if expected and not isinstance(primary, expected):
                return (
                    f"Step '{step.name}' expects {schema} input, "
                    f"got {type(primary).__name__}"
                )

        return None

    # -- Output validation ---------------------------------------------------

    _PLANNING_PROSE_PREFIXES = re.compile(
        r"^\s*(let me|i'll|i will|first,?\s+i|okay,?\s|sure,?\s|alright|"
        r"to (begin|start|do this)|my (plan|approach|strategy))",
        re.IGNORECASE,
    )
    _STRUCTURED_INDICATORS = re.compile(
        r"(^##\s|\n##\s|```|\{[\s\S]*\}|^\s*[-*]\s)",
        re.MULTILINE,
    )

    _YAML_TYPE_TO_JSON = {
        "string": "string", "str": "string",
        "integer": "integer", "int": "integer",
        "number": "number", "float": "number",
        "boolean": "boolean", "bool": "boolean",
        "array": "array", "object": "object",
    }

    @staticmethod
    def _extract_required_keys(schema: dict | Any) -> list[str]:
        """Extract top-level required field names from an output_schema dict."""
        if not isinstance(schema, dict):
            return []
        return [k for k in schema.keys() if not str(k).startswith("_")]

    @classmethod
    def _schema_to_jsonschema(cls, schema: dict) -> dict:
        """Convert a YAML output_schema to a JSON Schema dict.

        Handles:
          field: string           → {"type": "string"}
          field: string (desc)    → {"type": "string"}
          field:                  → nested object (recurse)
            sub: string
          field:                  → {"type": "array"}
            type: array
            items:
              sub: string
        """
        properties: dict = {}
        required: list[str] = []

        for key, value in schema.items():
            if str(key).startswith("_"):
                continue
            required.append(key)

            if isinstance(value, str):
                base = value.split("(")[0].strip().split(" ")[0].strip().lower()
                json_type = cls._YAML_TYPE_TO_JSON.get(base)
                if json_type:
                    properties[key] = {"type": json_type}
                else:
                    properties[key] = {}
            elif isinstance(value, dict):
                if "type" in value:
                    base = str(value["type"]).split("(")[0].strip().lower()
                    json_type = cls._YAML_TYPE_TO_JSON.get(base, base)
                    prop: dict = {"type": json_type}
                    if json_type == "array" and "items" in value and isinstance(value["items"], dict):
                        item_schema = cls._schema_to_jsonschema(value["items"])
                        if item_schema.get("properties"):
                            prop["items"] = item_schema
                    properties[key] = prop
                else:
                    properties[key] = cls._schema_to_jsonschema(value)
            else:
                properties[key] = {}

        result: dict = {"type": "object"}
        if required:
            result["required"] = required
        if properties:
            result["properties"] = properties
        return result

    def _validate_step_output(self, output: Any, step: StepSpec) -> str | None:
        """Return a failure reason string, or None if output is acceptable."""
        if isinstance(output, dict) and ("error" in output or output.get("skipped")):
            return None

        op = step.op

        if op in ("Classify", "Synthesize", "Extract") and step.output_schema is not None:
            parsed = None
            if isinstance(output, str):
                text = output.strip()
                if not text:
                    return f"{op} step returned empty output; expected JSON matching output_schema."
                try:
                    parsed = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    if not text.startswith(("{", "[")):
                        return (
                            f"{op} step returned unstructured text instead of JSON. "
                            f"Output starts with: {text[:120]!r}"
                        )
            elif isinstance(output, dict):
                parsed = output
            elif output is None:
                return f"{op} step returned None; expected structured JSON output."

            if isinstance(parsed, dict):
                if _jsonschema is not None:
                    json_schema = self._schema_to_jsonschema(step.output_schema)
                    try:
                        _jsonschema.validate(parsed, json_schema)
                    except _jsonschema.ValidationError as e:
                        path = ".".join(str(p) for p in e.absolute_path) if e.absolute_path else "(root)"
                        return (
                            f"Schema validation failed at '{path}': {e.message}. "
                            f"Required keys: {json_schema.get('required', [])}. "
                            f"Got keys: {sorted(parsed.keys())}"
                        )
                else:
                    required_keys = self._extract_required_keys(step.output_schema)
                    if required_keys:
                        present = set(parsed.keys())
                        missing = [k for k in required_keys if k not in present]
                        if missing:
                            return (
                                f"Missing required fields: {missing}. "
                                f"Schema requires: {required_keys}. "
                                f"Got keys: {sorted(present)}"
                            )

        if op == "Classify" and isinstance(output, str) and not output.strip():
            return "Classify step returned empty string; expected a JSON classification."

        if op in ("Extract", "Translate") and isinstance(output, str) and len(output.strip()) < 20:
            return f"{op} step returned suspiciously short output ({len(output.strip())} chars): {output.strip()!r}"

        if isinstance(output, str) and len(output) > 30:
            text = output.strip()
            if (self._PLANNING_PROSE_PREFIXES.match(text)
                    and not self._STRUCTURED_INDICATORS.search(text[:500])):
                return (
                    "Step returned planning/meta text instead of actual content. "
                    f"Output starts with: {text[:120]!r}"
                )

        return None

    # -- Step execution ----------------------------------------------------

    def execute_step(self, step: StepSpec, inputs: dict) -> Any:
        if (step.op == "Classify"
                and "atomic" in (step.output_schema or {})):
            if self.plan.config.get("_force_atomic"):
                print(f"    [force_atomic] Skipping classify — max recursion depth",
                      file=sys.stderr)
                return {"atomic": True, "rationale": "Forced atomic at max recursion depth",
                        "estimated_files": 1}
            if not self.plan.config.get("_has_scoped_files", False):
                print(f"    [force_decompose] Unscoped task always decomposes",
                      file=sys.stderr)
                return {"atomic": False, "rationale": "Unscoped task always decomposes",
                        "estimated_files": 4}

        if step.op == "Pipeline":
            return self._exec_pipeline(step, inputs)

        if step.op == "Foreach" and step.body:
            return self._exec_foreach_body(step, inputs)

        if step.deterministic:
            return self._exec_deterministic(step, inputs)

        if step.op == "While":
            return self._exec_while(step, inputs)

        validate_cfg = step.validate
        max_retries = validate_cfg.get("max_retries", 2) if validate_cfg.get("enabled", True) else 0

        for attempt in range(1 + max_retries):
            if attempt > 0:
                schema_hint = ""
                if step.output_schema is not None and (
                    "Missing required fields" in failure_reason
                    or "Schema validation failed" in failure_reason
                ):
                    required = self._extract_required_keys(step.output_schema)
                    schema_hint = (
                        f"\nYour output MUST be a JSON object with these top-level keys: "
                        f"{required}. Each field must match the expected type from the schema."
                    )
                feedback = (
                    f"\n\n[RETRY — previous attempt rejected]\n"
                    f"Reason: {failure_reason}\n"
                    f"Please produce well-structured output this time. "
                    f"Do NOT start with planning text or meta-commentary."
                    f"{schema_hint}"
                )
                inputs = {**inputs, "_validation_feedback": feedback}
                step = dataclasses.replace(
                    step,
                    prompt_vars={**step.prompt_vars, "_validation_feedback": feedback},
                )
                print(f"    [retry {attempt}/{max_retries}] {failure_reason[:100]}",
                      file=sys.stderr)

            if step.agentic:
                output = self._exec_agentic(step, inputs)
            elif step.foreach:
                output = self._exec_foreach(step, inputs)
            elif step.batch_size > 0:
                output = self._exec_batched(step, inputs)
            elif self._should_auto_chunk(step, inputs,
                    self.config_for_step(step, input_chars=self._estimate_input_chars(inputs))):
                output = self._exec_chunked(step, inputs)
            else:
                output = self._exec_single(step, inputs)

            failure_reason = self._validate_step_output(output, step)
            if failure_reason is None:
                return output

        print(f"    [warning] Validation failed after {max_retries} retries, "
              f"using last output anyway: {failure_reason[:100]}", file=sys.stderr)
        return output

    @staticmethod
    def _estimate_input_chars(inputs: dict) -> int:
        total = 0
        for v in inputs.values():
            if isinstance(v, str):
                total += len(v)
            elif isinstance(v, (dict, list)):
                total += len(json.dumps(v, ensure_ascii=False))
        return total

    # -- Chunk-map-merge for oversized inputs --------------------------------

    def _get_backend_budget(self, config: dict) -> int:
        """Return max input tokens for a backend, accounting for output and overhead."""
        model = config.get("model", "")
        ctx = None
        if _is_local_vllm_config(config):
            ctx = _get_vllm_max_model_len(config)
        if not ctx:
            ctx = MODEL_CONTEXT_LIMITS.get(model)
        if not ctx:
            ctx = 131_072
        max_out = config.get("max_output_tokens", 32768)
        thinking = REASONING_THINKING_BUDGETS.get(
            config.get("reasoning_effort", ""), 0)
        return max(ctx - max_out - thinking - CONTEXT_SAFETY_MARGIN, 8192)

    def _should_auto_chunk(self, step: StepSpec, inputs: dict, config: dict) -> bool:
        """Check if a step's total input exceeds the backend budget and should be chunked."""
        if step.chunk is False:
            return False
        if step.op not in ("Extract", "Classify", "Translate"):
            return False
        if step.deterministic or step.agentic or step.foreach or step.batch_size:
            return False
        budget_tokens = self._get_backend_budget(config)
        # Same conservative ratio as llm_call's context cap — if the trigger
        # were laxer (e.g. 4 chars/token), an input could skip chunking here
        # and still starve the output budget at call time.
        def _ctx_tokens(v) -> int:
            text = (json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (dict, list)) else str(v))
            return int(len(text) / CONTEXT_EST_CHARS_PER_TOKEN)
        total_input_tokens = sum(
            _ctx_tokens(v)
            for k, v in inputs.items()
            if k != "primary" and isinstance(v, (str, dict, list))
        )
        prompt_overhead_tokens = 1000  # system prompt + schema + template
        total_input_tokens += prompt_overhead_tokens
        primary = inputs.get("primary")
        if primary is not None:
            total_input_tokens += _ctx_tokens(primary)
        return total_input_tokens > budget_tokens

    @staticmethod
    def _split_text_chunks(text: str, max_chars: int, overlap: int = 500) -> list[str]:
        """Split text into chunks at section boundaries with overlap.

        Tries to split at section headers first, then double-newlines,
        then hard-splits at max_chars as a last resort.  Adjacent chunks
        share ``overlap`` trailing/leading characters so items straddling
        a boundary are seen by at least one chunk (dedup in merge handles
        any duplicated extractions).
        """
        if len(text) <= max_chars:
            return [text]

        section_re = re.compile(r'\n(?=#{1,3}\s|\n[A-Z][A-Z ]{3,}\n)')
        parts = section_re.split(text)
        if len(parts) < 2:
            parts = text.split("\n\n")

        raw_chunks: list[str] = []
        current = ""
        for part in parts:
            if len(current) + len(part) + 2 > max_chars and current:
                raw_chunks.append(current.strip())
                current = part
            else:
                current = current + "\n\n" + part if current else part

        if current:
            while len(current) > max_chars:
                cut = current[:max_chars].rfind("\n")
                if cut < max_chars // 2:
                    cut = max_chars
                raw_chunks.append(current[:cut].strip())
                current = current[cut:].strip()
            if current:
                raw_chunks.append(current.strip())

        if not raw_chunks:
            return [text]

        if overlap <= 0 or len(raw_chunks) <= 1:
            return raw_chunks

        chunks: list[str] = [raw_chunks[0]]
        for i in range(1, len(raw_chunks)):
            prefix = raw_chunks[i - 1][-overlap:]
            chunks.append(prefix + "\n\n" + raw_chunks[i])
        return chunks

    @staticmethod
    def _split_list_chunks(items: list, max_chars: int) -> list[list]:
        """Split a list of items into chunks whose JSON serialization fits max_chars."""
        if not items:
            return [items]
        chunks: list[list] = []
        current: list = []
        current_size = 2  # []
        for item in items:
            item_size = len(json.dumps(item, ensure_ascii=False)) + 2
            if current_size + item_size > max_chars and current:
                chunks.append(current)
                current = [item]
                current_size = 2 + item_size
            else:
                current.append(item)
                current_size += item_size
        if current:
            chunks.append(current)
        return chunks or [items]

    def _merge_chunk_results(self, op: str, results: list[Any],
                             output_schema: dict | None) -> Any:
        """Merge chunked results based on op type.

        Extract: deduplicate list fields, merge dicts
        Classify: union of all classification fields
        Translate: concatenate strings or lists
        """
        if not results:
            return {}

        parsed = []
        for r in results:
            if isinstance(r, str):
                from pipeline_utils import extract_json_result
                pr = extract_json_result(r)
                parsed.append(pr.data if pr.is_json else r)
            else:
                parsed.append(r)

        if op == "Translate":
            if all(isinstance(p, str) for p in parsed):
                return "\n\n".join(parsed)
            if all(isinstance(p, list) for p in parsed):
                merged: list = []
                for p in parsed:
                    merged.extend(p)
                return merged

        if op in ("Extract", "Classify"):
            if all(isinstance(p, dict) for p in parsed):
                merged_dict: dict = {}
                for p in parsed:
                    for key, val in p.items():
                        if key not in merged_dict:
                            merged_dict[key] = val
                        elif isinstance(val, list) and isinstance(merged_dict[key], list):
                            seen = set()
                            deduped = list(merged_dict[key])
                            for item in merged_dict[key]:
                                sig = json.dumps(item, sort_keys=True, ensure_ascii=False) if isinstance(item, dict) else str(item)
                                seen.add(sig)
                            for item in val:
                                sig = json.dumps(item, sort_keys=True, ensure_ascii=False) if isinstance(item, dict) else str(item)
                                if sig not in seen:
                                    seen.add(sig)
                                    deduped.append(item)
                            merged_dict[key] = deduped
                        elif isinstance(val, str) and isinstance(merged_dict[key], str):
                            if val and val != merged_dict[key]:
                                merged_dict[key] = merged_dict[key] + "\n\n" + val
                return merged_dict

        return parsed[-1] if parsed else {}

    @staticmethod
    def _find_chunk_target(inputs: dict) -> str | None:
        """Pick the binding to split when auto-chunking: the largest str/list
        input by serialized size. Returns the binding key, or None if no
        input is chunkable.

        Named bindings are preferred over the bare "primary" alias so the
        per-chunk replacement updates the key that build_prompt renders
        (DATA FROM {key}); "primary" is only used when it has no named
        source (e.g. input: [task])."""
        best_key: str | None = None
        best_size = 0
        for k, v in inputs.items():
            if k in ("primary", "_validation_feedback"):
                continue
            if isinstance(v, str):
                size = len(v)
            elif isinstance(v, list):
                size = len(json.dumps(v, ensure_ascii=False))
            else:
                continue
            if size > best_size:
                best_key, best_size = k, size
        if best_key is not None:
            return best_key
        if isinstance(inputs.get("primary"), (str, list)):
            return "primary"
        return None

    @staticmethod
    def _find_primary_source_key(inputs: dict) -> str | None:
        """Find the original input key that 'primary' aliases.

        resolve_inputs() sets inputs["primary"] as a reference to the first
        input binding's data.  This method finds that original key so we can
        exclude it from secondary-input accounting and update it when chunking.
        """
        primary = inputs.get("primary")
        if primary is None:
            return None
        for k, v in inputs.items():
            if k != "primary" and v is primary:
                return k
        return None

    def _exec_chunked(self, step: StepSpec, inputs: dict) -> Any:
        """Split the oversized input binding into chunks, run each through
        _exec_single, then merge results.

        The chunk target is the LARGEST str/list binding, not necessarily the
        primary (first) one. E.g. classify_by_dimension takes
        [extract_claims, merge_all.papers]: primary is a small claims dict
        while the 300K-char papers list rides along as a secondary binding.
        Chunking only the primary silently fell back to a single call whose
        input consumed nearly the whole model context, starving the output
        budget to the 1,024-token floor (output_20260605_170109: 6 calls,
        all truncated mid-reasoning, no JSON ever emitted)."""
        config = self.config_for_step(step, input_chars=self._estimate_input_chars(inputs))
        budget_tokens = self._get_backend_budget(config)

        target_key = self._find_chunk_target(inputs)
        if target_key is None:
            return self._exec_single(step, inputs)
        target = inputs[target_key]
        primary_is_target = inputs.get("primary") is target

        exclude_keys = {"primary", target_key}
        secondary_chars = sum(
            len(json.dumps(v, ensure_ascii=False)) if isinstance(v, (dict, list)) else len(str(v))
            for k, v in inputs.items()
            if k not in exclude_keys
        )
        overhead_chars = secondary_chars + 4000  # prompt template + schema overhead
        max_target_chars = max(
            int(budget_tokens * CONTEXT_EST_CHARS_PER_TOKEN) - overhead_chars,
            8000,
        )

        if isinstance(target, str):
            chunks = self._split_text_chunks(target, max_target_chars)
        else:
            chunks = self._split_list_chunks(target, max_target_chars)

        if len(chunks) <= 1:
            return self._exec_single(step, inputs)

        print(f"    [auto-chunk] {step.name}: splitting '{target_key}' into {len(chunks)} "
              f"chunks (budget {budget_tokens:,} tokens, "
              f"target {_estimate_tokens(json.dumps(target, ensure_ascii=False) if not isinstance(target, str) else target):,} tokens)",
              file=sys.stderr)

        chunk_results = []
        for ci, chunk in enumerate(chunks):
            chunk_inputs = dict(inputs)
            chunk_inputs[target_key] = chunk
            if primary_is_target:
                chunk_inputs["primary"] = chunk
            print(f"    [auto-chunk] chunk {ci + 1}/{len(chunks)} "
                  f"({_estimate_tokens(json.dumps(chunk, ensure_ascii=False) if not isinstance(chunk, str) else chunk):,} tokens)",
                  file=sys.stderr)
            result = self._exec_single(step, chunk_inputs)
            chunk_results.append(result)

        merged = self._merge_chunk_results(step.op, chunk_results, step.output_schema)
        print(f"    [auto-chunk] merged {len(chunk_results)} chunk results",
              file=sys.stderr)
        return merged

    def _exec_single(self, step: StepSpec, inputs: dict) -> Any:
        config = self.config_for_step(step, input_chars=self._estimate_input_chars(inputs))
        if _is_codex_backend(config):
            return self._exec_codex(step, inputs, config, agentic=False)
        if _is_cc_backend(config):
            return self._exec_cc(step, inputs, config, agentic=False)
        if _is_ci_backend(config):
            return self._exec_cc_interactive(step, inputs, config, agentic=False)
        system, user = self.build_prompt(step, inputs)
        max_tok = step.budget.get("max_output_tokens",
                                  config.get("max_output_tokens", 131072))
        text = self.llm_call(user, system, config, max_tokens=max_tok)
        if step.op in ("Extract", "Classify", "Synthesize"):
            result = extract_json_result(text)
            if not result.is_json:
                print(f"    [json-parse] {step.name}: parse failed, retrying LLM call "
                      f"({len(text)} chars)", file=sys.stderr)
                # Retrying the identical prompt at temperature 0 reproduces the
                # identical failure. Qwen in particular front-loads reasoning
                # prose before (or instead of) the JSON — make the retry
                # explicitly forbid that.
                retry_user = (
                    user
                    + "\n\n[FORMAT ERROR] Your previous response was not valid "
                    "JSON. Respond with ONLY the JSON matching the OUTPUT "
                    "SCHEMA above — no reasoning, no preamble, no markdown "
                    "fences. Your first character must be '{' or '['."
                )
                text = self.llm_call(retry_user, system, config, max_tokens=max_tok)
                result = extract_json_result(text)
            if not result.is_json:
                print(f"    [json-parse] {step.name}: all parse strategies failed after retry, "
                      f"returning raw text ({len(text)} chars)", file=sys.stderr)
                debug_path = os.path.join(self.output_dir, f"{step.name}_raw_response.txt")
                with open(debug_path, "w") as _dbg:
                    _dbg.write(text)
                print(f"    [json-parse] raw response saved to {debug_path}", file=sys.stderr)
            elif result.source != "direct":
                print(f"    [json-parse] {step.name}: parsed via {result.source} "
                      f"(warnings: {result.warnings})", file=sys.stderr)
            # str→object coercion: when a step's schema expects an object but the
            # parsed result is a string (e.g. the model emitted a string-wrapped
            # JSON literal "{...}", or json.loads returned a str), unwrap it so
            # downstream input-contract checks see the object, not a str.
            if (_schema_output_type(step.output_schema) == "object"
                    and isinstance(result.data, str)):
                inner = extract_json_result(result.data)
                if inner.is_json and isinstance(inner.data, dict):
                    print(f"    [json-coerce] {step.name}: unwrapped string-wrapped "
                          f"object (via {inner.source})", file=sys.stderr)
                    result = inner
            post_proc = self._step_post_processors.get(step.name)
            if post_proc:
                return post_proc(result.data, inputs.get("primary"))
            return result.data
        return text

    _ADAPTIVE_BATCH_MAX = 48
    _ADAPTIVE_CONTEXT_FILL = 0.65

    def _adaptive_batch_size(self, step: StepSpec, config: dict,
                             primary: list, inputs: dict,
                             primary_source_key: str | None) -> int:
        """Estimate an optimal batch size based on the backend's context window."""
        bs = step.batch_size
        if not step.adaptive_batch or len(primary) == 0:
            return bs

        context_budget = config.get("max_input_tokens", 0)
        if not context_budget:
            if _is_local_vllm_config(config):
                context_budget = _get_vllm_max_model_len(config) or 32768
            else:
                context_budget = 65536

        pilot = primary[:min(bs, len(primary))]
        pilot_inputs = dict(inputs)
        pilot_inputs["primary"] = pilot
        if primary_source_key is not None:
            pilot_inputs[primary_source_key] = pilot
        _, user = self.build_prompt(step, pilot_inputs)
        pilot_tokens = int(len(user) / CONTEXT_EST_CHARS_PER_TOKEN)

        secondary_tokens = max(pilot_tokens - int(len(str(pilot)) / CONTEXT_EST_CHARS_PER_TOKEN), 200)
        tokens_per_item = max((pilot_tokens - secondary_tokens) / max(len(pilot), 1), 50)
        usable = int(context_budget * self._ADAPTIVE_CONTEXT_FILL) - secondary_tokens

        new_bs = max(1, min(int(usable / tokens_per_item), self._ADAPTIVE_BATCH_MAX))

        if new_bs != bs:
            print(f"    [adaptive] batch_size {bs} -> {new_bs} "
                  f"(est. {int(tokens_per_item)} tok/item, "
                  f"budget {context_budget})", file=sys.stderr)
        return new_bs

    def _format_batch_items(self, batch: list) -> list[str]:
        """Format batch items for the prompt."""
        numbered = []
        for j, item in enumerate(batch):
            if isinstance(item, dict):
                parts = [f"[{j+1}]"]
                for k in ("id", "title", "abstract"):
                    if k in item:
                        v = item[k]
                        if isinstance(v, str) and len(v) > 500:
                            v = v[:500] + "..."
                        parts.append(f"{k.upper()}: {v}")
                numbered.append("\n".join(parts))
            else:
                numbered.append(f"[{j+1}] {item}")
        return numbered

    def _run_single_batch(self, step: StepSpec, config: dict, inputs: dict,
                          batch: list, batch_num: int, total_batches: int,
                          primary_source_key: str | None) -> list:
        """Execute a single batch and return its results."""
        print(f"    Batch {batch_num}/{total_batches} ({len(batch)} items)...",
              file=sys.stderr)

        numbered = self._format_batch_items(batch)

        batch_inputs = dict(inputs)
        batch_inputs["primary"] = batch
        if primary_source_key is not None:
            batch_inputs[primary_source_key] = batch
        batch_inputs["_batch_items"] = "\n\n".join(numbered)

        system, user = self.build_prompt(step, batch_inputs)
        user += f"\n\nITEMS TO PROCESS:\n{chr(10).join(numbered)}"

        max_tok = step.budget.get("max_output_tokens",
                                  config.get("max_output_tokens", 131072))
        text = self.llm_call(user, system, config, max_tokens=max_tok)

        try:
            batch_results = extract_json(text)
            flatten_fn = self._batch_post_processors.get("flatten_tiered")
            if flatten_fn and flatten_fn(batch_results, batch):
                return list(batch)
            elif isinstance(batch_results, list):
                for entry in batch_results:
                    if not isinstance(entry, dict):
                        continue
                    idx = entry.get("index", 0) - 1
                    if 0 <= idx < len(batch) and isinstance(batch[idx], dict):
                        for k, v in entry.items():
                            if k != "index":
                                batch[idx][k] = v
                return list(batch)
            else:
                return list(batch)
        except (ValueError, TypeError):
            print(f"    [warning] Batch {batch_num}/{total_batches}: JSON "
                  f"parse failed — {len(batch)} items pass through "
                  f"unmodified", file=sys.stderr)
            return list(batch)

    def _exec_batched(self, step: StepSpec, inputs: dict) -> Any:
        config = self.config_for_step(step)
        primary = inputs.get("primary", [])
        if not isinstance(primary, list):
            return self._exec_single(step, inputs)

        primary_source_key = self._find_primary_source_key(inputs)

        bs = self._adaptive_batch_size(step, config, primary, inputs,
                                       primary_source_key)

        batches = []
        for batch_start in range(0, len(primary), bs):
            batches.append(primary[batch_start:batch_start + bs])
        total_batches = len(batches)

        max_workers = max(1, step.batch_parallel)

        if max_workers <= 1:
            all_results: list = []
            for batch_num, batch in enumerate(batches, 1):
                results = self._run_single_batch(
                    step, config, inputs, batch, batch_num,
                    total_batches, primary_source_key)
                all_results.extend(results)
        else:
            print(f"    [parallel] Running {total_batches} batches with "
                  f"max_workers={max_workers}", file=sys.stderr)
            ordered_results: dict[int, list] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {}
                for batch_num, batch in enumerate(batches, 1):
                    future = executor.submit(
                        self._run_single_batch,
                        step, config, inputs, batch, batch_num,
                        total_batches, primary_source_key)
                    future_to_idx[future] = batch_num

                for future in as_completed(future_to_idx):
                    batch_num = future_to_idx[future]
                    try:
                        ordered_results[batch_num] = future.result()
                    except Exception as e:
                        print(f"    [error] Batch {batch_num}/{total_batches} "
                              f"failed: {e}", file=sys.stderr)
                        ordered_results[batch_num] = list(batches[batch_num - 1])

            all_results = []
            for i in range(1, total_batches + 1):
                all_results.extend(ordered_results.get(i, []))

        return all_results

    def _exec_foreach(self, step: StepSpec, inputs: dict) -> list:
        config = self.config_for_step(step)
        primary = inputs.get("primary", [])

        if isinstance(primary, dict) and "clusters" in primary:
            items = primary["clusters"]
            for cl in items:
                self._normalize_cluster_paper_ids(cl)
        elif isinstance(primary, list):
            items = primary
        else:
            items = [primary]

        results = []
        for i, item in enumerate(items):
            label = ""
            if isinstance(item, dict):
                label = item.get("cluster_label", item.get("theme", item.get("name", f"item {i+1}")))
            print(f"    Foreach {i+1}/{len(items)}: {label}...", file=sys.stderr)

            item_inputs: dict[str, Any] = {"primary": item}
            if "task" in inputs:
                item_inputs["task"] = inputs["task"]

            cluster_ids = set()
            if isinstance(item, dict) and "paper_ids" in item:
                cluster_ids = self._paper_identifier_set(item["paper_ids"])

            for key, data in inputs.items():
                if key not in ("primary", "task"):
                    if cluster_ids and self._looks_like_paper_list(data):
                        filtered = [
                            p for p in data
                            if self._paper_identifier_set(p) & cluster_ids
                        ]
                        print(
                            f"      [{key}] filtered {len(data)} → "
                            f"{len(filtered)} papers for cluster",
                            file=sys.stderr,
                        )
                        item_inputs[key] = filtered
                        continue
                    item_inputs[key] = data

            system, user = self.build_prompt(step, item_inputs)
            max_tok = step.budget.get("max_output_tokens",
                                      config.get("max_output_tokens", 131072))
            text = self.llm_call(user, system, config, max_tokens=max_tok)
            results.append(text)

        return results

    @staticmethod
    def _normalize_cluster_paper_ids(cluster: dict) -> None:
        """If a cluster has 'papers' with full objects instead of 'paper_ids'
        with strings, extract the IDs and set paper_ids."""
        if "paper_ids" in cluster and cluster["paper_ids"]:
            return
        papers = cluster.get("papers", [])
        if papers and isinstance(papers[0], dict):
            cluster["paper_ids"] = [
                p.get("id", p.get("arxiv_id", p.get("title", "")))
                for p in papers
            ]
            del cluster["papers"]

    @staticmethod
    def _compact_paper_list(papers: list[dict]) -> list[dict]:
        """Strip large fields from paper dicts for prompt compaction."""
        _KEEP = {"id", "title", "year", "tier", "relevance_score", "source"}
        compact = []
        for p in papers:
            entry = {k: v for k, v in p.items() if k in _KEEP}
            abstract = p.get("abstract", "")
            if abstract:
                entry["abstract_snippet"] = abstract[:200]
            compact.append(entry)
        return compact

    @staticmethod
    def _looks_like_paper_list(data: Any) -> bool:
        if not isinstance(data, list) or not data:
            return False
        id_fields = {"id", "paper_id", "s2_id", "arxiv_id", "doi"}
        for entry in data[:5]:
            if isinstance(entry, dict) and id_fields & set(entry.keys()):
                return True
        return False

    @staticmethod
    def _paper_identifier_set(value: Any) -> set[str]:
        """Return normalized identifiers for matching papers across steps."""
        values: set[str] = set()

        def add(raw: Any) -> None:
            if raw is None:
                return
            text = str(raw).strip()
            if text:
                values.add(text)

        if isinstance(value, list):
            for item in value:
                values.update(PipelineCompiler._paper_identifier_set(item))
            return values

        if not isinstance(value, dict):
            add(value)
            return values

        add(value.get("id"))
        add(value.get("paper_id"))
        add(value.get("s2_id"))
        add(value.get("arxiv_id"))
        add(value.get("doi"))
        add(value.get("title"))

        if value.get("s2_id"):
            add(f"s2:{value['s2_id']}")
        if value.get("arxiv_id"):
            add(f"arxiv:{value['arxiv_id']}")
        if value.get("doi"):
            add(f"doi:{value['doi']}")

        return values

    def _exec_foreach_body(self, step: StepSpec, inputs: dict) -> list:
        """Execute a Foreach with nested body steps.

        Resolves the iteration array from foreach_over, then for each item
        binds it to foreach_as and runs the body steps sequentially.
        """
        over_ref = step.foreach_over
        as_name = step.foreach_as or "item"
        body_steps = step.body or []

        # Resolve the iteration array from a dotted reference like "plan_changes.changes"
        parts = over_ref.split(".")
        base = parts[0]
        if base not in self.outputs and not self._ensure_output_loaded(base):
            raise ValueError(
                f"Foreach '{step.name}': over reference '{base}' not in outputs"
            )
        items = self.outputs[base]
        for attr in parts[1:]:
            if isinstance(items, dict):
                items = items[attr]
            else:
                raise ValueError(
                    f"Foreach '{step.name}': cannot traverse '{attr}' on {type(items).__name__}"
                )

        if not isinstance(items, list):
            raise ValueError(
                f"Foreach step '{step.name}': 'over' resolved to "
                f"{type(items).__name__}, expected list"
            )

        print(f"    [Foreach] {len(items)} items over '{over_ref}', "
              f"{len(body_steps)} body steps", file=sys.stderr)

        all_results = []
        for i, item in enumerate(items):
            label = ""
            if isinstance(item, dict):
                label = item.get("file", item.get("name", ""))
            print(f"\n    [Foreach {i+1}/{len(items)}] {label}", file=sys.stderr)

            # Bind the current item so body steps can reference it
            self.outputs[as_name] = item

            # Track body step outputs for cross-reference within this iteration
            body_outputs = {}
            iteration_result = {}

            for bs in body_steps:
                prefixed = dataclasses.replace(bs, name=f"{step.name}_i{i}_{bs.name}")

                # Resolve inputs for this body step: can reference as_name,
                # other body steps (by unprefixed name), outer steps, or "task"
                body_inputs: dict[str, Any] = {}
                for j, binding in enumerate(prefixed.input):
                    binding = binding.strip()
                    if binding == "task":
                        body_inputs["task"] = self.task_text
                        if j == 0:
                            body_inputs["primary"] = self.task_text
                        continue

                    ref_base = binding.split("[")[0].split(".")[0].strip()
                    ref_rest = binding[len(ref_base):]

                    # Check body-local outputs first, then foreach var, then outer
                    if ref_base in body_outputs:
                        data = body_outputs[ref_base]
                    elif ref_base == as_name:
                        data = self.outputs[as_name]
                    elif ref_base in self.outputs:
                        data = self.outputs[ref_base]
                    else:
                        print(f"      [warning] Body input '{ref_base}' not available",
                              file=sys.stderr)
                        continue

                    # Traverse dotted sub-paths (e.g. "change_entry.file")
                    if ref_rest.startswith("."):
                        for attr in ref_rest[1:].split("."):
                            if isinstance(data, dict) and attr in data:
                                data = data[attr]

                    body_inputs[ref_base] = data
                    if j == 0:
                        body_inputs["primary"] = data

                if not body_inputs:
                    body_inputs = {"primary": item, "task": self.task_text}

                print(f"      [{prefixed.name}] op={bs.op}", file=sys.stderr)

                # Validate body step inputs against schema (warn, don't abort)
                if bs.input_schema:
                    err = self._validate_step_input(bs, body_inputs)
                    if err:
                        print(f"      [input-warn] {bs.name}: {err}", file=sys.stderr)

                try:
                    # execute_step's internal resolve_inputs finds body outputs
                    # in self.outputs by unprefixed name; the compile-time
                    # collision check in _validate_plan_edges ensures safety.
                    output = self.execute_step(prefixed, body_inputs)
                except Exception as e:
                    print(f"      [ERROR] {prefixed.name}: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    output = {"error": str(e)}

                body_outputs[bs.name] = output
                self.outputs[bs.name] = output
                self.save_step(prefixed.name, output)
                iteration_result[bs.name] = output

            all_results.append(iteration_result)

            # Clean up body step outputs from self.outputs between iterations
            for bs in body_steps:
                self.outputs.pop(bs.name, None)

        # Clean up the foreach variable
        self.outputs.pop(as_name, None)

        return all_results

    def _exec_agentic(self, step: StepSpec, inputs: dict) -> Any:
        config = self.config_for_step(step)
        if _is_codex_backend(config):
            return self._exec_codex(step, inputs, config, agentic=True)
        if _is_cc_backend(config):
            return self._exec_cc(step, inputs, config, agentic=True)
        if _is_ci_backend(config):
            return self._exec_cc_interactive(step, inputs, config, agentic=True)
        system, user = self.build_prompt(step, inputs)
        budget = step.budget or {}
        tool_definitions, tool_dispatcher = self._resolve_tools_for_step(step)

        soft_limit = budget.get("soft_limit", 500_000)
        max_out = budget.get("max_output_tokens",
                             config.get("max_output_tokens", 131072))
        model_name = config["model"]
        ctx_limit = MODEL_CONTEXT_LIMITS.get(model_name)
        if ctx_limit:
            safe_limit = max(ctx_limit - max_out - CONTEXT_SAFETY_MARGIN, 0)
            if soft_limit > safe_limit:
                print(f"    [cap] soft_limit {soft_limit:,} → {safe_limit:,} "
                      f"(model {model_name} ctx={ctx_limit:,})", file=sys.stderr)
                soft_limit = safe_limit

        final_text = run_agent(
            prompt=user,
            system_prompt=system,
            model=model_name,
            base_url=config.get("base_url", "https://api.deepseek.com/v1"),
            api_key=config.get("api_key", ""),
            max_output_tokens=max_out,
            max_iters=budget.get("max_iterations", 30),
            soft_limit=soft_limit,
            thinking_budget=config.get("thinking_budget", 0),
            reasoning_effort=config.get("reasoning_effort"),
            http_timeout=config.get("agent_timeout", 900),
            step_name=step.name,
            stream_events=True,
            output_dir=self.output_dir,
            tool_definitions=tool_definitions,
            tool_dispatcher=tool_dispatcher,
        )
        usage = agent_loop.last_usage
        if usage.get("input_tokens") or usage.get("output_tokens"):
            backend = "deep" if config is self.deep_config else "light"
            self._token_records.append({
                "step": self._current_step_name,
                "backend": backend,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_tokens": 0,
            })
        if not isinstance(final_text, str):
            print(f"    [warning] run_agent returned {type(final_text).__name__}, "
                  f"coercing to str", file=sys.stderr)
            if isinstance(final_text, list):
                final_text = "\n".join(str(x) for x in final_text)
            else:
                final_text = str(final_text)
        if step.op in ("Extract", "Classify", "Synthesize"):
            return extract_json(final_text)
        return final_text

    # -- Pipeline (sub-pipeline) execution ------------------------------------

    def _resolve_plan_path(self, ref: str) -> str:
        """Resolve a plan reference to an absolute file path.

        Tries: relative to plan_dir, relative to plans/ sibling of engine/,
        then absolute.
        """
        if os.path.isabs(ref):
            return ref
        if not ref.endswith(".yaml"):
            ref = ref + ".yaml"

        candidates = []
        if self.plan_dir:
            candidates.append(os.path.join(self.plan_dir, ref))
        engine_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(engine_dir)
        candidates.append(os.path.join(repo_root, "plans", ref))
        candidates.append(os.path.join(repo_root, ref))

        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"Pipeline plan '{ref}' not found. Searched: {candidates}"
        )

    def _exec_pipeline(self, step: StepSpec, inputs: dict) -> Any:
        """Execute a sub-pipeline by loading a referenced plan and running it
        in a child PipelineCompiler.

        The child inherits backends, tool providers, prompt library, and
        handler modules from the parent.  Depth is tracked via config._depth
        and enforced against max_recursion_depth.
        """
        plan_ref = step.pipeline_ref
        if not plan_ref:
            raise ValueError(f"Pipeline step '{step.name}' has no 'plan' reference")

        plan_path = self._resolve_plan_path(plan_ref)
        child_plan = load_plan(plan_path)

        current_depth = self.plan.config.get("_depth", 0)
        max_depth = self.plan.config.get("max_recursion_depth", 3)
        child_depth = current_depth + 1

        if child_depth > max_depth:
            print(f"    [Pipeline] Max depth {max_depth} reached — forcing atomic",
                  file=sys.stderr)
            child_plan.config["_force_atomic"] = True

        # Build child task text from the Pipeline step's primary input. If the
        # caller explicitly passes task, preserve the historical raw-task
        # behavior; otherwise a sub-pipeline should see the bound input as its
        # task payload.
        if "task" in inputs:
            task_text = inputs["task"]
        elif "primary" in inputs:
            primary = inputs["primary"]
            if isinstance(primary, str):
                task_text = primary
            else:
                task_text = json.dumps(primary, indent=2, ensure_ascii=False)
        else:
            task_text = self.task_text

        # Detect change_entry (from Foreach) by explicit key name
        change_entry = inputs.get("change_entry")
        if change_entry and not (isinstance(change_entry, dict)
                                 and "file" in change_entry
                                 and "change" in change_entry):
            change_entry = None

        if change_entry:
            task_text = (
                f"SUBTASK: {change_entry.get('change', '')}\n"
                f"FILE: {change_entry.get('file', '')}\n"
                f"LOCATION: {change_entry.get('location', '')}\n"
                f"RATIONALE: {change_entry.get('rationale', '')}"
            )
            scoped_files = [change_entry["file"]]
            if change_entry.get("read_context"):
                scoped_files.extend(change_entry["read_context"])
            child_plan.config["target_files"] = scoped_files

        # Propagate key config from parent
        for key in ("codebase_root", "handler_module", "max_recursion_depth",
                     "context_escalation_threshold", "test_context",
                     "test_context_path", "test_command"):
            if key in self.plan.config and key not in child_plan.config:
                child_plan.config[key] = self.plan.config[key]

        # Propagate backend configs if child doesn't define its own
        if "light_backend" not in child_plan.config:
            child_plan.config["light_backend"] = self.plan.config.get("light_backend", {})
        if "deep_backend" not in child_plan.config:
            child_plan.config["deep_backend"] = self.plan.config.get("deep_backend", {})

        child_plan.config["_depth"] = child_depth
        child_plan.config["_has_scoped_files"] = bool(change_entry)
        if self._deadline:
            child_plan.config["_deadline"] = self._deadline

        # Propagate escalation override from parent While-body retries
        escalation_override = inputs.get("_escalation_override")
        if escalation_override:
            child_plan.config["_escalation_override"] = escalation_override
            print(f"    [Pipeline] propagating escalation to child: {escalation_override}",
                  file=sys.stderr)

        # Build output directory name — include file basename when available
        dir_suffix = uuid.uuid4().hex[:6]
        if change_entry and change_entry.get("file"):
            file_stem = os.path.splitext(os.path.basename(change_entry["file"]))[0]
            file_stem = re.sub(r'[^a-zA-Z0-9_-]', '_', file_stem)
            dir_name = f"{step.name}_d{child_depth}_{file_stem}_{uuid.uuid4().hex[:4]}"
        else:
            dir_name = f"{step.name}_d{child_depth}_{dir_suffix}"

        child_output = os.path.join(
            self.output_dir,
            dir_name
        )
        os.makedirs(child_output, exist_ok=True)

        child = PipelineCompiler(
            plan=child_plan,
            task_text=task_text,
            output_dir=child_output,
            light_config=self.light_config,
            deep_config=self.deep_config,
            tool_providers=self.tool_providers,
            prompt_library=self.prompt_library,
            plan_dir=os.path.dirname(plan_path),
        )

        # Propagate parent outputs the child might need (e.g. extract_intent)
        for key, val in inputs.items():
            if key in ("task", "primary", "_validation_feedback",
                        "_cached_inner_outputs", "_escalation_override"):
                continue
            child.outputs[key] = val

        # Inject cached gather/assemble/classify from a prior retry round.
        # These go into _precached_steps (not outputs) so run() can distinguish
        # "skip this step" from "parent propagated this value for consumption."
        cached = inputs.get("_cached_inner_outputs", {})
        if cached:
            for cache_key, cache_val in cached.items():
                child._precached_steps[cache_key] = cache_val
            print(f"    [Pipeline] pre-cached {list(cached.keys())} — will skip those steps",
                  file=sys.stderr)

        print(f"    [Pipeline] d{child_depth} → {plan_ref} "
              f"(task: {len(task_text)} chars, "
              f"files: {child_plan.config.get('target_files', [])})",
              file=sys.stderr)

        try:
            child.run()
        except Exception as e:
            print(f"    [Pipeline] d{child_depth} FAILED: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            self._token_records.extend(child._token_records)
            return {
                "_pipeline_depth": child_depth,
                "_pipeline_plan": plan_ref,
                "_error": str(e),
            }

        self._token_records.extend(child._token_records)
        result = dict(child.outputs)
        result["_pipeline_depth"] = child_depth
        result["_pipeline_plan"] = plan_ref
        return result

    # -- While-with-body execution -------------------------------------------

    def _exec_while_body(self, step: StepSpec, inputs: dict) -> Any:
        """Execute a While loop with a multi-step body.

        Each round: body steps → classify. On RETRY, the planner gets
        verify feedback + the current diff (changes stay on disk) and
        produces a corrective plan. Backend escalates on retries.
        Returns the last round's body outputs as a dict.
        """
        ws = step.while_spec
        if not ws:
            raise ValueError(f"While step '{step.name}' has no while_spec")

        body_steps = step.body or []
        max_rounds = ws.get("max_rounds", 3)
        classify_cfg = ws.get("classify", {})
        classify_backend = classify_cfg.get("backend", "light")
        classify_prompt_ref = classify_cfg.get("prompt", "")
        classify_schema = classify_cfg.get("output_schema")

        print(f"    [While-body] {len(body_steps)} body steps, max {max_rounds} rounds",
              file=sys.stderr)

        retry_history: list[dict] = []
        last_body_outputs: dict = {}
        cached_inner_outputs: dict = {}

        escalation_rules = ws.get("escalation", [])
        parent_escalation = self.plan.config.get("_escalation_override")

        for round_num in range(1, max_rounds + 1):
            escalation = None
            for rule in escalation_rules:
                if round_num >= rule.get("round", 999):
                    escalation = rule
            # Parent escalation override applies from round 1 as a floor
            if parent_escalation and not escalation:
                escalation = parent_escalation
            if escalation:
                print(f"    [While-body] escalation for round {round_num}: {escalation}",
                      file=sys.stderr)

            print(f"\n    [While-body round {round_num}/{max_rounds}]", file=sys.stderr)

            effective_task = self.task_text
            if retry_history:
                feedback_lines = []
                for r in retry_history:
                    feedback_lines.append(f"Round {r['round']} REJECTED: {r['rationale']}")
                    if r.get("current_diff"):
                        diff_preview = r["current_diff"][:2000]
                        feedback_lines.append(f"  Diff produced:\n{diff_preview}")
                effective_task = (
                    "--- PRIOR ATTEMPT(S) REJECTED BY VERIFIER ---\n"
                    + "\n".join(feedback_lines)
                    + "\n--- END PRIOR ATTEMPTS ---\n\n"
                    + self.task_text
                )

            # --- Phase 1: Execute body steps sequentially ---
            body_outputs: dict = {}
            for bs in body_steps:
                prefixed = dataclasses.replace(bs, name=f"{step.name}_r{round_num}_{bs.name}")

                body_inputs: dict[str, Any] = {}
                for j, binding in enumerate(bs.input):
                    binding = binding.strip()
                    if binding == "task":
                        body_inputs["task"] = effective_task
                        if j == 0:
                            body_inputs["primary"] = effective_task
                        continue

                    ref_base = binding.split("[")[0].split(".")[0].strip()
                    ref_rest = binding[len(ref_base):]

                    if ref_base in body_outputs:
                        data = body_outputs[ref_base]
                    elif ref_base in self.outputs:
                        data = self.outputs[ref_base]
                    else:
                        print(f"      [warning] Body input '{ref_base}' not available",
                              file=sys.stderr)
                        continue

                    if ref_rest.startswith("."):
                        resolved_ok = True
                        for attr in ref_rest[1:].split("."):
                            if isinstance(data, dict) and attr in data:
                                data = data[attr]
                            else:
                                print(f"      [warning] Cannot resolve '{binding}': "
                                      f"'{attr}' not found in {ref_base}",
                                      file=sys.stderr)
                                resolved_ok = False
                                break
                        if not resolved_ok:
                            continue

                    body_inputs[binding] = data
                    if j == 0:
                        body_inputs["primary"] = data

                if not body_inputs:
                    body_inputs = {"primary": effective_task, "task": effective_task}

                # Inject retry history only into LLM-facing steps (the planner)
                if retry_history and bs.op in ("Synthesize", "Translate", "Extract"):
                    body_inputs["_retry_history"] = retry_history

                print(f"      [{prefixed.name}] op={bs.op}", file=sys.stderr)

                if bs.input_schema:
                    err = self._validate_step_input(bs, body_inputs)
                    if err:
                        print(f"      [input-warn] {bs.name}: {err}", file=sys.stderr)

                # On retry rounds, inject cached gather outputs so inner pipeline skips re-gather
                if cached_inner_outputs and retry_history and bs.op == "Pipeline":
                    body_inputs["_cached_inner_outputs"] = cached_inner_outputs
                    print(f"      [injecting cached gather → skip re-gather]", file=sys.stderr)

                try:
                    # Apply retry escalation to body steps
                    if escalation:
                        if escalation.get("use_deep"):
                            prefixed = dataclasses.replace(prefixed, backend_override="deep")
                        if escalation.get("reasoning_effort"):
                            prefixed = dataclasses.replace(
                                prefixed, reasoning_effort=escalation["reasoning_effort"])
                        # Propagate escalation into Pipeline child config so
                        # child While loops (gather/extract + edit) also escalate
                        if bs.op == "Pipeline":
                            body_inputs["_escalation_override"] = escalation
                    output = self.execute_step(prefixed, body_inputs)
                except Exception as e:
                    print(f"      [ERROR] {prefixed.name}: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    output = {"error": str(e)}

                body_outputs[bs.name] = output
                self.outputs[bs.name] = output
                self.save_step(prefixed.name, output)

            last_body_outputs = body_outputs

            # Cache gather/assemble/classify_atomic from the inner pipeline after round 1.
            # On retry rounds, these get injected so the inner pipeline skips re-gather.
            # Note: cached gather context describes the pre-edit codebase state;
            # retry_history carries post-edit state (verifier rationale + current diff).
            if not cached_inner_outputs:
                for bs_name, bs_output in body_outputs.items():
                    if not isinstance(bs_output, dict):
                        continue
                    for cache_key in ("gather_context", "assemble_context", "context_summary", "classify_atomic"):
                        if cache_key in bs_output:
                            cached_inner_outputs[cache_key] = bs_output[cache_key]
                if cached_inner_outputs:
                    print(f"    [While-body] cached {list(cached_inner_outputs.keys())} "
                          f"for retry rounds (skip re-gather)", file=sys.stderr)

            # --- Syntax gate: catch compile errors before expensive LLM verify ---
            collect_data = body_outputs.get("collect_results", {})
            if isinstance(collect_data, dict) and collect_data.get("files_changed"):
                codebase_root = self.plan.config.get("codebase_root", ".")
                syntax_errors = []
                for rel_path in collect_data["files_changed"]:
                    if not rel_path.endswith(".py"):
                        continue
                    abs_path = os.path.join(codebase_root, rel_path)
                    if not os.path.exists(abs_path):
                        continue
                    try:
                        with open(abs_path, "r", encoding="utf-8", errors="replace") as sf:
                            ast.parse(sf.read(), filename=rel_path)
                    except SyntaxError as se:
                        syntax_errors.append(
                            f"{rel_path}:{se.lineno}: {se.msg}"
                            + (f" — {se.text.rstrip()}" if se.text else ""))
                if syntax_errors:
                    err_list = "\n".join(f"  - {e}" for e in syntax_errors)
                    classify_result = {
                        "sufficient": False,
                        "rationale": f"SYNTAX ERRORS in modified files (compile check failed):\n{err_list}",
                        "missing": [f"Fix syntax error: {e}" for e in syntax_errors],
                    }
                    self.save_step(f"{step.name}_classify_r{round_num}", classify_result)
                    print(f"    [While-body] SYNTAX GATE FAILED — skipping verifier LLM call",
                          file=sys.stderr)
                    for e in syntax_errors:
                        print(f"      {e}", file=sys.stderr)

                    retry_entry = {
                        "round": round_num,
                        "rationale": classify_result["rationale"],
                        "missing": classify_result["missing"],
                        "current_diff": collect_data.get("cumulative_diff", ""),
                        "files_changed": collect_data.get("files_changed", []),
                        "previous_plan": body_outputs.get("plan_changes", {}),
                    }
                    retry_history.append(retry_entry)
                    if round_num < max_rounds:
                        print(f"    [While-body] RETRY — looping to round {round_num + 1} "
                              f"(syntax error, changes preserved on disk)", file=sys.stderr)
                    continue

            # --- Test execution gate: run tests if available ---
            test_output = ""
            test_context_path = self.plan.config.get("test_context_path", "")
            codebase_root_for_test = self.plan.config.get("codebase_root", ".")
            if test_context_path and os.path.isfile(test_context_path) and os.path.isdir(codebase_root_for_test):
                try:
                    tc = self.plan.config.get("test_context", "")
                    test_files = [m.group(1) for m in re.finditer(r'^\+\+\+ b/(.+\.py)$', tc, re.M)]
                    # Always apply — patch may add new tests to existing files
                    subprocess.run(
                        ["git", "apply", test_context_path],
                        cwd=codebase_root_for_test, capture_output=True, text=True, timeout=10,
                    )
                    # Extract test function names from added lines
                    test_names = []
                    for line in tc.splitlines():
                        if line.startswith('+') and not line.startswith('+++'):
                            m = re.match(r'\+\s+(?:async\s+)?def (test_\w+)\(', line)
                            if m:
                                test_names.append(m.group(1))
                    custom_test_cmd = self.plan.config.get("test_command")
                    if custom_test_cmd:
                        cmd = shlex.split(custom_test_cmd)
                    elif test_files:
                        venv_python = os.path.join(codebase_root_for_test, ".venv", "bin", "python3")
                        pytest_cmd = venv_python if os.path.isfile(venv_python) else sys.executable
                        cmd = [pytest_cmd, "-m", "pytest", "-x", "--tb=short", "-q"]
                        if test_names:
                            cmd += ["-k", " or ".join(test_names)]
                        cmd += test_files
                    else:
                        cmd = None
                    if cmd:
                        test_proc = subprocess.run(
                            cmd,
                            cwd=codebase_root_for_test, capture_output=True, text=True, timeout=120,
                        )
                        if test_proc.returncode == 0:
                            print(f"    [test-gate] ALL TESTS PASSED — skipping LLM verify",
                                  file=sys.stderr)
                            classify_result = {
                                "sufficient": True, "missing": [],
                                "rationale": "All tests passed (pytest exit code 0).",
                            }
                            self.save_step(f"{step.name}_classify_r{round_num}", classify_result)
                            break
                        else:
                            test_output = (test_proc.stdout + test_proc.stderr)[-5000:]
                            print(f"    [test-gate] Tests FAILED (exit {test_proc.returncode}) — "
                                  f"feeding output to LLM verifier", file=sys.stderr)
                except Exception as e:
                    print(f"    [test-gate] Error running tests: {e}", file=sys.stderr)

            # --- Phase 2: Classify (verify) ---
            problem_text = inputs.get("primary") or inputs.get("task", "")
            if isinstance(problem_text, (dict, list)):
                problem_text = json.dumps(problem_text, indent=2, ensure_ascii=False)

            classify_prompt_raw = classify_prompt_ref
            ref = PromptLibrary.reference_name(classify_prompt_ref)
            if ref and self.prompt_library:
                classify_prompt_raw = self.prompt_library.resolve(ref, {
                    "toolset": "",
                    "_validation_feedback": "",
                })

            classify_user = classify_prompt_raw.strip()
            classify_user += f"\n\nPROBLEM STATEMENT:\n{problem_text}"

            # Inject extract_intent if available (gives verifier the structured intent)
            intent_data = self.outputs.get("extract_intent") or inputs.get("extract_intent")
            if intent_data:
                formatted_intent = intent_data
                if isinstance(formatted_intent, (dict, list)):
                    formatted_intent = json.dumps(formatted_intent, indent=2, ensure_ascii=False)
                classify_user += f"\n\nINTENT ANALYSIS:\n{formatted_intent}"

            # Inject test expectations if available in config
            test_context = self.plan.config.get("test_context", "")
            if not test_context:
                test_patch = self.plan.config.get("test_patch", "")
                if test_patch:
                    test_context = test_patch
            if test_context:
                if len(test_context) > 20000:
                    test_context = test_context[:20000] + "\n[truncated]"
                classify_user += f"\n\nTEST EXPECTATIONS:\n{test_context}"

            if test_output:
                classify_user += (f"\n\nTEST EXECUTION RESULTS (pytest FAILED):\n"
                                  f"{test_output}\n\n"
                                  f"The tests above FAILED. You MUST return sufficient=false "
                                  f"and describe what needs to change to make them pass.")

            # Only pass the collect_results output (diff summary) if available,
            # otherwise fall back to the last body step's output. The diff is what
            # verify actually needs — not the raw iteration artifacts from all steps.
            collect_key = next(
                (k for k in body_outputs if "collect" in k.lower()), None)
            verify_data = {collect_key: body_outputs[collect_key]} if collect_key else {
                list(body_outputs.keys())[-1]: list(body_outputs.values())[-1]
            } if body_outputs else {}
            for bname, bdata in verify_data.items():
                formatted = bdata
                if isinstance(formatted, (dict, list)):
                    formatted = json.dumps(formatted, indent=2, ensure_ascii=False)
                if isinstance(formatted, str) and len(formatted) > 50000:
                    formatted = formatted[:50000] + "\n[truncated]"
                classify_user += f"\n\nDATA FROM {bname}:\n{formatted}"

            if retry_history:
                history_text = json.dumps(retry_history, indent=2, ensure_ascii=False)
                if len(history_text) > 10000:
                    history_text = history_text[-10000:]
                classify_user += f"\n\nPRIOR RETRY HISTORY:\n{history_text}"

            if classify_schema:
                schema_text = yaml.safe_dump(classify_schema, sort_keys=False).strip()
                classify_user += (
                    f"\n\nOUTPUT SCHEMA:\nReturn valid JSON matching this schema.\n"
                    f"```yaml\n{schema_text}\n```"
                )

            classify_system = (
                "You are a classification tool. Apply the given criteria consistently. "
                "Output valid JSON only — no preamble, no commentary. " + MATH_INSTRUCTION
            )

            # Classify always uses deep on retries
            if escalation and classify_backend == "light":
                c_config = self.deep_config
            else:
                c_config = self.light_config if classify_backend == "light" else self.deep_config
            max_tok = classify_cfg.get("max_output_tokens",
                                       c_config.get("max_output_tokens", 16384))
            classify_text = self.llm_call(classify_user, classify_system, c_config,
                                          max_tokens=max_tok)
            classify_result = extract_json(classify_text)

            if not isinstance(classify_result, dict):
                print(f"    [While-body] classify returned non-dict: {type(classify_result).__name__}",
                      file=sys.stderr)
                classify_result = {"sufficient": False, "rationale": str(classify_result)}

            sufficient = classify_result.get("sufficient", False)
            rationale = classify_result.get("rationale", "")
            missing = classify_result.get("missing", [])

            self.save_step(f"{step.name}_classify_r{round_num}", classify_result)
            print(f"    [While-body] classify: sufficient={sufficient}", file=sys.stderr)
            if rationale:
                print(f"    [While-body] rationale: {rationale[:200]}", file=sys.stderr)

            if sufficient:
                print(f"    [While-body] PASS after round {round_num}", file=sys.stderr)
                break

            # Record this round for the next iteration — changes stay on disk
            retry_entry = {
                "round": round_num,
                "rationale": rationale,
                "missing": missing,
            }
            collect_data = body_outputs.get("collect_results", {})
            if isinstance(collect_data, dict):
                retry_entry["current_diff"] = collect_data.get("cumulative_diff", "")
                retry_entry["files_changed"] = collect_data.get("files_changed", [])
                retry_entry["previous_plan"] = body_outputs.get("plan_changes", {})
            retry_history.append(retry_entry)

            if round_num < max_rounds:
                print(f"    [While-body] RETRY — looping to round {round_num + 1} "
                      f"(changes preserved on disk)", file=sys.stderr)

        # Clean up body step outputs from self.outputs
        for bs in body_steps:
            self.outputs.pop(bs.name, None)

        result = dict(last_body_outputs)
        result["_verify"] = classify_result
        result["_rounds"] = round_num
        return result

    # -- While loop execution -----------------------------------------------

    def _exec_while(self, step: StepSpec, inputs: dict) -> Any:
        """Execute a While loop.

        Two modes:
        - Standard (no body): classify → translate → execute per round.
          Returns concatenated evidence from all rounds.
        - With body: execute body steps → classify per round.
          Returns the last round's body outputs as a dict.
        """
        if step.body:
            return self._exec_while_body(step, inputs)

        ws = step.while_spec
        if not ws:
            raise ValueError(f"While step '{step.name}' has no while_spec")

        # --- Gather vs Extract: scope-aware config selection ---
        # Use extract mode when the parent Pipeline call provided scoped files
        # (i.e., we're inside a Foreach with a specific change_entry).
        # Use gather mode for the initial discovery call even at depth > 0.
        current_depth = self.plan.config.get("_depth", 0)
        has_scoped_files = self.plan.config.get("_has_scoped_files", False)
        use_extract = (has_scoped_files
                       and ws.get("extract_classify")
                       and ws.get("extract_translate"))

        if use_extract:
            extract_mr = ws.get("extract_max_rounds")
            max_rounds = extract_mr if extract_mr is not None else ws.get("max_rounds", 3)
            classify_cfg = ws["extract_classify"]
            translate_cfg = ws["extract_translate"]
            toolset = ws.get("extract_toolset") or ws.get("toolset", [])
            print(f"    [While] depth={current_depth} → EXTRACT mode "
                  f"(scoped_files=True, max_rounds={max_rounds}, tools={toolset})",
                  file=sys.stderr)
        else:
            max_rounds = ws.get("max_rounds", 3)
            classify_cfg = ws.get("classify", {})
            translate_cfg = ws.get("translate", {})
            toolset = ws.get("toolset", [])
            if current_depth > 0:
                print(f"    [While] depth={current_depth} → GATHER mode "
                      f"(scoped_files={has_scoped_files})", file=sys.stderr)

        execute_cfg = ws.get("execute", {})

        classify_backend = classify_cfg.get("backend", "light")
        translate_backend = translate_cfg.get("backend", "light")
        classify_prompt_ref = classify_cfg.get("prompt", "")
        translate_prompt_ref = translate_cfg.get("prompt", "")
        classify_schema = classify_cfg.get("output_schema")
        translate_schema = translate_cfg.get("output_schema")
        execute_handler_name = execute_cfg.get("handler", "")

        execute_handler = self._handlers.get(execute_handler_name)
        if not execute_handler:
            raise ValueError(
                f"While step '{step.name}': unknown handler '{execute_handler_name}'. "
                f"Available: {list(self._handlers.keys())}"
            )

        toolset_description = ", ".join(toolset) if toolset else "(none)"
        all_evidence: list[dict] = []
        problem_text = inputs.get("primary") or inputs.get("task", "")
        if isinstance(problem_text, (dict, list)):
            problem_text = json.dumps(problem_text, indent=2, ensure_ascii=False)

        refresh_handler_name = ws.get("refresh_handler", "")
        refresh_handler = self._handlers.get(refresh_handler_name) if refresh_handler_name else None

        for round_num in range(1, max_rounds + 1):
            print(f"    [While round {round_num}/{max_rounds}]", file=sys.stderr)

            # --- Phase 0: Refresh context (if configured) ---
            if refresh_handler:
                refresh_inputs = dict(inputs)
                refresh_inputs["primary"] = inputs.get("primary") or inputs.get("task", "")
                refreshed = refresh_handler(refresh_inputs, self)
                if isinstance(refreshed, str) and refreshed:
                    problem_text = refreshed
                    print(f"    [While] refresh_handler updated problem_text "
                          f"({len(problem_text)} chars)", file=sys.stderr)
                elif isinstance(refreshed, dict):
                    problem_text = json.dumps(refreshed, indent=2, ensure_ascii=False)
                    print(f"    [While] refresh_handler updated problem_text "
                          f"({len(problem_text)} chars)", file=sys.stderr)

            # --- Phase 1: Classify ---
            max_evidence_chars = ws.get("max_evidence_chars", 50000)
            evidence_text = ""
            if all_evidence:
                evidence_text = json.dumps(all_evidence, indent=2, ensure_ascii=False)
                if len(evidence_text) > max_evidence_chars:
                    evidence_text = (
                        f"[evidence truncated — showing last {max_evidence_chars} chars]\n"
                        + evidence_text[-max_evidence_chars:]
                    )

            classify_prompt_raw = classify_prompt_ref
            ref = PromptLibrary.reference_name(classify_prompt_ref)
            if ref and self.prompt_library:
                classify_prompt_raw = self.prompt_library.resolve(ref, {
                    "toolset": toolset_description,
                    "_validation_feedback": "",
                })

            classify_user = classify_prompt_raw.strip()

            # Structured formatting: if primary input is an assemble_context dict
            # (has 'files' key), separate background context from extracted evidence
            # so the classifier can distinguish what this loop collected vs. what
            # was already available from a prior gather phase.
            raw_primary = inputs.get("primary") or inputs.get("task", "")
            if (isinstance(raw_primary, dict) and "files" in raw_primary
                    and classify_cfg.get("separate_background", False)):
                task_text = raw_primary.get("task", "")
                if isinstance(task_text, (dict, list)):
                    task_text = json.dumps(task_text, indent=2, ensure_ascii=False)
                classify_user += f"\n\nTASK DESCRIPTION:\n{task_text}"

                bg_files = raw_primary.get("files", {})
                if isinstance(bg_files, dict):
                    bg_summary = []
                    for fpath in sorted(bg_files.keys()):
                        content = bg_files[fpath]
                        nlines = content.count("\n") + 1 if isinstance(content, str) else 0
                        bg_summary.append(f"  - {fpath} ({nlines} lines)")
                    bg_text = "\n".join(bg_summary) if bg_summary else "(none)"
                else:
                    bg_text = str(bg_files)[:2000]
                classify_user += (
                    f"\n\nBACKGROUND FILE LIST (from prior gather — "
                    f"for orientation only, NOT extracted evidence):\n{bg_text}"
                )
            else:
                classify_user += f"\n\nPROBLEM STATEMENT:\n{problem_text}"

            n_items = len(all_evidence)
            if evidence_text:
                classify_user += (
                    f"\n\nEXTRACTED EVIDENCE ({n_items} items collected by this loop):"
                    f"\n{evidence_text}"
                )
            else:
                classify_user += (
                    f"\n\nEXTRACTED EVIDENCE (0 items collected by this loop):\n"
                    f"(none yet — this loop has not collected any evidence. "
                    f"You MUST return sufficient=false and request targeted reads.)"
                )

            test_context = self.plan.config.get("test_context", "")
            if not test_context:
                test_context = self.plan.config.get("test_patch", "")
            if test_context:
                tc = test_context[:20000] + ("\n[truncated]" if len(test_context) > 20000 else "")
                classify_user += f"\n\nTEST EXPECTATIONS:\n{tc}"

            classify_system = (
                "You are a classification tool. Apply the given criteria consistently. "
                "Output valid JSON only — no preamble, no commentary. " + MATH_INSTRUCTION
            )

            if classify_schema:
                schema_text = yaml.safe_dump(classify_schema, sort_keys=False).strip()
                classify_user += (
                    f"\n\nOUTPUT SCHEMA:\nReturn valid JSON matching this schema.\n"
                    f"```yaml\n{schema_text}\n```"
                )

            config = self.light_config if classify_backend == "light" else self.deep_config
            max_tok = classify_cfg.get("max_output_tokens",
                                       config.get("max_output_tokens", 16384))
            classify_text = self.llm_call(classify_user, classify_system, config, max_tokens=max_tok)
            classify_result = extract_json(classify_text)

            if not isinstance(classify_result, dict):
                print(f"    [While] classify returned non-dict: {type(classify_result).__name__}",
                      file=sys.stderr)
                classify_result = {"sufficient": False, "rationale": str(classify_result)}

            if classify_schema and _jsonschema is not None:
                json_schema = self._schema_to_jsonschema(classify_schema)
                try:
                    _jsonschema.validate(classify_result, json_schema)
                except _jsonschema.ValidationError as e:
                    print(f"    [While] classify schema error: {e.message}", file=sys.stderr)

            sufficient = classify_result.get("sufficient", False)
            rationale = classify_result.get("rationale", "")

            self.save_step(f"{step.name}_classify_r{round_num}", classify_result)
            print(f"    [While] classify: sufficient={sufficient}", file=sys.stderr)
            if rationale:
                print(f"    [While] rationale: {rationale[:200]}", file=sys.stderr)

            min_rounds = ws.get("min_rounds", 2)
            if sufficient and round_num >= min_rounds:
                print(f"    [While] Evidence sufficient after {round_num - 1} rounds",
                      file=sys.stderr)
                break

            # --- Phase 2: Translate ---
            translate_prompt_raw = translate_prompt_ref
            ref = PromptLibrary.reference_name(translate_prompt_ref)
            if ref and self.prompt_library:
                translate_prompt_raw = self.prompt_library.resolve(ref, {
                    "toolset": toolset_description,
                    "_validation_feedback": "",
                })

            translate_user = translate_prompt_raw.strip()
            if (isinstance(raw_primary, dict) and "files" in raw_primary
                    and classify_cfg.get("separate_background", False)):
                task_text_t = raw_primary.get("task", "")
                if isinstance(task_text_t, (dict, list)):
                    task_text_t = json.dumps(task_text_t, indent=2, ensure_ascii=False)
                translate_user += f"\n\nTASK DESCRIPTION:\n{task_text_t}"
                bg_files_t = raw_primary.get("files", {})
                if isinstance(bg_files_t, dict):
                    bg_lines = []
                    for fpath in sorted(bg_files_t.keys()):
                        content = bg_files_t[fpath]
                        nlines = content.count("\n") + 1 if isinstance(content, str) else 0
                        bg_lines.append(f"  - {fpath} ({nlines} lines)")
                    bg_text_t = "\n".join(bg_lines) if bg_lines else "(none)"
                else:
                    bg_text_t = str(bg_files_t)[:2000]
                translate_user += f"\n\nFILES AVAILABLE FOR READING:\n{bg_text_t}"
            else:
                translate_user += f"\n\nPROBLEM STATEMENT:\n{problem_text}"
            translate_user += f"\n\nRATIONALE (what information is needed):\n{rationale}"
            if toolset:
                translate_user += f"\n\nAVAILABLE TOOLS:\n{toolset_description}"
            if evidence_text:
                translate_user += f"\n\nEVIDENCE ALREADY COLLECTED:\n{evidence_text}"

            translate_system = ws.get("translate_system", "")
            if not translate_system:
                translate_system = (
                    "You are a transformation tool. Convert the rationale into concrete "
                    "tool call requests. Output valid JSON only — no preamble. " + MATH_INSTRUCTION
                )
            else:
                translate_system += " " + MATH_INSTRUCTION

            if translate_schema:
                schema_text = yaml.safe_dump(translate_schema, sort_keys=False).strip()
                translate_user += (
                    f"\n\nOUTPUT SCHEMA:\nReturn valid JSON matching this schema.\n"
                    f"```yaml\n{schema_text}\n```"
                )

            escalation_rules = ws.get("escalation", [])
            round_escalation = None
            for rule in escalation_rules:
                if round_num >= rule.get("round", 999):
                    round_escalation = rule
            # Parent escalation override applies from round 1 as a floor
            parent_esc = self.plan.config.get("_escalation_override")
            if parent_esc and not round_escalation:
                round_escalation = parent_esc

            if round_escalation and round_escalation.get("use_deep"):
                translate_config = dict(self.deep_config)
                if round_escalation.get("reasoning_effort"):
                    translate_config["reasoning_effort"] = round_escalation["reasoning_effort"]
                print(f"    [While] escalation round {round_num}: deep"
                      f"{'/' + round_escalation['reasoning_effort'] if round_escalation.get('reasoning_effort') else ''}",
                      file=sys.stderr)
            else:
                translate_config = self.light_config if translate_backend == "light" else self.deep_config

            max_tok = translate_cfg.get("max_output_tokens",
                                        translate_config.get("max_output_tokens", 16384))
            translate_text = self.llm_call(translate_user, translate_system, translate_config, max_tokens=max_tok)
            translate_result = extract_json(translate_text)

            if not isinstance(translate_result, dict):
                if isinstance(translate_result, list):
                    translate_result = {"requests": translate_result}
                else:
                    print(f"    [While] translate returned non-dict: {type(translate_result).__name__}",
                          file=sys.stderr)
                    translate_result = {"requests": []}

            requests = translate_result.get("requests", [])
            self.save_step(f"{step.name}_translate_r{round_num}", translate_result)
            print(f"    [While] translate: {len(requests)} requests", file=sys.stderr)

            if not requests:
                print(f"    [While] No requests produced — stopping", file=sys.stderr)
                break

            # --- Phase 3: Execute (deterministic) ---
            execute_inputs = {
                "primary": requests,
                "toolset": toolset,
                "_rationale": rationale,
                "_translate_prompt": translate_prompt_raw,
                "_translate_backend": translate_backend,
            }
            round_evidence = execute_handler(execute_inputs, self)

            if isinstance(round_evidence, list):
                all_evidence.extend(round_evidence)
            elif isinstance(round_evidence, dict):
                all_evidence.append(round_evidence)
            elif isinstance(round_evidence, str):
                all_evidence.append({"content": round_evidence})

            self.save_step(f"{step.name}_execute_r{round_num}", round_evidence)
            print(f"    [While] execute: {len(round_evidence) if isinstance(round_evidence, list) else 1} "
                  f"evidence items (total: {len(all_evidence)})", file=sys.stderr)

        return all_evidence

    # Canonical source: mesh/cli/mesh_tool.py:AGENT_ROUTED_TOOLS
    # This superset adds memory/map tools that also route via the agent socket.
    _SOCKET_ROUTED_TOOLS = frozenset({
        "send_message", "mesh_status", "agent_status",
        "channel_list", "channel_members", "schedule_list",
        "gmail_send_message", "gmail_reply_to", "account_set_current",
        "memory_search", "memory_get", "memory_list", "memory_add",
        "todo_list", "todo_add", "todo_update", "todo_toggle",
        "todo_remove", "todo_reorder",
        "map_get", "map_list", "map_edit", "map_create", "set_project_context",
    })

    def _preflight_mesh_socket_for_step(self, step: StepSpec, proc_env: dict[str, str]) -> None:
        """Warn (but don't block) when an agentic step's tool list includes
        mesh-routed tools but the socket isn't reachable.  Individual tool
        calls will fail gracefully via the circuit breaker in mesh_tool.py."""
        if not any(tool in self._SOCKET_ROUTED_TOOLS for tool in step.tools):
            return

        socket_path = proc_env.get("MESH_SOCKET_PATH", "")
        node_id = proc_env.get("MESH_NODE_ID", "")
        home = proc_env.get("HOME", "")

        def _warn(reason: str) -> None:
            print(f"    [socket-preflight] WARNING: {reason} "
                  f"(MESH_NODE_ID={node_id or '(not set)'}, "
                  f"MESH_SOCKET_PATH={socket_path or '(not set)'}, "
                  f"HOME={home or '(not set)'}). "
                  f"Socket-routed tools will fail individually if invoked.",
                  file=sys.stderr)

        if not node_id:
            _warn("MESH_NODE_ID is not set")
            return
        if not socket_path:
            _warn("MESH_SOCKET_PATH is not set")
            return
        if not os.path.exists(socket_path):
            _warn(f"MESH_SOCKET_PATH does not exist ({socket_path})")
            return
        if not stat.S_ISSOCK(os.stat(socket_path).st_mode):
            _warn(f"MESH_SOCKET_PATH is not a Unix socket ({socket_path})")
            return

        try:
            transport = httpx.HTTPTransport(uds=socket_path)
            with httpx.Client(transport=transport, timeout=5.0) as client:
                resp = client.get("http://localhost/tools")
                resp.raise_for_status()
        except Exception as exc:
            _warn(f"/tools request failed ({exc}; socket={socket_path})")

    def _exec_codex(self, step: StepSpec, inputs: dict, config: dict,
                    *, agentic: bool = True) -> Any:
        """Execute a step via Codex CLI subprocess.

        Args:
            agentic: If True, full tool access (shell, files). If False,
                     sandbox mode — text-only output, no tools.
        """
        model_name = config["model"]
        system, user = self.build_prompt(step, inputs)
        prompt = f"{system}\n\n{user}"

        codex_bin = config.get("codex_binary") or shutil.which("codex") or "codex"

        cmd = [codex_bin, "exec", "-", "-m", model_name, "--ephemeral", "--json"]
        if agentic:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            cmd.extend(["--sandbox", "read-only"])

        budget = step.budget or {}
        timeout = budget.get("agent_timeout", 900)

        real_home = pwd.getpwuid(os.getuid()).pw_dir
        proc_env = dict(os.environ)
        proc_env["HOME"] = real_home
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        proc_env["PATH"] = (
            repo_root + ":"
            + os.path.join(real_home, ".local/share/node-v22/bin") + ":"
            + os.path.join(real_home, ".local/bin") + ":"
            + proc_env.get("PATH", "")
        )
        if agentic:
            self._preflight_mesh_socket_for_step(step, proc_env)

        print(f"    [codex] {codex_bin} exec -m {model_name} "
              f"({'agentic' if agentic else 'sandbox'}) "
              f"step={step.name}", file=sys.stderr)

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=proc_env,
            start_new_session=True,
        )

        try:
            proc.stdin.write(prompt.encode("utf-8"))
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

        text_blocks: list[str] = []
        error_messages: list[str] = []

        try:
            for raw_line in proc.stdout:
                line_str = raw_line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue
                try:
                    event = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")

                if event_type == "agent_message":
                    message = event.get("message", "")
                    if isinstance(message, str) and message.strip():
                        text_blocks.append(message.strip())

                elif event_type in ("item.completed", "item_completed"):
                    item = event.get("item", {})
                    item_type = item.get("type")
                    if item_type in (None, "agent_message"):
                        text = item.get("text", "")
                        if isinstance(text, str) and text.strip():
                            text_blocks.append(text.strip())

                elif event_type == "error":
                    msg = event.get("message", str(event))
                    error_messages.append(msg)
                    print(f"    [codex] error event: {msg[:300]}",
                          file=sys.stderr)

                elif event_type == "turn.failed":
                    err = event.get("error", {})
                    msg = err.get("message", "") if isinstance(err, dict) else str(err)
                    if msg:
                        error_messages.append(msg)
                    print(f"    [codex] turn.failed: {msg[:300]}",
                          file=sys.stderr)

            proc.wait(timeout=timeout)

            if proc.returncode != 0:
                stderr_data = proc.stderr.read().decode("utf-8", errors="replace").strip()
                detail = "; ".join(error_messages) or stderr_data or "(no detail)"
                print(f"    [codex] exit {proc.returncode}: {detail[:500]}",
                      file=sys.stderr)
                partial_note = (
                    f" (partial output: {len(text_blocks)} blocks, "
                    f"{sum(len(b) for b in text_blocks)} chars discarded)"
                    if text_blocks else ""
                )
                raise RuntimeError(
                    f"Codex failed (exit {proc.returncode}): "
                    f"{detail[:500]}{partial_note}"
                )

        except subprocess.TimeoutExpired:
            print(f"    [codex] timeout after {timeout}s, killing", file=sys.stderr)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            partial_note = (
                f" (partial output: {len(text_blocks)} blocks, "
                f"{sum(len(b) for b in text_blocks)} chars discarded)"
                if text_blocks else ""
            )
            raise RuntimeError(
                f"Codex subprocess timed out after {timeout}s{partial_note}"
            )

        finally:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass

        final_text = "\n\n".join(text_blocks)
        print(f"    [codex] done: {len(final_text)} chars, "
              f"{len(text_blocks)} blocks", file=sys.stderr)

        if step.op in ("Extract", "Classify", "Synthesize"):
            return extract_json(final_text)
        return final_text

    def _exec_cc(self, step: StepSpec, inputs: dict, config: dict,
                  *, agentic: bool = True) -> Any:
        """Execute a step via Claude Code CLI subprocess (claude -p).

        Args:
            agentic: If True, full tool access. If False, read-only tools
                     only. Pipe mode always uses --dangerously-skip-permissions;
                     read-only mode constrains it with --allowedTools.
        """
        model_name = config["model"]
        system, user = self.build_prompt(step, inputs)
        prompt = f"{system}\n\n{user}"

        cc_bin = config.get("cc_binary") or shutil.which("claude") or "claude"
        cc_effort = config.get("cc_effort", "xhigh")

        cmd = [
            cc_bin, "-p",
            "--model", model_name,
            "--output-format", "stream-json",
            "--verbose",
            "--no-session-persistence",
        ]
        if cc_effort:
            cmd.extend(["--effort", cc_effort])
        cmd.append("--dangerously-skip-permissions")
        if not agentic:
            cmd.extend(["--tools", ""])

        budget = step.budget or {}
        timeout = budget.get("agent_timeout", 900)

        real_home = pwd.getpwuid(os.getuid()).pw_dir
        proc_env = dict(os.environ)
        proc_env["HOME"] = real_home
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        proc_env["PATH"] = (
            repo_root + ":"
            + os.path.join(real_home, ".local/share/node-v22/bin") + ":"
            + os.path.join(real_home, ".local/bin") + ":"
            + proc_env.get("PATH", "")
        )
        if agentic:
            self._preflight_mesh_socket_for_step(step, proc_env)

        print(f"    [cc] {cc_bin} -p --model {model_name} --effort {cc_effort} "
              f"({'agentic' if agentic else 'no-tools'}) "
              f"step={step.name}", file=sys.stderr)

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=proc_env,
            start_new_session=True,
        )

        try:
            proc.stdin.write(prompt.encode("utf-8"))
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

        text_blocks: list[str] = []
        error_messages: list[str] = []
        result_text: str | None = None

        timeout_fired = threading.Event()
        cleanup_lock = threading.Lock()

        def partial_output_note() -> str:
            return (
                f" (partial output: {len(text_blocks)} blocks, "
                f"{sum(len(b) for b in text_blocks)} chars discarded)"
                if text_blocks else ""
            )

        def terminate_process_group() -> None:
            with cleanup_lock:
                if proc.poll() is not None:
                    return
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass

        def on_stdout_timeout() -> None:
            timeout_fired.set()
            print(f"    [cc] timeout after {timeout}s, killing", file=sys.stderr)
            terminate_process_group()

        stdout_watchdog = threading.Timer(timeout, on_stdout_timeout)
        stdout_watchdog.daemon = True
        stdout_watchdog.start()

        try:
            for raw_line in proc.stdout:
                line_str = raw_line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue
                try:
                    event = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")

                if event_type == "assistant":
                    message = event.get("message", {})
                    if isinstance(message, dict):
                        for block in message.get("content", []):
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "")
                                if text.strip():
                                    text_blocks.append(text.strip())

                elif event_type == "result":
                    if event.get("is_error"):
                        msg = event.get("result", str(event))
                        error_messages.append(str(msg))
                        print(f"    [cc] error result: {str(msg)[:300]}",
                              file=sys.stderr)
                    else:
                        result_text = event.get("result", "")

                elif event_type == "error":
                    msg = event.get("error", event.get("message", str(event)))
                    error_messages.append(str(msg))
                    print(f"    [cc] error event: {str(msg)[:300]}",
                          file=sys.stderr)

            if not timeout_fired.is_set():
                stdout_watchdog.cancel()
            proc.wait(timeout=timeout)

            if timeout_fired.is_set():
                raise RuntimeError(
                    f"Claude Code subprocess timed out after {timeout}s"
                    f"{partial_output_note()}"
                )

            if proc.returncode != 0:
                stderr_data = proc.stderr.read().decode("utf-8", errors="replace").strip()
                detail = "; ".join(error_messages) or stderr_data or "(no detail)"
                print(f"    [cc] exit {proc.returncode}: {detail[:500]}",
                      file=sys.stderr)
                raise RuntimeError(
                    f"Claude Code failed (exit {proc.returncode}): "
                    f"{detail[:500]}{partial_output_note()}"
                )

        except subprocess.TimeoutExpired:
            print(f"    [cc] timeout after {timeout}s, killing", file=sys.stderr)
            terminate_process_group()
            raise RuntimeError(
                f"Claude Code subprocess timed out after {timeout}s"
                f"{partial_output_note()}"
            )

        finally:
            stdout_watchdog.cancel()
            terminate_process_group()

        if result_text is not None:
            final_text = result_text
        else:
            final_text = "\n\n".join(text_blocks)

        print(f"    [cc] done: {len(final_text)} chars, "
              f"{len(text_blocks)} blocks", file=sys.stderr)

        if step.op in ("Extract", "Classify", "Synthesize"):
            return extract_json(final_text)
        return final_text

    def _exec_cc_interactive(self, step: StepSpec, inputs: dict, config: dict,
                             *, agentic: bool = True) -> Any:
        """Execute a step via claude_interactive.py (tmux-based interactive wrapper).

        All steps in one compiler instance share a tmux session for context reuse.
        """
        model_name = config["model"]
        system, user = self.build_prompt(step, inputs)
        prompt = f"{system}\n\n{user}"

        ci_bin = config.get("ci_binary") or ""
        ci_script = config.get("ci_script") or shutil.which("claude_interactive.py") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "claude_interactive.py",
        )

        budget = step.budget or {}
        timeout = budget.get("agent_timeout", 900)
        effort = config.get("cc_effort", "high")
        permission = "agentic" if agentic else "restricted"

        if not hasattr(self, "_ci_session_name"):
            self._ci_session_name = f"ci-{uuid.uuid4().hex[:8]}"

        is_last_step = (
            step.name == self.plan.steps[-1].name
            if self.plan and self.plan.steps else False
        )

        cmd = [sys.executable, ci_script]
        if model_name:
            cmd.extend(["--model", model_name])
        if effort:
            cmd.extend(["--effort", effort])
        cmd.extend(["--permission-mode", permission])
        cmd.extend(["--timeout", str(timeout)])
        cmd.extend(["--session-name", self._ci_session_name])
        if ci_bin:
            cmd.extend(["--cc-binary", ci_bin])
        cmd.extend(["--working-dir", self.output_dir])
        if not is_last_step:
            cmd.append("--keep-session")

        real_home = pwd.getpwuid(os.getuid()).pw_dir
        proc_env = dict(os.environ)
        proc_env["HOME"] = real_home
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        proc_env["PATH"] = (
            repo_root + ":"
            + os.path.join(real_home, ".local/share/node-v22/bin") + ":"
            + os.path.join(real_home, ".local/bin") + ":"
            + proc_env.get("PATH", "")
        )
        if agentic:
            self._preflight_mesh_socket_for_step(step, proc_env)

        print(f"    [ci] claude_interactive --model {model_name} --effort {effort} "
              f"--permission-mode {permission} "
              f"step={step.name} session={self._ci_session_name}",
              file=sys.stderr)

        try:
            proc = subprocess.run(
                cmd,
                input=prompt.encode("utf-8"),
                capture_output=True,
                env=proc_env,
                timeout=timeout + 30,
            )
        except (subprocess.TimeoutExpired, Exception):
            subprocess.run(
                ["tmux", "kill-session", "-t", self._ci_session_name],
                capture_output=True, timeout=5,
            )
            raise

        stdout_text = proc.stdout.decode("utf-8", errors="replace").strip()
        stderr_text = proc.stderr.decode("utf-8", errors="replace").strip()

        if proc.returncode == 2:
            subprocess.run(
                ["tmux", "kill-session", "-t", self._ci_session_name],
                capture_output=True, timeout=5,
            )
            raise RuntimeError(
                f"Claude interactive timed out after {timeout}s "
                f"(step={step.name})"
            )
        elif proc.returncode != 0:
            subprocess.run(
                ["tmux", "kill-session", "-t", self._ci_session_name],
                capture_output=True, timeout=5,
            )
            detail = stderr_text[-500:] if stderr_text else "(no detail)"
            raise RuntimeError(
                f"Claude interactive failed (exit {proc.returncode}): "
                f"{detail}"
            )

        print(f"    [ci] done: {len(stdout_text)} chars", file=sys.stderr)

        if step.op in ("Extract", "Classify", "Synthesize"):
            return extract_json(stdout_text)
        return stdout_text

    def _exec_deterministic(self, step: StepSpec, inputs: dict) -> Any:
        handler = self._handlers.get(step.handler)
        if not handler:
            raise ValueError(
                f"Unknown handler: {step.handler}. "
                f"Available: {list(self._handlers.keys())}. "
                f"Did the plan specify handler_module in its config?"
            )
        try:
            return handler(inputs, self)
        except ContractError as e:
            print(f"    [CONTRACT VIOLATION] step '{step.name}': {e}",
                  file=sys.stderr)
            raise ValueError(
                f"Handler contract violation in step '{step.name}': {e}"
            ) from e

    # -- Output management -------------------------------------------------

    def save_step(self, name: str, data: Any):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{name}.json")
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved {path}", file=sys.stderr)

    def load_step(self, name: str) -> Any:
        path = os.path.join(self.output_dir, f"{name}.json")
        with open(path) as f:
            return json.load(f)

    def step_cached(self, name: str) -> bool:
        return os.path.exists(os.path.join(self.output_dir, f"{name}.json"))

    def _resolve_cache_from(self, path: str) -> str | None:
        """Resolve a step's cache_from path. Tries the path as given
        (absolute or CWD-relative), then relative to the pipeline root so
        plans work regardless of invocation directory."""
        candidates = [path]
        if not os.path.isabs(path):
            pipeline_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidates.append(os.path.join(pipeline_root, path))
        for cand in candidates:
            if os.path.isfile(cand):
                return cand
        return None

    def _ensure_output_loaded(self, name: str) -> bool:
        """Load a cached step output into memory if it is available on disk."""
        if name in self.outputs:
            return True
        if not self.step_cached(name):
            return False
        self.outputs[name] = self.load_step(name)
        print(f"    [cache] Loaded {name}.json", file=sys.stderr)
        return True

    def _resolve_step_index(self, selector: str | int) -> int:
        """Resolve a --step selector to a 1-based step index.

        Numeric selectors are 1-based to match --resume. Selector 0 is accepted
        as an alias for the first step so either common indexing convention can
        start a step-by-step smoke cleanly.
        """
        raw = str(selector).strip()
        if not raw:
            raise ValueError("--step requires a step name or numeric index")
        if raw.isdigit():
            idx = int(raw)
            if idx == 0:
                return 1
            if 1 <= idx <= len(self.plan.steps):
                return idx
            raise ValueError(
                f"Step index {idx} out of range; plan has {len(self.plan.steps)} steps"
            )
        for i, step in enumerate(self.plan.steps, 1):
            if step.name == raw:
                return i
        names = ", ".join(step.name for step in self.plan.steps)
        raise ValueError(f"Unknown step '{raw}'. Available steps: {names}")

    # -- Gate evaluation ----------------------------------------------------

    def _evaluate_gate(self, step: StepSpec) -> bool:
        """Evaluate a Gate condition against current outputs.

        Supports:
          step[pred].length > 0    — filter + length check
          step.field == value      — field-level comparison (bool, int, string)
        Connectors: OR, AND
        """
        condition = step.condition.strip()
        if not condition:
            return True

        # Split into clauses on OR / AND (AND binds tighter)
        or_groups = [g.strip() for g in re.split(r'\bOR\b', condition)]
        for or_group in or_groups:
            and_clauses = [c.strip() for c in re.split(r'\bAND\b', or_group)]
            group_pass = True
            for clause in and_clauses:
                if not self._evaluate_clause(clause):
                    group_pass = False
                    break
            if group_pass:
                return True
        return False

    def _evaluate_clause(self, clause: str) -> bool:
        """Evaluate a single clause.

        Supported forms:
          step_name[pred].length OP N    — filter + length check (existing)
          step_name.field OP value       — field-level comparison (new)
        """
        # --- Form 1: step_name[pred].length OP N ---
        m = re.match(
            r'^(\w+)(\[([^\]]*)\])?\.length\s*(>|>=|==|<|<=|!=)\s*(\d+)$',
            clause.strip()
        )
        if m:
            step_name = m.group(1)
            predicates_str = m.group(3)
            cmp_op = m.group(4)
            cmp_val = int(m.group(5))

            self._ensure_output_loaded(step_name)
            data = self.outputs.get(step_name)
            if data is None:
                print(f"    [GATE] No output for step {step_name!r}", file=sys.stderr)
                return False

            if predicates_str:
                _, predicates = parse_binding(f"{step_name}[{predicates_str}]")
                filtered = apply_filter(data, predicates)
            else:
                filtered = data if isinstance(data, list) else [data]

            length = len(filtered) if isinstance(filtered, list) else 0

            ops = {'>': lambda a, b: a > b, '>=': lambda a, b: a >= b,
                   '==': lambda a, b: a == b, '<': lambda a, b: a < b,
                   '<=': lambda a, b: a <= b, '!=': lambda a, b: a != b}
            return ops[cmp_op](length, cmp_val)

        # --- Form 2: step_name.field OP value ---
        m2 = re.match(
            r'^(\w+)\.(\w+)\s*(==|!=)\s*(.+)$',
            clause.strip()
        )
        if m2:
            step_name = m2.group(1)
            field_name = m2.group(2)
            cmp_op = m2.group(3)
            raw_value = m2.group(4).strip()

            self._ensure_output_loaded(step_name)
            data = self.outputs.get(step_name)
            if data is None:
                print(f"    [GATE] No output for step {step_name!r}", file=sys.stderr)
                return False

            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except (json.JSONDecodeError, ValueError):
                    print(f"    [GATE] Output for {step_name!r} is not a dict", file=sys.stderr)
                    return False

            if not isinstance(data, dict):
                print(f"    [GATE] Output for {step_name!r} is not a dict", file=sys.stderr)
                return False

            actual = data.get(field_name)
            if raw_value.lower() == "true":
                expected: Any = True
            elif raw_value.lower() == "false":
                expected = False
            elif raw_value.lower() == "null" or raw_value.lower() == "none":
                expected = None
            elif raw_value.isdigit():
                expected = int(raw_value)
            else:
                expected = raw_value.strip("'\"")

            if cmp_op == "==":
                return actual == expected
            else:
                return actual != expected

        print(f"    [GATE] Cannot parse clause: {clause!r}", file=sys.stderr)
        return False

    # -- Dependency graph ----------------------------------------------------

    def _build_dep_graph(self, steps: list[StepSpec]) -> dict[str, set[str]]:
        """Build {step_name: set(upstream_step_names)} from input bindings.

        Also adds implicit Gate→next-step edges: a Gate step makes
        the immediately following step (by declaration order) depend on it,
        since the gate decides whether to skip that step.
        """
        step_names = {s.name for s in steps}
        deps: dict[str, set[str]] = {}
        for idx, step in enumerate(steps):
            upstream: set[str] = set()
            for binding in step.input:
                binding = binding.strip()
                if "." in binding and "[" not in binding:
                    base = binding.split(".")[0]
                else:
                    base = binding.split("[")[0].split(".")[0].strip()
                if base == "task":
                    continue
                if base in step_names:
                    upstream.add(base)
            # Gate condition references
            if step.op == "Gate" and step.condition:
                for m in re.finditer(r'\b(\w+)(?:\[|\.|\.length)', step.condition):
                    ref = m.group(1)
                    if ref in step_names:
                        upstream.add(ref)
            deps[step.name] = upstream

        # Implicit Gate edges: a Gate always blocks the next declared step.
        for idx, step in enumerate(steps):
            if step.op == "Gate" and idx + 1 < len(steps):
                next_name = steps[idx + 1].name
                deps[next_name].add(step.name)

        return deps

    def _print_dag_schedule(self, steps: list[StepSpec],
                            deps: dict[str, set[str]],
                            max_concurrency: int) -> None:
        """Print the parallel execution schedule without running anything."""
        remaining = {s.name for s in steps}
        completed: set[str] = set()
        step_idx = {s.name: i + 1 for i, s in enumerate(steps)}
        total = len(steps)
        wave = 0
        while remaining:
            ready = [
                name for name in remaining
                if deps.get(name, set()).issubset(completed)
            ]
            if not ready:
                print(f"  [DAG ERROR] Circular dependency among: {remaining}",
                      file=sys.stderr)
                break
            ready.sort(key=lambda n: step_idx[n])
            wave += 1
            batch = ready[:max_concurrency]
            overflow = ready[max_concurrency:]
            indices = ", ".join(f"{step_idx[n]}:{n}" for n in batch)
            print(f"  Wave {wave}: [{len(batch)} step(s)] {indices}",
                  file=sys.stderr)
            for name in batch:
                remaining.discard(name)
                completed.add(name)
            # overflow stays in remaining for next wave
        print(f"  Total waves: {wave} (max_concurrency={max_concurrency})",
              file=sys.stderr)

    # -- Main loop ---------------------------------------------------------

    def run(self, resume_from: int = 1, step_only: str | int | None = None,
            max_concurrency: int | None = None):
        steps = self.plan.steps
        total_steps = len(steps)
        target_index: int | None = None

        if max_concurrency is None:
            max_concurrency = self.plan.config.get("max_concurrency", 4)
        max_concurrency = max(1, max_concurrency)

        if step_only is not None:
            target_index = self._resolve_step_index(step_only)
            resume_from = target_index
            run_steps = steps[:target_index]
        else:
            run_steps = steps

        print(f"\nPipeline: {self.plan.name}", file=sys.stderr)
        print(f"  Steps: {total_steps}", file=sys.stderr)
        print(f"  Output: {self.output_dir}", file=sys.stderr)
        print(f"  Task input: {_estimate_tokens(self.task_text):,} est. tokens",
              file=sys.stderr)
        if target_index is not None:
            target = steps[target_index - 1]
            print(
                f"  Single-step mode: executing step {target_index}/{total_steps}: "
                f"{target.name}",
                file=sys.stderr,
            )
        if resume_from > 1:
            print(f"  Resuming from step {resume_from}", file=sys.stderr)
        if max_concurrency > 1:
            print(f"  Parallel: max_concurrency={max_concurrency}", file=sys.stderr)
        print(f"  Light: {self.light_config['model']} @ {self.light_config['base_url']}",
              file=sys.stderr)
        print(f"  Deep:  {self.deep_config['model']} @ {self.deep_config['base_url']}",
              file=sys.stderr)

        t_pipeline = time.time()

        # Build dependency graph for DAG scheduling
        dep_graph = self._build_dep_graph(run_steps)
        step_index = {s.name: i + 1 for i, s in enumerate(run_steps)}

        # Phase 1: load cached/skipped steps before resume_from
        for i, step in enumerate(run_steps, 1):
            if i < resume_from:
                if self.step_cached(step.name):
                    cached = self.load_step(step.name)
                    self.outputs[step.name] = cached
                    if (step.op == "Gate" and isinstance(cached, dict)
                            and cached.get("gate_passed") is False
                            and i < total_steps):
                        self._gate_skipped.add(steps[i].name)
                    print(f"\n  [{i}/{total_steps}] {step.name} ({step.op}) — cached",
                          file=sys.stderr)
                else:
                    cache_path = os.path.join(self.output_dir, f"{step.name}.json")
                    if target_index is not None:
                        raise FileNotFoundError(
                            f"Cannot run --step {step_only}: missing cached output "
                            f"for prior step {i} ({step.name}) at {cache_path}. "
                            f"Run --step {i} first, or run a full pipeline through "
                            f"that step, then retry."
                        )
                    print(f"\n  [{i}/{total_steps}] {step.name} ({step.op}) — SKIP (no cache)",
                          file=sys.stderr)

        # Phase 2: DAG-scheduled execution of remaining steps
        executable = [s for s in run_steps if step_index[s.name] >= resume_from]
        remaining = {s.name for s in executable}
        completed: set[str] = set(self.outputs.keys())
        step_by_name = {s.name: s for s in executable}
        failed_steps: set[str] = set()
        outputs_lock = threading.Lock()

        def _has_failed_ancestor(name: str, visited: set[str] | None = None) -> bool:
            if visited is None:
                visited = set()
            for dep in dep_graph.get(name, set()):
                if dep in visited:
                    continue
                visited.add(dep)
                if dep in failed_steps:
                    return True
                if _has_failed_ancestor(dep, visited):
                    return True
            return False

        def _is_ready(name: str) -> bool:
            return dep_graph.get(name, set()).issubset(completed)

        def _run_one_step(step: StepSpec) -> tuple[str, Any, float, str, str]:
            """Execute a single step. Returns (name, output, elapsed, backend, mode)."""
            i = step_index[step.name]

            if step.name in self._gate_skipped:
                print(f"\n  [{i}/{total_steps}] {step.name} — SKIP (gate blocked)",
                      file=sys.stderr)
                return step.name, {"error": "skipped: gate blocked", "skipped": True}, 0.0, "", "skipped"

            if step.name in self._precached_steps:
                existing = self._precached_steps[step.name]
                self.save_step(step.name, existing)
                print(f"\n  [{i}/{total_steps}] {step.name} — SKIP (pre-cached)",
                      file=sys.stderr)
                return step.name, existing, 0.0, "", "pre-cached"

            if step.cache_from:
                if self._ensure_output_loaded(step.name):
                    print(f"\n  [{i}/{total_steps}] {step.name} ({step.op}) — cached",
                          file=sys.stderr)
                    return step.name, self.outputs[step.name], 0.0, "", "cached"
                src = self._resolve_cache_from(step.cache_from)
                if src is None:
                    raise FileNotFoundError(
                        f"Step '{step.name}' declares cache_from "
                        f"'{step.cache_from}' but no such file exists "
                        f"(tried as given and relative to the pipeline root)"
                    )
                with open(src) as f:
                    seeded = json.load(f)
                self.save_step(step.name, seeded)
                print(f"\n  [{i}/{total_steps}] {step.name} ({step.op}) — "
                      f"seeded from {src}", file=sys.stderr)
                return step.name, seeded, 0.0, "", "seeded"

            if self._deadline and time.time() > self._deadline:
                elapsed_total = time.time() - t_pipeline
                print(f"\n  [TIMEOUT] Wall-clock deadline exceeded "
                      f"({elapsed_total / 60:.1f} min) — aborting at {step.name}",
                      file=sys.stderr)
                err = {
                    "error": "pipeline timeout exceeded",
                    "elapsed_minutes": round(elapsed_total / 60, 1),
                }
                self.save_step(step.name, err)
                return step.name, err, 0.0, "", "timeout"

            t_step = time.time()
            backend = "deterministic" if step.deterministic else (
                "deep" if self.select_config(step) is self.deep_config else "light"
            )
            if step.op == "Pipeline":
                depth = self.plan.config.get("_depth", 0)
                mode = f"pipeline({step.pipeline_ref}, d{depth})"
            elif step.op == "Foreach" and step.body:
                mode = f"foreach({len(step.body)} body steps)"
            elif step.agentic:
                mode = "agentic"
            elif step.foreach:
                mode = "foreach"
            elif step.batch_size:
                parts = [f"batched({step.batch_size})"]
                if step.batch_parallel > 1:
                    parts.append(f"parallel({step.batch_parallel})")
                if step.adaptive_batch:
                    parts.append("adaptive")
                mode = " ".join(parts)
            elif step.deterministic:
                mode = f"handler:{step.handler}"
            else:
                mode = "single"

            print(f"\n{'=' * 60}", file=sys.stderr)
            print(f"  [{i}/{total_steps}] {step.name}", file=sys.stderr)
            print(f"  Op: {step.op}  Type: {step.type_sig}", file=sys.stderr)
            print(f"  Input: {step.input}", file=sys.stderr)
            print(f"  Backend: {backend}  Mode: {mode}", file=sys.stderr)
            if step.reasoning_effort:
                print(f"  Reasoning: {step.reasoning_effort}", file=sys.stderr)

            if step.op == "Gate":
                gate_passed = self._evaluate_gate(step)
                output = {"gate_passed": gate_passed, "condition": step.condition}
                self.save_step(step.name, output)
                elapsed = time.time() - t_step
                print(f"  Condition: {step.condition}", file=sys.stderr)
                print(f"  Gate {'PASSED' if gate_passed else 'BLOCKED'}  ({elapsed:.1f}s)",
                      file=sys.stderr)
                if not gate_passed:
                    next_idx = step_index[step.name]
                    if next_idx < total_steps:
                        skipped = steps[next_idx]
                        print(f"  Gate skipping next step: {skipped.name}", file=sys.stderr)
                        skip_output = {
                            "error": f"skipped: gate '{step.name}' blocked",
                            "skipped": True,
                        }
                        self.save_step(skipped.name, skip_output)
                        self._gate_skipped.add(skipped.name)
                return step.name, output, elapsed, backend, mode

            try:
                inputs = self.resolve_inputs(step)
            except EmptyStepOutputError:
                if step.allow_empty_input:
                    print(f"    [{step.name}] empty upstream input — allowed, skipping step",
                          file=sys.stderr)
                    output = {"error": "skipped: empty input (allowed)", "skipped": True}
                    self.save_step(step.name, output)
                    return step.name, output, 0.0, backend, mode
                raise

            if not inputs:
                print(f"    [SKIP] All inputs for '{step.name}' are empty or errored — skipping",
                      file=sys.stderr)
                output = {"error": f"skipped: all inputs empty/errored",
                          "skipped": True}
                self.save_step(step.name, output)
                return step.name, output, 0.0, backend, mode

            if not step.deterministic and not step.batch_size:
                input_tokens = sum(
                    _estimate_tokens(json.dumps(v) if not isinstance(v, str) else v)
                    for v in inputs.values()
                )
                if (input_tokens > CONTEXT_UPGRADE_THRESHOLD
                        and self.select_config(step) is self.light_config):
                    step.backend_override = "deep"
                    print(f"    Input ~{input_tokens:,} tokens — upgrading to deep backend",
                          file=sys.stderr)

            input_error = self._validate_step_input(step, inputs)
            if input_error:
                if step.deterministic:
                    print(f"    [INPUT CONTRACT] {input_error}", file=sys.stderr)
                    output = {"error": f"input contract violation: {input_error}"}
                    self.save_step(step.name, output)
                    return step.name, output, 0.0, backend, mode
                else:
                    print(f"    [input-warn] {input_error}", file=sys.stderr)

            if not step.agentic and not step.deterministic:
                if self.should_compress(step, inputs):
                    threshold = self.plan.config.get("defaults", {}).get(
                        "compress_threshold", 500_000,
                    )
                    total_tok = sum(
                        _estimate_tokens(json.dumps(d, ensure_ascii=False) if isinstance(d, (list, dict)) else d)
                        for k, d in inputs.items()
                        if k not in ("primary", "task") and isinstance(d, (str, list, dict))
                    )
                    msg = (
                        f"Input exceeds context threshold ({total_tok:,} tokens > "
                        f"{threshold:,} limit). "
                        f"Reduce input size or raise compress_threshold in the pipeline config."
                    )
                    print(f"    [CONTEXT-OVERFLOW] {step.name}: {msg}", file=sys.stderr)
                    output = {"error": msg}
                    self.save_step(step.name, output)
                    return step.name, output, 0.0, backend, mode

            if step.op == "Synthesize" and not step.deterministic:
                config = self.config_for_step(step, input_chars=self._estimate_input_chars(inputs))
                budget = self._get_backend_budget(config)
                input_est = sum(
                    _estimate_tokens(json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
                    for v in inputs.values() if isinstance(v, (str, dict, list))
                )
                if input_est > budget:
                    print(f"    [BUDGET-WARNING] Synthesize step '{step.name}' input "
                          f"~{input_est:,} tokens exceeds backend budget {budget:,}. "
                          f"Add an upstream Extract step to reduce input size.",
                          file=sys.stderr)

            self._current_step_name = step.name
            try:
                output = self.execute_step(step, inputs)
            except Exception as e:
                print(f"    [ERROR] {step.name}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                output = {"error": str(e)}

            elapsed = time.time() - t_step
            self.save_step(step.name, output)
            if isinstance(output, list):
                summary = f"{len(output)} items"
            elif isinstance(output, dict):
                keys = list(output.keys())[:5]
                summary = f"dict({', '.join(keys)})"
            elif isinstance(output, str):
                summary = f"{len(output)} chars"
            else:
                summary = type(output).__name__
            print(f"  Result: {summary}  ({elapsed:.1f}s)", file=sys.stderr)

            return step.name, output, elapsed, backend, mode

        def _record_timing(step: StepSpec, elapsed: float, backend: str, mode: str):
            step_tokens = [r for r in self._token_records if r["step"] == step.name]
            step_in = sum(r["input_tokens"] for r in step_tokens)
            step_out = sum(r["output_tokens"] for r in step_tokens)
            step_cache = sum(r["cache_tokens"] for r in step_tokens)
            self.step_timings.append({
                "step": step.name, "op": step.op, "backend": backend,
                "mode": mode, "reasoning_effort": step.reasoning_effort,
                "elapsed_s": round(elapsed, 1),
                "input_tokens": step_in,
                "output_tokens": step_out,
                "cache_tokens": step_cache,
                "llm_calls": len(step_tokens),
            })

        def _handle_step_result(sname, output, elapsed, backend, mode, step):
            with outputs_lock:
                self.outputs[sname] = output
            remaining.discard(sname)
            completed.add(sname)
            _record_timing(step, elapsed, backend, mode)

            is_error = (isinstance(output, dict) and "error" in output
                        and not output.get("skipped"))
            is_timeout = (isinstance(output, dict)
                          and output.get("error") == "pipeline timeout exceeded")
            if is_error or is_timeout:
                failed_steps.add(sname)
                i = step_index.get(sname, 0)
                print(f"\n    [FAILED] Step '{sname}' returned error: "
                      f"{str(output['error'])[:200]}", file=sys.stderr)
                print(f"    Downstream steps will be skipped. To retry: --resume {i}",
                      file=sys.stderr)

        # DAG scheduler loop
        while remaining:
            # Skip steps whose ancestors have failed
            blocked_by_failure = set()
            for name in list(remaining):
                if name not in self._gate_skipped and _has_failed_ancestor(name):
                    blocked_by_failure.add(name)
                    skip_output = {
                        "error": "skipped: upstream step failed",
                        "skipped": True,
                    }
                    with outputs_lock:
                        self.outputs[name] = skip_output
                    remaining.discard(name)
                    completed.add(name)
                    print(f"  [{step_index.get(name, '?')}/{total_steps}] {name} — SKIP (upstream failed)",
                          file=sys.stderr)

            ready = [
                name for name in remaining
                if _is_ready(name) and name not in self._gate_skipped
            ]
            # Also include gate-skipped steps so they get processed and removed
            gate_skipped_remaining = [
                name for name in remaining
                if name in self._gate_skipped
            ]
            ready.extend(gate_skipped_remaining)
            if not ready:
                orphaned = [
                    name for name in remaining
                    if name not in self._gate_skipped
                ]
                if orphaned:
                    missing = {
                        name: dep_graph[name] - completed
                        for name in orphaned
                    }
                    print(f"\n  [DAG ERROR] Steps blocked on missing deps: {missing}",
                          file=sys.stderr)
                break

            ready.sort(key=lambda n: step_index.get(n, 0))
            batch = ready[:max_concurrency]

            if max_concurrency > 1 and len(batch) > 1:
                names = ", ".join(batch)
                print(f"\n  [parallel] Starting {len(batch)} steps: {names}",
                      file=sys.stderr)

            if max_concurrency <= 1 or len(batch) == 1:
                # Sequential execution
                for name in batch:
                    step = step_by_name[name]
                    sname, output, elapsed, backend, mode = _run_one_step(step)
                    _handle_step_result(sname, output, elapsed, backend, mode, step)
            else:
                # Parallel execution
                with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                    future_to_name = {}
                    for name in batch:
                        step = step_by_name[name]
                        future = executor.submit(_run_one_step, step)
                        future_to_name[future] = name

                    for future in as_completed(future_to_name):
                        name = future_to_name[future]
                        step = step_by_name[name]
                        try:
                            sname, output, elapsed, backend, mode = future.result()
                        except Exception as e:
                            print(f"    [ERROR] {name}: {e}", file=sys.stderr)
                            traceback.print_exc(file=sys.stderr)
                            output = {"error": str(e)}
                            sname, elapsed, backend, mode = name, 0.0, "", "error"
                        _handle_step_result(sname, output, elapsed, backend, mode, step)

        if failed_steps:
            print(f"\n  [SUMMARY] {len(failed_steps)} step(s) failed: {', '.join(sorted(failed_steps))}",
                  file=sys.stderr)
            print(f"  Independent branches completed normally.", file=sys.stderr)

        total = time.time() - t_pipeline

        light_in = sum(r["input_tokens"] for r in self._token_records if r["backend"] == "light")
        light_out = sum(r["output_tokens"] for r in self._token_records if r["backend"] == "light")
        light_cache = sum(r["cache_tokens"] for r in self._token_records if r["backend"] == "light")
        light_calls = sum(1 for r in self._token_records if r["backend"] == "light")
        deep_in = sum(r["input_tokens"] for r in self._token_records if r["backend"] == "deep")
        deep_out = sum(r["output_tokens"] for r in self._token_records if r["backend"] == "deep")
        deep_cache = sum(r["cache_tokens"] for r in self._token_records if r["backend"] == "deep")
        deep_calls = sum(1 for r in self._token_records if r["backend"] == "deep")

        def _fmt(n: int) -> str:
            if n >= 1_000_000:
                return f"{n / 1_000_000:.2f}M"
            if n >= 1_000:
                return f"{n / 1_000:.1f}K"
            return str(n)

        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"Pipeline complete in {total / 60:.1f} minutes", file=sys.stderr)
        if self._token_records:
            print(f"[tokens] light: {_fmt(light_in)} in / {_fmt(light_out)} out ({light_calls} calls)"
                  f" | deep: {_fmt(deep_in)} in / {_fmt(deep_out)} out ({deep_calls} calls)"
                  f" | total: {_fmt(light_in + deep_in)} in / {_fmt(light_out + deep_out)} out",
                  file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)

        token_usage = {
            "light": {
                "model": self.light_config.get("model"),
                "input_tokens": light_in,
                "output_tokens": light_out,
                "cache_tokens": light_cache,
                "calls": light_calls,
            },
            "deep": {
                "model": self.deep_config.get("model"),
                "input_tokens": deep_in,
                "output_tokens": deep_out,
                "cache_tokens": deep_cache,
                "calls": deep_calls,
            },
            "total_input_tokens": light_in + deep_in,
            "total_output_tokens": light_out + deep_out,
            "total_cache_tokens": light_cache + deep_cache,
            "total_calls": light_calls + deep_calls,
        }

        timing_file = os.path.join(self.output_dir, "pipeline_timings.json")
        with open(timing_file, "w") as f:
            json.dump({
                "pipeline": self.plan.name,
                "total_seconds": round(total, 1),
                "single_step": step_only,
                "single_step_index": target_index,
                "light_model": self.light_config.get("model"),
                "deep_model": self.deep_config.get("model"),
                "steps": self.step_timings,
                "token_usage": token_usage,
            }, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_tool_providers(spec: str, output_dir: str) -> list[ToolProvider]:
    providers: list[ToolProvider] = []
    for raw_name in spec.split(","):
        name = raw_name.strip().lower()
        if not name:
            continue
        if name == "standalone":
            providers.append(StandaloneToolProvider())
        elif name == "mesh":
            providers.append(MeshToolProvider())
        else:
            raise ValueError(
                f"Unknown tool provider '{raw_name}'. "
                "Expected comma-separated values from: standalone, mesh"
            )
    return providers or [StandaloneToolProvider()]


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Compiler — execute YAML pipeline plans",
    )
    parser.add_argument("plan_pos", nargs="?", help="Pipeline plan YAML file")
    parser.add_argument("--plan", dest="plan_opt", default=None,
                        help="Pipeline plan YAML file (alternative to positional plan)")
    parser.add_argument("--input", "-i", default=None,
                        help="Input file (text or PDF)")
    parser.add_argument("--text", default=None,
                        help="Input text string (instead of --input)")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory")
    parser.add_argument("--config", default="pipeline_config.yaml",
                        help="Pipeline config YAML (for deep backend defaults)")
    parser.add_argument("--resume", type=int, default=1,
                        help="Resume from step N")
    parser.add_argument("--step", default=None,
                        help="Execute only this step name or 1-based index "
                             "(0 aliases the first step)")
    parser.add_argument("--list-steps", action="store_true",
                        help="List step names and execution modes, then exit")
    parser.add_argument("--light-url", default=None)
    parser.add_argument("--light-model", default=None)
    parser.add_argument("--deep-url", default=None)
    parser.add_argument("--deep-model", default=None)
    parser.add_argument("--backend-config", default=None,
                        help="YAML file with light_backend/deep_backend overrides "
                             "(applied after the plan's inline config, before CLI args)")
    parser.add_argument("--all-light", action="store_true",
                        help="Force ALL steps to use light backend (no deep)")
    parser.add_argument("--tool-providers", default="standalone",
                        help="Comma-separated tool providers: standalone,mesh")
    parser.add_argument("--prompt-dir", default=None,
                        help="Prompt template directory (default: prompts/ beside the plan)")
    parser.add_argument("--test-context", default=None,
                        help="Test patch or test expectations for verifier (file path or inline text)")
    parser.add_argument("--test-command", default=None,
                        help="Custom test command (e.g. 'cargo test', 'npx jest'). Default: pytest")
    parser.add_argument("--codebase-root", default=None,
                        help="Root directory of the codebase to edit (required for code_edit pipelines)")
    parser.add_argument("--max-concurrency", type=int, default=None,
                        help="Max parallel steps in DAG scheduler (default: plan config or 4)")
    parser.add_argument("--show-dag", action="store_true",
                        help="Print the DAG execution schedule and exit")
    args = parser.parse_args()

    plan_path = args.plan_opt or args.plan_pos
    if not plan_path:
        parser.error("Provide a plan YAML path as a positional argument or with --plan")

    if args.step is not None and args.resume != 1:
        parser.error("--step loads prior cached outputs itself; do not combine it with --resume")

    plan = load_plan(plan_path)

    if args.list_steps:
        print_step_list(plan)
        return

    if args.show_dag:
        mc = args.max_concurrency or plan.config.get("max_concurrency", 4)
        compiler_tmp = PipelineCompiler.__new__(PipelineCompiler)
        compiler_tmp.plan = plan
        compiler_tmp._gate_skipped = set()
        deps = compiler_tmp._build_dep_graph(plan.steps)
        compiler_tmp._print_dag_schedule(plan.steps, deps, mc)
        return

    if args.codebase_root:
        plan.config["codebase_root"] = os.path.abspath(args.codebase_root)
    elif plan.config.get("handler_module") == "code_edit":
        parser.error("--codebase-root is required for code_edit pipelines")

    if args.test_context:
        if os.path.isfile(args.test_context):
            plan.config["test_context_path"] = os.path.abspath(args.test_context)
            with open(args.test_context) as f:
                plan.config["test_context"] = f.read()
        else:
            plan.config["test_context"] = args.test_context

    if args.test_command:
        plan.config["test_command"] = args.test_command

    output_dir = args.output_dir or f"./output_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    task_path = os.path.join(output_dir, "_pipeline_task.txt")

    if args.text:
        task_text = args.text
        with open(task_path, "w", encoding="utf-8") as f:
            f.write(task_text)
    elif args.input:
        task_text = _read_input(args.input)
        with open(task_path, "w", encoding="utf-8") as f:
            f.write(task_text)
    elif os.path.exists(task_path):
        with open(task_path, encoding="utf-8") as f:
            task_text = f.read()
    else:
        parser.error(
            "Provide --input or --text, or reuse an --output-dir containing "
            "_pipeline_task.txt from a prior run"
        )
        return

    if not task_text.strip():
        source = "--text" if args.text else (args.input or task_path)
        parser.error(f"Task input from {source} is empty — nothing to send to the LLM")
    if args.input and args.input.lower().endswith('.pdf') and len(task_text.split()) < 100:
        parser.error(
            f"--input {args.input} extracted only {len(task_text.split())} words — "
            f"too short to be a real document. Check the PDF."
        )

    plan_dir = os.path.dirname(os.path.abspath(plan_path))
    _pipeline_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_dir = args.prompt_dir or os.path.join(_pipeline_root, "prompts")

    light_config: dict[str, Any] = {
        "base_url": args.light_url,
        "model": args.light_model,
        "api_key": "",
        "max_output_tokens": 131072,
    }

    deep_config: dict[str, Any] = {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-v4-pro",
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "max_output_tokens": 131072,
        "thinking_budget": 0,
        "agent_timeout": 900,
    }

    if os.path.exists(args.config):
        with open(args.config) as f:
            raw = _env_substitute(f.read())
        file_cfg = yaml.safe_load(raw) or {}
        for k in ("base_url", "model", "api_key", "max_output_tokens",
                   "thinking_budget", "soft_limit", "agent_timeout"):
            if k in file_cfg:
                deep_config[k] = file_cfg[k]

    plan_cfg = plan.config
    if plan_cfg.get("light_backend"):
        light_config.update(plan_cfg["light_backend"])
    if plan_cfg.get("deep_backend") and not args.all_light:
        deep_config.update(plan_cfg["deep_backend"])

    if args.backend_config and os.path.exists(args.backend_config):
        with open(args.backend_config) as f:
            raw = _env_substitute(f.read())
        bc = yaml.safe_load(raw) or {}
        if bc.get("light_backend"):
            light_config.update(bc["light_backend"])
        if bc.get("deep_backend") and not args.all_light:
            deep_config.update(bc["deep_backend"])

    if args.light_url:
        light_config["base_url"] = args.light_url
    if args.light_model:
        light_config["model"] = args.light_model
    if args.deep_url:
        deep_config["base_url"] = args.deep_url
    if args.deep_model:
        deep_config["model"] = args.deep_model

    if args.all_light:
        deep_config = light_config.copy()

    compiler = PipelineCompiler(
        plan=plan,
        task_text=task_text,
        output_dir=output_dir,
        light_config=light_config,
        deep_config=deep_config,
        tool_providers=build_tool_providers(args.tool_providers, output_dir),
        prompt_library=PromptLibrary(prompt_dir),
        plan_dir=plan_dir,
    )
    compiler.run(
        resume_from=args.resume,
        step_only=args.step,
        max_concurrency=args.max_concurrency,
    )


if __name__ == "__main__":
    main()
