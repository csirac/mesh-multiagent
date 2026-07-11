"""Pipeline-backed mesh router mode.

This module adapts the vendored pipeline engine in ``mesh/pipeline_engine`` to
RouterV2's runtime contract.  It is intentionally a thin bridge: RouterV2 owns
message sending, history write-back, and worker dispatch; PipelineRouter owns
compiling/running the pipeline and normalizing its final output.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .protocol import Message


_ENGINE_DIR = Path(__file__).resolve().parent / "pipeline_engine"
_HANDLERS_DIR = _ENGINE_DIR / "handlers"
_MESH_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLAN_PATH = _MESH_ROOT / "mesh" / "plans" / "router_pipeline.yaml"
DEFAULT_BACKEND_DIR = _MESH_ROOT / "mesh" / "plans" / "backends"
DEFAULT_PROMPTS_DIR = _MESH_ROOT / "mesh" / "prompts"

for _path in (_ENGINE_DIR, _HANDLERS_DIR):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from pipeline_compiler import (  # type: ignore  # noqa: E402
    PipelineCompiler,
    build_tool_providers,
    load_plan,
)
from prompt_library import PromptLibrary  # type: ignore  # noqa: E402


def _env_substitute(text: str) -> str:
    return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), text)


class PipelineRouter:
    """Run the typed router pipeline and return RouterV2-compatible results."""

    def __init__(
        self,
        llm_backend_config: str | dict[str, Any] | None = "deepseek",
        agent_name: str = "",
        nickname: str = "",
        history_dir: str | Path | None = None,
        plan_path: str | Path = DEFAULT_PLAN_PATH,
    ) -> None:
        self.agent_name = agent_name
        self.nickname = nickname
        self.history_dir = Path(history_dir) if history_dir else Path.home() / ".mesh" / "history"
        self.plan_path = Path(plan_path)
        self.llm_backend_config = llm_backend_config

    async def process(self, msg: Message, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Process one mesh message through the router pipeline.

        Returns both the native pipeline fields and the RouterV2 parse contract:
        ``no_response``, ``response``, ``dispatch_worker``, and ``task``.
        """
        context = dict(context or {})
        context.setdefault("agent_name", self.agent_name)
        context.setdefault("nickname", self.nickname)
        context.setdefault("history_dir", str(self.history_dir))

        payload = {
            "message": self._message_to_dict(msg),
            "context": context,
        }

        result = await asyncio.to_thread(self._run_pipeline, payload)
        validated = result.get("validate_response") or {}
        if not isinstance(validated, dict):
            validated = {}

        action = str(validated.get("action") or "respond").strip().lower()
        if action == "tool_call":
            # The stage-3 loop should normally terminate before validation. If it
            # does not, treat the synthesized text as a direct response rather
            # than launching work from an incomplete intermediate state.
            action = "respond"
        if action not in {"respond", "dispatch", "sleep"}:
            action = "respond"

        response_text = str(validated.get("response_text") or "").strip()
        task_spec = str(validated.get("task_spec") or "").strip()
        issues = validated.get("issues") or []
        valid = bool(validated.get("valid", not issues))

        return {
            "action": action,
            "response_text": response_text,
            "task_spec": task_spec,
            "valid": valid,
            "issues": issues,
            "no_response": action == "sleep",
            "response": "" if action == "sleep" else response_text,
            "dispatch_worker": action == "dispatch",
            "task": task_spec,
            "raw_outputs": result,
        }

    def _run_pipeline(self, payload: dict[str, Any]) -> dict[str, Any]:
        os.environ["MESH_HISTORY_DIR"] = str(self.history_dir)

        plan = load_plan(str(self.plan_path))
        light_config, deep_config = self._backend_configs(plan.config)
        output_dir = tempfile.mkdtemp(prefix=f"router_pipeline_{self.nickname or 'agent'}_")
        task_text = json.dumps(payload, ensure_ascii=False)

        compiler = PipelineCompiler(
            plan=plan,
            task_text=task_text,
            output_dir=output_dir,
            light_config=light_config,
            deep_config=deep_config,
            tool_providers=build_tool_providers("standalone,mesh", output_dir),
            prompt_library=PromptLibrary(DEFAULT_PROMPTS_DIR),
            plan_dir=str(self.plan_path.parent),
        )
        compiler.run()
        return dict(compiler.outputs)

    def _backend_configs(self, plan_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        light_config: dict[str, Any] = {
            "base_url": None,
            "model": None,
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

        if plan_config.get("light_backend"):
            light_config.update(plan_config["light_backend"])
        if plan_config.get("deep_backend"):
            deep_config.update(plan_config["deep_backend"])

        backend_cfg = self._load_backend_config(self.llm_backend_config)
        if backend_cfg.get("light_backend"):
            light_config.update(backend_cfg["light_backend"])
        if backend_cfg.get("deep_backend"):
            deep_config.update(backend_cfg["deep_backend"])

        return light_config, deep_config

    def _load_backend_config(self, spec: str | dict[str, Any] | None) -> dict[str, Any]:
        if spec is None:
            spec = "deepseek"
        if isinstance(spec, dict):
            return spec

        raw_spec = str(spec)
        candidates = [Path(raw_spec)]
        if not raw_spec.endswith(".yaml"):
            candidates.append(DEFAULT_BACKEND_DIR / f"{raw_spec}.yaml")
        candidates.append(DEFAULT_BACKEND_DIR / raw_spec)

        for path in candidates:
            if path.exists():
                text = _env_substitute(path.read_text(encoding="utf-8"))
                return yaml.safe_load(text) or {}
        return {}

    @staticmethod
    def _message_to_dict(msg: Message) -> dict[str, Any]:
        data = asdict(msg)
        msg_type = data.get("type")
        if hasattr(msg_type, "value"):
            data["type"] = msg_type.value
        else:
            data["type"] = str(msg_type)
        return data
