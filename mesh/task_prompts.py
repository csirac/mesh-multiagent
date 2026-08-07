"""Load and compose task-level prompt bundles.

Configuration parsing lives in :mod:`mesh.config`; this module owns filesystem
resolution and exact prompt composition so synchronous tools, ordinary
workers, and PEV workers cannot drift into separate loading behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import TaskPromptConfig


@dataclass(frozen=True)
class ResolvedTaskPromptBundle:
    worker_system_prompt: str = ""
    base_instructions: str = ""
    sync_instructions: str = ""
    plan_instructions: str = ""
    execute_instructions: str = ""
    verify_instructions: str = ""
    sync_backend: str | None = None
    thinking_budget: int | None = None
    phase_mesh_tools: tuple[tuple[str, tuple[str, ...]], ...] = ()
    phase_harness_tools: tuple[tuple[str, tuple[str, ...]], ...] = ()
    verify_read_only: bool = False

    def tools_for_phase(self, phase: str) -> tuple[str, ...]:
        for configured_phase, tool_names in self.phase_mesh_tools:
            if configured_phase == phase:
                return tool_names
        return ()

    def harness_tools_for_phase(self, phase: str) -> tuple[str, ...]:
        for configured_phase, tool_names in self.phase_harness_tools:
            if configured_phase == phase:
                return tool_names
        return ()


def _resolve_prompt_path(repo_root: Path, configured_path: str) -> Path:
    root = repo_root.resolve()
    candidate = Path(configured_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"prompt path escapes repository root: {configured_path}"
        ) from exc
    if not resolved.is_file():
        raise ValueError(f"prompt file does not exist: {configured_path}")
    return resolved


def _read_prompt(repo_root: Path, configured_path: str | None) -> str:
    if not configured_path:
        return ""
    path = _resolve_prompt_path(repo_root, configured_path)
    try:
        content = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(f"prompt file is not valid UTF-8: {configured_path}") from exc

    # Compatibility asset resolution for the current writing prompt. New
    # domain prompts should keep assets explicit and avoid hidden includes.
    if "<style-guide/>" in content:
        style_path = path.parent / "writing_style.md"
        if not style_path.is_file():
            raise ValueError(
                f"prompt references <style-guide/> but asset is missing: {style_path}"
            )
        style = style_path.read_text(encoding="utf-8").strip()
        content = content.replace(
            "<style-guide/>",
            f"<style-guide>\n{style}\n</style-guide>",
        )
    if "<writing-examples/>" in content:
        examples_path = path.parent / "writing_examples.md"
        if not examples_path.is_file():
            raise ValueError(
                f"prompt references <writing-examples/> but asset is missing: {examples_path}"
            )
        examples = examples_path.read_text(encoding="utf-8").strip()
        content = content.replace(
            "<writing-examples/>",
            f"<writing-examples>\n{examples}\n</writing-examples>",
        )
    return content


def resolve_task_prompt_bundle(
    config: TaskPromptConfig,
    repo_root: str | Path,
) -> ResolvedTaskPromptBundle:
    """Resolve every configured prompt file into immutable text content."""
    root = Path(repo_root)
    return ResolvedTaskPromptBundle(
        worker_system_prompt=_read_prompt(
            root, config.worker_system_prompt_file
        ),
        base_instructions=_read_prompt(
            root, config.base_instructions_file
        ),
        sync_instructions=_read_prompt(
            root, config.sync_instructions_file
        ),
        plan_instructions=_read_prompt(
            root, config.plan_instructions_file
        ),
        execute_instructions=_read_prompt(
            root, config.execute_instructions_file
        ),
        verify_instructions=_read_prompt(
            root, config.verify_instructions_file
        ),
        sync_backend=config.sync_backend,
        thinking_budget=config.thinking_budget,
        phase_mesh_tools=config.phase_mesh_tools,
        phase_harness_tools=config.phase_harness_tools,
        verify_read_only=config.verify_read_only,
    )


def instruction_block(label: str, content: str | None) -> str:
    """Wrap one additive domain instruction segment with an auditable label."""
    normalized = str(content or "").strip()
    if not normalized:
        return ""
    return (
        f'<domain-instructions role="{label}">\n'
        f"{normalized}\n"
        "</domain-instructions>"
    )


def compose_task_instructions(
    *,
    base: str = "",
    plan: str = "",
    execute: str = "",
    verify: str = "",
    sync: str = "",
) -> str:
    """Compose configured segments in deterministic lifecycle order."""
    segments = (
        ("base", base),
        ("plan", plan),
        ("execute", execute),
        ("verify", verify),
        ("sync", sync),
    )
    return "\n\n".join(
        block
        for label, content in segments
        if (block := instruction_block(label, content))
    )
