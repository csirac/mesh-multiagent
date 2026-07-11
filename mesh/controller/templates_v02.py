"""
Template loader for v0.2 controller phase instructions.

Loads markdown instruction templates for each phase and provides
utilities for injecting them into LLM prompts.
"""

from pathlib import Path
from functools import lru_cache
from enum import Enum

from .models_v02 import FlowPhase


# Template directory relative to this file
TEMPLATE_DIR = Path(__file__).parent.parent / "prompts" / "controller_v02"


class TemplateError(Exception):
    """Raised when a template cannot be loaded."""
    pass


@lru_cache(maxsize=10)
def load_template(phase: FlowPhase) -> str:
    """
    Load the instruction template for a given phase.

    Args:
        phase: The flow phase to load instructions for

    Returns:
        The template content as a string

    Raises:
        TemplateError: If the template file doesn't exist or can't be read
    """
    # Map phases to template filenames
    phase_to_file = {
        FlowPhase.INFO: "info.md",
        FlowPhase.PLAN: "plan.md",
        FlowPhase.EXECUTE: "execute.md",
        FlowPhase.VALIDATE: "validate.md",
        FlowPhase.DOCUMENT: "document.md",
        FlowPhase.DONE: "done.md",
        FlowPhase.FAILED: "done.md",  # FAILED uses same template as DONE
    }

    filename = phase_to_file.get(phase)
    if not filename:
        raise TemplateError(f"No template defined for phase: {phase}")

    template_path = TEMPLATE_DIR / filename

    if not template_path.exists():
        raise TemplateError(f"Template file not found: {template_path}")

    try:
        return template_path.read_text(encoding="utf-8")
    except IOError as e:
        raise TemplateError(f"Failed to read template {template_path}: {e}")


def get_phase_instructions(phase: FlowPhase, context: dict | None = None) -> str:
    """
    Get phase instructions with optional context substitution.

    Args:
        phase: The flow phase
        context: Optional dict of values to substitute in the template
                 (for future use - currently templates don't use substitution)

    Returns:
        The instruction text ready for injection into the LLM prompt
    """
    template = load_template(phase)

    # Future: could add template variable substitution here
    # For now, return as-is
    if context:
        # Simple placeholder substitution if needed
        for key, value in context.items():
            template = template.replace(f"{{{key}}}", str(value))

    return template


def format_phase_block(phase: FlowPhase, context: dict | None = None) -> str:
    """
    Format phase instructions as a tagged block for injection into prompts.

    Args:
        phase: The flow phase
        context: Optional context for template substitution

    Returns:
        Instructions wrapped in XML-style tags for clear delineation
    """
    instructions = get_phase_instructions(phase, context)

    return f"""<controller_phase phase="{phase.value}">
{instructions}
</controller_phase>"""


def list_available_templates() -> list[str]:
    """
    List all available template files.

    Returns:
        List of template filenames that exist
    """
    if not TEMPLATE_DIR.exists():
        return []

    return [f.name for f in TEMPLATE_DIR.glob("*.md")]


def validate_templates() -> dict[str, bool]:
    """
    Check that all required templates exist and are readable.

    Returns:
        Dict mapping phase names to whether their template is valid
    """
    results = {}

    for phase in FlowPhase:
        try:
            load_template(phase)
            results[phase.value] = True
        except TemplateError:
            results[phase.value] = False

    return results
