"""
Tests for v0.2 controller template loader.
"""

import pytest
from pathlib import Path

from mesh.controller.models_v02 import FlowPhase
from mesh.controller.templates_v02 import (
    load_template,
    get_phase_instructions,
    format_phase_block,
    list_available_templates,
    validate_templates,
    TemplateError,
    TEMPLATE_DIR,
)


class TestTemplateDirectory:
    """Tests for template directory setup."""

    def test_template_dir_exists(self):
        """Template directory should exist."""
        assert TEMPLATE_DIR.exists(), f"Template dir not found: {TEMPLATE_DIR}"

    def test_template_dir_contains_files(self):
        """Template directory should contain markdown files."""
        templates = list(TEMPLATE_DIR.glob("*.md"))
        assert len(templates) >= 5, f"Expected at least 5 templates, found {len(templates)}"


class TestLoadTemplate:
    """Tests for load_template function."""

    def test_load_info_template(self):
        """Should load INFO phase template."""
        template = load_template(FlowPhase.INFO)
        assert "INFO" in template
        assert "<assessment>" in template
        assert "complexity" in template

    def test_load_plan_template(self):
        """Should load PLAN phase template."""
        template = load_template(FlowPhase.PLAN)
        assert "PLAN" in template
        assert "<plan>" in template
        assert "steps" in template

    def test_load_execute_template(self):
        """Should load EXECUTE phase template."""
        template = load_template(FlowPhase.EXECUTE)
        assert "EXECUTE" in template
        assert "execution" in template.lower() or "execute" in template.lower()

    def test_load_validate_template(self):
        """Should load VALIDATE phase template."""
        template = load_template(FlowPhase.VALIDATE)
        assert "VALIDATE" in template
        assert "<validation>" in template
        assert "task_accomplished" in template

    def test_load_document_template(self):
        """Should load DOCUMENT phase template."""
        template = load_template(FlowPhase.DOCUMENT)
        assert "DOCUMENT" in template
        assert "documentation" in template.lower()

    def test_load_done_template(self):
        """Should load DONE phase template."""
        template = load_template(FlowPhase.DONE)
        assert "DONE" in template
        assert "summary" in template.lower() or "complete" in template.lower()

    def test_failed_uses_done_template(self):
        """FAILED phase should use the same template as DONE."""
        done_template = load_template(FlowPhase.DONE)
        failed_template = load_template(FlowPhase.FAILED)
        assert done_template == failed_template

    def test_template_caching(self):
        """Templates should be cached after first load."""
        # Clear cache
        load_template.cache_clear()

        # Load twice
        template1 = load_template(FlowPhase.INFO)
        template2 = load_template(FlowPhase.INFO)

        # Should be same object due to caching
        assert template1 is template2

        # Check cache info
        cache_info = load_template.cache_info()
        assert cache_info.hits >= 1


class TestGetPhaseInstructions:
    """Tests for get_phase_instructions function."""

    def test_returns_template_content(self):
        """Should return template content."""
        instructions = get_phase_instructions(FlowPhase.INFO)
        assert len(instructions) > 100  # Should be substantial content
        assert "assessment" in instructions.lower()

    def test_with_context_substitution(self):
        """Should substitute context variables if present."""
        # Note: Current templates don't use substitution,
        # but the function should handle it
        instructions = get_phase_instructions(FlowPhase.INFO, context={})
        assert len(instructions) > 0


class TestFormatPhaseBlock:
    """Tests for format_phase_block function."""

    def test_wraps_in_tags(self):
        """Should wrap instructions in controller_phase tags."""
        block = format_phase_block(FlowPhase.INFO)
        assert block.startswith('<controller_phase phase="info">')
        assert block.endswith("</controller_phase>")

    def test_includes_full_content(self):
        """Should include full template content."""
        block = format_phase_block(FlowPhase.PLAN)
        assert "<plan>" in block
        assert "steps" in block

    def test_all_phases_format_correctly(self):
        """All phases should format without error."""
        for phase in FlowPhase:
            block = format_phase_block(phase)
            assert f'phase="{phase.value}"' in block


class TestListAvailableTemplates:
    """Tests for list_available_templates function."""

    def test_returns_list_of_files(self):
        """Should return list of template filenames."""
        templates = list_available_templates()
        assert isinstance(templates, list)
        assert "info.md" in templates
        assert "plan.md" in templates

    def test_only_markdown_files(self):
        """Should only list .md files."""
        templates = list_available_templates()
        for template in templates:
            assert template.endswith(".md")


class TestValidateTemplates:
    """Tests for validate_templates function."""

    def test_all_phases_have_templates(self):
        """All phases should have valid templates."""
        results = validate_templates()

        for phase in FlowPhase:
            assert phase.value in results, f"Missing result for {phase.value}"
            assert results[phase.value] is True, f"Invalid template for {phase.value}"

    def test_returns_dict(self):
        """Should return a dictionary."""
        results = validate_templates()
        assert isinstance(results, dict)
        assert len(results) == len(FlowPhase)


class TestTemplateContent:
    """Tests for template content quality."""

    def test_info_has_xml_format(self):
        """INFO template should show expected XML format."""
        template = load_template(FlowPhase.INFO)
        assert "<assessment>" in template
        assert "<complexity>" in template
        assert "<need_clarification>" in template

    def test_plan_has_step_structure(self):
        """PLAN template should show step structure."""
        template = load_template(FlowPhase.PLAN)
        assert "<steps>" in template
        assert "<step>" in template
        assert "<number>" in template
        assert "<description>" in template

    def test_validate_has_scoring(self):
        """VALIDATE template should show scoring structure."""
        template = load_template(FlowPhase.VALIDATE)
        assert "<task_accomplished>" in template
        assert "<verified>" in template
        assert "<issues>" in template

    def test_templates_have_guidelines(self):
        """All instruction templates should have guidelines section."""
        for phase in [FlowPhase.INFO, FlowPhase.PLAN, FlowPhase.VALIDATE]:
            template = load_template(phase)
            assert "## Guidelines" in template or "## Guideline" in template

    def test_templates_have_examples(self):
        """Instruction templates should have examples."""
        for phase in [FlowPhase.INFO, FlowPhase.PLAN, FlowPhase.VALIDATE, FlowPhase.DONE]:
            template = load_template(phase)
            assert "Example" in template or "example" in template
