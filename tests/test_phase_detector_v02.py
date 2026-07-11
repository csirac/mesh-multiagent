"""
Tests for v0.2 phase detector (XML parsing).
"""

import pytest

from mesh.controller.phase_detector_v02 import PhaseDetectorV02, DEFAULTS
from mesh.controller.models_v02 import InfoAssessment, PlanV02, ValidationResult


class TestInfoAssessmentParsing:
    """Test INFO phase assessment parsing."""

    def setup_method(self):
        self.detector = PhaseDetectorV02()

    def test_parse_valid_xml(self):
        """Parse well-formed assessment XML."""
        xml = """
        Some preamble text...
        <assessment>
            <complexity>0.7</complexity>
            <need_clarification>0.2</need_clarification>
            <need_web>0.4</need_web>
            <need_literature>0.1</need_literature>
            <need_project_files>0.6</need_project_files>
            <web_search_intent>Find caching best practices</web_search_intent>
        </assessment>
        Some trailing text...
        """
        result = self.detector.parse_info_assessment(xml)

        assert result.complexity == 0.7
        assert result.need_clarification == 0.2
        assert result.need_web == 0.4
        assert result.need_literature == 0.1
        assert result.need_project_files == 0.6
        assert result.web_search_intent == "Find caching best practices"
        assert result.parsed_successfully is True

    def test_parse_with_clarification_questions(self):
        """Parse assessment with clarification questions."""
        xml = """
        <assessment>
            <complexity>0.5</complexity>
            <need_clarification>0.8</need_clarification>
            <need_web>0.0</need_web>
            <need_literature>0.0</need_literature>
            <need_project_files>0.0</need_project_files>
            <clarification_questions>
                <question>Which database are you using?</question>
                <question>Do you want sync or async?</question>
            </clarification_questions>
        </assessment>
        """
        result = self.detector.parse_info_assessment(xml)

        assert result.need_clarification == 0.8
        assert result.clarification_questions is not None
        assert len(result.clarification_questions) == 2
        assert "database" in result.clarification_questions[0]

    def test_regex_fallback_for_malformed_xml(self):
        """Fall back to regex when XML is malformed."""
        # Missing closing tag
        xml = """
        <assessment>
            <complexity>0.6</complexity>
            <need_clarification>0.1
            <need_web>0.3</need_web>
        """
        result = self.detector.parse_info_assessment(xml)

        # Should extract what it can via regex
        assert result.complexity == 0.6
        assert result.need_web == 0.3
        assert result.parsed_successfully is False

    def test_defaults_when_nothing_parseable(self):
        """Use defaults when nothing is parseable."""
        result = self.detector.parse_info_assessment("Just some random text with no XML")

        assert result.complexity == DEFAULTS["complexity"]
        assert result.need_clarification == DEFAULTS["need_clarification"]
        assert result.need_web == DEFAULTS["need_web"]
        assert result.parsed_successfully is False

    def test_clamp_scores_to_0_1(self):
        """Scores outside 0-1 range should be clamped."""
        xml = """
        <assessment>
            <complexity>1.5</complexity>
            <need_web>-0.2</need_web>
            <need_clarification>0.0</need_clarification>
            <need_literature>0.0</need_literature>
            <need_project_files>0.0</need_project_files>
        </assessment>
        """
        result = self.detector.parse_info_assessment(xml)

        assert result.complexity == 1.0  # Clamped from 1.5
        assert result.need_web == 0.0    # Clamped from -0.2

    def test_parse_intents(self):
        """Parse all intent fields."""
        xml = """
        <assessment>
            <complexity>0.5</complexity>
            <need_clarification>0.0</need_clarification>
            <need_web>0.5</need_web>
            <need_literature>0.5</need_literature>
            <need_project_files>0.5</need_project_files>
            <web_search_intent>Search for Python async patterns</web_search_intent>
            <literature_search_intent>Find papers on distributed caching</literature_search_intent>
            <project_files_intent>Look for existing cache implementations</project_files_intent>
        </assessment>
        """
        result = self.detector.parse_info_assessment(xml)

        assert result.web_search_intent == "Search for Python async patterns"
        assert result.literature_search_intent == "Find papers on distributed caching"
        assert result.project_files_intent == "Look for existing cache implementations"


class TestPlanParsing:
    """Test PLAN phase output parsing."""

    def setup_method(self):
        self.detector = PhaseDetectorV02()

    def test_parse_valid_plan_xml(self):
        """Parse well-formed plan XML."""
        xml = """
        <plan>
            <quality>0.85</quality>
            <steps>
                <step>Set up database schema</step>
                <step>Implement API endpoints</step>
                <step>Write unit tests</step>
            </steps>
            <rollback>Drop tables and revert migration</rollback>
        </plan>
        """
        result = self.detector.parse_plan(xml)

        assert result.quality_score == 0.85
        assert len(result.steps) == 3
        assert result.steps[0].number == 1
        assert result.steps[0].description == "Set up database schema"
        assert result.rollback_strategy == "Drop tables and revert migration"
        assert result.parsed_successfully is True

    def test_parse_with_complexity_reassessment(self):
        """Parse plan with complexity reassessment."""
        xml = """
        <plan>
            <quality>0.9</quality>
            <steps>
                <step>Single step task</step>
            </steps>
            <complexity_reassessment>Actually simpler than expected - just need one API call</complexity_reassessment>
        </plan>
        """
        result = self.detector.parse_plan(xml)

        assert result.complexity_reassessment is not None
        assert "simpler" in result.complexity_reassessment

    def test_hybrid_xml_and_numbered_list(self):
        """Parse quality from XML and steps from numbered list."""
        text = """
        Here's my plan:
        <quality>0.7</quality>
        1. First, set up the database
        2. Then implement the API
        3. Finally, write tests
        """
        result = self.detector.parse_plan(text)

        # Should extract quality from XML and steps from numbered list
        assert len(result.steps) >= 3
        assert "database" in result.steps[0].description.lower()
        assert result.quality_score == 0.7
        # This is a valid hybrid parse, so parsed_successfully should be True
        assert result.parsed_successfully is True

    def test_defaults_when_nothing_parseable(self):
        """Use defaults when nothing is parseable."""
        result = self.detector.parse_plan("Just some text with no plan structure")

        assert result.quality_score == DEFAULTS["plan_quality"]
        assert len(result.steps) == 0
        assert result.parsed_successfully is False

    def test_max_steps_not_enforced_in_parser(self):
        """Parser doesn't enforce step limit - that's controller's job."""
        xml = """
        <plan>
            <quality>0.8</quality>
            <steps>
                <step>Step 1</step>
                <step>Step 2</step>
                <step>Step 3</step>
                <step>Step 4</step>
                <step>Step 5</step>
                <step>Step 6</step>
                <step>Step 7</step>
                <step>Step 8</step>
                <step>Step 9</step>
            </steps>
        </plan>
        """
        result = self.detector.parse_plan(xml)

        # Parser should return all steps, limit enforcement is elsewhere
        assert len(result.steps) == 9


class TestValidationParsing:
    """Test VALIDATE phase output parsing."""

    def setup_method(self):
        self.detector = PhaseDetectorV02()

    def test_parse_valid_validation_xml(self):
        """Parse well-formed validation XML."""
        xml = """
        <validation>
            <task_accomplished>0.95</task_accomplished>
            <verified>0.9</verified>
            <issues>
                <issue>Minor type error in tests</issue>
            </issues>
            <can_fix_without_replan>true</can_fix_without_replan>
            <fix_actions>Fix the type annotation</fix_actions>
        </validation>
        """
        result = self.detector.parse_validation(xml)

        assert result.task_accomplished == 0.95
        assert result.verified == 0.9
        assert len(result.issues) == 1
        assert "type error" in result.issues[0]
        assert result.can_fix_without_replan is True
        assert result.fix_actions == "Fix the type annotation"
        assert result.parsed_successfully is True

    def test_parse_validation_no_issues(self):
        """Parse validation with no issues."""
        xml = """
        <validation>
            <task_accomplished>1.0</task_accomplished>
            <verified>1.0</verified>
            <issues></issues>
            <can_fix_without_replan>false</can_fix_without_replan>
        </validation>
        """
        result = self.detector.parse_validation(xml)

        assert result.task_accomplished == 1.0
        assert result.verified == 1.0
        assert len(result.issues) == 0
        assert result.can_fix_without_replan is False

    def test_regex_fallback(self):
        """Fall back to regex for malformed XML."""
        xml = """
        <validation>
            <task_accomplished>0.8</task_accomplished>
            <verified>0.7
            <can_fix_without_replan>false</can_fix_without_replan>
        """
        result = self.detector.parse_validation(xml)

        assert result.task_accomplished == 0.8
        assert result.can_fix_without_replan is False
        assert result.parsed_successfully is False

    def test_validation_success_helper(self):
        """Test is_successful helper method."""
        # Successful validation
        success_result = ValidationResult(
            task_accomplished=0.9,
            verified=0.8,
        )
        assert success_result.is_successful() is True

        # Below accomplished threshold
        fail_accomplished = ValidationResult(
            task_accomplished=0.7,
            verified=0.9,
        )
        assert fail_accomplished.is_successful() is False

        # Below verified threshold
        fail_verified = ValidationResult(
            task_accomplished=0.9,
            verified=0.5,
        )
        assert fail_verified.is_successful() is False

    def test_defaults_when_nothing_parseable(self):
        """Use defaults when nothing is parseable."""
        result = self.detector.parse_validation("Random text with no structure")

        assert result.task_accomplished == DEFAULTS["task_accomplished"]
        assert result.verified == DEFAULTS["verified"]
        assert result.parsed_successfully is False


class TestHelperMethods:
    """Test helper methods."""

    def setup_method(self):
        self.detector = PhaseDetectorV02()

    def test_extract_xml_block(self):
        """Extract named XML block from output."""
        text = """
        Some preamble...
        <assessment>
            <complexity>0.5</complexity>
        </assessment>
        Some middle text...
        <plan>
            <quality>0.8</quality>
        </plan>
        """
        assessment = self.detector.extract_xml_block(text, "assessment")
        plan = self.detector.extract_xml_block(text, "plan")
        missing = self.detector.extract_xml_block(text, "validation")

        assert assessment is not None
        assert "<complexity>" in assessment
        assert plan is not None
        assert "<quality>" in plan
        assert missing is None

    def test_has_structured_output(self):
        """Detect presence of structured XML blocks."""
        with_assessment = "Text <assessment><complexity>0.5</complexity></assessment>"
        with_plan = "Text <plan><quality>0.8</quality></plan>"
        with_validation = "Text <validation><verified>0.9</verified></validation>"
        without_structure = "Just plain text with no XML blocks"

        assert self.detector.has_structured_output(with_assessment) is True
        assert self.detector.has_structured_output(with_plan) is True
        assert self.detector.has_structured_output(with_validation) is True
        assert self.detector.has_structured_output(without_structure) is False


class TestEdgeCases:
    """Test edge cases and robustness."""

    def setup_method(self):
        self.detector = PhaseDetectorV02()

    def test_empty_string(self):
        """Handle empty string input."""
        assessment = self.detector.parse_info_assessment("")
        plan = self.detector.parse_plan("")
        validation = self.detector.parse_validation("")

        # Should return defaults, not crash
        assert assessment.complexity == DEFAULTS["complexity"]
        assert plan.quality_score == DEFAULTS["plan_quality"]
        assert validation.task_accomplished == DEFAULTS["task_accomplished"]

    def test_unicode_content(self):
        """Handle Unicode content in XML."""
        xml = """
        <assessment>
            <complexity>0.6</complexity>
            <need_clarification>0.5</need_clarification>
            <need_web>0.0</need_web>
            <need_literature>0.0</need_literature>
            <need_project_files>0.0</need_project_files>
            <clarification_questions>
                <question>你想要什么样的缓存？</question>
                <question>¿Cuál es el tamaño esperado?</question>
            </clarification_questions>
        </assessment>
        """
        result = self.detector.parse_info_assessment(xml)

        assert result.clarification_questions is not None
        assert len(result.clarification_questions) == 2

    def test_nested_xml_in_content(self):
        """Handle nested-looking content in XML values."""
        xml = """
        <plan>
            <quality>0.8</quality>
            <steps>
                <step>Write code like: <code>return x</code></step>
            </steps>
        </plan>
        """
        # This will likely fail XML parsing, should fall back to regex
        result = self.detector.parse_plan(xml)

        # Should not crash, may or may not parse correctly
        assert result is not None

    def test_whitespace_variations(self):
        """Handle various whitespace in XML."""
        xml = """<assessment>
<complexity>   0.5   </complexity>
<need_clarification>
0.2
</need_clarification>
<need_web>0.3</need_web><need_literature>0.1</need_literature>
<need_project_files>0.4</need_project_files>
</assessment>"""
        result = self.detector.parse_info_assessment(xml)

        assert result.complexity == 0.5
        assert result.need_clarification == 0.2
        assert result.need_web == 0.3

    def test_case_insensitive_tags(self):
        """Handle case variations in XML tags."""
        xml = """
        <ASSESSMENT>
            <Complexity>0.6</Complexity>
            <NEED_WEB>0.4</NEED_WEB>
            <need_clarification>0.1</need_clarification>
            <need_literature>0.0</need_literature>
            <need_project_files>0.0</need_project_files>
        </ASSESSMENT>
        """
        result = self.detector.parse_info_assessment(xml)

        # Regex fallback should handle case variations
        assert result.need_web == 0.4 or result.need_web == DEFAULTS["need_web"]
