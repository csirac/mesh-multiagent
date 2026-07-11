"""
Phase detector v0.2 - Parse structured XML output from LLM phases.

The v0.2 controller expects structured XML output from the LLM at specific phases:
- INFO phase: InfoAssessment scores (complexity, need_* scores)
- PLAN phase: Plan with steps, rollback, quality score
- VALIDATE phase: Validation results (accomplished, verified, issues)

This module provides parsers with fallback behavior:
1. Primary: Standard XML parsing
2. Fallback: Lenient regex extraction for malformed XML
3. Last resort: Default values
"""

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

from .models_v02 import (
    InfoAssessment,
    PlanStepV02,
    PlanV02,
    ValidationResult,
)

logger = logging.getLogger(__name__)


# Default values when parsing fails completely
DEFAULTS = {
    "complexity": 0.5,
    "need_clarification": 0.3,
    "need_web": 0.3,
    "need_literature": 0.3,
    "need_project_files": 0.3,
    "plan_quality": 0.5,
    "task_accomplished": 0.5,
    "verified": 0.5,
}


class PhaseDetectorV02:
    """
    Parse structured XML output from LLM phases.

    Expected XML formats:

    INFO Assessment:
    ```
    <assessment>
        <complexity>0.7</complexity>
        <need_clarification>0.2</need_clarification>
        <need_web>0.4</need_web>
        <need_literature>0.1</need_literature>
        <need_project_files>0.6</need_project_files>
        <clarification_questions>
            <question>Which database do you want to use?</question>
        </clarification_questions>
        <web_search_intent>Find current best practices for caching</web_search_intent>
        <project_files_intent>Look for existing cache implementations</project_files_intent>
    </assessment>
    ```

    Plan Output:
    ```
    <plan>
        <quality>0.85</quality>
        <steps>
            <step>Set up database schema</step>
            <step>Implement API endpoints</step>
        </steps>
        <rollback>Revert migration file and drop tables</rollback>
        <complexity_reassessment>Actually simpler than expected</complexity_reassessment>
    </plan>
    ```

    Validation Output:
    ```
    <validation>
        <task_accomplished>0.95</task_accomplished>
        <verified>0.9</verified>
        <issues>
            <issue>Minor type error in tests</issue>
        </issues>
        <can_fix_without_replan>true</can_fix_without_replan>
        <fix_actions>Fix the type annotation</fix_actions>
    </validation>
    ```
    """

    def __init__(self):
        """Initialize the phase detector with compiled regex patterns."""
        # Regex patterns for fallback extraction
        self._score_patterns = {
            "complexity": re.compile(r"<complexity>\s*([0-9.]+)\s*</complexity>", re.IGNORECASE),
            "need_clarification": re.compile(r"<need_clarification>\s*([0-9.]+)\s*</need_clarification>", re.IGNORECASE),
            "need_web": re.compile(r"<need_web>\s*([0-9.]+)\s*</need_web>", re.IGNORECASE),
            "need_literature": re.compile(r"<need_literature>\s*([0-9.]+)\s*</need_literature>", re.IGNORECASE),
            "need_project_files": re.compile(r"<need_project_files>\s*([0-9.]+)\s*</need_project_files>", re.IGNORECASE),
            "quality": re.compile(r"<quality>\s*([0-9.]+)\s*</quality>", re.IGNORECASE),
            "task_accomplished": re.compile(r"<task_accomplished>\s*([0-9.]+)\s*</task_accomplished>", re.IGNORECASE),
            "verified": re.compile(r"<verified>\s*([0-9.]+)\s*</verified>", re.IGNORECASE),
        }

        # Pattern to extract assessment block
        self._assessment_block_re = re.compile(
            r"<assessment>(.*?)</assessment>",
            re.DOTALL | re.IGNORECASE
        )

        # Pattern to extract plan block
        self._plan_block_re = re.compile(
            r"<plan>(.*?)</plan>",
            re.DOTALL | re.IGNORECASE
        )

        # Pattern to extract validation block
        self._validation_block_re = re.compile(
            r"<validation>(.*?)</validation>",
            re.DOTALL | re.IGNORECASE
        )

        # Pattern to extract steps
        self._step_re = re.compile(r"<step>(.*?)</step>", re.DOTALL | re.IGNORECASE)

        # Pattern to extract issues
        self._issue_re = re.compile(r"<issue>(.*?)</issue>", re.DOTALL | re.IGNORECASE)

        # Pattern to extract questions
        self._question_re = re.compile(r"<question>(.*?)</question>", re.DOTALL | re.IGNORECASE)

    def parse_info_assessment(self, llm_output: str) -> InfoAssessment:
        """
        Parse INFO phase assessment from LLM output.

        Args:
            llm_output: Raw LLM output containing <assessment> block

        Returns:
            InfoAssessment with parsed or default values
        """
        # Try to extract assessment block
        assessment_match = self._assessment_block_re.search(llm_output)
        assessment_text = assessment_match.group(1) if assessment_match else llm_output

        # Try XML parsing first
        result = self._try_xml_parse_assessment(assessment_text if assessment_match else llm_output)
        if result:
            logger.debug("Successfully parsed INFO assessment via XML")
            return result

        # Fall back to regex extraction
        logger.info("XML parsing failed for INFO assessment, using regex fallback")
        return self._regex_parse_assessment(assessment_text)

    def _try_xml_parse_assessment(self, text: str) -> InfoAssessment | None:
        """Try to parse assessment using XML parser."""
        try:
            # Wrap in root element if needed
            if not text.strip().startswith("<assessment"):
                text = f"<assessment>{text}</assessment>"

            root = ET.fromstring(text)

            # Track if we found any expected fields
            fields_found = 0

            def get_float(tag: str, default: float) -> float:
                nonlocal fields_found
                elem = root.find(tag)
                if elem is not None and elem.text:
                    try:
                        fields_found += 1
                        return max(0.0, min(1.0, float(elem.text.strip())))
                    except ValueError:
                        pass
                return default

            def get_text(tag: str) -> str | None:
                nonlocal fields_found
                elem = root.find(tag)
                if elem is not None and elem.text:
                    fields_found += 1
                    return elem.text.strip()
                return None

            # Extract questions
            questions = []
            questions_elem = root.find("clarification_questions")
            if questions_elem is not None:
                for q in questions_elem.findall("question"):
                    if q.text:
                        questions.append(q.text.strip())
                        fields_found += 1

            result = InfoAssessment(
                complexity=get_float("complexity", DEFAULTS["complexity"]),
                need_clarification=get_float("need_clarification", DEFAULTS["need_clarification"]),
                need_web=get_float("need_web", DEFAULTS["need_web"]),
                need_literature=get_float("need_literature", DEFAULTS["need_literature"]),
                need_project_files=get_float("need_project_files", DEFAULTS["need_project_files"]),
                clarification_questions=questions if questions else None,
                web_search_intent=get_text("web_search_intent"),
                literature_search_intent=get_text("literature_search_intent"),
                project_files_intent=get_text("project_files_intent"),
                raw_output=text,
                parsed_successfully=(fields_found > 0),  # Only successful if we found something
            )

            # If no fields found, return None to trigger regex fallback
            if fields_found == 0:
                return None

            return result
        except ET.ParseError as e:
            logger.debug(f"XML parse error for assessment: {e}")
            return None

    def _regex_parse_assessment(self, text: str) -> InfoAssessment:
        """Parse assessment using regex fallback."""
        def extract_score(key: str) -> float:
            pattern = self._score_patterns.get(key)
            if pattern:
                match = pattern.search(text)
                if match:
                    try:
                        return max(0.0, min(1.0, float(match.group(1))))
                    except ValueError:
                        pass
            return DEFAULTS.get(key, 0.5)

        # Extract questions via regex
        questions = self._question_re.findall(text)
        questions = [q.strip() for q in questions if q.strip()]

        # Extract intents via simple tag search
        def extract_intent(tag: str) -> str | None:
            pattern = re.compile(f"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
            match = pattern.search(text)
            if match and match.group(1).strip():
                return match.group(1).strip()
            return None

        return InfoAssessment(
            complexity=extract_score("complexity"),
            need_clarification=extract_score("need_clarification"),
            need_web=extract_score("need_web"),
            need_literature=extract_score("need_literature"),
            need_project_files=extract_score("need_project_files"),
            clarification_questions=questions if questions else None,
            web_search_intent=extract_intent("web_search_intent"),
            literature_search_intent=extract_intent("literature_search_intent"),
            project_files_intent=extract_intent("project_files_intent"),
            raw_output=text,
            parsed_successfully=False,  # Regex fallback means not fully parsed
        )

    def parse_plan(self, llm_output: str) -> PlanV02:
        """
        Parse PLAN phase output from LLM.

        Args:
            llm_output: Raw LLM output containing <plan> block

        Returns:
            PlanV02 with parsed or default values
        """
        # Try to extract plan block
        plan_match = self._plan_block_re.search(llm_output)
        plan_text = plan_match.group(1) if plan_match else llm_output

        # Try XML parsing first
        result = self._try_xml_parse_plan(plan_text if plan_match else llm_output)
        if result:
            logger.debug("Successfully parsed PLAN via XML")
            return result

        # Fall back to regex extraction
        logger.info("XML parsing failed for PLAN, using regex fallback")
        return self._regex_parse_plan(plan_text)

    def _try_xml_parse_plan(self, text: str) -> PlanV02 | None:
        """Try to parse plan using XML parser."""
        try:
            # Wrap in root element if needed
            if not text.strip().startswith("<plan"):
                text = f"<plan>{text}</plan>"

            root = ET.fromstring(text)

            # Track if we found any expected fields
            fields_found = 0

            # Extract quality score
            quality = DEFAULTS["plan_quality"]
            quality_elem = root.find("quality")
            if quality_elem is not None and quality_elem.text:
                try:
                    quality = max(0.0, min(1.0, float(quality_elem.text.strip())))
                    fields_found += 1
                except ValueError:
                    pass

            # Extract steps from <steps> block
            steps = []
            steps_elem = root.find("steps")
            if steps_elem is not None:
                for i, step_elem in enumerate(steps_elem.findall("step"), 1):
                    if step_elem.text:
                        steps.append(PlanStepV02(
                            number=i,
                            description=step_elem.text.strip(),
                        ))
                        fields_found += 1

            # If no <step> tags found, also try numbered list in the raw text
            if not steps:
                numbered_pattern = re.compile(r"^\s*(\d+)[\.\)]\s+(.+)$", re.MULTILINE)
                for match in numbered_pattern.finditer(text):
                    steps.append(PlanStepV02(
                        number=int(match.group(1)),
                        description=match.group(2).strip(),
                    ))
                    fields_found += 1

            # Extract rollback
            rollback = None
            rollback_elem = root.find("rollback")
            if rollback_elem is not None and rollback_elem.text:
                rollback = rollback_elem.text.strip()
                fields_found += 1

            # Extract complexity reassessment
            reassessment = None
            reassess_elem = root.find("complexity_reassessment")
            if reassess_elem is not None and reassess_elem.text:
                reassessment = reassess_elem.text.strip()
                fields_found += 1

            # If no fields found, return None to trigger regex fallback
            if fields_found == 0:
                return None

            return PlanV02(
                steps=steps,
                quality_score=quality,
                rollback_strategy=rollback,
                complexity_reassessment=reassessment,
                raw_output=text,
                parsed_successfully=(fields_found > 0),
            )
        except ET.ParseError as e:
            logger.debug(f"XML parse error for plan: {e}")
            return None

    def _regex_parse_plan(self, text: str) -> PlanV02:
        """Parse plan using regex fallback."""
        # Extract quality
        quality = DEFAULTS["plan_quality"]
        quality_match = self._score_patterns["quality"].search(text)
        if quality_match:
            try:
                quality = max(0.0, min(1.0, float(quality_match.group(1))))
            except ValueError:
                pass

        # Extract steps
        steps = []
        step_texts = self._step_re.findall(text)
        for i, step_text in enumerate(step_texts, 1):
            if step_text.strip():
                steps.append(PlanStepV02(
                    number=i,
                    description=step_text.strip(),
                ))

        # If no <step> tags, try numbered list fallback
        if not steps:
            numbered_pattern = re.compile(r"^\s*(\d+)[\.\)]\s+(.+)$", re.MULTILINE)
            for match in numbered_pattern.finditer(text):
                steps.append(PlanStepV02(
                    number=int(match.group(1)),
                    description=match.group(2).strip(),
                ))

        # Extract rollback
        rollback = None
        rollback_match = re.search(r"<rollback>(.*?)</rollback>", text, re.DOTALL | re.IGNORECASE)
        if rollback_match and rollback_match.group(1).strip():
            rollback = rollback_match.group(1).strip()

        # Extract complexity reassessment
        reassessment = None
        reassess_match = re.search(r"<complexity_reassessment>(.*?)</complexity_reassessment>", text, re.DOTALL | re.IGNORECASE)
        if reassess_match and reassess_match.group(1).strip():
            reassessment = reassess_match.group(1).strip()

        return PlanV02(
            steps=steps,
            quality_score=quality,
            rollback_strategy=rollback,
            complexity_reassessment=reassessment,
            raw_output=text,
            parsed_successfully=False,
        )

    def parse_validation(self, llm_output: str) -> ValidationResult:
        """
        Parse VALIDATE phase output from LLM.

        Args:
            llm_output: Raw LLM output containing <validation> block

        Returns:
            ValidationResult with parsed or default values
        """
        # Try to extract validation block
        val_match = self._validation_block_re.search(llm_output)
        val_text = val_match.group(1) if val_match else llm_output

        # Try XML parsing first
        result = self._try_xml_parse_validation(val_text if val_match else llm_output)
        if result:
            logger.debug("Successfully parsed VALIDATION via XML")
            return result

        # Fall back to regex extraction
        logger.info("XML parsing failed for VALIDATION, using regex fallback")
        return self._regex_parse_validation(val_text)

    def _try_xml_parse_validation(self, text: str) -> ValidationResult | None:
        """Try to parse validation using XML parser."""
        try:
            # Wrap in root element if needed
            if not text.strip().startswith("<validation"):
                text = f"<validation>{text}</validation>"

            root = ET.fromstring(text)

            # Track if we found any expected fields
            fields_found = 0

            def get_float(tag: str, default: float) -> float:
                nonlocal fields_found
                elem = root.find(tag)
                if elem is not None and elem.text:
                    try:
                        fields_found += 1
                        return max(0.0, min(1.0, float(elem.text.strip())))
                    except ValueError:
                        pass
                return default

            def get_bool(tag: str, default: bool) -> bool:
                nonlocal fields_found
                elem = root.find(tag)
                if elem is not None and elem.text:
                    text_val = elem.text.strip().lower()
                    fields_found += 1
                    return text_val in ("true", "yes", "1")
                return default

            def get_text(tag: str) -> str | None:
                nonlocal fields_found
                elem = root.find(tag)
                if elem is not None and elem.text:
                    fields_found += 1
                    return elem.text.strip()
                return None

            # Extract issues
            issues = []
            issues_elem = root.find("issues")
            if issues_elem is not None:
                for issue in issues_elem.findall("issue"):
                    if issue.text:
                        issues.append(issue.text.strip())
                        fields_found += 1

            task_accomplished = get_float("task_accomplished", DEFAULTS["task_accomplished"])
            verified = get_float("verified", DEFAULTS["verified"])
            can_fix = get_bool("can_fix_without_replan", False)
            fix_actions = get_text("fix_actions")

            # If no fields found, return None to trigger regex fallback
            if fields_found == 0:
                return None

            return ValidationResult(
                task_accomplished=task_accomplished,
                verified=verified,
                issues=issues,
                can_fix_without_replan=can_fix,
                fix_actions=fix_actions,
                raw_output=text,
                parsed_successfully=(fields_found > 0),
            )
        except ET.ParseError as e:
            logger.debug(f"XML parse error for validation: {e}")
            return None

    def _regex_parse_validation(self, text: str) -> ValidationResult:
        """Parse validation using regex fallback."""
        # Extract scores
        task_accomplished = DEFAULTS["task_accomplished"]
        ta_match = self._score_patterns["task_accomplished"].search(text)
        if ta_match:
            try:
                task_accomplished = max(0.0, min(1.0, float(ta_match.group(1))))
            except ValueError:
                pass

        verified = DEFAULTS["verified"]
        ver_match = self._score_patterns["verified"].search(text)
        if ver_match:
            try:
                verified = max(0.0, min(1.0, float(ver_match.group(1))))
            except ValueError:
                pass

        # Extract issues
        issues = self._issue_re.findall(text)
        issues = [i.strip() for i in issues if i.strip()]

        # Extract can_fix_without_replan
        can_fix = False
        can_fix_match = re.search(r"<can_fix_without_replan>\s*(true|yes|1|false|no|0)\s*</can_fix_without_replan>", text, re.IGNORECASE)
        if can_fix_match:
            can_fix = can_fix_match.group(1).lower() in ("true", "yes", "1")

        # Extract fix actions
        fix_actions = None
        fix_match = re.search(r"<fix_actions>(.*?)</fix_actions>", text, re.DOTALL | re.IGNORECASE)
        if fix_match and fix_match.group(1).strip():
            fix_actions = fix_match.group(1).strip()

        return ValidationResult(
            task_accomplished=task_accomplished,
            verified=verified,
            issues=issues,
            can_fix_without_replan=can_fix,
            fix_actions=fix_actions,
            raw_output=text,
            parsed_successfully=False,
        )

    def extract_xml_block(self, llm_output: str, block_name: str) -> str | None:
        """
        Extract a named XML block from LLM output.

        Args:
            llm_output: Raw LLM output
            block_name: Name of the block (e.g., "assessment", "plan", "validation")

        Returns:
            Content of the block, or None if not found
        """
        pattern = re.compile(f"<{block_name}>(.*?)</{block_name}>", re.DOTALL | re.IGNORECASE)
        match = pattern.search(llm_output)
        if match:
            return match.group(1).strip()
        return None

    def has_structured_output(self, llm_output: str) -> bool:
        """
        Check if LLM output contains any of the expected structured blocks.

        Args:
            llm_output: Raw LLM output

        Returns:
            True if assessment, plan, or validation block is found
        """
        return bool(
            self._assessment_block_re.search(llm_output) or
            self._plan_block_re.search(llm_output) or
            self._validation_block_re.search(llm_output)
        )
