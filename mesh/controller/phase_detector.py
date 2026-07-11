"""
Phase detection - analyze LLM output to infer task phase transitions.

The Phase Detector analyzes LLM responses and tool calls to determine
what phase a task should be in:

- PLANNING → EXECUTING when LLM starts making file edits
- EXECUTING → NEEDS_CLARIFICATION when LLM asks questions
- EXECUTING → DONE when LLM confirms completion
- * → BLOCKED when LLM identifies external dependencies

This enables automatic task lifecycle management without explicit LLM prompting.
"""

import logging
import re
from typing import Any

from .models import TaskPhase

logger = logging.getLogger(__name__)


class PhaseDetector:
    """
    Detects task phase transitions from LLM output.

    Uses heuristics based on:
    - Tool calls (e.g., file_write → EXECUTING)
    - Response patterns (e.g., "I need to know..." → NEEDS_CLARIFICATION)
    - Completion signals (e.g., "Task complete" → DONE)
    """

    # Patterns indicating the LLM is asking for clarification
    CLARIFICATION_PATTERNS = [
        r"(?:i need to know|please clarify|can you tell me|which .+ should i|what .+ do you want)",
        r"(?:unclear|ambiguous|not sure|need more information|need to understand)",
        r"(?:before .+ proceed|before .+ continue|first .+ need to)",
    ]

    # Stronger patterns - LLM is definitely asking for user input
    CLARIFICATION_REQUEST_PATTERNS = [
        r"could you (?:clarify|explain|tell me|provide)",
        r"what do you mean by",
        r"can you (?:provide|share|give) more (?:details|information|context)",
        r"i need more information",
        r"which (?:option|approach|method) (?:would you|do you) prefer",
        r"do you want me to",
        r"should i .+\?",
        r"would you like me to",
        r"please (?:specify|clarify|confirm)",
        r"are you referring to",
        r"did you mean",
    ]

    # Patterns indicating task completion
    COMPLETION_PATTERNS = [
        r"(?:task (?:is )?(?:complete|done|finished)|implementation (?:is )?complete)",
        r"(?:successfully (?:completed|finished|implemented))",
        r"(?:all (?:done|finished|complete|set))",
    ]

    # Patterns indicating external blocking
    BLOCKING_PATTERNS = [
        r"(?:blocked (?:by|on)|waiting (?:for|on)|depends on .+ (?:not available|missing))",
        r"(?:requires (?:approval|permission|access))",
        r"(?:cannot proceed (?:without|until))",
    ]

    # Patterns indicating step completion
    STEP_COMPLETION_PATTERNS = [
        r"(?:step .+ (?:complete|done|finished))",
        r"(?:completed step)",
        r"(?:that's done|that is done)",
        r"(?:moving (?:on )?to (?:the )?next)",
        r"(?:now (?:let's|i'll|we'll) (?:move on|proceed))",
        r"(?:first step (?:is )?(?:complete|done))",
        r"(?:✓|✔|☑️)",  # Checkmark indicators
    ]

    def __init__(self):
        """Initialize the phase detector."""
        # Compile regex patterns
        self._clarification_re = re.compile(
            "|".join(f"(?:{p})" for p in self.CLARIFICATION_PATTERNS),
            re.IGNORECASE
        )
        self._clarification_request_re = re.compile(
            "|".join(f"(?:{p})" for p in self.CLARIFICATION_REQUEST_PATTERNS),
            re.IGNORECASE
        )
        self._completion_re = re.compile(
            "|".join(f"(?:{p})" for p in self.COMPLETION_PATTERNS),
            re.IGNORECASE
        )
        self._blocking_re = re.compile(
            "|".join(f"(?:{p})" for p in self.BLOCKING_PATTERNS),
            re.IGNORECASE
        )
        self._step_completion_re = re.compile(
            "|".join(f"(?:{p})" for p in self.STEP_COMPLETION_PATTERNS),
            re.IGNORECASE
        )

    def detect_phase(
        self,
        response: str,
        tool_calls: list[Any],
        current_phase: TaskPhase,
    ) -> TaskPhase | None:
        """
        Detect what phase the task should be in based on LLM output.

        Args:
            response: The LLM's text response
            tool_calls: List of tool calls from the LLM
            current_phase: The current task phase

        Returns:
            New phase if a transition is detected, None otherwise
        """
        # Check for blocking conditions first (highest priority)
        if self._blocking_re.search(response):
            logger.info("Detected BLOCKED signal in response")
            return TaskPhase.BLOCKED

        # Check for completion signals
        if self._completion_re.search(response):
            logger.info("Detected DONE signal in response")
            return TaskPhase.DONE

        # Check for clarification requests
        if self._clarification_re.search(response):
            # Only transition to NEEDS_CLARIFICATION if not already executing
            # (during execution, questions might be rhetorical or explanatory)
            if current_phase in (TaskPhase.PLANNING, TaskPhase.NEEDS_CLARIFICATION):
                logger.info("Detected NEEDS_CLARIFICATION signal in response")
                return TaskPhase.NEEDS_CLARIFICATION

        # Check tool calls for phase hints
        file_writing_tools = {"file_write", "file_create", "file_edit", "file_diff"}
        has_file_writes = any(
            getattr(call, "name", "") in file_writing_tools
            for call in tool_calls
        )

        if has_file_writes:
            # File writes indicate execution
            if current_phase == TaskPhase.PLANNING:
                logger.info("Detected file writes - transitioning PLANNING → EXECUTING")
                return TaskPhase.EXECUTING

        # Check for test execution (also indicates executing)
        test_tools = {"bash_exec"}  # Commands that might run tests
        has_test_execution = any(
            getattr(call, "name", "") in test_tools
            for call in tool_calls
        )

        if has_test_execution and current_phase == TaskPhase.PLANNING:
            # Bash execution after planning suggests we're executing the plan
            logger.info("Detected bash execution - transitioning PLANNING → EXECUTING")
            return TaskPhase.EXECUTING

        # No phase transition detected
        return None

    def detect_step_completion(self, response: str, tool_calls: list[Any]) -> bool:
        """
        Detect if the LLM has completed a step in the execution plan.

        This is used to auto-advance the current_step_id when a step is done.

        Detection priority:
        1. Explicit [STEP_DONE] signal (most reliable)
        2. Heuristic pattern matching (fallback)

        Args:
            response: The LLM's text response
            tool_calls: List of tool calls from the LLM

        Returns:
            True if the LLM appears to have completed a step
        """
        # 1. Check for explicit [STEP_DONE] signal (highest priority)
        if "[STEP_DONE]" in response:
            logger.info("Detected explicit [STEP_DONE] signal in LLM output")
            return True

        # 2. Fallback: Check for explicit step completion patterns
        if self._step_completion_re.search(response):
            logger.info("Detected step completion signal in LLM output (heuristic)")
            return True

        # 3. Fallback: If there were file writes and response indicates success
        file_writing_tools = {"file_write", "file_create", "file_edit", "file_diff"}
        has_file_writes = any(
            getattr(call, "name", "") in file_writing_tools
            for call in tool_calls
        )

        if has_file_writes:
            success_patterns = [
                r"(?:successfully|done|complete|updated|created|wrote|edited)",
                r"(?:file (?:has been|was) (?:created|updated|written))",
            ]
            for pattern in success_patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    logger.info("Detected file write success - step likely complete (heuristic)")
                    return True

        return False

    def detect_clarification_request(self, response: str) -> bool:
        """
        Detect if the LLM is asking the user for clarification.

        This is used to trigger a transition to NEEDS_CLARIFICATION phase
        when the LLM itself determines it needs more information.

        Args:
            response: The LLM's text response

        Returns:
            True if the LLM appears to be asking for clarification
        """
        # Check for strong clarification request patterns
        if self._clarification_request_re.search(response):
            logger.info("Detected clarification request in LLM output")
            return True

        # Also check if response ends with a question mark (heuristic)
        # Only count if the last sentence is a question
        sentences = response.strip().split(". ")
        if sentences:
            last_sentence = sentences[-1].strip()
            if last_sentence.endswith("?") and len(last_sentence) > 10:
                # Check it's not a rhetorical question by looking for clarification words
                clarification_words = ["which", "what", "how", "should", "would", "could", "do you"]
                if any(word in last_sentence.lower() for word in clarification_words):
                    logger.info("Detected question ending in LLM output (likely clarification)")
                    return True

        return False

    def extract_plan_steps(self, response: str) -> list[str]:
        """
        Extract plan steps from an LLM response.

        Looks for numbered lists, bullet points, or step-by-step instructions.

        Args:
            response: The LLM's text response

        Returns:
            List of step descriptions
        """
        steps = []

        # Look for numbered steps (1., 2., 3. or 1), 2), 3))
        numbered_pattern = r"^\s*(?:\d+[\.\)])\s+(.+)$"
        for line in response.split("\n"):
            match = re.match(numbered_pattern, line)
            if match:
                step_text = match.group(1).strip()
                if step_text:
                    steps.append(step_text)

        # Look for bullet points (-, *, •)
        if not steps:
            bullet_pattern = r"^\s*[-\*•]\s+(.+)$"
            for line in response.split("\n"):
                match = re.match(bullet_pattern, line)
                if match:
                    step_text = match.group(1).strip()
                    if step_text:
                        steps.append(step_text)

        return steps
