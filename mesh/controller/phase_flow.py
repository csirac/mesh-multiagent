"""
Phase Flow Controller - v0.2 implementation.

This controller implements a stateless per-message phase flow:
- No task persistence or routing (each message is independent)
- INFO phase assesses complexity and info needs
- Complexity routing determines which phases to execute
- All state is ephemeral and scoped to a single message

Flow:
    Message → INFO → [PLAN] → EXECUTE → [VALIDATE] → [DOCUMENT] → DONE

Unlike v0.1 (TaskFSMController), there is:
- No RouterLLM (INFO phase handles all classification)
- No task IDs or task persistence
- No edit approval (handled by global sandboxing)

Observability:
- Phase transitions are emitted in real-time
- Assessment results, plans, and validation are streamed
- Users can follow progress and debug issues as they occur
"""

import logging
from datetime import datetime
from typing import Any, TYPE_CHECKING

from .base import BaseController, ControllerDecision, ControllerContext
from .models_v02 import (
    FlowPhase,
    FlowState,
    FlowMetrics,
    ComplexityLevel,
    InfoAssessment,
    PlanV02,
    ValidationResult,
)
from .phase_detector_v02 import PhaseDetectorV02
from .templates_v02 import format_phase_block, get_phase_instructions
from .observability import (
    ObservabilityEmitter,
    LoggingObserver,
    PhaseEvent,
    make_phase_start_event,
    make_flow_start_event,
    make_flow_complete_event,
    make_flow_failed_event,
    make_info_assessment_event,
    make_plan_created_event,
    make_plan_revision_event,
    make_validation_event,
    make_clarification_event,
)

if TYPE_CHECKING:
    from ..config import ControllerConfigV02

logger = logging.getLogger(__name__)


class PhaseFlowController(BaseController):
    """
    Stateless phase-flow controller (v0.2).

    Each message goes through an ephemeral flow:
    1. INFO phase - assess complexity and info needs
    2. Complexity router - determine phase path
    3. [PLAN phase] - for MODERATE/HIGH complexity
    4. EXECUTE phase - produce the response
    5. [VALIDATE phase] - for MODERATE/HIGH complexity
    6. [DOCUMENT phase] - for HIGH complexity only
    7. DONE - return result to user

    No state persists between messages. Conversation history
    (managed by agent_node.py) provides continuity.
    """

    def __init__(
        self,
        config: "ControllerConfigV02 | None" = None,
        observer: ObservabilityEmitter | None = None,
    ):
        """
        Initialize the phase flow controller.

        Args:
            config: Controller configuration. If None, uses defaults.
            observer: Optional observability emitter for streaming events.
                     If None, uses LoggingObserver by default.
        """
        from ..config import ControllerConfigV02 as ConfigClass

        if config is None:
            config = ConfigClass()

        self.config = config

        # Initialize phase detector for XML parsing
        self._phase_detector = PhaseDetectorV02()

        # Observability - default to logging observer
        self._observer: ObservabilityEmitter = observer or LoggingObserver()

        # Current flow state (ephemeral, reset for each message)
        self._current_flow: FlowState | None = None

    # -------------------------------------------------------------------------
    # Observability
    # -------------------------------------------------------------------------

    async def _emit(self, event: PhaseEvent) -> None:
        """Emit an observability event to the configured observer."""
        try:
            await self._observer.emit(event)
        except Exception as e:
            # Don't let observability failures break the flow
            logger.warning(f"Observability emit failed: {e}")

    def set_observer(self, observer: ObservabilityEmitter) -> None:
        """Set a new observer (for runtime reconfiguration)."""
        self._observer = observer

    # -------------------------------------------------------------------------
    # BaseController interface
    # -------------------------------------------------------------------------

    async def on_message(
        self, message: Any, context: ControllerContext
    ) -> ControllerDecision:
        """
        Process an incoming message through the phase flow.

        This starts a new ephemeral flow for each message:
        1. Create fresh FlowState
        2. Run INFO phase to assess complexity
        3. Return decision with appropriate system_addendum

        Args:
            message: The incoming Message object
            context: ControllerContext with history, agent_id, etc.

        Returns:
            ControllerDecision with action and system_addendum
        """
        # Extract message text
        message_text = self._extract_message_text(message)
        if not message_text:
            # No text content, pass through without phase flow
            return ControllerDecision(
                action="PROCESS_WITH_LLM",
                payload={"message": message},
            )

        # Create fresh flow state
        self._current_flow = FlowState(
            phase=FlowPhase.INFO,
            original_message=message_text,
            metrics=FlowMetrics() if self.config.enable_metrics else None,
        )

        logger.info(f"Starting phase flow for message: {message_text[:100]}...")

        # Emit observability events
        await self._emit(make_flow_start_event(message_text))
        await self._emit(make_phase_start_event(FlowPhase.INFO))

        # Generate INFO phase instructions
        info_instructions = format_phase_block(FlowPhase.INFO)

        return ControllerDecision(
            action="PROCESS_WITH_LLM",
            payload={"message": message},
            phase=FlowPhase.INFO.value,
            system_addendum=info_instructions,
        )

    async def on_llm_response(
        self,
        response: str,
        tool_calls: list[Any],
        context: ControllerContext,
    ) -> ControllerDecision:
        """
        Process an LLM response and advance the phase flow.

        This is called after each LLM turn. Based on the current phase:
        1. Parse structured output (if expected)
        2. Determine next phase based on complexity routing
        3. Return decision with next phase's system_addendum

        Args:
            response: The LLM's text response
            tool_calls: Any tool calls from the LLM
            context: Current context

        Returns:
            ControllerDecision for next action
        """
        if not self._current_flow:
            # No active flow, pass through
            return ControllerDecision(
                action="PROCESS_WITH_LLM",
                payload={"response": response, "tool_calls": tool_calls},
            )

        flow = self._current_flow
        current_phase = flow.phase

        logger.debug(f"on_llm_response in phase: {current_phase.value}")

        # Handle based on current phase
        if current_phase == FlowPhase.INFO:
            return await self._handle_info_response(response, tool_calls, context)
        elif current_phase == FlowPhase.PLAN:
            return await self._handle_plan_response(response, tool_calls, context)
        elif current_phase == FlowPhase.EXECUTE:
            return await self._handle_execute_response(response, tool_calls, context)
        elif current_phase == FlowPhase.VALIDATE:
            return await self._handle_validate_response(response, tool_calls, context)
        elif current_phase == FlowPhase.DOCUMENT:
            return await self._handle_document_response(response, tool_calls, context)
        else:
            # Terminal phase, just pass through
            return ControllerDecision(
                action="DONE",
                payload={"response": response},
                phase=current_phase.value,
            )

    async def load_state(self) -> None:
        """
        No-op for v0.2 - controller is stateless.
        """
        logger.debug("PhaseFlowController.load_state() - no persistent state")

    async def save_state(self) -> None:
        """
        No-op for v0.2 - controller is stateless.
        """
        logger.debug("PhaseFlowController.save_state() - no persistent state")

    # -------------------------------------------------------------------------
    # Phase handlers
    # -------------------------------------------------------------------------

    async def _handle_info_response(
        self,
        response: str,
        tool_calls: list[Any],
        context: ControllerContext,
    ) -> ControllerDecision:
        """
        Handle response from INFO phase.

        Parses the assessment, determines complexity, and routes to next phase.
        """
        flow = self._current_flow
        assert flow is not None

        # Record iteration
        flow.info_iterations += 1
        if flow.metrics:
            flow.metrics.info_iterations = flow.info_iterations

        # Parse assessment from LLM output
        assessment = self._phase_detector.parse_info_assessment(response)
        flow.info_assessment = assessment

        if not assessment.parsed_successfully:
            logger.warning("INFO assessment parsing used fallback - may have defaults")
            if flow.metrics:
                flow.metrics.xml_parse_failures += 1

        logger.info(
            f"INFO assessment: complexity={assessment.complexity:.2f}, "
            f"clarification={assessment.need_clarification:.2f}, "
            f"web={assessment.need_web:.2f}"
        )

        # Emit assessment event
        await self._emit(make_info_assessment_event(assessment))

        # Check if clarification is needed (blocks other info gathering)
        clarification_threshold = self.config.get_threshold("info")
        if assessment.need_clarification >= clarification_threshold:
            # Need to ask user for clarification
            # The LLM response should already contain the questions
            # Emit clarification event
            questions = assessment.clarification_questions or []
            await self._emit(make_clarification_event(questions))

            # Just mark as done and return the response
            flow.complete()
            await self._emit(make_flow_complete_event(flow))
            return ControllerDecision(
                action="DONE",
                payload={"response": response, "needs_clarification": True},
                phase=FlowPhase.DONE.value,
            )

        # Check if info gathering is needed (and we haven't exceeded iterations)
        if assessment.any_info_needed(clarification_threshold):
            if flow.info_iterations < self.config.max_info_iterations:
                # Let LLM continue with tool calls for info gathering
                # The response likely contains tool calls
                if tool_calls:
                    return ControllerDecision(
                        action="EXECUTE_TOOLS",
                        payload={"tool_calls": tool_calls},
                        phase=FlowPhase.INFO.value,
                        system_addendum=format_phase_block(FlowPhase.INFO),
                    )
                # No tool calls but info needed - this is unexpected
                # Fall through to complexity routing

        # Determine complexity tier
        flow.complexity = self._determine_complexity(assessment.complexity)
        logger.info(f"Complexity routing: {flow.complexity.value}")

        # Record phase completion
        if flow.metrics:
            flow.metrics.record_phase(FlowPhase.INFO.value, 0)  # TODO: actual timing

        # Route based on complexity
        return await self._route_by_complexity(flow, response, tool_calls)

    def _determine_complexity(self, score: float) -> ComplexityLevel:
        """
        Determine complexity tier from score.

        Uses thresholds from config:
        - score < low_threshold -> LOW
        - score >= high_threshold -> HIGH
        - otherwise -> MODERATE
        """
        low_threshold = self.config.get_threshold("complexity_low")
        high_threshold = self.config.get_threshold("complexity_high")

        if score < low_threshold:
            return ComplexityLevel.LOW
        elif score >= high_threshold:
            return ComplexityLevel.HIGH
        else:
            return ComplexityLevel.MODERATE

    async def _route_by_complexity(
        self,
        flow: FlowState,
        response: str,
        tool_calls: list[Any],
    ) -> ControllerDecision:
        """
        Route to appropriate next phase based on complexity.

        LOW: Skip PLAN, go straight to EXECUTE
        MODERATE/HIGH: Go to PLAN first
        """
        previous_phase = flow.phase

        if flow.complexity == ComplexityLevel.LOW:
            # LOW complexity: skip PLAN, go to EXECUTE
            flow.phase = FlowPhase.EXECUTE
            await self._emit(make_phase_start_event(FlowPhase.EXECUTE, previous_phase))
            return ControllerDecision(
                action="PROCESS_WITH_LLM",
                payload={"continue": True},
                phase=FlowPhase.EXECUTE.value,
                system_addendum=self._get_execute_instructions(flow),
            )
        else:
            # MODERATE/HIGH: go to PLAN phase
            flow.phase = FlowPhase.PLAN
            await self._emit(make_phase_start_event(FlowPhase.PLAN, previous_phase))
            return ControllerDecision(
                action="PROCESS_WITH_LLM",
                payload={"continue": True},
                phase=FlowPhase.PLAN.value,
                system_addendum=format_phase_block(FlowPhase.PLAN),
            )

    async def _handle_plan_response(
        self,
        response: str,
        tool_calls: list[Any],
        context: ControllerContext,
    ) -> ControllerDecision:
        """
        Handle response from PLAN phase.

        Parses the plan, checks quality, and either revises or proceeds.
        """
        flow = self._current_flow
        assert flow is not None

        # Record iteration
        flow.plan_iterations += 1
        if flow.metrics:
            flow.metrics.plan_iterations = flow.plan_iterations

        # Parse plan from LLM output
        plan = self._phase_detector.parse_plan(response)

        # Update revision count
        plan.revision_count = flow.plan_iterations

        if not plan.parsed_successfully:
            logger.warning("PLAN parsing used fallback - may have defaults")
            if flow.metrics:
                flow.metrics.xml_parse_failures += 1

        flow.plan = plan

        logger.info(
            f"PLAN parsed: {len(plan.steps)} steps, quality={plan.quality_score:.2f}"
        )

        # Emit plan created event
        await self._emit(make_plan_created_event(plan))

        # Check plan quality threshold
        quality_threshold = self.config.get_threshold("plan_quality")

        if plan.quality_score < quality_threshold:
            # Plan needs revision
            if flow.plan_iterations < self.config.max_plan_iterations:
                logger.info(
                    f"Plan quality {plan.quality_score:.2f} < {quality_threshold}, "
                    f"requesting revision (iteration {flow.plan_iterations})"
                )

                # Emit plan revision event
                await self._emit(make_plan_revision_event(plan, flow.plan_iterations))

                # Request revision with context about what to improve
                revision_context = {
                    "quality_score": plan.quality_score,
                    "threshold": quality_threshold,
                    "iteration": flow.plan_iterations,
                }
                return ControllerDecision(
                    action="PROCESS_WITH_LLM",
                    payload={"revision_needed": True, "context": revision_context},
                    phase=FlowPhase.PLAN.value,
                    system_addendum=self._get_plan_revision_instructions(plan),
                )
            else:
                # Max iterations reached, proceed with current plan
                logger.warning(
                    f"Max plan iterations ({self.config.max_plan_iterations}) reached, "
                    f"proceeding with quality={plan.quality_score:.2f}"
                )

        # Check for complexity reassessment
        if plan.complexity_reassessment:
            logger.info(f"Complexity reassessment: {plan.complexity_reassessment}")
            # Could adjust flow.complexity here if reassessment indicates change

        # Record phase completion
        if flow.metrics:
            flow.metrics.record_phase(FlowPhase.PLAN.value, 0)

        # Proceed to EXECUTE
        previous_phase = flow.phase
        flow.phase = FlowPhase.EXECUTE
        await self._emit(make_phase_start_event(FlowPhase.EXECUTE, previous_phase))
        return ControllerDecision(
            action="PROCESS_WITH_LLM",
            payload={"plan": plan.to_dict()},
            phase=FlowPhase.EXECUTE.value,
            system_addendum=self._get_execute_instructions(flow),
        )

    async def _handle_execute_response(
        self,
        response: str,
        tool_calls: list[Any],
        context: ControllerContext,
    ) -> ControllerDecision:
        """
        Handle response from EXECUTE phase.

        For LOW complexity: mark done
        For MODERATE/HIGH: proceed to VALIDATE
        """
        flow = self._current_flow
        assert flow is not None

        # Record phase completion
        if flow.metrics:
            flow.metrics.record_phase(FlowPhase.EXECUTE.value, 0)

        # If there are tool calls, let them execute
        if tool_calls:
            return ControllerDecision(
                action="EXECUTE_TOOLS",
                payload={"tool_calls": tool_calls},
                phase=FlowPhase.EXECUTE.value,
                system_addendum=self._get_execute_instructions(flow),
            )

        # Route based on complexity
        previous_phase = flow.phase
        if flow.complexity == ComplexityLevel.LOW:
            # LOW complexity: done after execute
            flow.complete()
            await self._emit(make_flow_complete_event(flow))
            return ControllerDecision(
                action="DONE",
                payload={"response": response},
                phase=FlowPhase.DONE.value,
                system_addendum=format_phase_block(FlowPhase.DONE),
            )
        else:
            # MODERATE/HIGH: proceed to VALIDATE
            flow.phase = FlowPhase.VALIDATE
            await self._emit(make_phase_start_event(FlowPhase.VALIDATE, previous_phase))
            return ControllerDecision(
                action="PROCESS_WITH_LLM",
                payload={"continue": True},
                phase=FlowPhase.VALIDATE.value,
                system_addendum=format_phase_block(FlowPhase.VALIDATE),
            )

    async def _handle_validate_response(
        self,
        response: str,
        tool_calls: list[Any],
        context: ControllerContext,
    ) -> ControllerDecision:
        """
        Handle response from VALIDATE phase.

        Checks validation results and either:
        - Loops back for fixes
        - Proceeds to DOCUMENT (HIGH only)
        - Completes (MODERATE)
        """
        flow = self._current_flow
        assert flow is not None

        # Parse validation from LLM output
        validation = self._phase_detector.parse_validation(response)
        flow.validation = validation

        if not validation.parsed_successfully:
            logger.warning("VALIDATE parsing used fallback - may have defaults")
            if flow.metrics:
                flow.metrics.xml_parse_failures += 1

        logger.info(
            f"VALIDATE parsed: accomplished={validation.task_accomplished:.2f}, "
            f"verified={validation.verified:.2f}, issues={len(validation.issues)}"
        )

        # Emit validation event
        await self._emit(make_validation_event(validation))

        # Record phase completion
        if flow.metrics:
            flow.metrics.record_phase(FlowPhase.VALIDATE.value, 0)

        # Check if validation passed
        previous_phase = flow.phase
        if validation.is_successful():
            # Validation passed
            if flow.complexity == ComplexityLevel.HIGH:
                # HIGH complexity: proceed to DOCUMENT
                flow.phase = FlowPhase.DOCUMENT
                await self._emit(make_phase_start_event(FlowPhase.DOCUMENT, previous_phase))
                return ControllerDecision(
                    action="PROCESS_WITH_LLM",
                    payload={"continue": True},
                    phase=FlowPhase.DOCUMENT.value,
                    system_addendum=format_phase_block(FlowPhase.DOCUMENT),
                )
            else:
                # MODERATE: done
                flow.complete()
                await self._emit(make_flow_complete_event(flow))
                return ControllerDecision(
                    action="DONE",
                    payload={"response": response, "validation": validation.to_dict()},
                    phase=FlowPhase.DONE.value,
                )
        else:
            # Validation failed
            if validation.can_fix_without_replan and validation.fix_actions:
                # Can fix in place - loop back to execute
                logger.info(f"Validation failed but can fix: {validation.fix_actions}")
                flow.phase = FlowPhase.EXECUTE
                await self._emit(make_phase_start_event(FlowPhase.EXECUTE, previous_phase))
                return ControllerDecision(
                    action="PROCESS_WITH_LLM",
                    payload={
                        "fix_needed": True,
                        "issues": validation.issues,
                        "fix_actions": validation.fix_actions,
                    },
                    phase=FlowPhase.EXECUTE.value,
                    system_addendum=self._get_fix_instructions(validation),
                )
            else:
                # Cannot fix - fail the flow
                error_msg = f"Validation failed: {', '.join(validation.issues)}"
                flow.fail(error_msg)
                await self._emit(make_flow_failed_event(flow, error_msg))
                return ControllerDecision(
                    action="DONE",
                    payload={
                        "response": response,
                        "error": error_msg,
                        "validation": validation.to_dict(),
                    },
                    phase=FlowPhase.FAILED.value,
                )

    async def _handle_document_response(
        self,
        response: str,
        tool_calls: list[Any],
        context: ControllerContext,
    ) -> ControllerDecision:
        """
        Handle response from DOCUMENT phase.

        This is the final phase for HIGH complexity flows.
        """
        flow = self._current_flow
        assert flow is not None

        # Record phase completion
        if flow.metrics:
            flow.metrics.record_phase(FlowPhase.DOCUMENT.value, 0)

        # If there are tool calls (e.g., file writes for docs), let them execute
        if tool_calls:
            return ControllerDecision(
                action="EXECUTE_TOOLS",
                payload={"tool_calls": tool_calls},
                phase=FlowPhase.DOCUMENT.value,
                system_addendum=format_phase_block(FlowPhase.DOCUMENT),
            )

        # Documentation complete, finish the flow
        flow.complete()
        await self._emit(make_flow_complete_event(flow))
        return ControllerDecision(
            action="DONE",
            payload={"response": response},
            phase=FlowPhase.DONE.value,
        )

    # -------------------------------------------------------------------------
    # Instruction helpers
    # -------------------------------------------------------------------------

    def _get_execute_instructions(self, flow: FlowState) -> str:
        """Get EXECUTE phase instructions with context."""
        base_instructions = get_phase_instructions(FlowPhase.EXECUTE)

        # Add context based on complexity and plan
        context_parts = [base_instructions]

        if flow.complexity:
            context_parts.append(f"\nComplexity: {flow.complexity.value.upper()}")

        if flow.plan and flow.plan.steps:
            context_parts.append("\nPlan to execute:")
            for step in flow.plan.steps:
                context_parts.append(f"  {step.number}. {step.description}")

        return "\n".join(context_parts)

    def _get_plan_revision_instructions(self, plan: PlanV02) -> str:
        """Get instructions for plan revision."""
        base = get_phase_instructions(FlowPhase.PLAN)
        return f"""{base}

---

**REVISION NEEDED**

Your previous plan scored {plan.quality_score:.2f}, below the required threshold.

Please revise your plan to:
1. Be more concrete and actionable
2. Cover all aspects of the request
3. Include appropriate error handling/rollback
4. Self-assess with a higher quality score

Your revised plan:"""

    def _get_fix_instructions(self, validation: ValidationResult) -> str:
        """Get instructions for fixing validation issues."""
        issues_text = "\n".join(f"- {issue}" for issue in validation.issues)
        return f"""**FIX REQUIRED**

The validation phase found issues that need to be fixed:

{issues_text}

Recommended fix actions:
{validation.fix_actions}

Please apply the fixes and then proceed."""

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def _extract_message_text(self, message: Any) -> str:
        """Extract text content from a message."""
        if isinstance(message, str):
            return message
        if hasattr(message, "content"):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, dict):
                return content.get("text", content.get("body", ""))
        if hasattr(message, "text"):
            return message.text
        return str(message) if message else ""

    def get_current_flow(self) -> FlowState | None:
        """Get the current flow state (for testing/debugging)."""
        return self._current_flow

    def get_flow_metrics(self) -> dict | None:
        """Get metrics from the current flow (for monitoring)."""
        if self._current_flow and self._current_flow.metrics:
            return self._current_flow.metrics.to_dict()
        return None
