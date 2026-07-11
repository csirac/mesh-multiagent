"""
Router V3 - Planning pipeline extension of RouterV2.

Subclasses RouterV2 to add a plan-refine loop triggered by regex
detection of "plan" in user messages. All V2 behavior (DIRECT,
NEEDS_WORKER) is inherited unchanged; V3 only overrides on_message()
to intercept plan triggers before falling through to super().

State machine extension:
- PLANNING state with sub-phases: GENERATING, VALIDATING, REVISING,
  AWAITING_INPUT, COMPLETE
- Plan trigger via regex (word-boundary match on "plan")
- Abort via regex (abort/cancel/stop/nevermind)

See docs/router_v3_planning_pipeline_spec.md for full spec.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Awaitable, Any, TYPE_CHECKING

from .router_v2 import RouterV2, RouterState, RouterV2Config
from .protocol import Message
from .conversation_history import Turn

if TYPE_CHECKING:
    from .llm import LLMClient, HistoryMessage
    from .tools import ToolRegistry, ToolCall

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLAN_TRIGGER_PATTERN = re.compile(r'\bplan\s+it\s+out\b', re.IGNORECASE)

ABORT_PATTERN = re.compile(
    r'\b(abort|cancel|stop|nevermind|never\s*mind)\b',
    re.IGNORECASE,
)

# Tools available during plan phases (planner, validator, reviser all share this set).
# file_edit and send_message are excluded; file_write is prompt-restricted to plan dir.
PLANNING_TOOL_NAMES = [
    "file_read", "file_write", "file_create",
    "grep", "glob", "bash_exec",
    "exa_search", "exa_fetch_full",
    "notes_search", "notes_get", "notes_list",
    "memory_search",
]

PLANNING_INSTRUCTIONS = """\
You are a planning agent. Your job is to create a detailed, actionable plan
based on the conversation above.

## Guidelines

1. Identify the user's planning request from the conversation history
2. Explore the codebase using your tools to understand the current state
   of relevant components before planning
3. Pick a short, descriptive name for this plan (e.g., "auth_redesign",
   "rolling_window_history", "plaid_integration"). This will be the filename.
4. Break the task into numbered steps
5. For each step, specify:
   - What needs to be done
   - Which files/components are involved (if applicable)
   - What the expected outcome is
6. Identify dependencies between steps
7. Call out any assumptions you're making
8. Note anything that needs user clarification

## Output

Write the plan to: ~/.mesh/plans/{nickname}/{your_plan_name}.md
Use whatever structure best fits the task. Common formats:
- Numbered step lists for sequential tasks
- Dependency graphs for parallel work
- Decision trees for tasks with conditional paths

IMPORTANT: You MUST write the plan to a file. Do NOT just output the plan
as text — write it to the path above using file_write or file_create.\
"""

VALIDATION_INSTRUCTIONS = """\
You are a plan validator. Your job is to actively verify the plan — not
just read it passively.

First, read the plan file at {plan_file_path}. If no plan file exists
at this path, immediately return REVISE with feedback instructing the
planner to write the plan to a file.

## Validation Criteria

1. **Internal consistency**: Do the steps follow logically? Are there
   contradictions, circular dependencies, or missing steps?

2. **External consistency**: If the plan involves components that interface
   with other parts of the system (APIs, data models, existing code),
   verify those interfaces are correctly accounted for. USE YOUR TOOLS
   to read the relevant source files and confirm assumptions.

3. **Coherence**: Does the plan actually address the user's request?
   Is the scope right — not too narrow, not too broad?

4. **Executability**: Can each step be concretely carried out with the
   available tools and context? Are there blockers or prerequisites
   that aren't addressed?

## Validation Process

- Read relevant source files to verify interface assumptions
- Check that referenced functions, classes, and config fields exist
- Verify that the plan's modifications are compatible with existing code
- Identify any ambiguities that need user input

## Output

End your response with a JSON block on its own line with one of these verdicts:

PASS — plan is validated and ready for execution:
{{"verdict": "PASS"}}

REVISE — plan has issues that can be fixed without user input:
{{"verdict": "REVISE", "feedback": "Specific issues found and what to fix..."}}

NEEDS_INPUT — something requires user clarification before proceeding:
{{"verdict": "NEEDS_INPUT", "question": "The specific question for the user..."}}\
"""

REVISION_INSTRUCTIONS = """\
You are revising a plan based on the feedback in the conversation above.
The reviewer's findings (including any source files they checked) are
already in context.

## Instructions

1. Read the current plan from: {plan_file_path}
2. Address each issue raised by the reviewer
3. Use your tools to verify any additional details if needed
4. Write the complete updated plan back to the same file (overwrite)
5. Maintain the same format as the original\
"""

MAX_PLAN_TOOL_ITERATIONS = 30

# ---------------------------------------------------------------------------
# Router LLM prompt templates for planning responses (Phase 2)
# ---------------------------------------------------------------------------

ROUTER_INSTRUCTIONS_PLAN_ACK = """
MODE: PLANNING — The user just triggered a planning request.

The user's request is in your conversation history. Acknowledge that
you're starting the planning process. Be brief and natural. Let them
know they can type "abort" to cancel.

Respond with plain text (no JSON).
"""

ROUTER_INSTRUCTIONS_PLAN_INPUT_ACK = """
MODE: PLANNING — The user has provided clarifying input for the plan.

The user responded to a question the planning pipeline asked. Their
input is in your conversation history. Acknowledge their input briefly
and let them know that planning is resuming. They can still type "abort"
to cancel.

Respond with plain text (no JSON).
"""

ROUTER_INSTRUCTIONS_PLANNING_STATUS = """
MODE: PLANNING — Generating/validating a plan.

Planning request: "{trigger_summary}"
Current phase: {phase}
Elapsed time: {elapsed:.0f}s
Revisions so far: {revision_count}
Plan file: {plan_file}

Your conversation history includes <worker_activity> entries showing
what the planning pipeline is currently doing (identified by worker_id
"planning-{nickname}"). Use those to give an informed status update.

Briefly acknowledge the new message. Mention what phase planning is in
and what progress has been made. Let them know they can type "abort" to
cancel.

Keep your response short and friendly. Respond with plain text (no JSON).
"""

ROUTER_INSTRUCTIONS_PLAN_COMPLETE = """
Planning is complete.

Plan file: {plan_file}
Revisions: {revision_count}
Validator verdict: {verdict}

The plan content is in your conversation history as a <worker_activity>
entry (worker_id "planning-{nickname}"). Present the plan to the user
clearly. If there's a caveat (max revisions exceeded), note it.

Respond with plain text (no JSON).
"""

ROUTER_INSTRUCTIONS_PLAN_CANCELLED = """
The user cancelled the planning process.

Phase at cancellation: {phase}
Elapsed time: {elapsed:.0f}s

Acknowledge the cancellation briefly. If partial work was done
(plan file exists), mention it's available at {plan_file}.

Respond with plain text (no JSON).
"""

ROUTER_INSTRUCTIONS_PLAN_NEEDS_INPUT = """
MODE: PLANNING — The planning pipeline needs user input.

The validator has a question that requires the user's input before
planning can continue. The question is:

{question}

Present this question to the user clearly. Let them know they can
type "abort" to cancel planning.

Respond with plain text (no JSON).
"""


class PlanPhase(Enum):
    """Sub-states within the PLANNING state."""
    GENERATING = "generating"
    VALIDATING = "validating"
    REVISING = "revising"
    AWAITING_INPUT = "awaiting_input"
    COMPLETE = "complete"


# ---------------------------------------------------------------------------
# RouterV3
# ---------------------------------------------------------------------------

class RouterV3(RouterV2):
    """RouterV2 subclass with planning pipeline.

    Overrides on_message() to intercept plan triggers before falling
    through to super().on_message() for normal V2 classification.
    """

    def __init__(
        self,
        worker_fn: Callable[[list[Any], Message], Awaitable[Any]],
        send_fn: Callable[[str, Message | None], Awaitable[None]],
        config: RouterV2Config | None = None,
        node_id: str = "",
        nickname: str = "",
        agent_type: str = "",
        llm_client: "LLMClient | None" = None,
        system_prompt: str = "",
        identity_block: str = "",
        tools_block: str = "",
        worker_context_fn: Callable[[], list[Any]] | None = None,
        cc_events_fn: Callable[[], list[Any]] | None = None,
        memory_system: Any | None = None,
        session_gap_secs: int = 900,
        # RouterV3-specific args:
        worker_llm_client: "LLMClient | None" = None,
        tool_registry: "ToolRegistry | None" = None,
        execute_tool_fn: Callable[..., Awaitable[list[str]]] | None = None,
        # Pass through any additional RouterV2 kwargs (e.g. flush_interval_tools)
        **kwargs,
    ):
        super().__init__(
            worker_fn=worker_fn,
            send_fn=send_fn,
            config=config,
            node_id=node_id,
            nickname=nickname,
            agent_type=agent_type,
            llm_client=llm_client,
            system_prompt=system_prompt,
            identity_block=identity_block,
            tools_block=tools_block,
            worker_context_fn=worker_context_fn,
            cc_events_fn=cc_events_fn,
            memory_system=memory_system,
            session_gap_secs=session_gap_secs,
            **kwargs,
        )

        # Planning infrastructure from AgentNode
        self._worker_llm_client = worker_llm_client
        self._tool_registry = tool_registry
        self._execute_tool_fn = execute_tool_fn

        # Planning state
        self._plan_phase: PlanPhase | None = None
        self._plan_trigger: Message | None = None
        self._plan_revision_count: int = 0
        self._plan_max_revisions: int = 3
        self._plan_cancelled: bool = False
        self._plan_file_path: str | None = None
        self._planning_task: asyncio.Task | None = None
        self._planning_start_time: float | None = None

    # ------------------------------------------------------------------
    # on_message override
    # ------------------------------------------------------------------

    async def on_message(self, msg: Message) -> None:
        """Handle incoming message with plan trigger detection.

        Order of checks:
        1. If in PLANNING state, handle planning-specific messages
        2. If IDLE and message matches plan trigger, enter planning
        3. Otherwise fall through to RouterV2.on_message()
        """
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        async with self._state_lock:
            # --- Store in history (same as V2's on_message preamble) ---
            ts = msg.timestamp
            if isinstance(ts, str) and ts:
                try:
                    ts = datetime.fromisoformat(ts)
                except ValueError:
                    ts = datetime.now(timezone.utc)
            elif not ts:
                ts = datetime.now(timezone.utc)

            self._history.append(Turn(
                role="incoming",
                content=content,
                timestamp=ts,
                from_node=msg.from_node or "",
                to_node=msg.to_node,
            ))

            if msg.from_node and msg.from_node.startswith("user:"):
                self._latest_user_message = content

            self._check_and_trigger_summarization()

            logger.debug(
                f"RouterV3 on_message: state={self._state}, "
                f"plan_phase={self._plan_phase}, from={msg.from_node}"
            )

            # 1. PLANNING state — handle planning messages
            if self._state == RouterState.PLANNING:
                await self._handle_planning_message(msg, content)
                return

            # 2. IDLE + plan trigger from user → enter planning
            if (
                self._state == RouterState.IDLE
                and msg.from_node
                and msg.from_node.startswith("user:")
                and PLAN_TRIGGER_PATTERN.search(content)
            ):
                await self._handle_planning(msg)
                return

        # 3. Fall through to V2 for everything else
        # V2's on_message also acquires _state_lock, so we must release
        # ours first. But we already stored the message in history above,
        # so we need to skip V2's history-append. Use the parent's handler
        # methods directly instead.
        async with self._state_lock:
            logger.debug(f"RouterV3 falling through to V2 handlers")
            if self._state == RouterState.IDLE:
                if self._config.llm_enabled and self._llm_client:
                    await self._handle_idle_with_llm(msg)
                else:
                    await self._start_worker(msg)
            elif self._state == RouterState.BUSY:
                worker_id = self._current_worker_id
                pending_trigger = self._pending_trigger
                worker_start_time = self._worker_start_time

                if self._config.llm_enabled and self._llm_client:
                    await self._handle_busy_with_llm(
                        msg, worker_id, pending_trigger, worker_start_time
                    )
                else:
                    await self._handle_busy(
                        msg, worker_id, pending_trigger, worker_start_time
                    )

    # ------------------------------------------------------------------
    # Planning state handlers
    # ------------------------------------------------------------------

    async def _handle_planning(self, msg: Message) -> None:
        """Enter the plan-refine loop.

        Sets state, sends ack via router LLM, and launches the planning
        pipeline as a background task — releasing the lock so new messages
        can be processed during planning.
        """
        self._state = RouterState.PLANNING
        self._plan_phase = PlanPhase.GENERATING
        self._plan_trigger = msg
        self._plan_revision_count = 0
        self._plan_cancelled = False
        self._plan_file_path = None
        self._planning_start_time = time.monotonic()

        # Router LLM ack
        await self._planning_send_ack(msg)

        # Launch planning as background task (releases the lock)
        self._planning_task = asyncio.create_task(
            self._run_planning_pipeline(msg)
        )

    async def _run_planning_pipeline(self, trigger: Message) -> None:
        """Run the full planning pipeline as a background task.

        Runs WITHOUT _state_lock. Only acquires the lock for brief state
        transitions via lock-acquiring methods (_plan_complete,
        _plan_await_input, error handler cleanup).
        """
        try:
            # Step 1: Generate plan
            await self._plan_generate(trigger)

            if self._plan_cancelled:
                return  # cancel_planning() handles cleanup and response

            # Step 2: Validate/revise loop
            await self._plan_validate_revise_loop(trigger)

        except asyncio.CancelledError:
            # Do NOT acquire _state_lock here — cancel_planning() already
            # holds it and will handle cleanup. Just log and re-raise.
            logger.info("Planning pipeline cancelled")
            raise

        except Exception as e:
            logger.exception(f"Planning failed: {e}")
            async with self._state_lock:
                await self._planning_send_error(trigger, e)
                self._cleanup_planning_state()

    async def _handle_planning_message(self, msg: Message, content: str) -> None:
        """Handle user message while in PLANNING state.

        Called from on_message() under _state_lock. Uses
        _cancel_planning_inner() (not cancel_planning()) because the
        lock is already held.
        """
        is_from_user = msg.from_node and msg.from_node.startswith("user:")
        if not is_from_user:
            return

        # AWAITING_INPUT — user response to a NEEDS_INPUT question
        if self._plan_phase == PlanPhase.AWAITING_INPUT:
            if ABORT_PATTERN.search(content):
                await self._cancel_planning_inner()
                return
            await self._handle_planning_input(msg)
            return

        # Active phase (GENERATING, VALIDATING, REVISING) — check for abort
        if ABORT_PATTERN.search(content):
            await self._cancel_planning_inner()
            return

        # Non-abort message during active planning — classify first (Phase 6)
        try:
            classification = await self._classify_message(msg)
            if not classification.get("needs_response", True):
                return
        except Exception:
            pass  # Default to responding

        # Router LLM status update
        await self._planning_send_status(msg)

    async def _handle_planning_input(self, msg: Message) -> None:
        """Handle user response to a NEEDS_INPUT question during planning.

        Acks the input via router LLM, then resumes the planning pipeline
        as a new background task.
        """
        # Ack with input-specific prompt (not the plan-trigger ack)
        await self._planning_send_ack(
            msg, instructions=ROUTER_INSTRUCTIONS_PLAN_INPUT_ACK
        )

        # Resume planning pipeline as background task
        self._planning_task = asyncio.create_task(
            self._resume_planning_after_input(msg)
        )

    async def _resume_planning_after_input(self, msg: Message) -> None:
        """Resume planning after user input (runs as background task).

        Runs WITHOUT _state_lock, matching _run_planning_pipeline().
        """
        try:
            self._plan_phase = PlanPhase.REVISING
            self._plan_revision_count += 1
            await self._plan_revise(self._plan_trigger)

            if self._plan_cancelled:
                return  # cancel_planning() handles cleanup and response

            await self._plan_validate_revise_loop(self._plan_trigger)

        except asyncio.CancelledError:
            logger.info("Planning pipeline cancelled during input handling")
            raise

        except Exception as e:
            logger.exception(f"Planning revision after input failed: {e}")
            async with self._state_lock:
                await self._planning_send_error(self._plan_trigger, e)
                self._cleanup_planning_state()

    # ------------------------------------------------------------------
    # Core plan call — LLM with tool loop
    # ------------------------------------------------------------------

    def _is_cc_backend(self) -> bool:
        """Check if the worker LLM client uses the claude-code backend."""
        return (
            self._worker_llm_client is not None
            and self._worker_llm_client.config.backend == "claude-code"
        )

    async def _plan_call(
        self, system_instructions: str, tool_names: list[str]
    ) -> str:
        """Direct LLM call with tool loop for plan phases.

        For claude-code backend: makes a single complete() call and lets CC
        use its own internal tools (Read, Bash, Write, etc.). No external
        tool loop needed.

        For other backends: builds history, calls complete_with_tools() in
        a loop until no more tool calls, executing tools via the AgentNode
        callback.

        Returns the LLM's final text response.
        """
        if self._is_cc_backend():
            return await self._plan_call_cc(system_instructions)
        return await self._plan_call_tools(system_instructions, tool_names)

    async def _plan_call_cc(self, system_instructions: str) -> str:
        """Plan call for claude-code backend.

        CC uses its own internal tools (Read, Bash, Write, Grep, Glob)
        so we make a single complete() call. CC handles all tool execution
        internally and returns a final text response.

        Uses _build_router_prompt() so planner/validator/reviser get the
        full agent prompt (identity, memory, preferences) with mode-specific
        instructions in the <instructions> block.
        """
        prompt = await self._build_router_prompt(system_instructions)

        logger.info("Plan call (CC backend): single complete() call, CC handles tools internally")

        response_text = await self._worker_llm_client.complete(prompt)

        # Store in shared history
        self._history.append(Turn(
            role="assistant",
            content=response_text,
            timestamp=datetime.now(timezone.utc),
            from_node=self._node_id,
        ))

        return response_text

    async def _plan_call_tools(
        self, system_instructions: str, tool_names: list[str]
    ) -> str:
        """Plan call for backends with external tool loop (OpenAI, Anthropic, etc.).

        Builds history from shared ConversationHistory, calls the worker
        LLM client with the full agent prompt (via _build_router_prompt()),
        and loops on tool calls until the LLM produces a final text response.

        Tool call results are appended to both the local messages list
        (for the current call's context) and the shared history (so
        subsequent phases see them).
        """
        from .llm import HistoryMessage

        messages = self._history.build_context_for_llm()

        # Build the full system prompt with identity, memory, preferences,
        # and mode-specific instructions in the <instructions> block.
        system_prompt = await self._build_router_prompt(system_instructions)

        for iteration in range(MAX_PLAN_TOOL_ITERATIONS):
            if self._plan_cancelled:
                return ""

            response_text, tool_calls = (
                await self._worker_llm_client.complete_with_tools(
                    history=messages,
                    node_id=self._node_id,
                    system_prompt=system_prompt,
                    tool_registry=self._tool_registry,
                    tool_names=tool_names,
                )
            )

            if not tool_calls:
                # Final text response — store in shared history
                self._history.append(Turn(
                    role="assistant",
                    content=response_text,
                    timestamp=datetime.now(timezone.utc),
                    from_node=self._node_id,
                ))
                return response_text

            # Track plan file path from file_write/file_create tool calls
            for tc in tool_calls:
                if tc.name in ("file_write", "file_create"):
                    path = tc.arguments.get("path", "")
                    if "/.mesh/plans/" in path:
                        from .paths import resolve_path
                        self._plan_file_path = resolve_path(path)

            # Execute tool calls via the AgentNode callback
            tool_results_str = await self._execute_tool_fn(tool_calls)

            # Check for cancellation after each tool execution
            if self._plan_cancelled:
                return ""

            ts = datetime.now(timezone.utc).isoformat()

            # For OpenAI native tools, response might be empty — synthesize
            response_for_history = response_text
            if not response_text and tool_calls:
                response_for_history = "\n".join(
                    tc.raw_xml for tc in tool_calls if hasattr(tc, "raw_xml")
                )

            # Append to local messages for this call's context
            messages.append(HistoryMessage(
                from_node=self._node_id,
                content=response_for_history,
                timestamp=ts,
                source="in_flight",
            ))
            messages.append(HistoryMessage(
                from_node="system",
                content=f"Tool execution results:\n{tool_results_str}",
                timestamp=ts,
                source="in_flight",
            ))

            # Append to shared history so subsequent phases see tool calls
            self._history.append(Turn(
                role="assistant",
                content=response_for_history,
                timestamp=datetime.now(timezone.utc),
                from_node=self._node_id,
            ))
            self._history.append(Turn(
                role="tool",
                content=f"Tool execution results:\n{tool_results_str}",
                timestamp=datetime.now(timezone.utc),
            ))

            logger.debug(
                f"Plan call iteration {iteration + 1}: "
                f"{len(tool_calls)} tool call(s)"
            )

        # Hit iteration limit
        logger.warning(
            f"Plan call hit max iterations ({MAX_PLAN_TOOL_ITERATIONS})"
        )
        return response_text

    # ------------------------------------------------------------------
    # Plan generation
    # ------------------------------------------------------------------

    async def _plan_generate(self, trigger: Message) -> None:
        """Generate a plan using the worker LLM with planning tools."""
        self._plan_phase = PlanPhase.GENERATING

        # Ensure plan directory exists
        from .paths import resolve_path
        plan_dir = resolve_path(f"~/.mesh/plans/{self._nickname}")
        os.makedirs(plan_dir, exist_ok=True)

        # Snapshot existing plan files before generation (for CC backend detection)
        existing_files = set()
        if os.path.isdir(plan_dir):
            existing_files = set(os.listdir(plan_dir))

        prompt = PLANNING_INSTRUCTIONS.replace("{nickname}", self._nickname)

        logger.info(f"Starting plan generation for trigger: "
                     f"{trigger.content[:80] if trigger.content else ''}...")

        await self._plan_call(prompt, PLANNING_TOOL_NAMES)

        # For CC backend: CC writes files internally, so we don't see tool
        # calls. Scan the plan directory for new files to set _plan_file_path.
        if not self._plan_file_path and os.path.isdir(plan_dir):
            current_files = set(os.listdir(plan_dir))
            new_files = current_files - existing_files
            if new_files:
                # Pick the most recently modified new file
                newest = max(
                    new_files,
                    key=lambda f: os.path.getmtime(os.path.join(plan_dir, f)),
                )
                self._plan_file_path = os.path.join(plan_dir, newest)
                logger.info(f"CC backend: detected new plan file: {self._plan_file_path}")

        logger.info(
            f"Plan generation complete. "
            f"Plan file: {self._plan_file_path or '(none written)'}"
        )

    # ------------------------------------------------------------------
    # Plan validation
    # ------------------------------------------------------------------

    async def _plan_validate(self, trigger: Message) -> dict:
        """Validate a plan. Returns dict with verdict and optional feedback."""
        import json as _json

        self._plan_phase = PlanPhase.VALIDATING

        plan_path = self._plan_file_path or "(no plan file written)"
        prompt = VALIDATION_INSTRUCTIONS.replace("{plan_file_path}", plan_path)

        logger.info(f"Starting plan validation for: {plan_path}")

        response = await self._plan_call(prompt, PLANNING_TOOL_NAMES)

        # Parse the verdict from the response — look for JSON block
        result = {"verdict": "PASS"}  # default if parsing fails
        for line in reversed(response.strip().splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    parsed = _json.loads(line)
                    if "verdict" in parsed:
                        result = parsed
                        break
                except _json.JSONDecodeError:
                    continue

        logger.info(f"Validation result: {result.get('verdict')}")
        return result

    # ------------------------------------------------------------------
    # Plan revision
    # ------------------------------------------------------------------

    async def _plan_revise(self, trigger: Message) -> None:
        """Revise a plan based on validator feedback in shared history."""
        self._plan_phase = PlanPhase.REVISING

        plan_path = self._plan_file_path or "(no plan file written)"
        prompt = REVISION_INSTRUCTIONS.replace("{plan_file_path}", plan_path)

        logger.info(f"Starting plan revision #{self._plan_revision_count}")

        await self._plan_call(prompt, PLANNING_TOOL_NAMES)

        logger.info(f"Plan revision #{self._plan_revision_count} complete")

    # ------------------------------------------------------------------
    # Validate/revise loop
    # ------------------------------------------------------------------

    async def _plan_validate_revise_loop(self, trigger: Message) -> None:
        """Shared validate/revise loop (runs in background task).

        Cancel checks just return — _cancel_planning_inner() handles
        cleanup and response. NEEDS_INPUT goes through _plan_await_input()
        which acquires the lock internally.
        """
        while self._plan_revision_count <= self._plan_max_revisions:
            # cancel_planning_inner() handles cleanup and response:
            if self._plan_cancelled:
                return

            result = await self._plan_validate(trigger)

            if result.get("verdict") == "PASS":
                self._plan_phase = PlanPhase.COMPLETE
                await self._plan_complete(trigger)
                return

            if result.get("verdict") == "NEEDS_INPUT":
                question = result.get("question", "I have a question about the plan.")
                await self._plan_await_input(trigger, question)
                # State stays PLANNING/AWAITING_INPUT — next user message
                # re-enters via _handle_planning_input()
                return

            # REVISE — feedback is already in shared history
            self._plan_revision_count += 1
            await self._plan_revise(trigger)

        # Max revisions exceeded — send best-effort plan with caveat
        await self._plan_complete(trigger, caveat=True)

    # ------------------------------------------------------------------
    # Plan completion
    # ------------------------------------------------------------------

    async def _plan_complete(
        self, trigger: Message, caveat: bool = False
    ) -> None:
        """Read plan from file, send to user via router LLM, return to IDLE.

        Acquires _state_lock internally — called from the background pipeline.
        """
        plan_text = ""

        if self._plan_file_path and os.path.exists(self._plan_file_path):
            try:
                with open(self._plan_file_path, "r") as f:
                    plan_text = f.read()
            except Exception as e:
                logger.error(f"Failed to read plan file: {e}")
                plan_text = "(Could not read plan file)"

            # Add metadata comment to plan file
            meta = {
                "status": "validated" if not caveat else "best_effort",
                "revisions": self._plan_revision_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": self._nickname,
            }
            try:
                import json as _json
                meta_comment = (
                    f"<!-- plan_meta: {_json.dumps(meta)} -->\n\n"
                )
                with open(self._plan_file_path, "w") as f:
                    f.write(meta_comment + plan_text)
            except Exception as e:
                logger.warning(f"Failed to write plan metadata: {e}")
        else:
            plan_text = "(No plan file was written during planning.)"

        # Store plan as a Turn with meta for downstream use
        self._history.append(Turn(
            role="assistant",
            content=plan_text,
            timestamp=datetime.now(timezone.utc),
            from_node=self._node_id,
            meta={"plan": True, "plan_path": self._plan_file_path or ""},
        ))

        # Router LLM generates the delivery response
        async with self._state_lock:
            # Guard against cancel that arrived while we were waiting for lock
            if self._plan_cancelled:
                return  # cancel_planning() already handled cleanup and response
            logger.info(
                f"Plan complete (revisions={self._plan_revision_count}, "
                f"caveat={caveat}, path={self._plan_file_path})"
            )
            await self._planning_send_complete(trigger, plan_text, caveat)
            self._cleanup_planning_state()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_planning_state(self) -> None:
        """Reset all planning state fields and return to IDLE."""
        self._state = RouterState.IDLE
        self._plan_phase = None
        self._plan_trigger = None
        self._plan_file_path = None
        self._plan_cancelled = False
        self._plan_revision_count = 0
        self._planning_task = None
        self._planning_start_time = None
        self._ephemeral_peeks.clear()

    # ------------------------------------------------------------------
    # Planning peek (Phase 2b)
    # ------------------------------------------------------------------

    def _store_planning_peek(self) -> None:
        """Store a snapshot of planning progress as an ephemeral peek.

        Analogous to _store_worker_peek() in RouterV2. Uses the same
        _ephemeral_peeks list with a distinct worker_id for planning.
        """
        elapsed = 0.0
        if self._planning_start_time:
            elapsed = time.monotonic() - self._planning_start_time

        phase_name = self._plan_phase.value if self._plan_phase else "unknown"
        plan_file = self._plan_file_path or "(not yet written)"

        activity_parts = [
            f"Phase: {phase_name}",
            f"Elapsed: {elapsed:.0f}s",
            f"Revisions: {self._plan_revision_count}",
            f"Plan file: {plan_file}",
        ]

        # Include last few turns of planning history for context
        recent_planning_turns = []
        for turn in reversed(self._history.window):
            if len(recent_planning_turns) >= 3:
                break
            if hasattr(turn, 'role') and turn.role in ('assistant', 'tool'):
                content = turn.content if isinstance(turn.content, str) else str(turn.content)
                if len(content) > 200:
                    content = content[:197] + "..."
                recent_planning_turns.append(content)

        if recent_planning_turns:
            activity_parts.append("Recent activity:")
            for entry in reversed(recent_planning_turns):
                activity_parts.append(f"  - {entry}")

        activity_text = "\n".join(activity_parts)
        self._ephemeral_peeks.append({
            "worker_activity": activity_text,
            "worker_id": f"planning-{self._nickname}",
        })

    # ------------------------------------------------------------------
    # Router LLM response helpers (Phase 2c)
    # ------------------------------------------------------------------

    async def _planning_send_ack(
        self, trigger: Message,
        instructions: str = ROUTER_INSTRUCTIONS_PLAN_ACK
    ) -> None:
        """Send router-LLM-generated planning ack.

        Args:
            trigger: The message to ack (plan trigger or user input).
            instructions: The prompt template. Defaults to plan-trigger ack.
                Pass ROUTER_INSTRUCTIONS_PLAN_INPUT_ACK for user input acks.
        """
        prompt = await self._build_router_prompt(instructions)
        response = await self._llm_client.complete(prompt)
        await self._send_and_store(response.strip(), in_reply_to=trigger)

    async def _planning_send_status(self, msg: Message) -> None:
        """Send router-LLM-generated planning status update."""
        self._store_planning_peek()

        trigger_summary = self._summarize_trigger(self._plan_trigger)
        elapsed = 0.0
        if self._planning_start_time:
            elapsed = time.monotonic() - self._planning_start_time

        instructions = ROUTER_INSTRUCTIONS_PLANNING_STATUS.format(
            trigger_summary=trigger_summary,
            phase=self._plan_phase.value if self._plan_phase else "unknown",
            elapsed=elapsed,
            revision_count=self._plan_revision_count,
            plan_file=self._plan_file_path or "(not yet written)",
            nickname=self._nickname,
        )
        prompt = await self._build_router_prompt(instructions)
        response = await self._llm_client.complete(prompt)
        await self._send_and_store(response.strip(), in_reply_to=msg)

    async def _planning_send_complete(
        self, trigger: Message, plan_text: str, caveat: bool = False
    ) -> None:
        """Send router-LLM-generated plan completion response."""
        # Store plan text as ephemeral peek so router LLM can see it
        self._ephemeral_peeks.append({
            "worker_activity": f"FINAL PLAN:\n{plan_text}",
            "worker_id": f"planning-{self._nickname}",
        })

        verdict = "best_effort (max revisions exceeded)" if caveat else "PASS"
        instructions = ROUTER_INSTRUCTIONS_PLAN_COMPLETE.format(
            plan_file=self._plan_file_path or "(unknown)",
            revision_count=self._plan_revision_count,
            verdict=verdict,
            nickname=self._nickname,
        )
        prompt = await self._build_router_prompt(instructions)
        response = await self._llm_client.complete(prompt)
        await self._send_and_store(response.strip(), in_reply_to=trigger)

    async def _planning_send_cancelled(self, trigger: Message) -> None:
        """Send router-LLM-generated cancellation acknowledgment."""
        elapsed = 0.0
        if self._planning_start_time:
            elapsed = time.monotonic() - self._planning_start_time

        instructions = ROUTER_INSTRUCTIONS_PLAN_CANCELLED.format(
            phase=self._plan_phase.value if self._plan_phase else "unknown",
            elapsed=elapsed,
            plan_file=self._plan_file_path or "(none)",
        )
        prompt = await self._build_router_prompt(instructions)
        response = await self._llm_client.complete(prompt)
        await self._send_and_store(response.strip(), in_reply_to=trigger)

    async def _planning_send_needs_input(
        self, trigger: Message, question: str
    ) -> None:
        """Send router-LLM-generated NEEDS_INPUT question to the user."""
        instructions = ROUTER_INSTRUCTIONS_PLAN_NEEDS_INPUT.format(
            question=question,
        )
        prompt = await self._build_router_prompt(instructions)
        response = await self._llm_client.complete(prompt)
        await self._send_and_store(response.strip(), in_reply_to=trigger)

    async def _planning_send_error(
        self, trigger: Message, error: Exception
    ) -> None:
        """Send router-LLM-generated error message, with hardcoded fallback."""
        try:
            prompt = await self._build_router_prompt(
                f"Planning failed with error: {error}. Apologize briefly and "
                f"suggest the user can try again."
            )
            response = await self._llm_client.complete(prompt)
            await self._send_and_store(response.strip(), in_reply_to=trigger)
        except Exception:
            # Fallback to hardcoded if router LLM also fails
            await self._send_and_store(
                f"Planning failed with an error: {error}",
                in_reply_to=trigger,
            )

    # ------------------------------------------------------------------
    # AWAITING_INPUT transition (Phase 2f)
    # ------------------------------------------------------------------

    async def _plan_await_input(
        self, trigger: Message, question: str
    ) -> None:
        """Transition to AWAITING_INPUT and send the question to the user.

        Acquires _state_lock internally — called from the background pipeline.
        Mirrors _plan_complete()'s lock-acquiring pattern.
        """
        async with self._state_lock:
            if self._plan_cancelled:
                return  # cancel_planning() already handled cleanup
            self._plan_phase = PlanPhase.AWAITING_INPUT
            self._store_planning_peek()
            await self._planning_send_needs_input(trigger, question)

    # ------------------------------------------------------------------
    # Cancellation (Phase 3)
    # ------------------------------------------------------------------

    async def _cancel_planning_inner(self) -> bool:
        """Cancel in-flight planning pipeline (no lock — caller must hold it).

        Called from _handle_planning_message() which already holds _state_lock.
        External callers should use cancel_planning() instead.

        Returns True if planning was cancelled, False if not running.
        """
        if self._planning_task and not self._planning_task.done():
            self._plan_cancelled = True
            self._planning_task.cancel()
            try:
                await self._planning_task
            except (asyncio.CancelledError, Exception):
                pass

            trigger = self._plan_trigger
            await self._planning_send_cancelled(trigger)
            self._cleanup_planning_state()
            logger.info("Planning pipeline cancelled")
            return True

        return False

    async def cancel_planning(self) -> bool:
        """Cancel in-flight planning pipeline (if any).

        Public API — acquires _state_lock. Do NOT call from code that
        already holds the lock (use _cancel_planning_inner() instead).

        Returns True if planning was cancelled, False if not running.
        """
        async with self._state_lock:
            return await self._cancel_planning_inner()
