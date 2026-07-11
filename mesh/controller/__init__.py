"""
Controller module for message routing and task management.

This module provides controller implementations that sit between
incoming messages and the LLM, allowing for task tracking, routing,
and workflow management.

Controller modes:
- passthrough: Default, preserves existing behavior (direct LLM pass-through)
- task-fsm-v0: Rule-based router + hardcoded phase FSM + small LLM for classification
- task-fsm-v1: Future learned components (RL router, learned phase transitions)
- phase-flow-v02: LLM-scored adaptive phase flow (v0.2, stateless)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Awaitable
    from ..config import ControllerConfig, ControllerConfigV02

from .base import BaseController, ControllerDecision, ControllerContext
from .passthrough import PassthroughController
from .task_fsm import TaskFSMController
from .models import Task, TaskPhase, PlanStep, StepStatus, Resource, EditProposal
from .persistence import TaskPersistence
from .router import RouterLLM
from .phase_detector import PhaseDetector
from .edit_interceptor import EditInterceptor

# v0.2 controller
from .phase_flow import PhaseFlowController
from .models_v02 import FlowPhase, FlowState, ComplexityLevel, InfoAssessment, PlanV02, ValidationResult, FlowMetrics
from .phase_detector_v02 import PhaseDetectorV02
from .observability import ObservabilityEmitter, PhaseEvent, PhaseEventType, CollectingObserver, LoggingObserver, CompositeObserver, StreamingObserver

__all__ = [
    "BaseController",
    "ControllerDecision",
    "ControllerContext",
    "PassthroughController",
    "TaskFSMController",
    "get_controller",
    # Models (v0.1)
    "Task",
    "TaskPhase",
    "PlanStep",
    "StepStatus",
    "Resource",
    "EditProposal",
    # Persistence
    "TaskPersistence",
    # Router
    "RouterLLM",
    # Phase Detection
    "PhaseDetector",
    # Edit Interception
    "EditInterceptor",
    # v0.2 Controller
    "PhaseFlowController",
    "FlowPhase",
    "FlowState",
    "ComplexityLevel",
    "InfoAssessment",
    "PlanV02",
    "ValidationResult",
    "FlowMetrics",
    "PhaseDetectorV02",
    # v0.2 Observability
    "ObservabilityEmitter",
    "PhaseEvent",
    "PhaseEventType",
    "CollectingObserver",
    "LoggingObserver",
    "CompositeObserver",
    "StreamingObserver",
    # Factory
    "get_controller_v02",
]


def get_controller(mode: str, config: "ControllerConfig | ControllerConfigV02 | None" = None) -> BaseController:
    """
    Factory function to get a controller by mode.

    Args:
        mode: Controller mode. Supported modes:
              - "passthrough": Direct LLM pass-through (default)
              - "task-fsm-v0": Rule-based router + FSM (v0.1)
              - "task-fsm-v1": Future learned components
              - "phase-flow-v02": LLM-scored adaptive phase flow (v0.2)
        config: Optional controller configuration (ControllerConfig for v0.1, ControllerConfigV02 for v0.2)

    Returns:
        A controller instance

    Raises:
        ValueError: If mode is unknown
    """
    if mode == "passthrough":
        return PassthroughController()
    elif mode == "task-fsm-v0":
        return TaskFSMController(config)
    elif mode == "task-fsm-v1":
        # Future: learned controller
        raise NotImplementedError("task-fsm-v1 is not yet implemented")
    elif mode == "phase-flow-v02":
        # v0.2 stateless phase-flow controller
        from ..config import ControllerConfigV02 as ConfigClass
        if config is None:
            config = ConfigClass()
        elif not isinstance(config, ConfigClass):
            # Config was passed but is wrong type - create default
            config = ConfigClass()
        return PhaseFlowController(config=config)
    else:
        raise ValueError(f"Unknown controller mode: {mode}")


def get_controller_v02(
    config: "ControllerConfigV02 | None" = None,
    observer: ObservabilityEmitter | None = None,
    stream_callback: "Callable[[str], Awaitable[None]] | None" = None,
) -> PhaseFlowController:
    """
    Factory function for the v0.2 phase-flow controller with observability.

    This is the preferred way to create a v0.2 controller as it allows
    configuring observability at creation time.

    Args:
        config: Controller configuration. If None, uses defaults.
        observer: Optional custom observer. If None, uses LoggingObserver or
                 StreamingObserver if stream_callback is provided.
        stream_callback: Optional async callback for streaming phase updates to users.
                        If provided and observer is None, creates a StreamingObserver.
                        Signature: async def send(message: str) -> None

    Returns:
        Configured PhaseFlowController instance

    Example:
        # Simple usage with logging
        controller = get_controller_v02()

        # With streaming to user
        async def send_to_user(msg: str):
            await connection.send(msg)
        controller = get_controller_v02(stream_callback=send_to_user)

        # With custom observer
        collector = CollectingObserver()
        controller = get_controller_v02(observer=collector)
    """
    from typing import Callable, Awaitable
    from ..config import ControllerConfigV02 as ConfigClass

    if config is None:
        config = ConfigClass()

    # Determine observer
    if observer is not None:
        # Use provided observer
        pass
    elif stream_callback is not None:
        # Create streaming observer
        observer = StreamingObserver(callback=stream_callback)
    else:
        # Default to logging
        observer = LoggingObserver()

    return PhaseFlowController(config=config, observer=observer)
