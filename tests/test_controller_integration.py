"""
Tests for controller factory integration.

Verifies that:
1. get_controller() supports phase-flow-v02 mode
2. get_controller_v02() factory works with streaming callbacks
3. StreamingObserver properly forwards events
"""

import pytest

from mesh.controller import (
    get_controller,
    get_controller_v02,
    PhaseFlowController,
    PassthroughController,
    TaskFSMController,
    StreamingObserver,
    CollectingObserver,
    LoggingObserver,
    PhaseEvent,
    PhaseEventType,
)
from mesh.config import ControllerConfig, ControllerConfigV02


class TestGetController:
    """Test the get_controller() factory function."""

    def test_passthrough_mode(self):
        """Passthrough mode returns PassthroughController."""
        controller = get_controller("passthrough")
        assert isinstance(controller, PassthroughController)

    def test_task_fsm_v0_mode(self):
        """task-fsm-v0 mode returns TaskFSMController."""
        controller = get_controller("task-fsm-v0")
        assert isinstance(controller, TaskFSMController)

    def test_phase_flow_v02_mode(self):
        """phase-flow-v02 mode returns PhaseFlowController."""
        controller = get_controller("phase-flow-v02")
        assert isinstance(controller, PhaseFlowController)

    def test_phase_flow_v02_with_config(self):
        """phase-flow-v02 mode respects ControllerConfigV02."""
        config = ControllerConfigV02(effort="high", enable_metrics=True)
        controller = get_controller("phase-flow-v02", config)
        assert isinstance(controller, PhaseFlowController)
        assert controller.config.effort == "high"
        assert controller.config.enable_metrics is True

    def test_phase_flow_v02_default_config(self):
        """phase-flow-v02 without config uses defaults."""
        controller = get_controller("phase-flow-v02")
        assert controller.config.effort == "medium"
        assert controller.config.enable_metrics is False

    def test_phase_flow_v02_wrong_config_type(self):
        """phase-flow-v02 with wrong config type uses defaults."""
        # Pass a v0.1 config to v0.2 controller
        v01_config = ControllerConfig(mode="task-fsm-v0")
        controller = get_controller("phase-flow-v02", v01_config)
        # Should use default v0.2 config, not the v0.1 config
        assert controller.config.effort == "medium"

    def test_unknown_mode_raises(self):
        """Unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown controller mode"):
            get_controller("unknown-mode")

    def test_task_fsm_v1_not_implemented(self):
        """task-fsm-v1 raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            get_controller("task-fsm-v1")


class TestGetControllerV02:
    """Test the get_controller_v02() factory function."""

    def test_default_returns_phase_flow_controller(self):
        """Default call returns PhaseFlowController."""
        controller = get_controller_v02()
        assert isinstance(controller, PhaseFlowController)

    def test_with_config(self):
        """Config is passed to controller."""
        config = ControllerConfigV02(effort="low", max_info_iterations=5)
        controller = get_controller_v02(config=config)
        assert controller.config.effort == "low"
        assert controller.config.max_info_iterations == 5

    def test_with_observer(self):
        """Custom observer is used."""
        collector = CollectingObserver()
        controller = get_controller_v02(observer=collector)
        assert controller._observer is collector

    def test_with_stream_callback(self):
        """Stream callback creates StreamingObserver."""
        messages = []
        async def callback(msg: str):
            messages.append(msg)

        controller = get_controller_v02(stream_callback=callback)
        assert isinstance(controller._observer, StreamingObserver)

    def test_observer_takes_precedence(self):
        """Explicit observer takes precedence over stream_callback."""
        collector = CollectingObserver()
        async def callback(msg: str):
            pass

        controller = get_controller_v02(observer=collector, stream_callback=callback)
        # Observer was provided, so it should be used even though callback was also given
        assert controller._observer is collector

    def test_default_uses_logging_observer(self):
        """No observer or callback uses LoggingObserver."""
        controller = get_controller_v02()
        assert isinstance(controller._observer, LoggingObserver)


class TestStreamingObserver:
    """Test the StreamingObserver implementation."""

    @pytest.mark.asyncio
    async def test_streams_phase_start(self):
        """PHASE_START events are streamed."""
        messages = []
        async def callback(msg: str):
            messages.append(msg)

        observer = StreamingObserver(callback=callback)
        from mesh.controller.models_v02 import FlowPhase
        event = PhaseEvent(
            event_type=PhaseEventType.PHASE_START,
            phase=FlowPhase.INFO,
        )
        await observer.emit(event)

        assert len(messages) == 1
        assert "[PHASE: INFO]" in messages[0]

    @pytest.mark.asyncio
    async def test_streams_flow_start(self):
        """FLOW_START events are streamed."""
        messages = []
        async def callback(msg: str):
            messages.append(msg)

        observer = StreamingObserver(callback=callback)
        event = PhaseEvent(
            event_type=PhaseEventType.FLOW_START,
            data={"message_preview": "test"},
        )
        await observer.emit(event)

        assert len(messages) == 1
        assert "[FLOW:" in messages[0]

    @pytest.mark.asyncio
    async def test_skips_tool_events_by_default(self):
        """TOOL_CALL and TOOL_RESULT are skipped by default."""
        messages = []
        async def callback(msg: str):
            messages.append(msg)

        observer = StreamingObserver(callback=callback)

        await observer.emit(PhaseEvent(event_type=PhaseEventType.TOOL_CALL))
        await observer.emit(PhaseEvent(event_type=PhaseEventType.TOOL_RESULT))

        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_includes_tool_events_when_enabled(self):
        """Tool events are streamed when include_tool_events=True."""
        messages = []
        async def callback(msg: str):
            messages.append(msg)

        observer = StreamingObserver(callback=callback, include_tool_events=True)

        await observer.emit(PhaseEvent(
            event_type=PhaseEventType.TOOL_CALL,
            data={"tool": "file_read"},
        ))

        assert len(messages) == 1
        assert "file_read" in messages[0]

    @pytest.mark.asyncio
    async def test_skips_phase_end(self):
        """PHASE_END events are skipped to reduce noise."""
        messages = []
        async def callback(msg: str):
            messages.append(msg)

        observer = StreamingObserver(callback=callback)
        from mesh.controller.models_v02 import FlowPhase
        await observer.emit(PhaseEvent(
            event_type=PhaseEventType.PHASE_END,
            phase=FlowPhase.INFO,
        ))

        assert len(messages) == 0


class TestStreamingObserverProtocol:
    """Test that StreamingObserver implements ObservabilityEmitter protocol."""

    def test_is_emitter(self):
        """StreamingObserver satisfies ObservabilityEmitter protocol."""
        from mesh.controller import ObservabilityEmitter

        async def callback(msg: str):
            pass

        observer = StreamingObserver(callback=callback)
        assert isinstance(observer, ObservabilityEmitter)
