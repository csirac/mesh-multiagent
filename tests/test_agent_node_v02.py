"""
Tests for agent_node.py v0.2 controller integration.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from mesh.config import NodeConfig, ControllerConfigV02, EffortPreset


class TestAgentNodeV02Integration:
    """Test v0.2 controller integration in agent_node."""

    def test_controller_config_parsing(self):
        """Test that ControllerConfigV02 is created correctly from config."""
        config_dict = {
            "mode": "phase-flow-v02",
            "effort": "high",
            "stream_phase_updates": True,
        }
        config = ControllerConfigV02(**config_dict)
        
        assert config.mode == "phase-flow-v02"
        assert config.effort == "high"
        assert config.stream_phase_updates is True
        assert config.get_effort_preset() == EffortPreset.HIGH

    def test_get_threshold_from_preset(self):
        """Test threshold values from effort preset."""
        config = ControllerConfigV02(effort="high")
        
        # HIGH effort has lower thresholds (more thorough)
        assert config.get_threshold("complexity_low") == 0.2
        assert config.get_threshold("complexity_high") == 0.5
        
        config_low = ControllerConfigV02(effort="low")
        # LOW effort has higher thresholds (quicker)
        assert config_low.get_threshold("complexity_low") == 0.4
        assert config_low.get_threshold("complexity_high") == 0.8

    def test_individual_threshold_override(self):
        """Test that individual threshold overrides work."""
        config = ControllerConfigV02(
            effort="medium",
            complexity_low=0.1,  # Override
        )
        
        # Override takes precedence
        assert config.get_threshold("complexity_low") == 0.1
        # Other thresholds use preset
        assert config.get_threshold("complexity_high") == 0.7  # MEDIUM default

    def test_node_config_controller_parsing(self):
        """Test NodeConfig parses controller config correctly."""
        from mesh.config import MeshConfig
        
        yaml_content = """
nodes:
  agent:test:v02:
    llm_backend: "default"
    tools: []
    controller:
      mode: "phase-flow-v02"
      effort: "high"
"""
        import yaml
        data = yaml.safe_load(yaml_content)
        config = MeshConfig.from_dict(data)
        
        node = config.nodes["agent:test:v02"]
        assert node.controller is not None
        assert isinstance(node.controller, ControllerConfigV02)
        assert node.controller.mode == "phase-flow-v02"
        assert node.controller.effort == "high"


class TestAgentNodeControllerInit:
    """Test controller initialization in AgentNode."""

    @pytest.mark.asyncio
    async def test_v02_controller_created(self):
        """Test that PhaseFlowController is created for v0.2 config."""
        from mesh.controller import PhaseFlowController
        from mesh.agent_node import AgentNode
        
        node_config = NodeConfig(
            id="agent:test:v02",
            controller=ControllerConfigV02(mode="phase-flow-v02"),
        )
        
        # Create agent node (without connecting)
        agent = AgentNode(
            config=node_config,
            persist=False,
        )
        
        assert isinstance(agent.controller, PhaseFlowController)

    @pytest.mark.asyncio
    async def test_passthrough_for_none_controller(self):
        """Test that passthrough controller is used when no config."""
        from mesh.controller import PassthroughController
        from mesh.agent_node import AgentNode
        
        node_config = NodeConfig(
            id="agent:test:passthrough",
            controller=None,
        )
        
        agent = AgentNode(
            config=node_config,
            persist=False,
        )
        
        assert isinstance(agent.controller, PassthroughController)


class TestStreamingObserverSetup:
    """Test streaming observer setup for v0.2 controller."""

    def test_streaming_observer_callback_format(self):
        """Test that StreamingObserver formats events correctly."""
        from mesh.controller import StreamingObserver, PhaseEvent, PhaseEventType, FlowPhase

        received = []

        async def callback(msg: str):
            received.append(msg)

        observer = StreamingObserver(callback=callback)

        # Create a phase start event with proper FlowPhase enum
        event = PhaseEvent(
            event_type=PhaseEventType.PHASE_START,
            phase=FlowPhase.INFO,
            data={}
        )

        # Run emit
        asyncio.get_event_loop().run_until_complete(observer.emit(event))

        assert len(received) == 1
        assert "PHASE: INFO" in received[0] or "INFO" in received[0]
