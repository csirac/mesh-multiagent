#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Start an agent node.

Usage:
    python run_agent.py --agent <type> [--nickname <nick>] [--config PATH]

Examples:
    python run_agent.py --agent coder --nickname alice
    python run_agent.py --agent researcher -n bob
    python run_agent.py --agent coder -n alice --fresh  # Start fresh (no history)
    python run_agent.py agent:coder  # Legacy: node_id style

By default, agents resume their previous session (load and persist history).
Use --fresh or --no-resume to start without history.
"""

import argparse
import asyncio
import logging
import secrets
import signal
import sys
from pathlib import Path

from mesh.config import load_config, NodeConfig, backend_config_to_llm_config
from mesh.agent_node import AgentNode, SimpleAgentNode
from mesh.llm import LLMConfig, LLMClient
from mesh.protocol import build_agent_node_id

# Import tool implementations to register all tools in the global registry
import mesh.tool_implementations  # noqa: F401 - side effect: registers tools

# Reconnect parameters
RECONNECT_DELAY_INITIAL = 1.0    # Start with 1 second
RECONNECT_DELAY_MAX = 60.0       # Cap at 60 seconds
RECONNECT_DELAY_FACTOR = 2.0     # Exponential backoff


def setup_logging(node_id: str, log_dir: str = "logs"):
    """Configure logging to both file and console."""
    # Sanitize node_id for filename (replace : with -)
    safe_name = node_id.replace(":", "-")
    log_path = Path(log_dir) / f"{safe_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # File handler - detailed logs
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)

    # Console handler - less verbose
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    console_handler.setFormatter(console_format)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return str(log_path)


logger = logging.getLogger(__name__)


async def main(
    agent_type: str,
    nickname: str | None = None,
    description: str | None = None,
    config_path: str | None = None,
    legacy_node_id: str | None = None,
    fresh: bool = False,
    history_file: str | None = None,
    soft_limit: int | None = None,
    target_ratio: float | None = None,
    # TLS and auth overrides
    use_tls: bool | None = None,
    router_host: str | None = None,
    router_port: int | None = None,
    auth_token: str | None = None,
    # Backend override
    backend: str | None = None,
    # Model override (within backend)
    model: str | None = None,
    # Conversation loading
    load_conversation: str | None = None,
    # Sandbox settings
    sandboxed: bool = False,
    allowed_dirs: list[str] | None = None,
    allow_network: bool = True,
    # Controller settings
    controller_mode: str | None = None,
    effort: str | None = None,
    # Relevance router settings
    relevance_router: bool = False,
    relevance_threshold: float = 0.7,
):
    config = load_config(config_path)

    # Handle legacy node_id format (e.g., "agent:coder")
    if legacy_node_id:
        # Parse legacy format
        parts = legacy_node_id.split(":")
        if len(parts) >= 2 and parts[0] == "agent":
            agent_type = parts[1]
            if len(parts) >= 3:
                nickname = parts[2]

    # Generate nickname if not provided
    if not nickname:
        nickname = secrets.token_hex(2)

    # Build the full node ID
    node_id = build_agent_node_id(agent_type, nickname)

    # Look up config by agent type (e.g., "agent:coder") for backwards compatibility
    legacy_config_id = f"agent:{agent_type}"

    # Get node config, or create a default one
    if node_id in config.nodes:
        node_config = config.nodes[node_id]
    elif legacy_config_id in config.nodes:
        # Use config from legacy ID but update with new node ID
        node_config = config.nodes[legacy_config_id]
        node_config.id = node_id
    else:
        node_config = NodeConfig(
            id=node_id,
            router_host=config.router.host,
            router_port=config.router.port,
        )

    # Store agent_type and nickname in config
    node_config.agent_type = agent_type
    node_config.nickname = nickname

    # Apply CLI overrides for TLS and auth
    if router_host is not None:
        node_config.router_host = router_host
    if router_port is not None:
        node_config.router_port = router_port
    if use_tls is not None:
        node_config.use_tls = use_tls
    if auth_token is not None:
        node_config.auth_token = auth_token

    # Apply CLI override for controller mode
    if controller_mode is not None:
        from mesh.config import ControllerConfig, ControllerConfigV02
        if controller_mode == "phase-flow-v02":
            # v0.2 controller
            node_config.controller = ControllerConfigV02(
                mode=controller_mode,
                effort=effort or "medium",
            )
            logger.info(f"Controller override: {controller_mode} (effort={effort or 'medium'})")
        else:
            # v0.1 controllers (passthrough, task-fsm-v0)
            node_config.controller = ControllerConfig(mode=controller_mode)
            logger.info(f"Controller override: {controller_mode}")

    # Get LLM config for this node (try both new and legacy IDs)
    # CLI --backend overrides config file
    if backend:
        llm_backend_config = config.llm_backends.get(backend)
        if not llm_backend_config:
            logger.error(f"Backend '{backend}' not found in config. Available: {list(config.llm_backends.keys())}")
            return
    else:
        llm_backend_config = config.get_llm_config_for_node(node_id)
        if not llm_backend_config:
            llm_backend_config = config.get_llm_config_for_node(legacy_config_id)

    llm_config = None

    if llm_backend_config:
        # Convert backend config to LLM config
        llm_config = backend_config_to_llm_config(llm_backend_config)
        # Override model: CLI --model takes highest precedence, then explicit
        # node config, then backend default_model (already set by backend_config_to_llm_config)
        if model:
            llm_config.model = model
        elif node_config.llm_model and not backend:
            # Only use node config model if explicitly set and --backend wasn't specified
            llm_config.model = node_config.llm_model

        logger.info(
            f"LLM configured: backend={llm_config.backend}, "
            f"model={llm_config.model}"
        )
        # Sandbox: CLI flags override config, config overrides defaults
        effective_sandboxed = sandboxed or node_config.sandboxed
        effective_allowed_dirs = allowed_dirs if allowed_dirs else node_config.allowed_dirs
        effective_allow_network = allow_network if not sandboxed else allow_network  # Only apply network flag if sandbox enabled

        # Relevance router config (if enabled)
        relevance_router_config = None
        if relevance_router:
            from mesh.config import RelevanceRouterConfig
            relevance_router_config = RelevanceRouterConfig(
                threshold=relevance_threshold,
                bypass_direct=True,      # Direct messages always process (no LLM call)
                bypass_mentions=True,    # @nickname mentions always process (skip LLM scoring)
            )
            logger.info(f"Relevance router enabled: threshold={relevance_threshold}")

        agent = AgentNode(
            node_config,
            llm_config=llm_config,
            description=description,
            history_file=history_file,
            persist=not fresh or bool(history_file),
            soft_limit=soft_limit,
            target_ratio=target_ratio,
            # Preference extraction settings from config
            pref_message_threshold=node_config.pref_message_threshold,
            pref_context_limit=node_config.pref_context_limit,
            pref_stale_hours=node_config.pref_stale_hours,
            pref_extraction_model=node_config.pref_extraction_model,
            pref_extraction_backend=node_config.pref_extraction_backend,
            # Sandbox settings
            sandboxed=effective_sandboxed,
            allowed_dirs=effective_allowed_dirs,
            allow_network=effective_allow_network,
            # Relevance router
            relevance_router_config=relevance_router_config,
        )

        # Configure separate router V2 LLM if specified
        if node_config.router_v2_llm_backend:
            router_backend_config = config.llm_backends.get(node_config.router_v2_llm_backend)
            if router_backend_config:
                router_llm_config = backend_config_to_llm_config(router_backend_config)
                if node_config.router_v2_llm_model:
                    router_llm_config.model = node_config.router_v2_llm_model
                agent._router_v2_llm_config = router_llm_config
                logger.info(
                    f"RouterV2 LLM configured: backend={router_llm_config.backend}, "
                    f"model={router_llm_config.model}"
                )
            else:
                logger.warning(
                    f"RouterV2 LLM backend '{node_config.router_v2_llm_backend}' "
                    f"not found in config. Available: {list(config.llm_backends.keys())}"
                )

        # Configure the native harness session backend if specified. Resolves
        # the named llm_backends block (e.g. mesh-harness-qwen36) into an
        # LLMConfig that HarnessSessionManager reads to build the session
        # subprocess argv.
        if getattr(node_config, 'harness_session_backend', ''):
            hs_backend_config = config.llm_backends.get(node_config.harness_session_backend)
            if hs_backend_config:
                agent._harness_session_llm_config = backend_config_to_llm_config(hs_backend_config)
                logger.info(
                    f"Harness session backend configured: "
                    f"{node_config.harness_session_backend} "
                    f"(model={agent._harness_session_llm_config.model})"
                )
            else:
                logger.warning(
                    f"Harness session backend '{node_config.harness_session_backend}' "
                    f"not found in config. Available: {list(config.llm_backends.keys())}"
                )

        # Configure separate memory LLM if specified
        if node_config.memory_llm_backend:
            memory_backend_config = config.llm_backends.get(node_config.memory_llm_backend)
            if memory_backend_config:
                memory_llm_config = backend_config_to_llm_config(memory_backend_config)
                agent._memory_llm_config = memory_llm_config
                logger.info(
                    f"Memory LLM configured: backend={memory_llm_config.backend}, "
                    f"model={memory_llm_config.model}"
                )
            else:
                logger.warning(
                    f"Memory LLM backend '{node_config.memory_llm_backend}' "
                    f"not found in config. Available: {list(config.llm_backends.keys())}"
                )
    else:
        logger.warning(f"No LLM backend configured for {node_id}, using SimpleAgentNode (echo)")
        agent = SimpleAgentNode(node_config)

    # Load history: either from mesh conversation or from file
    if load_conversation:
        # Load from mesh storage
        from mesh.storage import MessageStore
        from mesh.paths import resolve_path
        db_path = resolve_path("~/log/chats/mesh-storage/messages.db")
        store = MessageStore(db_path)

        # Normalize conversation name (add "chat:" prefix if needed)
        conv_id = load_conversation if load_conversation.startswith("chat:") else f"chat:{load_conversation}"

        loaded = agent.load_history_from_store(store, conv_id)
        if loaded > 0:
            logger.info(f"Loaded {loaded} messages from mesh conversation: {conv_id}")
            # Also check for summary in mesh storage
            summary_info = store.get_summary(conv_id)
            if summary_info:
                logger.info(f"Loaded summary covering {summary_info.get('messages_summarized', '?')} messages")
        else:
            logger.warning(f"No messages found for conversation: {conv_id}")
    elif not fresh or history_file:
        loaded = agent.load_history()
        if loaded > 0:
            logger.info(f"Resumed with {loaded} history entries from {agent.history_file}")

            # Also load summary if available
            if hasattr(agent, 'load_summary_from_disk') and agent.load_summary_from_disk():
                logger.info(f"Loaded saved summary covering {agent._summary.messages_summarized} messages")
        else:
            logger.info(f"No previous history (will persist to {agent.history_file})")

    logger.info(f"Agent starting as: {node_id} (nickname: {nickname})")

    # Set auth token for remote shutdown validation
    if node_config.auth_token:
        agent.set_auth_token(node_config.auth_token)
        logger.debug("Auth token configured for remote shutdown support")

    # Handle shutdown signals
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def shutdown():
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown)

    # Reconnect loop with exponential backoff
    reconnect_delay = RECONNECT_DELAY_INITIAL

    while not stop_event.is_set():
        try:
            router_addr = f"{node_config.router_host}:{node_config.router_port}"
            logger.info(f"Connecting to router at {router_addr}...")
            await agent.connect()
            logger.info("Connected successfully")

            # Flush any pending sends from before the disconnect
            if agent.pending_send_count > 0:
                logger.info(f"Flushing {agent.pending_send_count} pending sends...")
                flushed = await agent.flush_pending_sends()
                logger.info(f"Flushed {flushed} pending sends")

            # Reset backoff on successful connection
            reconnect_delay = RECONNECT_DELAY_INITIAL

            # Run receive loop until disconnect (pass stop_event for graceful shutdown)
            await agent.receive_loop(stop_event=stop_event)

            # receive_loop exited = disconnected
            logger.warning("Disconnected from router")

        except ConnectionRefusedError:
            logger.warning(f"Connection refused, retrying in {reconnect_delay:.1f}s...")
        except asyncio.TimeoutError:
            logger.warning(f"Connection timed out, retrying in {reconnect_delay:.1f}s...")
        except ConnectionError as e:
            # Registration failures, protocol errors, etc.
            logger.warning(f"Connection error: {e}, retrying in {reconnect_delay:.1f}s...")
        except OSError as e:
            # Network errors (including "Multiple exceptions" from asyncio)
            logger.warning(f"Network error: {e}, retrying in {reconnect_delay:.1f}s...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}, retrying in {reconnect_delay:.1f}s...")
        finally:
            # Ensure we're disconnected before reconnecting
            if agent.is_connected:
                await agent.disconnect()

        # Wait before reconnect (unless shutdown requested)
        if not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=reconnect_delay)
            except asyncio.TimeoutError:
                pass  # Normal - timeout means we should retry
            reconnect_delay = min(reconnect_delay * RECONNECT_DELAY_FACTOR, RECONNECT_DELAY_MAX)

    # Final cleanup
    if agent.is_connected:
        await agent.disconnect()
    logger.info("Agent shutdown complete")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start an agent node")
    parser.add_argument("--agent", "-a", help="Agent type (e.g., coder, researcher)")
    parser.add_argument("--nickname", "-n", help="Agent nickname for display/addressing")
    parser.add_argument("--description", "-d", help="Agent description (e.g., project name)")
    parser.add_argument("--config", "-c", help="Path to config file")
    parser.add_argument("--log-dir", default="logs", help="Log directory")
    parser.add_argument(
        "--fresh", "--no-resume",
        action="store_true",
        dest="fresh",
        help="Start fresh (don't load or persist history)"
    )
    parser.add_argument(
        "--history-file",
        help="Path to history file (implies persistence)"
    )
    parser.add_argument(
        "--soft-limit",
        type=int,
        default=None,
        help="Token limit for history window (overrides config if set)"
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.25,
        help="Target context ratio after summarization (default: 0.25)"
    )
    # TLS and auth settings
    parser.add_argument(
        "--tls",
        action="store_true",
        dest="use_tls",
        help="Enable TLS for router connection"
    )
    parser.add_argument(
        "--router-host",
        help="Router hostname (overrides config)"
    )
    parser.add_argument(
        "--router-port",
        type=int,
        help="Router port (overrides config)"
    )
    parser.add_argument(
        "--auth-token",
        help="Auth token for router authentication"
    )
    parser.add_argument(
        "--backend", "-b",
        help="LLM backend name (overrides config, e.g., openai-reasoning-medium)"
    )
    parser.add_argument(
        "--model", "-m",
        help="Model name within the backend (overrides backend default, e.g., opus, gpt-4o)"
    )
    # Conversation loading
    parser.add_argument(
        "--load-conversation",
        help="Load a conversation from mesh storage (e.g., 'research-notes', 'chat:project-x')"
    )
    parser.add_argument(
        "--list-conversations",
        action="store_true",
        help="List available conversations in mesh storage and exit"
    )
    # Sandbox settings
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Enable bwrap sandboxing (restricts file/bash access). On Ubuntu 24.04+, run: sudo ./scripts/setup_bwrap_apparmor.sh"
    )
    parser.add_argument(
        "--allowed-dirs",
        nargs="+",
        default=None,
        help="Directories writable in sandbox (default: cwd + /tmp)"
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Block network access in sandbox"
    )
    # Controller settings
    parser.add_argument(
        "--controller",
        choices=["passthrough", "task-fsm-v0", "phase-flow-v02"],
        help="Controller mode (passthrough=no controller, phase-flow-v02=v0.2 adaptive phases)"
    )
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Effort preset for v0.2 controller (default: medium)"
    )
    # Relevance router settings (LLM-based channel filtering)
    parser.add_argument(
        "--relevance-router",
        action="store_true",
        help="Enable LLM-based relevance router for channel messages (smarter than nickname matching)"
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.7,
        help="Relevance score threshold for processing channel messages (0.0-1.0, default: 0.7)"
    )
    # Legacy positional argument for backwards compatibility
    parser.add_argument("node_id", nargs="?", help="Legacy: Node ID (e.g., agent:researcher)")
    args = parser.parse_args()

    # Determine agent type from args
    agent_type = args.agent
    nickname = args.nickname
    legacy_node_id = None

    if args.node_id:
        # Legacy mode: parse from node_id
        legacy_node_id = args.node_id
        parts = args.node_id.split(":")
        if len(parts) >= 2 and parts[0] == "agent":
            agent_type = parts[1]
            if len(parts) >= 3:
                nickname = parts[2]

    # Handle --list-conversations before agent type validation
    if args.list_conversations:
        from mesh.storage import MessageStore
        from mesh.paths import resolve_path
        db_path = resolve_path("~/log/chats/mesh-storage/messages.db")
        store = MessageStore(db_path)
        convos = store.get_all_conversations()
        if not convos:
            print("No conversations found in mesh storage")
        else:
            print(f"Available conversations ({len(convos)}):")
            print("")
            for conv in convos:
                conv_id = conv.get("conversation_id", "")
                msg_count = conv.get("message_count", 0)
                last_time = conv.get("last_timestamp", "")
                # Format: remove "chat:" prefix for display
                name = conv_id[5:] if conv_id.startswith("chat:") else conv_id
                print(f"  {name}")
                print(f"      {msg_count} messages, last: {last_time[:19] if last_time else 'unknown'}")
        sys.exit(0)

    if not agent_type:
        parser.error("Either --agent TYPE or a node_id argument is required")

    # Generate display ID for logging
    display_nick = nickname or secrets.token_hex(2)
    display_id = f"agent:{agent_type}:{display_nick}"

    log_file = setup_logging(display_id, args.log_dir)
    logger.info(f"Starting {display_id}, logging to {log_file}")

    asyncio.run(main(
        agent_type,
        nickname,
        description=args.description,
        config_path=args.config,
        legacy_node_id=legacy_node_id,
        fresh=args.fresh,
        history_file=args.history_file,
        soft_limit=args.soft_limit,
        target_ratio=args.target_ratio,
        use_tls=args.use_tls if args.use_tls else None,
        router_host=args.router_host,
        router_port=args.router_port,
        auth_token=args.auth_token,
        backend=args.backend,
        model=args.model,
        load_conversation=args.load_conversation,
        # Sandbox settings
        sandboxed=args.sandbox,
        allowed_dirs=args.allowed_dirs,
        allow_network=not args.no_network,
        # Controller settings
        controller_mode=args.controller,
        effort=args.effort,
        # Relevance router settings
        relevance_router=args.relevance_router,
        relevance_threshold=args.relevance_threshold,
    ))
