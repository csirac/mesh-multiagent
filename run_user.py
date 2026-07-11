#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Start a simple user node CLI.

This is a minimal CLI for testing. The full TUI will be adapted from chat-app.

Usage:
    python run_user.py [node_id] [--config PATH]

Example:
    python run_user.py user:yourname
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from mesh.config import load_config, NodeConfig
from mesh.node import Node
from mesh.protocol import Message, MessageType


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


class UserNode(Node):
    """Simple user node with CLI interface."""

    def __init__(self, config: NodeConfig):
        super().__init__(config)
        self._print_queue: asyncio.Queue[str] = asyncio.Queue()

    async def on_message(self, msg: Message) -> None:
        """Handle incoming messages by printing them."""
        if msg.type == MessageType.MESSAGE:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            await self._print_queue.put(f"\n[{msg.from_node}]: {content}\n> ")
        elif msg.type == MessageType.CONTROL:
            content = msg.content if isinstance(msg.content, dict) else {}
            action = content.get("action", "unknown")
            await self._print_queue.put(f"\n[control:{action}]: {content}\n> ")

    async def printer_loop(self):
        """Print messages from the queue."""
        while True:
            text = await self._print_queue.get()
            print(text, end="", flush=True)


async def input_loop(user: UserNode):
    """Read input and send messages."""
    loop = asyncio.get_running_loop()

    print("Connected. Commands:")
    print("  /list          - List connected nodes")
    print("  /to <node> msg - Send message to node")
    print("  /quit          - Disconnect")
    print()

    while True:
        print("> ", end="", flush=True)

        # Read input in executor to not block event loop
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break

        line = line.strip()
        if not line:
            continue

        if line == "/quit":
            break
        elif line == "/list":
            nodes = await user.request_node_list()
            print(f"Connected nodes: {nodes}")
        elif line.startswith("/to "):
            # Parse: /to node_id message
            parts = line[4:].split(" ", 1)
            if len(parts) < 2:
                print("Usage: /to <node_id> <message>")
                continue
            target, content = parts
            await user.send(target, content)
            print(f"Sent to {target}")
        else:
            print("Unknown command. Use /to <node> <message> to send.")


async def main(node_id: str, config_path: str | None = None):
    config = load_config(config_path)

    # Get node config or create default
    if node_id in config.nodes:
        node_config = config.nodes[node_id]
    else:
        node_config = NodeConfig(
            id=node_id,
            router_host=config.router.host,
            router_port=config.router.port,
        )

    user = UserNode(node_config)

    # Handle shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def shutdown():
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown)

    # Connect
    try:
        await user.connect()
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # Start background tasks
    receive_task = asyncio.create_task(user.receive_loop())
    printer_task = asyncio.create_task(user.printer_loop())

    # Run input loop
    try:
        await input_loop(user)
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await user.disconnect()
        receive_task.cancel()
        printer_task.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a user node CLI")
    parser.add_argument("node_id", nargs="?", default="user:yourname", help="Node ID")
    parser.add_argument("--config", "-c", help="Path to config file")
    parser.add_argument("--log-dir", default="logs", help="Log directory")
    args = parser.parse_args()

    log_file = setup_logging(args.node_id, args.log_dir)
    logger.info(f"Starting {args.node_id}, logging to {log_file}")

    asyncio.run(main(args.node_id, args.config))
