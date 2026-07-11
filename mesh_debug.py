#!/usr/bin/env python3
"""
Mesh Debug Client - Simple CLI for testing agent communication.

A lightweight tool for debugging agents without the full TUI.
Sends messages and prints raw responses with minimal formatting.

Usage:
    python mesh_debug.py [options] <target> <message>
    python mesh_debug.py [options] -i               # Interactive mode

Examples:
    # Send a single message
    python mesh_debug.py agent:assistant "What tools do you have?"

    # Interactive mode
    python mesh_debug.py -i
    > /target agent:assistant
    > Hello, list your tools
    > /to agent:coder Write a hello world

    # With timeout
    python mesh_debug.py -t 30 agent:researcher "Search for Python docs"
"""

import argparse
import asyncio
import sys
from datetime import datetime
from typing import Optional

from mesh.config import load_config, NodeConfig
from mesh.node import Node
from mesh.protocol import Message, MessageType, make_confirm_response, make_status_request, make_message


# Colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"


class DebugNode(Node):
    """Simple node that collects incoming messages."""

    def __init__(self, config: NodeConfig):
        super().__init__(config)
        self._messages: asyncio.Queue[Message] = asyncio.Queue()
        # Simple roster: nickname -> full node ID
        self._roster: dict[str, str] = {}

    async def on_message(self, msg: Message) -> None:
        # Track presence for nickname resolution
        if msg.type == MessageType.PRESENCE:
            content = msg.content if isinstance(msg.content, dict) else {}
            event = content.get("event", "")
            nickname = content.get("nickname", "")
            if event == "join" and nickname:
                self._roster[nickname.lower()] = msg.from_node
            elif event == "leave" and nickname:
                self._roster.pop(nickname.lower(), None)

        await self._messages.put(msg)

    def resolve_target(self, target: str) -> Optional[str]:
        """Resolve nickname to full node ID."""
        if ":" in target:
            return target
        return self._roster.get(target.lower())

    async def wait_for_response(self, timeout: float = 60.0) -> Optional[Message]:
        """Wait for a response with timeout."""
        try:
            return await asyncio.wait_for(self._messages.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None


async def send_and_receive(
    node_id: str,
    target: str,
    message: str,
    timeout: float = 60.0,
    config_path: Optional[str] = None,
    raw: bool = False,
    auth_token: Optional[str] = None,
) -> int:
    """Send a message and print the response."""
    config = load_config(config_path)

    node_config = NodeConfig(
        id=node_id,
        router_host=config.router.host,
        router_port=config.router.port,
        auth_token=auth_token,
    )

    node = DebugNode(node_config)

    try:
        await node.connect()
    except Exception as e:
        print(f"{RED}Failed to connect:{RESET} {e}", file=sys.stderr)
        return 1

    # Start receive loop in background
    receive_task = asyncio.create_task(node.receive_loop())

    # Wait briefly for presence messages to build the roster, then resolve nickname
    if ":" not in target:
        await asyncio.sleep(0.3)  # Brief wait for presence
        resolved = node.resolve_target(target)
        if resolved:
            target = resolved
        # Otherwise just use the nickname as-is, router may resolve it

    if not raw:
        print(f"{DIM}[connected as {node_id}]{RESET}", file=sys.stderr)
        print(f"{DIM}[sending to {target}]{RESET}", file=sys.stderr)

    # Handle /status command specially
    is_status_request = False
    if message.startswith("/status") or message.startswith("/s ") or message == "/s":
        is_status_request = True
        parts = message.split()
        # Parse optional count (e.g., "/status 10" or "/s 5")
        count = 5
        for p in parts[1:]:
            if p.isdigit():
                count = int(p)
        status_msg = make_status_request(
            from_node=node_id,
            to_node=target,
            num_messages=count,
        )
        await node._conn.send(status_msg)
    else:
        # Send regular message
        await node.send(target, message)

    # Wait for response, handle confirmations
    while True:
        response = await node.wait_for_response(timeout)

        if response is None:
            print(f"{RED}Timeout waiting for response{RESET}", file=sys.stderr)
            break

        if response.type == MessageType.CONFIRM_REQUEST:
            # Handle confirmation request (data is in content, not metadata)
            content_data = response.content if isinstance(response.content, dict) else {}
            tool_name = content_data.get("tool_name", "unknown")
            preview = content_data.get("preview", "")

            print(f"\n{YELLOW}━━━ Confirmation Required ━━━{RESET}")
            print(f"{BOLD}Tool:{RESET} {tool_name}")
            print(f"{BOLD}Action:{RESET}\n{preview}")
            print(f"{YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")

            try:
                answer = input(f"{BOLD}Confirm? [y/n]:{RESET} ").strip().lower()
            except EOFError:
                answer = "n"

            confirmed = answer in ("y", "yes")
            confirm_msg = make_confirm_response(
                from_node=node_id,
                to_node=response.from_node,
                in_reply_to=response.id,
                confirmed=confirmed,
            )
            await node._conn.send(confirm_msg)
            print(f"{DIM}[{'confirmed' if confirmed else 'rejected'}]{RESET}")
            # Continue waiting for actual response
            continue

        elif response.type == MessageType.MESSAGE:
            content = response.content if isinstance(response.content, str) else str(response.content)
            if raw:
                print(content)
            else:
                print(f"\n{BOLD}{MAGENTA}{response.from_node}:{RESET}")
                print(content)
            break
        elif response.type == MessageType.PRESENCE:
            # Skip presence messages in non-interactive mode
            print(f"{YELLOW}[{response.type}]{RESET} {response.content}", file=sys.stderr)
            continue
        elif response.type == MessageType.STATUS_RESPONSE:
            # Handle status response
            content_data = response.content if isinstance(response.content, dict) else {}
            print(f"\n{CYAN}━━━ Status: {response.from_node} ━━━{RESET}")
            if content_data.get("summary"):
                print(f"{DIM}Summary:{RESET} {content_data['summary'][:200]}")
                print()
            for ctx in content_data.get("context", []):
                sender = ctx.get("from", "?")
                ts_ctx = ctx.get("timestamp", "")
                content_str = ctx.get("content", "")
                entry_type = ctx.get("type", "message")

                # Format based on entry type
                if entry_type == "tool_call":
                    print(f"{GREEN}{BOLD}{sender} 🔧{RESET} {DIM}({ts_ctx}){RESET}:")
                elif entry_type == "tool_result":
                    print(f"{CYAN}⚙ system{RESET} {DIM}({ts_ctx}){RESET}:")
                else:
                    print(f"{BOLD}{sender}{RESET} {DIM}({ts_ctx}){RESET}:")

                # Truncate long content
                lines = content_str.split("\n")
                if len(lines) > 15:
                    lines = lines[:15] + ["... (truncated)"]
                print("\n".join(lines[:15]))
                print()
            print(f"{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
            break
        else:
            print(f"{YELLOW}[{response.type}]{RESET} {response.content}", file=sys.stderr)
            break

    # Cleanup
    receive_task.cancel()
    try:
        await receive_task
    except asyncio.CancelledError:
        pass
    await node.disconnect()

    return 0 if response else 1


async def interactive_mode(
    node_id: str,
    timeout: float = 60.0,
    config_path: Optional[str] = None,
    auth_token: Optional[str] = None,
):
    """Interactive mode with target selection."""
    config = load_config(config_path)

    node_config = NodeConfig(
        id=node_id,
        router_host=config.router.host,
        router_port=config.router.port,
        auth_token=auth_token,
    )

    node = DebugNode(node_config)

    try:
        await node.connect()
    except Exception as e:
        print(f"{RED}Failed to connect:{RESET} {e}", file=sys.stderr)
        return

    print(f"{GREEN}Connected as {node_id}{RESET}")
    print(f"{DIM}Commands: /target <node>, /to <node> <msg>, /list, /quit{RESET}")
    print()

    # Start receive loop
    receive_task = asyncio.create_task(node.receive_loop())

    # Pending confirmation requests
    pending_confirms: dict[str, Message] = {}

    # Track attached agent for tool activity filtering (mutable container for closure)
    state = {"attached_agent": None}

    # Background task to print incoming messages
    async def print_responses():
        while True:
            try:
                msg = await node._messages.get()
                ts = datetime.now().strftime("%H:%M:%S")

                if msg.type == MessageType.CONFIRM_REQUEST:
                    # Store and prompt for confirmation (data is in content, not metadata)
                    pending_confirms[msg.id] = msg
                    content_data = msg.content if isinstance(msg.content, dict) else {}
                    tool_name = content_data.get("tool_name", "unknown")
                    preview = content_data.get("preview", "")

                    print(f"\n{YELLOW}━━━ Confirmation Required ({msg.from_node}) ━━━{RESET}")
                    print(f"{BOLD}Tool:{RESET} {tool_name}")
                    print(f"{BOLD}Action:{RESET}\n{preview}")
                    print(f"{YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
                    print(f"Reply with: /confirm {msg.id[:8]} y  or  /confirm {msg.id[:8]} n")
                    print()
                    sys.stdout.write("> ")
                    sys.stdout.flush()

                elif msg.type == MessageType.STATUS_RESPONSE:
                    content_data = msg.content if isinstance(msg.content, dict) else {}
                    print(f"\n{CYAN}━━━ Status: {msg.from_node} ━━━{RESET}")
                    if content_data.get("summary"):
                        print(f"{DIM}Summary:{RESET} {content_data['summary'][:200]}")
                        print()
                    for ctx in content_data.get("context", []):
                        sender = ctx.get("from", "?")
                        ts_ctx = ctx.get("timestamp", "")
                        content_str = ctx.get("content", "")
                        entry_type = ctx.get("type", "message")

                        # Format based on entry type
                        if entry_type == "tool_call":
                            print(f"{GREEN}{BOLD}{sender} 🔧{RESET} {DIM}({ts_ctx}){RESET}:")
                        elif entry_type == "tool_result":
                            print(f"{CYAN}⚙ system{RESET} {DIM}({ts_ctx}){RESET}:")
                        else:
                            print(f"{BOLD}{sender}{RESET} {DIM}({ts_ctx}){RESET}:")

                        # Truncate long content
                        lines = content_str.split("\n")
                        if len(lines) > 15:
                            lines = lines[:15] + ["... (truncated)"]
                        print("\n".join(lines[:15]))
                        print()
                    print(f"{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
                    sys.stdout.write("> ")
                    sys.stdout.flush()

                elif msg.type == MessageType.TOOL_ACTIVITY:
                    # Handle tool activity events (real-time tool call/result streaming)
                    content_data = msg.content if isinstance(msg.content, dict) else {}
                    event_type = content_data.get("event_type", "unknown")
                    tool_name = content_data.get("tool_name", "unknown")
                    is_mesh_tool = content_data.get("is_mesh_tool", False)
                    from_nick = msg.from_node.split(":")[-1] if msg.from_node else "?"

                    # Skip if we're not attached to this agent
                    if state["attached_agent"] and msg.from_node != state["attached_agent"]:
                        continue

                    if event_type == "tool_call":
                        args = content_data.get("tool_args", {})
                        args_preview = ", ".join(f"{k}: {repr(str(v)[:50])}" for k, v in list(args.items())[:3])
                        mesh_tag = f"{CYAN}[mesh]{RESET} " if is_mesh_tool else ""
                        print(f"\n{GREEN}{from_nick}:{RESET} {mesh_tag}● {BOLD}{tool_name}{RESET}({DIM}{args_preview}{RESET})")
                        sys.stdout.write("> ")
                        sys.stdout.flush()
                    elif event_type == "tool_result":
                        result = content_data.get("result", "")
                        result_preview = str(result)[:200]
                        if len(str(result)) > 200:
                            result_preview += "..."
                        print(f"  {DIM}⎿{RESET}  {result_preview}")
                        sys.stdout.write("> ")
                        sys.stdout.flush()

                elif msg.type == MessageType.MESSAGE:
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    print(f"\n{BOLD}{MAGENTA}{msg.from_node}{RESET} {DIM}({ts}){RESET}:")
                    print(content)
                    print()
                    sys.stdout.write("> ")
                    sys.stdout.flush()
                else:
                    print(f"\n{YELLOW}[{msg.type}]{RESET} from {msg.from_node}: {msg.content}")
                    print()
                    sys.stdout.write("> ")
                    sys.stdout.flush()
            except asyncio.CancelledError:
                break

    print_task = asyncio.create_task(print_responses())

    default_target: Optional[str] = None

    try:
        while True:
            try:
                # Use simple input (no readline/prompt_toolkit for simplicity)
                target_hint = f"→{default_target}" if default_target else "no target"
                line = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input(f"[{target_hint}]> ")
                )
            except EOFError:
                break

            line = line.strip()
            if not line:
                continue

            # Commands
            if line.startswith("/"):
                parts = line.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in ("/quit", "/q"):
                    break

                elif cmd in ("/target", "/t"):
                    if arg:
                        target_input = arg.strip()
                        resolved = node.resolve_target(target_input)
                        if resolved:
                            default_target = resolved
                            print(f"{GREEN}Target set to:{RESET} {default_target}")
                        else:
                            # Could be a new node - keep as-is
                            default_target = target_input
                            print(f"{GREEN}Target set to:{RESET} {default_target} {YELLOW}(unresolved){RESET}")
                    else:
                        print(f"{CYAN}Current target:{RESET} {default_target or '(none)'}")

                elif cmd == "/to":
                    if " " not in arg:
                        print(f"{YELLOW}Usage: /to <node> <message>{RESET}")
                    else:
                        target_input, msg = arg.split(" ", 1)
                        resolved = node.resolve_target(target_input.strip())
                        target = resolved or target_input.strip()
                        await node.send(target, msg)
                        print(f"{DIM}→ Sent to {target}{RESET}")

                elif cmd in ("/list", "/ls"):
                    nodes = await node.request_node_list()
                    print(f"{BOLD}Nodes:{RESET} {', '.join(nodes) if nodes else '(none)'}")

                elif cmd == "/confirm":
                    parts2 = arg.split()
                    if len(parts2) < 2:
                        print(f"{YELLOW}Usage: /confirm <id_prefix> y|n{RESET}")
                    else:
                        id_prefix = parts2[0]
                        answer = parts2[1].lower()
                        # Find matching pending confirm
                        matching = [
                            (mid, msg)
                            for mid, msg in pending_confirms.items()
                            if mid.startswith(id_prefix)
                        ]
                        if not matching:
                            print(f"{YELLOW}No pending confirmation with prefix: {id_prefix}{RESET}")
                        elif len(matching) > 1:
                            print(f"{YELLOW}Ambiguous prefix, matches: {[m[0][:12] for m in matching]}{RESET}")
                        else:
                            msg_id, req_msg = matching[0]
                            confirmed = answer in ("y", "yes")
                            confirm_msg = make_confirm_response(
                                from_node=node_id,
                                to_node=req_msg.from_node,
                                in_reply_to=req_msg.id,
                                confirmed=confirmed,
                            )
                            await node._conn.send(confirm_msg)
                            del pending_confirms[msg_id]
                            print(f"{GREEN if confirmed else RED}{'Confirmed' if confirmed else 'Rejected'}{RESET}")

                elif cmd in ("/status", "/s"):
                    # /status [target] [count]
                    parts2 = arg.split()
                    target = None
                    count = 5
                    for p in parts2:
                        if p.isdigit():
                            count = int(p)
                        else:
                            # Try to resolve nickname to full node ID
                            resolved = node.resolve_target(p)
                            if resolved:
                                target = resolved
                            else:
                                print(f"{YELLOW}Unknown target '{p}'. Use /list to see available nodes.{RESET}")
                                target = None
                                break
                    if not target:
                        # Use default target (also resolve if needed)
                        if default_target:
                            target = node.resolve_target(default_target) or default_target
                    if not target:
                        print(f"{YELLOW}No target set. Use /status <node> or set target first{RESET}")
                    else:
                        status_msg = make_status_request(
                            from_node=node_id,
                            to_node=target,
                            num_messages=count,
                        )
                        await node._conn.send(status_msg)
                        print(f"{DIM}→ Requested status from {target} ({count} messages){RESET}")

                elif cmd == "/attach":
                    if arg:
                        resolved = node.resolve_target(arg.strip())
                        state["attached_agent"] = resolved or arg.strip()
                        print(f"{GREEN}Attached to {state['attached_agent']}{RESET} - you'll see tool activity in real-time")
                    else:
                        print(f"{YELLOW}Usage: /attach <agent>{RESET}")

                elif cmd == "/detach":
                    if state["attached_agent"]:
                        print(f"{CYAN}Detached from {state['attached_agent']}{RESET}")
                        state["attached_agent"] = None
                    else:
                        print(f"{YELLOW}Not attached to any agent{RESET}")

                elif cmd in ("/help", "/h"):
                    print(f"""
{BOLD}Commands:{RESET}
  /target, /t <node>  - Set default target
  /to <node> <msg>    - Send to specific node
  /attach <agent>     - Attach to agent (see real-time tool activity)
  /detach             - Detach from agent
  /status, /s [node] [n] - Get agent's recent context (default 5)
  /confirm <id> y|n   - Respond to confirmation request
  /list, /ls          - List connected nodes
  /quit, /q           - Exit
""")
                else:
                    print(f"{YELLOW}Unknown command:{RESET} {cmd}")

            else:
                # Send to default target
                if not default_target:
                    print(f"{YELLOW}No target set. Use /target <node> first{RESET}")
                else:
                    # Resolve default target if it's a nickname
                    target = node.resolve_target(default_target) or default_target
                    await node.send(target, line)
                    print(f"{DIM}→ Sent to {target}{RESET}")

    except KeyboardInterrupt:
        print()
    finally:
        print(f"{RED}Disconnecting...{RESET}")
        print_task.cancel()
        receive_task.cancel()
        try:
            await print_task
        except asyncio.CancelledError:
            pass
        try:
            await receive_task
        except asyncio.CancelledError:
            pass
        await node.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Mesh debug client for testing agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s agent:assistant "Hello"
  %(prog)s -i
  %(prog)s -t 30 agent:researcher "Search for X"
  %(prog)s --raw agent:coder "Write code" > output.txt
""",
    )
    parser.add_argument("target", nargs="?", help="Target node ID")
    parser.add_argument("message", nargs="?", help="Message to send")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("-t", "--timeout", type=float, default=60.0, help="Response timeout (seconds)")
    parser.add_argument("-n", "--node-id", default="debug:client", help="This client's node ID")
    parser.add_argument("-c", "--config", help="Path to mesh.yaml config")
    parser.add_argument("-a", "--auth-token", help="Authentication token (or use MESH_AUTH_TOKEN)")
    parser.add_argument("--raw", action="store_true", help="Raw output (no formatting)")

    args = parser.parse_args()

    # Get auth token from args or environment
    import os
    auth_token = args.auth_token or os.environ.get("MESH_AUTH_TOKEN")

    if args.interactive:
        asyncio.run(interactive_mode(args.node_id, args.timeout, args.config, auth_token))
    elif args.target and args.message:
        exit_code = asyncio.run(
            send_and_receive(
                args.node_id,
                args.target,
                args.message,
                args.timeout,
                args.config,
                args.raw,
                auth_token,
            )
        )
        sys.exit(exit_code)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
