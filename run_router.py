#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Start the mesh router.

Usage:
    python run_router.py [--config PATH]

User management:
    python run_router.py --create-user USERNAME
    python run_router.py --list-users
    python run_router.py --disable-user USERNAME
    python run_router.py --enable-user USERNAME
    python run_router.py --delete-user USERNAME
    python run_router.py --regen-token USERNAME
    python run_router.py --set-token USERNAME TOKEN
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from mesh.config import load_config
from mesh.router import Router
from mesh.storage import MessageStore


def setup_logging(log_file: str = "router.log"):
    """Configure logging to both file and console."""
    # Create logs directory if needed
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # File handler - detailed logs
    file_handler = logging.FileHandler(log_file, mode='a')
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

    return log_file


logger = logging.getLogger(__name__)


async def main(config_path: str | None = None, auth_token: str | None = None):
    config = load_config(config_path)

    # CLI auth_token overrides config
    if auth_token:
        config.router.auth_enabled = True
        config.router.auth_token = auth_token

    router = Router(config.router)

    if config.router.auth_enabled:
        logger.info("Authentication enabled")

    # Handle shutdown signals
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def shutdown():
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown)

    # Start router
    await router.start()

    # Wait for shutdown
    await stop_event.wait()

    # Cleanup
    await router.stop()


def handle_user_commands(args, config) -> bool:
    """
    Handle user management commands.

    Returns True if a user command was handled, False otherwise.
    """
    store = MessageStore(config.router.storage_path)

    if args.create_user:
        username = args.create_user
        try:
            token = store.create_user(username)
            print(f"User '{username}' created successfully.")
            print()
            print(f"Token: {token}")
            print()
            print("IMPORTANT: Save this token! It will not be shown again.")
            print()
            print("Usage:")
            print(f"  export MESH_AUTH_TOKEN={token}")
            print(f"  python run_user_tui.py --nickname {username}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return True

    if args.list_users:
        users = store.list_users()
        if not users:
            print("No users configured.")
        else:
            print(f"{'ID':<4} {'Username':<20} {'Created':<24} {'Status':<10} {'Prefixes'}")
            print("-" * 80)
            for user in users:
                status = "DISABLED" if user["disabled"] else "active"
                created = user["created_at"][:19] if user["created_at"] else "unknown"
                prefixes = ", ".join(user["allowed_prefixes"]) if user.get("allowed_prefixes") else "(any)"
                print(f"{user['id']:<4} {user['username']:<20} {created:<24} {status:<10} {prefixes}")
        return True

    if args.disable_user:
        username = args.disable_user
        if store.disable_user(username):
            print(f"User '{username}' disabled.")
        else:
            print(f"User '{username}' not found.", file=sys.stderr)
            sys.exit(1)
        return True

    if args.enable_user:
        username = args.enable_user
        if store.enable_user(username):
            print(f"User '{username}' enabled.")
        else:
            print(f"User '{username}' not found.", file=sys.stderr)
            sys.exit(1)
        return True

    if args.delete_user:
        username = args.delete_user
        if store.delete_user(username):
            print(f"User '{username}' deleted.")
        else:
            print(f"User '{username}' not found.", file=sys.stderr)
            sys.exit(1)
        return True

    if args.regen_token:
        username = args.regen_token
        token = store.regenerate_user_token(username)
        if token:
            print(f"New token for '{username}':")
            print()
            print(f"Token: {token}")
            print()
            print("IMPORTANT: Save this token! It will not be shown again.")
        else:
            print(f"User '{username}' not found.", file=sys.stderr)
            sys.exit(1)
        return True

    if args.set_token:
        if len(args.set_token) != 2:
            print("Error: --set-token requires USERNAME and TOKEN", file=sys.stderr)
            sys.exit(1)
        username, token = args.set_token
        if store.set_user_token(username, token):
            print(f"Token set for '{username}'.")
        else:
            print(f"User '{username}' not found.", file=sys.stderr)
            sys.exit(1)
        return True

    if args.set_prefixes:
        if len(args.set_prefixes) < 2:
            print("Error: --set-prefixes requires USERNAME followed by one or more prefixes",
                  file=sys.stderr)
            sys.exit(1)
        username = args.set_prefixes[0]
        prefixes = args.set_prefixes[1:]
        if store.set_allowed_prefixes(username, prefixes):
            print(f"Set allowed prefixes for '{username}': {prefixes}")
        else:
            print(f"User '{username}' not found.", file=sys.stderr)
            sys.exit(1)
        return True

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start the mesh router or manage users",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
User Management Examples:
  %(prog)s --create-user grad1        Create user 'grad1' and print token
  %(prog)s --list-users               List all users
  %(prog)s --disable-user grad1       Disable user 'grad1'
  %(prog)s --enable-user grad1        Re-enable user 'grad1'
  %(prog)s --delete-user grad1        Permanently delete user 'grad1'
  %(prog)s --regen-token grad1        Generate new random token for 'grad1'
  %(prog)s --set-token grad1 TOKEN   Set token to a specific value

To enable per-user auth, set auth_mode: per_user in your config.
        """
    )
    parser.add_argument("--config", "-c", help="Path to config file")
    parser.add_argument("--log", "-l", default="logs/router.log", help="Log file path")
    parser.add_argument(
        "--auth-token",
        help="Enable auth with this token (overrides config)"
    )

    # User management commands
    user_group = parser.add_argument_group("user management")
    user_group.add_argument(
        "--create-user",
        metavar="USERNAME",
        help="Create a new user and print their token"
    )
    user_group.add_argument(
        "--list-users",
        action="store_true",
        help="List all users"
    )
    user_group.add_argument(
        "--disable-user",
        metavar="USERNAME",
        help="Disable a user (revoke access)"
    )
    user_group.add_argument(
        "--enable-user",
        metavar="USERNAME",
        help="Re-enable a disabled user"
    )
    user_group.add_argument(
        "--delete-user",
        metavar="USERNAME",
        help="Permanently delete a user"
    )
    user_group.add_argument(
        "--regen-token",
        metavar="USERNAME",
        help="Generate a new random token for a user"
    )
    user_group.add_argument(
        "--set-token",
        nargs=2,
        metavar=("USERNAME", "TOKEN"),
        help="Set a user's token to a specific value"
    )
    user_group.add_argument(
        "--set-prefixes",
        nargs="+",
        metavar=("USERNAME", "PREFIX"),
        help="Set allowed identity prefixes for a user (e.g. --set-prefixes yourname 'user:')"
    )

    args = parser.parse_args()

    # Load config for user management or router
    config = load_config(args.config)

    # Handle user management commands (no logging needed)
    if handle_user_commands(args, config):
        sys.exit(0)

    # Start router with logging
    log_file = setup_logging(args.log)
    logger.info(f"Logging to {log_file}")

    asyncio.run(main(args.config, auth_token=args.auth_token))
