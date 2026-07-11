#!/usr/bin/env python3
"""
mesh-tool: Shell-based access to mesh tools.

Replaces MCP sidecar for agents with shell access. Any agent that can
run a command can use mesh tools — no MCP plumbing, no XML parsing.

Usage:
    mesh-tool                          # list all tools
    mesh-tool <name>                   # show usage for a tool
    mesh-tool <name> --help            # same
    mesh-tool <name> --arg1 val1 ...   # call the tool
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Tool surface — which tools are exposed via the CLI
# ---------------------------------------------------------------------------

GMAIL_TOOLS = frozenset({
    "gmail_search_emails",
    "gmail_list_from_date",
    "gmail_list_recent",
    "gmail_list_unread",
    "gmail_get_email",
    "gmail_send_message",
    "gmail_reply_to",
    "gmail_create_draft",
    "gmail_draft_reply",
})

# Tools that support --account switching (Gmail + Calendar)
ACCOUNT_TOOLS = GMAIL_TOOLS | {"calendar_list_on_date"}

CLI_TOOLS = frozenset({
    # Gmail
    "gmail_search_emails",
    "gmail_list_from_date",
    "gmail_list_recent",
    "gmail_list_unread",
    "gmail_get_email",
    "gmail_send_message",
    "gmail_reply_to",
    "gmail_create_draft",
    "gmail_draft_reply",
    # Web search
    "exa_search",
    "exa_fetch_full",
    "extract_url",
    # Notes
    "notes_search",
    "notes_list",
    "notes_get",
    "notes_read",
    "notes_add",
    # Literature
    "arxiv_search",
    "arxiv_get",
    "arxiv_fulltext",
    "pubmed_search",
    "pubmed_get",
    "pubmed_fulltext",
    "pubmed_related",
    "literature_search",
    "literature_fulltext",
    "openalex_search",
    # Calendar (read-only)
    "calendar_list_on_date",
    # Memory
    "memory_search",
    "memory_get",
    "memory_list",
    "memory_add",
    "memory_edit",
    "memory_delete",
    # Standing digest
    "digest_get",
    "digest_edit",
    # History (raw conversation history; executes locally — reads the
    # calling agent's history file via MESH_NODE_ID, no socket needed)
    "history_search",
    # Conversation todos
    "todo_list",
    "todo_add",
    "todo_update",
    "todo_toggle",
    "todo_remove",
    "todo_reorder",
    "todo_set_section_order",
    # Maps
    "map_get",
    "map_list",
    "map_edit",
    "map_create",
    "map_review",
    "set_project_context",
    # Mesh info
    "mesh_list",
    "mesh_status",
    "agent_status",
    "channel_list",
    "channel_members",
    # Messaging
    "send_message",
    # Utility
    "current_time",
    "tool_help",
    # Finance (read-only)
    "plaid_link_status",
    "plaid_accounts",
    "plaid_transactions",
    # Quota
    "synthetic_quota",
    "claude_code_usage",
    # Account
    "account_get_current",
    "account_list",
    "account_set_current",
    # Schedule (read-only)
    "schedule_list",
    # Personality
    "personality_get",
    # Canvas LMS
    "canvas_auth_status",
    "canvas_list_courses",
    "canvas_list_students",
    "canvas_list_assignments",
    "canvas_list_submissions",
    "canvas_get_grades",
    "canvas_list_announcements",
    "canvas_list_modules",
    "canvas_list_pages",
    "canvas_list_quizzes",
    "canvas_grade_submission",
    "canvas_post_announcement",
    "canvas_create_assignment",
    "canvas_create_module",
    "canvas_upload_file",
    "canvas_get_page",
    "canvas_update_page",
    "canvas_get_analytics",
    "canvas_list_module_items",
    "canvas_get_student",
    "canvas_create_page",
    # Worker
    "worker_stop",
})

# Tools whose registered handler is a placeholder — they need the agent's
# Unix socket to actually execute (the agent intercepts and routes them).
AGENT_ROUTED_TOOLS = {
    "send_message",
    "mesh_status",
    "agent_status",
    "channel_list",
    "channel_members",
    "schedule_list",
    "gmail_send_message",
    "gmail_reply_to",
    "account_set_current",
    "canvas_grade_submission",
    "canvas_post_announcement",
    "canvas_create_assignment",
    "canvas_create_module",
    "canvas_upload_file",
    "canvas_update_page",
    "canvas_create_page",
    "exa_search",
    "exa_fetch_full",
    "worker_stop",
    "todo_list",
    "todo_add",
    "todo_update",
    "todo_toggle",
    "todo_remove",
    "todo_reorder",
    "todo_set_section_order",
}


def _type_label(t: str) -> str:
    """Map JSON schema type to a CLI-friendly label."""
    return {"string": "TEXT", "integer": "INT", "number": "NUM",
            "boolean": "BOOL", "object": "JSON", "array": "JSON"}.get(t, t.upper())


def _coerce(value: str, param_type: str) -> Any:
    """Coerce a CLI string argument to the parameter's declared type."""
    if param_type == "integer":
        return int(value)
    if param_type == "number":
        return float(value)
    if param_type == "boolean":
        return value.lower() in ("true", "1", "yes")
    if param_type in ("object", "array"):
        return json.loads(value)
    return value


# ---------------------------------------------------------------------------
# Agent socket routing (for placeholder-handler tools)
# ---------------------------------------------------------------------------

async def _call_agent_socket(
    socket_path: str, name: str, arguments: dict, account: str | None = None,
) -> str:
    """Route a tool call to the running agent via its Unix socket."""
    import aiohttp
    payload: dict[str, Any] = {"name": name, "arguments": arguments}
    if account:
        payload["account"] = account
    connector = aiohttp.UnixConnector(path=socket_path)
    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.post(
            "http://localhost/tool",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            data = await resp.json()
            return data.get("result", json.dumps(data))


def _find_agent_socket() -> str | None:
    """Find the Unix socket for the agent identified by MESH_NODE_ID.

    Checks MESH_SOCKET_PATH first (set by agent node for worker subprocesses
    where HOME may be overridden), then falls back to path derivation.
    """
    explicit = os.environ.get("MESH_SOCKET_PATH", "")
    if explicit and os.path.exists(explicit):
        return explicit
    node_id = os.environ.get("MESH_NODE_ID", "")
    if not node_id:
        return None
    sock_name = f"{node_id.replace(':', '_')}.sock"
    # Primary: restricted directory under <real home>/.mesh/sockets/.
    # Real home comes from /etc/passwd (mesh.paths), NOT $HOME — $HOME is a
    # synthetic CC acct home when the caller runs under a CC session.
    from mesh.paths import real_home
    secure_sock = os.path.join(str(real_home()), ".mesh", "sockets", sock_name)
    if os.path.exists(secure_sock):
        return secure_sock
    # Compat: agents launched before the pathfix created their socket under
    # $HOME (possibly a synthetic CC home). Check the caller's $HOME-derived
    # path so workers of such agents still find it until the agent restarts.
    home_sock = os.path.join(os.path.expanduser("~"), ".mesh", "sockets", sock_name)
    if home_sock != secure_sock and os.path.exists(home_sock):
        return home_sock
    # Legacy fallback for agents started before the socket path change
    legacy_sock = f"/tmp/mesh_agent_{node_id.replace(':', '_')}.sock"
    if os.path.exists(legacy_sock):
        return legacy_sock
    return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_tool_list(registry) -> None:
    """Print all available CLI tools with one-line descriptions."""
    # Group by category
    categories = {
        "Gmail": ["gmail_list_recent", "gmail_list_unread", "gmail_search_emails",
                  "gmail_list_from_date", "gmail_get_email",
                  "gmail_send_message", "gmail_reply_to",
                  "gmail_create_draft", "gmail_draft_reply"],
        "Web Search": ["exa_search", "exa_fetch_full", "extract_url"],
        "Notes": ["notes_search", "notes_list", "notes_get", "notes_read", "notes_add"],
        "Literature": ["arxiv_search", "arxiv_get", "arxiv_fulltext",
                        "pubmed_search", "pubmed_get", "pubmed_fulltext",
                        "pubmed_related", "literature_search", "literature_fulltext",
                        "openalex_search"],
        "Calendar": ["calendar_list_on_date"],
        "Memory": ["memory_search", "memory_get", "memory_list", "memory_add",
                   "memory_edit", "memory_delete", "history_search"],
        "Digest": ["digest_get", "digest_edit"],
        "Todos": ["todo_list", "todo_add", "todo_update", "todo_toggle",
                  "todo_remove", "todo_reorder", "todo_set_section_order"],
        "Maps": ["map_get", "map_list", "map_edit", "map_create",
                 "map_review", "set_project_context"],
        "Mesh": ["mesh_list", "mesh_status", "agent_status",
                 "channel_list", "channel_members", "send_message"],
        "Canvas": ["canvas_auth_status", "canvas_list_courses", "canvas_list_students",
                   "canvas_get_student", "canvas_list_assignments", "canvas_list_submissions",
                   "canvas_get_grades", "canvas_list_announcements",
                   "canvas_list_modules", "canvas_list_module_items",
                   "canvas_list_pages", "canvas_get_page", "canvas_list_quizzes",
                   "canvas_get_analytics",
                   "canvas_grade_submission", "canvas_post_announcement",
                   "canvas_create_assignment", "canvas_create_module",
                   "canvas_create_page", "canvas_update_page",
                   "canvas_upload_file"],
        "Finance": ["plaid_link_status", "plaid_accounts", "plaid_transactions"],
        "Utility": ["current_time", "tool_help", "synthetic_quota",
                     "claude_code_usage", "account_get_current", "account_list",
                     "schedule_list", "personality_get"],
    }

    print("mesh-tool — shell access to mesh tools\n")
    print(f"  MESH_NODE_ID = {os.environ.get('MESH_NODE_ID', '(not set)')}\n")

    for cat, names in categories.items():
        available = [n for n in names if registry.get(n) is not None]
        if not available:
            continue
        print(f"  {cat}:")
        for name in available:
            td = registry.get(name)
            desc = td.description.split(".")[0].split("\n")[0]
            if len(desc) > 60:
                desc = desc[:57] + "..."
            print(f"    {name:<28s} {desc}")
        print()

    print("Run 'mesh-tool <name>' for usage, 'mesh-tool <name> --arg val' to call.")


def _print_tool_usage(tool_def) -> None:
    """Print usage for a single tool."""
    print(f"{tool_def.name} — {tool_def.description.split(chr(10))[0]}\n")

    if tool_def.description.count("\n") > 0:
        for line in tool_def.description.split("\n")[1:]:
            stripped = line.strip()
            if stripped:
                print(f"  {stripped}")
        print()

    required = [p for p in tool_def.parameters if p.required]
    optional = [p for p in tool_def.parameters if not p.required]

    if required:
        print("Required:")
        for p in required:
            print(f"  --{p.name:<20s} {_type_label(p.type):<6s}  {p.description}")

    if optional:
        print("Optional:")
        for p in optional:
            default_str = f" (default: {json.dumps(p.default)})" if p.default is not None else ""
            print(f"  --{p.name:<20s} {_type_label(p.type):<6s}  {p.description}{default_str}")

    if not tool_def.parameters:
        print("  (no parameters)")

    print(f"\nExample:\n  mesh-tool {tool_def.name}", end="")
    for p in required:
        print(f" --{p.name} <{_type_label(p.type).lower()}>", end="")
    print()


def _parse_args(argv: list[str], tool_def) -> dict[str, Any]:
    """Parse --key value pairs from argv into a dict, coercing types.

    Shell-safety: if a parameter value is literally "-", the value is read
    from stdin instead. Use with single-quoted heredocs to avoid shell
    interpolation of $, backticks, etc:
        mesh-tool gmail_send_message --to "x@y.com" --subject "Pay" --body - <<'EOF'
        The payment of $550 is outstanding.
        EOF
    """
    args: dict[str, Any] = {}
    i = 0
    known_params = {p.name: p for p in tool_def.parameters}

    while i < len(argv):
        token = argv[i]
        if not token.startswith("--"):
            print(f"Error: unexpected positional argument '{token}'", file=sys.stderr)
            print(f"Run 'mesh-tool {tool_def.name}' for usage.", file=sys.stderr)
            sys.exit(1)

        key = token[2:]

        if key == "help":
            _print_tool_usage(tool_def)
            sys.exit(0)

        if key not in known_params:
            # Suggest closest match
            from difflib import get_close_matches
            matches = get_close_matches(key, known_params.keys(), n=1, cutoff=0.5)
            hint = f" Did you mean --{matches[0]}?" if matches else ""
            print(f"Error: unknown argument '--{key}'.{hint}", file=sys.stderr)
            print(f"Run 'mesh-tool {tool_def.name}' for usage.", file=sys.stderr)
            sys.exit(1)

        param = known_params[key]

        # Boolean flags can be used without a value
        if param.type == "boolean":
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                i += 1
                args[key] = _coerce(argv[i], "boolean")
            else:
                args[key] = True
            i += 1
            continue

        if i + 1 >= len(argv):
            print(f"Error: --{key} requires a value.", file=sys.stderr)
            sys.exit(1)

        i += 1
        raw_value = argv[i]

        # Stdin mode: read value from stdin to avoid shell interpolation
        if raw_value == "-" and not sys.stdin.isatty():
            raw_value = sys.stdin.read().strip()

        try:
            args[key] = _coerce(raw_value, param.type)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Error: --{key} expected {_type_label(param.type)}, got '{raw_value}': {e}",
                  file=sys.stderr)
            sys.exit(1)
        i += 1

    # Check required parameters
    for p in tool_def.parameters:
        if p.required and p.name not in args:
            print(f"Error: missing required argument --{p.name}", file=sys.stderr)
            print(file=sys.stderr)
            _print_tool_usage(tool_def)
            sys.exit(1)

    return args


async def _execute(
    tool_def, args: dict[str, Any], account: str | None = None,
) -> str:
    """Execute a tool and return the result string."""
    name = tool_def.name
    sock = _find_agent_socket()

    # Agent-routed tools need the socket — their registered handlers are placeholders
    if name in AGENT_ROUTED_TOOLS:
        if sock:
            return await _call_agent_socket(sock, name, args, account=account)
        return (
            f"Error: {name} requires a running agent. "
            f"Set MESH_NODE_ID and ensure the agent is running."
        )

    # Memory and map tools need the agent's memory system (complex state).
    # Route through socket when available; fall through to local handler otherwise.
    if sock and (
        name.startswith("memory_")
        or name.startswith("map_")
        or name == "set_project_context"
    ):
        return await _call_agent_socket(sock, name, args)

    # Switch account for Gmail tools executed locally
    if account and name in ACCOUNT_TOOLS:
        from mesh.tool_implementations import _get_tool_host
        host = _get_tool_host()
        if host:
            host.set_current_account(account)

    # Execute via the registry's handler
    if tool_def.handler is None:
        return f"Error: tool '{name}' has no handler."

    if asyncio.iscoroutinefunction(tool_def.handler):
        result = await tool_def.handler(**args)
    else:
        result = tool_def.handler(**args)

    return str(result)


def main() -> None:
    # Suppress logging noise from imports
    import logging
    logging.basicConfig(level=logging.WARNING)

    # Import tool_implementations to register all tools
    import mesh.tool_implementations  # noqa: F401
    from mesh.tools import get_registry

    registry = get_registry()
    argv = sys.argv[1:]

    # No args → list tools
    if not argv:
        _print_tool_list(registry)
        sys.exit(0)

    tool_name = argv[0]

    # --help on the CLI itself
    if tool_name in ("--help", "-h", "help"):
        _print_tool_list(registry)
        sys.exit(0)

    # Look up the tool
    tool_def = registry.get(tool_name)
    if tool_def is None:
        # Suggest close matches
        from difflib import get_close_matches
        all_names = [n for n in registry.list_names() if n in CLI_TOOLS]
        matches = get_close_matches(tool_name, all_names, n=3, cutoff=0.4)
        print(f"Error: unknown tool '{tool_name}'.", file=sys.stderr)
        if matches:
            print(f"Did you mean: {', '.join(matches)}?", file=sys.stderr)
        print(f"\nRun 'mesh-tool' to see all available tools.", file=sys.stderr)
        sys.exit(1)

    if tool_name not in CLI_TOOLS:
        print(f"Error: '{tool_name}' is not available via mesh-tool.", file=sys.stderr)
        print(f"Run 'mesh-tool' to see available tools.", file=sys.stderr)
        sys.exit(1)

    remaining = argv[1:]

    # Extract --account before tool-specific arg parsing (Gmail + Calendar)
    account: str | None = None
    if tool_name in ACCOUNT_TOOLS and "--account" in remaining:
        idx = remaining.index("--account")
        if idx + 1 < len(remaining):
            account = remaining[idx + 1]
            remaining = remaining[:idx] + remaining[idx + 2:]
        else:
            print("Error: --account requires a value.", file=sys.stderr)
            sys.exit(1)

    # No args or --help → show usage
    if not remaining or remaining == ["--help"] or remaining == ["-h"]:
        has_required = any(p.required for p in tool_def.parameters)
        if not remaining and not has_required and remaining != ["--help"]:
            # Tool has no required params — execute it
            pass
        else:
            if remaining in (["--help"], ["-h"]) or (not remaining and has_required):
                _print_tool_usage(tool_def)
                sys.exit(0)

    # Parse arguments
    args = _parse_args(remaining, tool_def)

    # Execute
    try:
        result = asyncio.run(_execute(tool_def, args, account=account))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Output result
    is_error = result.startswith("Error:")
    try:
        parsed = json.loads(result)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except (json.JSONDecodeError, TypeError):
        out = sys.stderr if is_error else sys.stdout
        print(result, file=out)

    if is_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
