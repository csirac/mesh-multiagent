"""
CLI entry point for the standalone harness.

Usage:
    python -m mesh.harness exec \\
        --backend openai --model gpt-5 \\
        --prompt "Fix the bug in app.py" \\
        [--system-prompt "You are a helpful coding assistant."] \\
        [--system-prompt-file sys.txt] \\
        [--max-iters 50] \\
        [--soft-limit 500000] \\
        [--api-key sk-...] \\
        [--base-url https://api.openai.com/v1] \\
        [--effort high] \\
        [--thinking-budget 50000] \\
        [--tools bash_exec,file_read,file_edit,file_write] \\
        [--cwd /path/to/workdir]

All output goes to stdout as JSONL events (see protocol.py).
Logs go to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import socket as _sock_mod

from ..llm import LLMClient, LLMConfig, HistoryMessage
from ..tools import get_registry, ToolParameter
from . import protocol
from .loop import run_loop

logger = logging.getLogger(__name__)


def _query_mesh_tools(socket_path: str) -> list[dict]:
    """Query the parent agent_node's tool socket for available mesh tools."""
    s = _sock_mod.socket(_sock_mod.AF_UNIX, _sock_mod.SOCK_STREAM)
    try:
        s.connect(socket_path)
        s.settimeout(5.0)
        s.sendall(b"GET /tools HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        chunks = []
        while True:
            chunk = s.recv(8192)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        idx = raw.find(b"\r\n\r\n")
        if idx < 0:
            return []
        body = raw[idx + 4:]
        # Handle chunked transfer encoding
        if b"Transfer-Encoding: chunked" in raw[:idx]:
            decoded = _decode_chunked(body)
        else:
            decoded = body
        return json.loads(decoded).get("tools", [])
    except Exception as e:
        logger.warning("Failed to query mesh tools from agent socket: %s", e)
        return []
    finally:
        s.close()


def _decode_chunked(data: bytes) -> bytes:
    """Decode HTTP chunked transfer encoding."""
    result = []
    pos = 0
    while pos < len(data):
        end = data.find(b"\r\n", pos)
        if end < 0:
            break
        size = int(data[pos:end], 16)
        if size == 0:
            break
        pos = end + 2
        result.append(data[pos:pos + size])
        pos += size + 2
    return b"".join(result)

DEFAULT_TOOLS_LEGACY = [
    "bash_exec",
    "file_read",
    "file_edit",
    "file_write",
    "file_diff",
    "set_working_directory",
    "get_working_directory",
]

DEFAULT_TOOLS_HARNESS = [
    "apply_patch",
    "shell",
    "file_read",
    "file_edit",
    "list_dir",
]


def build_llm_config(args: argparse.Namespace) -> LLMConfig:
    """Build LLMConfig from CLI arguments."""
    config = LLMConfig(
        backend=args.backend,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if args.api_key:
        config.api_key = args.api_key
        if args.backend == "anthropic":
            config.anthropic_api_key = args.api_key
        elif args.backend == "zai":
            config.zai_api_key = args.api_key
    if args.base_url:
        config.base_url = args.base_url
        if args.backend == "anthropic":
            config.anthropic_base_url = args.base_url
    if args.cc_binary:
        config.cc_binary = args.cc_binary
    if args.effort:
        config.cc_effort = args.effort
        config.reasoning_effort = args.effort  # type: ignore
    # Thinking budget: applies to both Anthropic and OpenAI-compatible (Qwen, etc.)
    if args.thinking_budget:
        config.anthropic_thinking_budget = args.thinking_budget
        config.thinking_budget = args.thinking_budget
    elif args.effort and args.backend == "anthropic":
        _effort_to_budget = {"low": 2000, "medium": 5000, "high": 10000, "xhigh": 50000}
        config.anthropic_thinking_budget = _effort_to_budget.get(args.effort)
    return config


def build_history(args: argparse.Namespace) -> list[HistoryMessage]:
    """Build initial conversation history from CLI arguments."""
    history: list[HistoryMessage] = []

    if args.history_file:
        with open(args.history_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                history.append(HistoryMessage(
                    from_node=obj.get("from", "user"),
                    content=obj["content"],
                    timestamp=obj.get("timestamp", ""),
                    source="persisted",
                ))

    # The prompt becomes the final user message
    if args.prompt:
        from datetime import datetime, timezone
        history.append(HistoryMessage(
            from_node="user",
            content=args.prompt,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="persisted",
        ))

    return history


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m mesh.harness",
        description="Standalone LLM auto-tool loop harness",
    )
    sub = parser.add_subparsers(dest="command")

    exec_p = sub.add_parser("exec", help="Run the tool loop")

    # LLM backend
    exec_p.add_argument("--backend", default="openai",
                        choices=["openai", "anthropic", "google", "claude-code"],
                        help="LLM backend (clean-room only, or claude-code for CC subprocess)")
    exec_p.add_argument("--model", default="gpt-4", help="Model name")
    exec_p.add_argument("--max-tokens", type=int, default=16384, help="Max output tokens")
    exec_p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    exec_p.add_argument("--api-key", default="", help="API key (or set env var)")
    exec_p.add_argument("--base-url", default="", help="API base URL override")
    exec_p.add_argument("--effort", default="", help="Reasoning effort (low/medium/high/xhigh)")
    exec_p.add_argument("--thinking-budget", type=int, default=0,
                        help="Anthropic thinking budget tokens (overrides --effort mapping)")
    exec_p.add_argument("--cc-binary", default="",
                        help="Path to Claude Code binary (for --backend claude-code)")

    # Prompt — pass "-" or omit to read from stdin (avoids OS argv length limit)
    exec_p.add_argument("--prompt", default="",
                        help="User prompt / task description. Use '-' or omit to read from stdin.")
    exec_p.add_argument("--system-prompt", default="", help="System prompt text")
    exec_p.add_argument("--system-prompt-file", default="", help="Read system prompt from file")
    exec_p.add_argument("--history-file", default="", help="JSONL file with prior conversation turns")

    # Tools
    exec_p.add_argument("--toolset", default="harness",
                        choices=["harness", "legacy"],
                        help="Tool set: 'harness' (codex-style) or 'legacy' (mesh-style)")
    exec_p.add_argument("--tools", default="",
                        help="Comma-separated tool names (overrides --toolset)")

    # Loop config
    exec_p.add_argument("--max-iters", type=int, default=50, help="Max loop iterations")
    exec_p.add_argument("--soft-limit", type=int, default=500_000, help="Token soft limit for context pruning")
    exec_p.add_argument("--node-id", default="harness", help="Agent identity string")

    # Agent socket (for routing agent-local tools to parent process)
    exec_p.add_argument("--agent-socket", default="",
                        help="Unix socket path for routing agent-local tools to parent agent_node")

    # Controller mode
    exec_p.add_argument("--controller-mode", default="standard",
                        choices=["standard", "plan_and_execute", "decompose"],
                        help="Controller mode: 'standard' (TAOR), 'plan_and_execute', or 'decompose'")

    # Assessor LLM (separate model for judge calls: triage, checkpoint, assessment)
    exec_p.add_argument("--assessor-backend", default="",
                        help="Assessor LLM backend (defaults to --backend)")
    exec_p.add_argument("--assessor-model", default="",
                        help="Assessor model name (defaults to --model)")
    exec_p.add_argument("--assessor-api-key", default="",
                        help="Assessor API key (defaults to --api-key)")
    exec_p.add_argument("--assessor-base-url", default="",
                        help="Assessor base URL (defaults to --base-url)")
    exec_p.add_argument("--assessor-effort", default="",
                        help="Assessor reasoning effort (defaults to --effort)")

    # Codex assessor (subprocess-based controller using codex exec)
    exec_p.add_argument("--codex-assessor", action="store_true",
                        help="Use codex exec as the assessor (subprocess, not chat API)")
    exec_p.add_argument("--codex-assessor-binary", default="",
                        help="Path to codex binary (auto-detected if empty)")
    exec_p.add_argument("--codex-assessor-model", default="o3",
                        help="Model for codex assessor (default: o3)")
    exec_p.add_argument("--codex-assessor-effort", default="high",
                        help="Reasoning effort for codex assessor")

    # Working directory
    exec_p.add_argument("--cwd", default="", help="Working directory for tool execution")

    # -------------------------------------------------------------------------
    # session: persistent interactive session (stays alive between turns)
    # -------------------------------------------------------------------------
    session_p = sub.add_parser(
        "session",
        help="Run a persistent interactive session driven by JSONL stdin commands",
    )
    session_p.add_argument("--backend", default="openai",
                           choices=["openai", "anthropic", "google", "claude-code"],
                           help="LLM backend")
    session_p.add_argument("--model", default="gpt-4", help="Model name")
    session_p.add_argument("--max-tokens", type=int, default=16384, help="Max output tokens")
    session_p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    session_p.add_argument("--api-key", default="", help="API key (or set env var)")
    session_p.add_argument("--base-url", default="", help="API base URL override")
    session_p.add_argument("--effort", default="", help="Reasoning effort (low/medium/high/xhigh)")
    session_p.add_argument("--thinking-budget", type=int, default=0,
                           help="Thinking budget tokens (overrides --effort mapping)")
    session_p.add_argument("--cc-binary", default="", help="Path to Claude Code binary")
    # The initial task. Use "-" or omit to read from stdin's FIRST line as a
    # plain string (NOT JSON) — convenient for manual launches.
    session_p.add_argument("--task", default="",
                           help="Initial task description. '-' or omit reads the first stdin line.")
    session_p.add_argument("--system-prompt", default="", help="System prompt text")
    session_p.add_argument("--system-prompt-file", default="", help="Read system prompt from file")
    session_p.add_argument("--toolset", default="harness",
                           choices=["harness", "legacy"], help="Tool set")
    session_p.add_argument("--tools", default="", help="Comma-separated tool names (overrides --toolset)")
    session_p.add_argument("--max-iters", type=int, default=100,
                           help="Max loop iterations per resumed run before yielding to the driver")
    session_p.add_argument("--soft-limit", type=int, default=200_000,
                           help="Token soft limit for context budget")
    session_p.add_argument("--node-id", default="harness-session", help="Agent identity string")
    session_p.add_argument("--agent-socket", default="",
                           help="Unix socket path for routing agent-local tools to parent agent_node")
    session_p.add_argument("--checkpoint-interval", type=int, default=0,
                           help="Yield a session.checkpoint every N iterations (0 = free-running)")
    session_p.add_argument("--cwd", default="", help="Working directory for tool execution")

    return parser.parse_args(argv)


def _resolve_session_tools(args: argparse.Namespace) -> list[str]:
    """Resolve the tool list for a session, discovering mesh tools via the
    agent socket (mirrors the exec path)."""
    if args.tools:
        tool_names = [t.strip() for t in args.tools.split(",") if t.strip()]
    elif args.toolset == "harness":
        tool_names = list(DEFAULT_TOOLS_HARNESS)
    else:
        tool_names = list(DEFAULT_TOOLS_LEGACY)

    if args.agent_socket:
        mesh_tool_defs = _query_mesh_tools(args.agent_socket)
        registry = get_registry()
        existing = set(tool_names)
        added = []
        for td in mesh_tool_defs:
            name = td["name"]
            if name in existing:
                continue
            if registry.get(name) is None:
                params = [
                    ToolParameter(
                        name=p["name"], type=p["type"],
                        description=p.get("description", ""),
                        required=p.get("required", True),
                        default=p.get("default"),
                    )
                    for p in td.get("parameters", [])
                ]
                registry.register(name, td.get("description", ""), params, handler=None)
            tool_names.append(name)
            added.append(name)
        logger.info("Session mesh tools: discovered=%d added=%d", len(mesh_tool_defs), len(added))
    else:
        logger.info("Session: no agent socket — mesh tools unavailable")
    return tool_names


def _run_session(args: argparse.Namespace) -> None:
    """Entry point for the persistent `session` subcommand."""
    from .session import run_session_loop

    ALLOWED_BACKENDS = {"openai", "anthropic", "google", "claude-code"}
    if args.backend not in ALLOWED_BACKENDS:
        print(f"Error: backend '{args.backend}' is not allowed.", file=sys.stderr)
        sys.exit(2)

    # Read system prompt before chdir (path may be relative to launch dir).
    system_prompt = args.system_prompt
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text()

    # Initial task is OPTIONAL. When provided via --task, it seeds the first
    # user turn and the session starts working immediately. When omitted, the
    # session starts in WARM_IDLE and waits for the driver's first
    # {"type":"task"} command on stdin — this is the path the manager uses,
    # avoiding OS argv length limits for large task descriptions.
    initial_task = "" if args.task == "-" else args.task

    # Trigger tool registrations (legacy first, harness wins on collision).
    import mesh.tool_implementations  # noqa: F401
    import mesh.harness.tools  # noqa: F401

    use_harness = args.toolset == "harness" and not args.tools
    if args.cwd:
        os.chdir(args.cwd)
        if not use_harness:
            from mesh.tool_implementations import set_working_directory
            set_working_directory(args.cwd)

    llm_config = build_llm_config(args)
    llm_client = LLMClient(llm_config)

    tool_names = _resolve_session_tools(args)

    logger.info(
        "Harness SESSION: backend=%s model=%s toolset=%s tools=%d "
        "soft_limit=%d checkpoint_interval=%d agent_socket=%s",
        args.backend, args.model, args.toolset, len(tool_names),
        args.soft_limit, args.checkpoint_interval, bool(args.agent_socket),
    )

    asyncio.run(run_session_loop(
        llm_client=llm_client,
        system_prompt=system_prompt,
        tool_registry=get_registry(),
        tool_names=tool_names,
        initial_task=initial_task,
        node_id=args.node_id,
        max_iterations=args.max_iters,
        soft_limit=args.soft_limit,
        agent_socket_path=args.agent_socket or None,
        checkpoint_interval=args.checkpoint_interval,
    ))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    args = parse_args(argv)

    if args.command == "session":
        _run_session(args)
        return

    if args.command != "exec":
        print("Usage: python -m mesh.harness exec --prompt '...' [options]", file=sys.stderr)
        sys.exit(1)

    # Read prompt from stdin when omitted or "-" (avoids OS argv length limit for large prompts)
    if not args.prompt or args.prompt == "-":
        args.prompt = sys.stdin.read()

    ALLOWED_BACKENDS = {"openai", "anthropic", "google", "claude-code"}
    if args.backend not in ALLOWED_BACKENDS:
        print(
            f"Error: backend '{args.backend}' is not allowed in the harness.\n"
            f"Allowed backends: {', '.join(sorted(ALLOWED_BACKENDS))}.\n"
            f"For z.ai: use --backend anthropic --base-url https://api.z.ai/api/anthropic/v1",
            file=sys.stderr,
        )
        sys.exit(2)

    # Read system prompt before chdir (path may be relative to launch dir)
    system_prompt = args.system_prompt
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text()

    # Import tools to trigger @tool registrations.
    # Legacy tools first, then harness tools (so harness versions win on name collisions).
    use_harness = args.toolset == "harness" and not args.tools
    import mesh.tool_implementations  # noqa: F401
    import mesh.harness.tools  # noqa: F401

    # Set working directory
    if args.cwd:
        os.chdir(args.cwd)
        if not use_harness:
            from mesh.tool_implementations import set_working_directory
            set_working_directory(args.cwd)

    # Build LLM config and client
    llm_config = build_llm_config(args)
    llm_client = LLMClient(llm_config)

    # Build separate assessor client when any assessor flag is provided
    assessor_llm_client = None
    if args.assessor_backend or args.assessor_model:
        assessor_config = LLMConfig(
            backend=args.assessor_backend or args.backend,
            model=args.assessor_model or args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        a_backend = assessor_config.backend
        a_key = args.assessor_api_key or args.api_key
        if a_key:
            assessor_config.api_key = a_key
            if a_backend == "anthropic":
                assessor_config.anthropic_api_key = a_key
        a_url = args.assessor_base_url or args.base_url
        if a_url:
            assessor_config.base_url = a_url
            if a_backend == "anthropic":
                assessor_config.anthropic_base_url = a_url
        a_effort = args.assessor_effort or args.effort
        if a_effort:
            assessor_config.cc_effort = a_effort
            assessor_config.reasoning_effort = a_effort  # type: ignore
        assessor_llm_client = LLMClient(assessor_config)
        logger.info("Assessor LLM: backend=%s model=%s", a_backend, assessor_config.model)

    # Resolve tool list
    if args.tools:
        tool_names = [t.strip() for t in args.tools.split(",") if t.strip()]
    elif use_harness:
        tool_names = list(DEFAULT_TOOLS_HARNESS)
    else:
        tool_names = list(DEFAULT_TOOLS_LEGACY)

    # When agent socket is available, discover mesh tools and register stubs
    if args.agent_socket:
        mesh_tool_defs = _query_mesh_tools(args.agent_socket)
        registry = get_registry()
        existing = set(tool_names)
        added_mesh = []
        for td in mesh_tool_defs:
            name = td["name"]
            if name in existing:
                continue
            if registry.get(name) is None:
                params = [
                    ToolParameter(
                        name=p["name"], type=p["type"],
                        description=p.get("description", ""),
                        required=p.get("required", True),
                        default=p.get("default"),
                    )
                    for p in td.get("parameters", [])
                ]
                registry.register(name, td.get("description", ""), params, handler=None)
            tool_names.append(name)
            added_mesh.append(name)
        logger.info(
            "Mesh tools from agent socket: discovered=%d, added=%d, skipped=%d (collisions with base toolset)",
            len(mesh_tool_defs), len(added_mesh), len(mesh_tool_defs) - len(added_mesh),
        )
        logger.info("Added mesh tools: %s", sorted(added_mesh))
    else:
        logger.info("No agent socket — mesh tools unavailable")

    # Startup banner: log full configuration for diagnosability
    logger.info(
        "Harness config: backend=%s model=%s toolset=%s controller=%s "
        "soft_limit=%d tools=%d agent_socket=%s",
        args.backend, args.model, args.toolset, args.controller_mode,
        args.soft_limit, len(tool_names), bool(args.agent_socket),
    )
    if assessor_llm_client:
        logger.info(
            "Assessor config: backend=%s model=%s",
            args.assessor_backend or args.backend,
            args.assessor_model or args.model,
        )

    # Build codex assessor config if requested
    codex_assessor_config = None
    if getattr(args, 'codex_assessor', False):
        codex_assessor_config = {
            "binary": args.codex_assessor_binary,
            "model": args.codex_assessor_model,
            "effort": args.codex_assessor_effort,
            "cwd": args.cwd or os.getcwd(),
        }
        logger.info(
            "Codex assessor config: binary=%s model=%s effort=%s",
            codex_assessor_config["binary"],
            codex_assessor_config["model"],
            codex_assessor_config["effort"],
        )

    # Build history
    history = build_history(args)

    if not history:
        print("Error: --prompt is required", file=sys.stderr)
        sys.exit(1)

    # Run the loop
    final = asyncio.run(run_loop(
        llm_client=llm_client,
        history=history,
        system_prompt=system_prompt,
        tool_registry=get_registry(),
        tool_names=tool_names,
        node_id=args.node_id,
        max_iterations=args.max_iters,
        soft_limit=args.soft_limit,
        agent_socket_path=args.agent_socket or None,
        controller_mode=args.controller_mode,
        assessor_llm_client=assessor_llm_client,
        codex_assessor_config=codex_assessor_config,
    ))

    # Exit 0 if we got a final response, 1 if empty (likely error)
    sys.exit(0 if final else 1)


if __name__ == "__main__":
    main()
